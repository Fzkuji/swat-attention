# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# Modified to use Lazy Attention Triton kernel

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.utils import logging

# Import Lazy Attention Triton kernel from flash-lazy-attention repo
import sys
sys.path.insert(0, '/home/fzkuji/PycharmProjects/flash-lazy-attention')
try:
    from lazy_attention_triton import lazy_attention_triton
    HAS_LAZY_ATTENTION = True
except ImportError:
    warnings.warn(
        "flash-lazy-attention is not available. Please check the path.",
        category=ImportWarning
    )
    lazy_attention_triton = None
    HAS_LAZY_ATTENTION = False


from fla.layers.utils import pad_input, unpad_input
from fla.modules import RMSNorm, RotaryEmbedding
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from fla.models.utils import Cache

logger = logging.get_logger(__name__)


class SWAttention(nn.Module):

    def __init__(
            self,
            hidden_size: int = 2048,
            num_heads: int = 32,
            num_kv_heads: Optional[int] = None,
            qkv_bias: bool = False,
            qk_norm: bool = False,
            window_size: Optional[int] = None,
            rope_theta: Optional[float] = 10000.,
            max_position_embeddings: Optional[int] = None,
            layer_idx: int = None,
            use_learnable_bias: bool = True,  # 保留用于兼容性，Lazy Attention 总是使用可学习 bias
            max_bias_length: int = 1024,  # bias 窗口大小
    ):
        super().__init__()

        if not HAS_LAZY_ATTENTION:
            raise ImportError("flash-lazy-attention is not available. Please check the path.")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm

        self.window_size = window_size  # 滑动窗口大小（可选）
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx
        self.max_bias_length = max_bias_length  # 可学习bias的最大长度

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # Lazy Attention 的可学习参数（与scratch分支保持一致）
        # 位置 bias: [num_heads, max_bias_length]
        # 距离范围 [0, max_bias_length)
        self.learnable_bias_diagonals = nn.Parameter(
            torch.zeros(self.num_heads, self.max_bias_length)
        )
        nn.init.normal_(self.learnable_bias_diagonals, mean=0.0, std=1e-3)

        # Elastic-Softmax 的 τ 参数: [num_heads]
        # τ_init = -1，对应论文公式 ReLU(Softmax + τ/i)
        self.tau = nn.Parameter(torch.full((self.num_heads,), -1.01))

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            output_attentions: bool = False,  # Lazy Attention 不返回 attention weights
            use_cache: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        batch_size, q_len, _ = hidden_states.size()

        q = rearrange(self.q_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        k = rearrange(self.k_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)
        v = rearrange(self.v_proj(hidden_states), '... (h d) -> ... h d', d=self.head_dim)

        if self.qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

        # equivalent to cu_seqlens in `flash_attn`
        cu_seqlens = kwargs.get('cu_seqlens', None)

        seqlen_offset, max_seqlen = 0, q_len
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)
            max_seqlen = q.shape[1] + seqlen_offset

            if attention_mask is not None:
                # to deliminate the offsets of padding tokens
                seqlen_offset = seqlen_offset + prepare_lens_from_mask(attention_mask) - attention_mask.shape[-1]
                max_seqlen = q.shape[1] + max(seqlen_offset)

        if self.max_position_embeddings is not None:
            max_seqlen = max(max_seqlen, self.max_position_embeddings)

        # Apply RoPE
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        # Handle GQA (Grouped Query Attention)
        # Lazy Attention requires all heads to have matching dimensions
        if self.num_kv_groups > 1:
            # [B, L, num_kv_heads, D] -> [B, L, num_heads, D]
            k = k.repeat_interleave(self.num_kv_groups, dim=2)
            v = v.repeat_interleave(self.num_kv_groups, dim=2)

        # Convert shape: [B, L, H, D] -> [B, H, L, D] for Lazy Attention
        q = rearrange(q, 'b l h d -> b h l d')
        k = rearrange(k, 'b l h d -> b h l d')
        v = rearrange(v, 'b l h d -> b h l d')

        # Call Lazy Attention Triton kernel
        # Triton kernel 内部已经实现了 causal mask，不需要传入 varlen
        # window_size 从 bias.shape[1] 自动推断
        attn_output = lazy_attention_triton(
            q, k, v,
            bias=self.learnable_bias_diagonals.to(q.dtype),
            tau=self.tau.to(q.dtype),
        )

        # Convert back: [B, H, L, D] -> [B, L, H*D]
        attn_output = rearrange(attn_output, 'b h l d -> b l (h d)')
        o = self.o_proj(attn_output)

        # Lazy Attention does not return attention weights
        attentions = None

        return o, attentions, past_key_values