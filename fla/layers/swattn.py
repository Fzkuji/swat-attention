# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import math
import warnings
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from transformers.utils import logging


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """  
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


from fla.layers.utils import pad_input, unpad_input
from fla.modules import RMSNorm, RotaryEmbedding
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from fla.models.utils import Cache

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    warnings.warn(
        "Flash Attention is not installed. Please install it via `pip install flash-attn --no-build-isolation`",
        category=ImportWarning
    )
    flash_attn_func = None

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
            use_learnable_bias: bool = True,  # 是否使用可学习的位置bias
            max_bias_length: int = 1024,  # bias矩阵的最大尺寸
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        if num_kv_heads is None:
            self.num_kv_heads = self.num_heads
        else:
            self.num_kv_heads = num_kv_heads
        self.num_kv_groups = num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm

        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx
        self.use_learnable_bias = use_learnable_bias
        self.max_bias_length = max_bias_length

        # No longer requiring flash attention
        # Don't pre-allocate mask to save memory
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # 初始化可学习的位置bias矩阵
        if self.use_learnable_bias:
            self._init_learnable_position_bias()

            # 每个头的softmax归一化偏置（加到分母上）
        # 初始化为0，保持标准softmax行为
        self.softmax_offset = nn.Parameter(torch.full((self.num_heads,), 0.0))

        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

    def apply_learnable_causal_avg_bias(self, attn_scores):
        """
        构造带可学习缩放系数的前缀平均分布。
        """
        batch_size, num_heads, seq_len_q, seq_len_k = attn_scores.shape
        device = attn_scores.device
        dtype = attn_scores.dtype

        # 构造前缀平均矩阵 [seq_len_q, seq_len_k]
        causal_avg = torch.zeros(seq_len_q, seq_len_k, device=device, dtype=dtype)
        for i in range(seq_len_q):
            causal_avg[i, :i + 1] = 1.0 / (i + 1)

        # 扩展维度 [1, 1, seq_len_q, seq_len_k]
        causal_avg = causal_avg.unsqueeze(0).unsqueeze(0)

        # 每个 head 的可学习缩放因子 [num_heads]
        offset = torch.abs(self.softmax_offset).view(1, num_heads, 1, 1)

        # 应用缩放
        causal_avg = causal_avg * offset

        # broadcast 到 batch
        causal_avg = causal_avg.expand(batch_size, num_heads, -1, -1)

        return causal_avg

    def _init_learnable_position_bias(self):
        """初始化可学习的位置bias参数，每个头都有独立的对角线参数（仅用于causal attention）"""
        # 创建可学习的对角线参数：[num_heads, max_bias_length]
        # 存储主对角线和所有下对角线的值
        self.learnable_bias_diagonals = nn.Parameter(
            torch.zeros(self.num_heads, self.max_bias_length)
        )

        # 用很小的随机值初始化 (例如正态分布, std=1e-3)
        nn.init.normal_(self.learnable_bias_diagonals, mean=0.0, std=1e-3)
        # # 使用ALiBi初始化
        # self._init_with_alibi_diagonals()

    def get_learnable_bias(self):
        """获取可学习的对角线bias参数"""
        # 直接返回对角线参数
        # [num_heads, max_bias_length]
        return self.learnable_bias_diagonals

    def apply_learnable_bias_efficient(self, attn_weights):
        """高效地应用对角线 bias，使用GPU并行操作，避免大内存占用"""
        batch_size, num_heads, seq_len_q, seq_len_k = attn_weights.shape

        # 一次性创建相对位置矩阵
        rel_pos = torch.arange(seq_len_q, device=attn_weights.device)[:, None] - \
                  torch.arange(seq_len_k, device=attn_weights.device)[None, :]

        # 创建有效位置mask (causal + 距离限制)
        valid_mask = (0 <= rel_pos) & (rel_pos < self.max_bias_length)

        # 限制索引范围并获取bias值
        indices = rel_pos.clamp(0, self.max_bias_length - 1)
        bias = self.learnable_bias_diagonals[:, indices] * valid_mask.to(attn_weights.dtype)

        # 直接广播加到attention weights上
        return attn_weights + bias[None, :, :, :]

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            output_attentions: bool = True,
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
        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        # Regular attention implementation (no flash attention)
        # Reshape for attention computation: (B, T, num_heads, head_dim) -> (B, num_heads, T, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Handle grouped multi-query attention using repeat_kv
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling

        # 应用可学习的位置bias
        if self.use_learnable_bias:
            attn_scores = self.apply_learnable_bias_efficient(attn_scores)

        # Apply attention mask if provided
        if attention_mask is not None:
            # attention_mask should be a binary mask: 1 = can attend, 0 = cannot attend
            causal_mask = attention_mask[:, :, :, :k.shape[-2]]
            # Use masked_fill: where mask is 0, set scores to -inf
            attn_scores = attn_scores.masked_fill(causal_mask == 0, float('-inf'))

        # Apply softmax with float32 for numerical stability
        attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(q.dtype)

        # 取绝对值确保非负
        offset = torch.abs(self.softmax_offset + 1).view(1, self.num_heads, 1, 1)

        # 每个位置的 adaptive offset
        positions = torch.arange(attn_weights.shape[-1], device=attn_weights.device)
        num_visible_tokens = positions.unsqueeze(0) + 1
        num_visible_tokens = num_visible_tokens.view(1, 1, -1, 1)
        adaptive_offset = offset / num_visible_tokens.float()

        # 应用 offset + ReLU
        attn_weights = F.relu(attn_weights - adaptive_offset)

        attentions = attn_weights

        # Apply attention to values
        # 确保 attn_weights 和 v dtype 一致
        attn_weights = attn_weights.to(v.dtype)
        attn_output = torch.matmul(attn_weights, v)

        # Reshape back: (B, num_heads, T, head_dim) -> (B, T, num_heads, head_dim) -> (B, T, hidden_size)
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(batch_size, q_len, -1)
        o = self.o_proj(attn_output)

        return o, attentions, past_key_values