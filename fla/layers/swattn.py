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
from fla.modules.fused_attention import attention
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
            window_size_for_bias: Optional[int] = None,  # sliding window size for DoG attention
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
        self.window_size_for_bias = window_size_for_bias

        # No longer requiring flash attention
        # Don't pre-allocate mask to save memory

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        # 初始化DoG bias参数和attention sinks
        self._init_dog_parameters()

    def _init_dog_parameters(self):
        """初始化DoG bias参数和attention sinks"""
        # DoG参数: 每个头有7个参数 (A1, s1, c1, A2, s2, c2, offset)
        # 形状: [num_heads, 7]
        self.dog_params = nn.Parameter(torch.randn(self.num_heads, 7, dtype=torch.float32))

        # 初始化DoG参数
        with torch.no_grad():
            # A1, A2: 振幅参数，初始化为小的随机值
            self.dog_params[:, 0] = torch.randn(self.num_heads) * 0.01  # A1
            self.dog_params[:, 3] = torch.randn(self.num_heads) * 0.01  # A2

            # s1, s2: 标准差，需要保持正值
            self.dog_params[:, 1] = torch.abs(torch.randn(self.num_heads)) * 5.0  # s1
            self.dog_params[:, 4] = torch.abs(torch.randn(self.num_heads)) * 10.0  # s2

            # c1, c2: 中心位置
            self.dog_params[:, 2] = torch.randn(self.num_heads) * 2.0 + 3.0  # c1
            self.dog_params[:, 5] = torch.randn(self.num_heads) * 8.0 + 2.0  # c2

            # offset: 偏置
            self.dog_params[:, 6] = torch.randn(self.num_heads) * 0.001  # offset

        # Attention sinks: 每个头一个可学习的sink值
        self.attention_sinks = nn.Parameter(torch.zeros(self.num_heads))

    def get_dog_parameters(self):
        """获取DoG参数"""
        return self.dog_params if hasattr(self, 'dog_params') else None

    def get_attention_sinks(self):
        """获取attention sinks"""
        return self.attention_sinks if hasattr(self, 'attention_sinks') else None

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

        # 使用DoG bias attention
        # 转换输入形状以匹配 DoG attention 的期望格式
        # q: (B, T, num_kv_heads, num_kv_groups, head_dim)
        # k, v: (B, T, num_kv_heads, head_dim)
        q_dog = q.reshape(batch_size, q_len, self.num_kv_heads, self.num_kv_groups, self.head_dim)
        k_dog = k
        v_dog = v

        # Create a transformed version of DoG parameters for the forward pass.
        # This keeps the original parameters unconstrained and maintains the gradient flow.
        dog_params_transformed = self.dog_params.clone()

        # Apply softplus to s1 and s2 indices (1 and 4) to ensure they are positive.
        # Adding a small epsilon for numerical stability is a good practice.
        dog_params_transformed[:, 1] = F.softplus(self.dog_params[:, 1])  # s1
        dog_params_transformed[:, 4] = F.softplus(self.dog_params[:, 4])  # s2

        # 设置参数
        start_q = torch.tensor([0], dtype=torch.int32, device=hidden_states.device)

        # 调用DoG attention (注意: autograd.Function.apply不支持关键字参数)
        attn_output = attention(
            q_dog, k_dog, v_dog,
            self.attention_sinks,
            dog_params_transformed,
            self.scaling,
            self.window_size_for_bias,
            start_q
        )

        # attn_output已经是 (B, T, hidden_size) 形状
        attentions = None  # DoG attention 不返回attention weights
        o = self.o_proj(attn_output)

        return o, attentions, past_key_values