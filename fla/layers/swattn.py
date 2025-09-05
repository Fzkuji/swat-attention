# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Optional, Tuple

import torch
import torch.nn as nn
from einops import rearrange
from transformers.utils import logging

from fla.layers.utils import pad_input, unpad_input
from fla.modules import RMSNorm, RotaryEmbedding
from fla.ops.utils.index import prepare_lens_from_mask

if TYPE_CHECKING:
    from fla.models.utils import Cache


logger = logging.get_logger(__name__)


def get_alibi_slope(num_heads):
    x = (2 ** 8) ** (1 / num_heads)
    return (
        torch.tensor([ 1 / x ** (i + 1) for i in range(num_heads)])
    )


# def get_alibi_slope(num_heads):
#     # 正值部分的斜率：衰减得快
#     pos_heads = int(num_heads * 3 / 4)

#     x_pos = (2 ** 8) ** (1 / num_heads)
#     pos_slopes = torch.tensor([1 / x_pos ** (i + 1) for i in range(pos_heads)])

#     # 负值部分的斜率：衰减得慢
#     neg_heads = num_heads - pos_heads

#     x_neg = (2 ** 8) ** (1 / num_heads)
#     neg_slopes = torch.tensor([-1 / x_neg ** (i + 1) for i in range(neg_heads)])

#     # 拼接正负 slope，形成不对称结构
#     full_slopes = torch.cat([pos_slopes, neg_slopes])

#     return full_slopes

# def get_alibi_slope(num_heads):
#     pos_heads = num_heads // 2          # 4
#     k = 10
#     x_pos = (2 ** k) ** (1 / num_heads) # ≈ 1.6726

#     # 0.018997257 是 RoPE 的最大瞬时衰减斜率 s_max
#     s_max = 1.8997257e-02
#     pos_slopes = torch.tensor([s_max / x_pos ** i for i in range(pos_heads)])

#     neg_slopes = -pos_slopes            # 保持对称
#     return torch.cat([pos_slopes, neg_slopes.flip(0)]).to(torch.bfloat16)


# def get_alibi_slope(num_heads):
#     # 计算标准 ALiBi slopes (num_heads-2 个)
#     n_alibi_heads = num_heads - 2
#     x = (2 ** 8) ** (1 / n_alibi_heads)

#     # 生成标准的 ALiBi slopes
#     alibi_slopes = torch.tensor([1 / x ** (i + 1) for i in range(n_alibi_heads)])

#     # 在开头添加 0，在末尾添加 2
#     slopes = torch.cat([
#         torch.tensor([0.0]),      # 第一个头使用 slope=0（无位置偏置）
#         alibi_slopes,             # 中间的头使用标准 ALiBi slopes
#         torch.tensor([2.0])       # 最后一个头使用 slope=2（强位置偏置）
#     ])

#     return slopes


# def get_alibi_slope(num_heads):  # 48.01 16 heads   / 32 heads 2048 3.0517
#     assert num_heads % 2 == 0, "num_heads 必须是偶数"
#     n_half = num_heads // 2

#     # 前一半：标准 ALiBi slope
#     x = (2 ** 8) ** (1 / n_half)
#     front_half = torch.tensor([2 / x ** (i + 1) for i in range(n_half)])

#     # 后一半：固定值 + 补零
#     fixed_values = [10, 8, 6.0, 4.0, 3.0, 2]
#     back_half = torch.zeros(n_half)
#     for i in range(n_half):
#         if i < len(fixed_values):
#             back_half[i] = fixed_values[i]
#         else:
#             back_half[i] = 0.0

#     # 拼接成完整的 slope 向量
#     slopes = torch.cat([front_half, back_half])
#     return slopes


# def get_alibi_slope(num_heads):  # 47.7925
#     assert num_heads == 16, "当前函数只支持 num_heads=16"
#     return torch.tensor([
#         0.08, 0.04, 0.02, 0.01,
#         0.005, 0.0025, -0.0025, 0,
#         0, 0.16, 0.33, 1.0,
#         14.0, 15.0, 16.0, 20.0
#     ])


# import torch

# def get_alibi_slope(num_heads):
#     assert num_heads % 2 == 0, "num_heads 必须是偶数"

#     # 固定的后半部分 slope 值
#     fixed_values = [10, 8, 6.0, 4.0, 3.0, 2, 0, 0, -0.05, -0.1]
#     back_half = torch.tensor(fixed_values, dtype=torch.float32)

#     # 前一半需要补多少
#     num_front = num_heads - len(fixed_values)

#     # 前一半 slope：对称指数增长
#     x = (2 ** 8) ** (1 / num_front)

#     front_half = torch.tensor(
#         [x ** (i - num_front / 2) for i in range(num_front)],
#         dtype=torch.float32
#     )

#     # 拼接为完整 slope 向量
#     slopes = torch.cat([front_half, back_half])
#     return slopes

# import torch

# def get_alibi_slope(num_heads):  # 3.0525 2048 16 heads
#     assert num_heads % 2 == 0, "num_heads 必须是偶数"

#     # 固定的后半部分 slope 值
#     fixed_values = []
#     back_half = torch.tensor(fixed_values, dtype=torch.float32)

#     # 前一半需要补多少
#     num_front = num_heads - len(fixed_values)

#     # 前一半 slope：对称指数增长
#     x = (2 ** 10) ** (1 / num_front)

#     front_half = torch.tensor(
#         [x ** (i - num_front / 2) for i in range(num_front)],
#         dtype=torch.float32
#     )

#     # 拼接为完整 slope 向量
#     slopes = torch.cat([back_half, front_half])
#     return slopes

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
        layer_idx: int = None
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
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm

        self.window_size = window_size
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        # Remove RoPE - we'll use learnable bias instead
        # self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
        if attention_mask is not None:
            assert len(attention_mask.shape) == 2, (
                "Expected attention_mask as a 0-1 matrix with shape [batch_size, seq_len] "
                "for padding purposes (0 indicating padding). "
                "Arbitrary attention masks of shape [batch_size, seq_len, seq_len] are not allowed."
            )

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

        # Remove RoPE encoding - we'll rely on ALiBi or learnable bias
        # if self.max_position_embeddings is not None:
        #     max_seqlen = max(max_seqlen, self.max_position_embeddings)
        # q, k = self.rotary(q, k, seqlen_offset=seqlen_offset, max_seqlen=max_seqlen, cu_seqlens=cu_seqlens)

        if past_key_values is not None:
            cache_has_content = past_key_values.get_seq_length(self.layer_idx) > 0
            k_cached, v_cached = past_key_values.update(
                attn_state=(k.flatten(-2, -1), v.flatten(-2, -1)),
                layer_idx=self.layer_idx,
                offset=q_len,
                cache_kwargs=dict(window_size=self.window_size)
            )['attn_state']
            if cache_has_content:
                k, v = k_cached, v_cached
                k = rearrange(k, '... (h d) -> ... h d', d=self.head_dim)
                v = rearrange(v, '... (h d) -> ... h d', d=self.head_dim)

        # Use our flash focus attention implementation
        from fla.layers.flash_attn import TritonAttention as flash_focus_attn
        
        sm_scale = self.head_dim ** -0.5
        causal = True
        o = flash_focus_attn.apply(q, k, v, causal, sm_scale)
        o = o.reshape(batch_size, q_len, -1)
        o = self.o_proj(o)

        attentions = None if not output_attentions else None
        
        return o, attentions, past_key_values