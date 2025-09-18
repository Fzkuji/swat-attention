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

# Triton imports
try:
    import triton
    import triton.language as tl

    HAS_TRITON = True
except ImportError:
    HAS_TRITON = False
    warnings.warn("Triton is not installed. Please install it via `pip install triton`")


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
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

logger = logging.get_logger(__name__)

# Triton Kernels for SWAttention
if HAS_TRITON:
    @triton.jit
    def _swattn_fwd_kernel(
            Q, K, V,
            learnable_bias_diagonals,
            softmax_offset,
            O, L, M,
            s_qz, s_qh, s_qt, s_qd,
            s_kz, s_kh, s_kt, s_kd,
            s_vz, s_vh, s_vt, s_vd,
            s_oz, s_oh, s_ot, s_od,
            batch_size, num_heads, seq_len,
            SM_SCALE: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            HEAD_DIM: tl.constexpr,
            MAX_BIAS_LENGTH: tl.constexpr,
            USE_LEARNABLE_BIAS: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_bh = tl.program_id(1)
        pid_b = pid_bh // num_heads
        pid_h = pid_bh % num_heads

        q_offset = pid_b * s_qz + pid_h * s_qh
        k_offset = pid_b * s_kz + pid_h * s_kh
        v_offset = pid_b * s_vz + pid_h * s_vh
        o_offset = pid_b * s_oz + pid_h * s_oh

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)
        q_ptrs = Q + q_offset + (offs_m[:, None] * s_qt + offs_d[None, :] * s_qd)

        acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
        l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
        m_i = tl.full([BLOCK_M], value=float('-inf'), dtype=tl.float32)

        mask_m = offs_m < seq_len
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

        # Pass 1: Compute Softmax Denominator
        for start_n in range(0, seq_len, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < seq_len
            k_ptrs = K + k_offset + (offs_n[None, :] * s_kt + offs_d[:, None] * s_kd)
            k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)

            s_ij = tl.dot(q, k.T) * SM_SCALE
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            s_ij = tl.where(causal_mask, s_ij, float('-inf'))

            if USE_LEARNABLE_BIAS:
                rel_pos = offs_m[:, None] - offs_n[None, :]
                valid_mask = (rel_pos >= 0) & (rel_pos < MAX_BIAS_LENGTH)
                indices = tl.where(valid_mask, rel_pos, 0)
                bias_ptr = learnable_bias_diagonals + pid_h * MAX_BIAS_LENGTH
                bias_val = tl.load(bias_ptr + indices, mask=valid_mask, other=0.0)
                s_ij += bias_val

            m_ij = tl.max(s_ij, axis=1)
            m_i_new = tl.maximum(m_i, m_ij)
            p_ij = tl.exp(s_ij - m_i_new[:, None])
            alpha = tl.exp(m_i - m_i_new)
            l_i = l_i * alpha + tl.sum(p_ij, axis=1)
            m_i = m_i_new

        l_i_inv = 1.0 / l_i

        # Pass 2: Compute Final Output
        for start_n in range(0, seq_len, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < seq_len
            k_ptrs = K + k_offset + (offs_n[None, :] * s_kt + offs_d[:, None] * s_kd)
            v_ptrs = V + v_offset + (offs_n[:, None] * s_vt + offs_d[None, :] * s_vd)
            k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

            s_ij = tl.dot(q, k.T) * SM_SCALE
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            s_ij = tl.where(causal_mask, s_ij, float('-inf'))

            if USE_LEARNABLE_BIAS:
                rel_pos = offs_m[:, None] - offs_n[None, :]
                valid_mask = (rel_pos >= 0) & (rel_pos < MAX_BIAS_LENGTH)
                indices = tl.where(valid_mask, rel_pos, 0)
                bias_ptr = learnable_bias_diagonals + pid_h * MAX_BIAS_LENGTH
                bias_val = tl.load(bias_ptr + indices, mask=valid_mask, other=0.0)
                s_ij += bias_val

            p_ij = tl.exp(s_ij - m_i[:, None]) * l_i_inv[:, None]

            offset_h = tl.load(softmax_offset + pid_h)
            offset_val = tl.abs(offset_h + 1.0)
            num_visible = (offs_n[None, :] + 1.0).to(tl.float32)
            adaptive_offset = offset_val / num_visible
            p_ij = p_ij - adaptive_offset
            p_ij = tl.where(p_ij > 0, p_ij, 0.0)

            acc += tl.dot(p_ij.to(Q.dtype.element_ty), v)

        o_ptrs = O + o_offset + (offs_m[:, None] * s_ot + offs_d[None, :] * s_od)
        tl.store(o_ptrs, acc, mask=mask_m[:, None])

        L_ptrs = L + pid_bh * seq_len + offs_m
        M_ptrs = M + pid_bh * seq_len + offs_m
        tl.store(L_ptrs, l_i, mask=mask_m)
        tl.store(M_ptrs, m_i, mask=mask_m)


    @triton.jit
    def _swattn_bwd_kernel(
            Q, K, V, O, dO,
            learnable_bias_diagonals, softmax_offset,
            L, M,
            dQ, dK, dV,
            d_learnable_bias, d_softmax_offset,
            s_qz, s_qh, s_qt, s_qd,
            s_kz, s_kh, s_kt, s_kd,
            s_vz, s_vh, s_vt, s_vd,
            batch_size, num_heads, seq_len,
            SM_SCALE: tl.constexpr,
            BLOCK_M: tl.constexpr,
            BLOCK_N: tl.constexpr,
            HEAD_DIM: tl.constexpr,
            MAX_BIAS_LENGTH: tl.constexpr,
            USE_LEARNABLE_BIAS: tl.constexpr,
    ):
        pid_m = tl.program_id(0)  # Parallelize over M
        pid_bh = tl.program_id(1)
        pid_b = pid_bh // num_heads
        pid_h = pid_bh % num_heads

        # Pointers
        q_offset = pid_b * s_qz + pid_h * s_qh
        k_offset = pid_b * s_kz + pid_h * s_kh
        v_offset = pid_b * s_vz + pid_h * s_vh

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_d = tl.arange(0, HEAD_DIM)

        # Load Q, dO for the current block
        mask_m = offs_m < seq_len
        q_ptrs = Q + q_offset + (offs_m[:, None] * s_qt + offs_d[None, :] * s_qd)
        q = tl.load(q_ptrs, mask=mask_m[:, None], other=0.0)

        do_ptrs = dO + q_offset + (offs_m[:, None] * s_qt + offs_d[None, :] * s_qd)
        do = tl.load(do_ptrs, mask=mask_m[:, None], other=0.0)

        # Load L, M
        l_ptrs = L + pid_bh * seq_len + offs_m
        m_ptrs = M + pid_bh * seq_len + offs_m
        l_i = tl.load(l_ptrs, mask=mask_m, other=1.0)
        m_i = tl.load(m_ptrs, mask=mask_m, other=float('-inf'))

        # Initialize dQ accumulator
        dq_acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

        # Loop over K, V blocks
        for start_n in range(0, seq_len, BLOCK_N):
            offs_n = start_n + tl.arange(0, BLOCK_N)
            mask_n = offs_n < seq_len

            # Load K, V
            k_ptrs = K + k_offset + (offs_n[None, :] * s_kt + offs_d[:, None] * s_kd)
            v_ptrs = V + v_offset + (offs_n[:, None] * s_vt + offs_d[None, :] * s_vd)
            k = tl.load(k_ptrs, mask=mask_n[None, :], other=0.0)
            v = tl.load(v_ptrs, mask=mask_n[:, None], other=0.0)

            # Recompute S_ij
            s_ij = tl.dot(q, k.T) * SM_SCALE
            causal_mask = offs_m[:, None] >= offs_n[None, :]
            s_ij = tl.where(causal_mask, s_ij, float('-inf'))

            if USE_LEARNABLE_BIAS:
                rel_pos = offs_m[:, None] - offs_n[None, :]
                valid_mask = (rel_pos >= 0) & (rel_pos < MAX_BIAS_LENGTH)
                indices = tl.where(valid_mask, rel_pos, 0)
                bias_ptr = learnable_bias_diagonals + pid_h * MAX_BIAS_LENGTH
                bias_val = tl.load(bias_ptr + indices, mask=valid_mask, other=0.0)
                s_ij += bias_val

                # Recompute P_ij
            p_raw = tl.exp(s_ij - m_i[:, None]) / l_i[:, None]

            offset_h = tl.load(softmax_offset + pid_h)
            offset_val = tl.abs(offset_h + 1.0)
            num_visible = (offs_n[None, :] + 1.0).to(tl.float32)
            adaptive_offset = offset_val / num_visible
            p_final = p_raw - adaptive_offset
            p_final = tl.where(p_final > 0, p_final, 0.0)

            # Compute dP
            dp = tl.dot(do, v.T)
            relu_mask = p_final > 0
            dp_raw = dp * relu_mask.to(dp.dtype)

            # Compute dS
            ds = p_raw * (dp_raw - tl.sum(dp_raw * p_raw, axis=1)[:, None])
            ds = ds * SM_SCALE
            ds = ds.to(Q.dtype.element_ty)

            # Accumulate dQ
            dq_acc += tl.dot(ds, k)

            # Accumulate dK, dV with atomic adds
            dk_ptrs = dK + k_offset + (offs_n[None, :] * s_kt + offs_d[:, None] * s_kd)
            dv_ptrs = dV + v_offset + (offs_n[:, None] * s_vt + offs_d[None, :] * s_vd)

            ds_t = tl.trans(ds)
            tl.atomic_add(dk_ptrs, tl.dot(ds_t, q))
            tl.atomic_add(dv_ptrs, tl.dot(p_final.to(V.dtype.element_ty).T, do))

            # Accumulate d_learnable_bias, d_softmax_offset
            if USE_LEARNABLE_BIAS:
                rel_pos = offs_m[:, None] - offs_n[None, :]
                valid_mask = (rel_pos >= 0) & (rel_pos < MAX_BIAS_LENGTH)
                indices = tl.where(valid_mask, rel_pos, 0)
                d_bias_ptr = d_learnable_bias + pid_h * MAX_BIAS_LENGTH
                value_to_add = (ds * valid_mask.to(ds.dtype)).to(d_learnable_bias.dtype.element_ty)
                tl.atomic_add(d_bias_ptr + indices, value_to_add)

            d_adaptive_offset = -dp_raw
            d_offset_val = tl.sum(d_adaptive_offset / num_visible)
            d_offset_h = d_offset_val * tl.where(offset_h + 1.0 > 0, 1.0, -1.0)
            tl.atomic_add(d_softmax_offset + pid_h, d_offset_h.to(d_softmax_offset.dtype.element_ty))

            # Write back dQ
        dq_ptrs = dQ + q_offset + (offs_m[:, None] * s_qt + offs_d[None, :] * s_qd)
        tl.store(dq_ptrs, dq_acc, mask=mask_m[:, None])


class SWAttentionTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, learnable_bias, softmax_offset, scaling):
        batch_size, num_heads, seq_len, head_dim = q.shape

        if head_dim not in [16, 32, 64, 128]:
            raise ValueError(f"Triton kernel only supports head dimensions of 16, 32, 64, or 128, but got {head_dim}")

        o = torch.empty_like(q)
        l = torch.empty((batch_size * num_heads, seq_len), device=q.device, dtype=torch.float32)
        m = torch.empty((batch_size * num_heads, seq_len), device=q.device, dtype=torch.float32)

        BLOCK_M = 128
        BLOCK_N = 64
        grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * num_heads)

        use_learnable_bias = learnable_bias is not None
        max_bias_length = learnable_bias.shape[1] if use_learnable_bias else 0

        _swattn_fwd_kernel[grid](
            q, k, v,
            learnable_bias, softmax_offset,
            o, l, m,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            o.stride(0), o.stride(1), o.stride(2), o.stride(3),
            batch_size, num_heads, seq_len,
            SM_SCALE=scaling,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            HEAD_DIM=head_dim,
            MAX_BIAS_LENGTH=max_bias_length,
            USE_LEARNABLE_BIAS=use_learnable_bias,
            num_warps=4,
        )

        ctx.save_for_backward(q, k, v, o, l, m, learnable_bias, softmax_offset)
        ctx.scaling = scaling
        ctx.max_bias_length = max_bias_length
        ctx.use_learnable_bias = use_learnable_bias
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, l, m, learnable_bias, softmax_offset = ctx.saved_tensors
        scaling = ctx.scaling
        max_bias_length = ctx.max_bias_length
        use_learnable_bias = ctx.use_learnable_bias

        batch_size, num_heads, seq_len, head_dim = q.shape

        dq = torch.empty_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)

        d_learnable_bias = torch.zeros_like(learnable_bias) if use_learnable_bias else None
        d_softmax_offset = torch.zeros_like(softmax_offset)

        BLOCK_M = 128
        BLOCK_N = 64
        grid = (triton.cdiv(seq_len, BLOCK_M), batch_size * num_heads)

        _swattn_bwd_kernel[grid](
            q, k, v, o, do,
            learnable_bias, softmax_offset,
            l, m,
            dq, dk, dv,
            d_learnable_bias, d_softmax_offset,
            q.stride(0), q.stride(1), q.stride(2), q.stride(3),
            k.stride(0), k.stride(1), k.stride(2), k.stride(3),
            v.stride(0), v.stride(1), v.stride(2), v.stride(3),
            batch_size, num_heads, seq_len,
            SM_SCALE=scaling,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            HEAD_DIM=head_dim,
            MAX_BIAS_LENGTH=max_bias_length,
            USE_LEARNABLE_BIAS=use_learnable_bias,
            num_warps=4,
        )
        return dq, dk, dv, d_learnable_bias, d_softmax_offset, None


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
            use_learnable_bias: bool = True,
            max_bias_length: int = 1024,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.num_kv_groups = self.num_heads // self.num_kv_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim ** -0.5
        self.qkv_bias = qkv_bias
        self.qk_norm = qk_norm
        self.window_size = window_size

        self.rope_theta = rope_theta
        self.layer_idx = layer_idx
        self.use_learnable_bias = use_learnable_bias
        self.max_bias_length = max_bias_length

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=self.qkv_bias)
        self.k_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.kv_dim, bias=self.qkv_bias)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        if self.use_learnable_bias:
            self.learnable_bias_diagonals = nn.Parameter(
                torch.zeros(self.num_heads, self.max_bias_length)
            )
            nn.init.normal_(self.learnable_bias_diagonals, mean=0.0, std=1e-3)
        else:
            self.register_parameter('learnable_bias_diagonals', None)

        self.softmax_offset = nn.Parameter(torch.full((self.num_heads,), 0.0))
        self.rotary = RotaryEmbedding(dim=self.head_dim, base=self.rope_theta)

    def forward(
            self,
            hidden_states: torch.Tensor,
            attention_mask: Optional[torch.LongTensor] = None,
            past_key_values: Optional[Cache] = None,
            output_attentions: bool = False,
            use_cache: bool = False,
            **kwargs,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:

        if not HAS_TRITON:
            raise ImportError("Triton is not installed, but is required for this SWAttention implementation.")
        if past_key_values is not None or use_cache:
            warnings.warn("Triton implementation of SWAttention does not currently support past_key_values (caching).")
        if attention_mask is not None and (attention_mask.ndim > 2 or (attention_mask == 0).any()):
            warnings.warn("Triton implementation of SWAttention only supports causal masking (implicit) and does not support padding masks.")

        batch_size, q_len, _ = hidden_states.size()

        q = rearrange(self.q_proj(hidden_states), 'b l (h d) -> b l h d', h=self.num_heads)
        k = rearrange(self.k_proj(hidden_states), 'b l (h d) -> b l h d', h=self.num_kv_heads)
        v = rearrange(self.v_proj(hidden_states), 'b l (h d) -> b l h d', h=self.num_kv_heads)

        seqlen_offset = 0
        if past_key_values is not None:
            seqlen_offset = past_key_values.get_seq_length(self.layer_idx)

        q, k = self.rotary(q, k, seqlen_offset=seqlen_offset)

        q = rearrange(q, 'b l h d -> b h l d')
        k = rearrange(k, 'b l h d -> b h l d')
        v = rearrange(v, 'b l h d -> b h l d')

        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        attn_output = SWAttentionTritonFunction.apply(
            q, k, v,
            self.learnable_bias_diagonals,
            self.softmax_offset,
            self.scaling
        )

        attn_output = rearrange(attn_output, 'b h l d -> b l (h d)')
        o = self.o_proj(attn_output)

        if output_attentions:
            warnings.warn("`output_attentions=True` is not supported with the Triton kernel. Returning `None` for attentions.")

        return o, None, past_key_values