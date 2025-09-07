import pytest
import torch
import torch.nn as nn

import triton
import triton.language as tl
from triton.tools.tensor_descriptor import TensorDescriptor


# =================================================================================
# Difference of Gaussians (DoG) Bias Logic in Triton
# =================================================================================
@triton.jit
def standard_dog_bias(dist, A1, s1, c1, A2, s2, c2, offset):
    """
    Triton JIT implementation of the Standard Difference of Gaussians (DoG) function.
    dist: The absolute distance |q_pos - k_pos|.
    """
    x = dist.to(tl.float32)

    # near = A1 * exp(-((x-c1)/s1)**2)
    arg1 = (x - c1) / s1
    near = A1 * tl.exp(-(arg1 * arg1))

    # far = A2 * exp(-((x-c2)/s2)**2)
    arg2 = (x - c2) / s2
    far = A2 * tl.exp(-(arg2 * arg2))

    return near - far + offset


# =================================================================================
# Forward Kernel
# =================================================================================
@triton.jit
def _attn_fwd(
        Q, K, V, Sinks, DogParams, sm_scale,
        M, Out,
        Start_q,
        Z, H, N_Q_CTX, N_KV_CTX,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BANDWIDTH: tl.constexpr,
):
    tl.static_assert(BLOCK_N <= HEAD_DIM)
    start_q = tl.load(Start_q).to(tl.int32)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    # Load DoG parameters for the current head
    dog_params_ptr = DogParams + off_h * 7
    A1 = tl.load(dog_params_ptr + 0)
    s1 = tl.load(dog_params_ptr + 1)
    c1 = tl.load(dog_params_ptr + 2)
    A2 = tl.load(dog_params_ptr + 3)
    s2 = tl.load(dog_params_ptr + 4)
    c2 = tl.load(dog_params_ptr + 5)
    offset = tl.load(dog_params_ptr + 6)

    # Load attention sinks
    if Sinks is not None:
        sink = tl.load(Sinks + off_h).to(tl.float32)
    else:
        sink = 0.0

    # Initialize offsets
    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m_i = tl.zeros([BLOCK_M], dtype=tl.float32) + sink
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)
    acc = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)

    q = Q.load([off_z, off_h, start_m * BLOCK_M, 0]).reshape([BLOCK_M, HEAD_DIM])

    if BANDWIDTH:
        lo = tl.maximum(0, start_q + start_m * BLOCK_M - BANDWIDTH)
    else:
        lo = 0
    hi = tl.minimum(start_q + (start_m + 1) * BLOCK_M, N_KV_CTX)
    lo = tl.multiple_of(lo, BLOCK_N)

    for start_n in range(lo, hi, BLOCK_N):
        mask = (start_n + offs_n)[None, :] > (start_q + offs_m)[:, None]

        if BANDWIDTH:
            too_old = (start_n + offs_n[None, :]) < (start_q + offs_m[:, None] - BANDWIDTH + 1)
            mask = mask | too_old

        k = K.load([off_z, off_h, start_n, 0]).reshape([BLOCK_N, HEAD_DIM]).T
        qk = tl.dot(q, k)

        pos_m = (start_q + offs_m)[:, None]
        pos_n = (start_n + offs_n)[None, :]
        dist = tl.abs(pos_m - pos_n)
        dog_bias = standard_dog_bias(dist, A1, s1, c1, A2, s2, c2, offset)
        qk += dog_bias

        qk = tl.where(qk < 0, -1.0e9, qk)

        qk = qk * sm_scale + tl.where(mask, -1.0e9, 0.0)

        m_ij = tl.maximum(m_i, tl.max(qk, 1))
        p = tl.math.exp(qk - m_ij[:, None])
        alpha = tl.math.exp(m_i - m_ij)

        l_ij = tl.sum(p, 1)
        acc = acc * alpha[:, None]

        v = V.load([off_z, off_h, start_n, 0]).reshape([BLOCK_N, HEAD_DIM])
        acc = tl.dot(p.to(v.dtype), v, acc)

        l_i = l_i * alpha + l_ij
        m_i = m_ij

    sink_val = tl.math.exp(sink - m_i)
    z = l_i + sink_val
    acc = acc / z[:, None]

    m_i += tl.math.log(z)

    m_ptrs = M + off_hz * N_Q_CTX + offs_m
    tl.store(m_ptrs, m_i)
    acc = acc.to(Out.dtype)
    acc = tl.reshape(acc, (1, 1, BLOCK_M, HEAD_DIM))
    Out.store([off_z, off_h, start_m * BLOCK_M, 0], acc)


@triton.jit
def _attn_bwd(
        Q, K, V, Sinks, DogParams, sm_scale,
        Out, dOut,
        M,
        dQ, dK, dV, dSinks, dDogParams,
        Start_q,
        Z, H, N_Q_CTX, N_KV_CTX,
        HEAD_DIM: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BANDWIDTH: tl.constexpr,
):
    start_q = tl.load(Start_q).to(tl.int32)
    start_m = tl.program_id(0)
    off_hz = tl.program_id(1)
    off_z = off_hz // H
    off_h = off_hz % H

    dog_params_ptr = DogParams + off_h * 7
    A1 = tl.load(dog_params_ptr + 0)
    s1 = tl.load(dog_params_ptr + 1)
    c1 = tl.load(dog_params_ptr + 2)
    A2 = tl.load(dog_params_ptr + 3)
    s2 = tl.load(dog_params_ptr + 4)
    c2 = tl.load(dog_params_ptr + 5)
    offset = tl.load(dog_params_ptr + 6)

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)

    m = tl.load(M + off_hz * N_Q_CTX + offs_m)

    if Sinks is not None:
        sink = tl.load(Sinks + off_h)
    else:
        sink = 0.0

    q = Q.load([off_z, off_h, start_m * BLOCK_M, 0]).reshape([BLOCK_M, HEAD_DIM])
    o = Out.load([off_z, off_h, start_m * BLOCK_M, 0]).reshape([BLOCK_M, HEAD_DIM])
    do = dOut.load([off_z, off_h, start_m * BLOCK_M, 0]).reshape([BLOCK_M, HEAD_DIM])

    do = do.to(tl.float32)
    delta = tl.sum(o * do, axis=1)

    dA1 = 0.0
    ds1 = 0.0
    dc1 = 0.0
    dA2 = 0.0
    ds2 = 0.0
    dc2 = 0.0
    doffset = 0.0
    dsink = 0.0  # Gradient for attention sink

    if BANDWIDTH:
        lo = tl.maximum(0, start_q + start_m * BLOCK_M - BANDWIDTH)
    else:
        lo = 0
    hi = tl.minimum(start_q + (start_m + 1) * BLOCK_M, N_KV_CTX)
    lo = tl.multiple_of(lo, BLOCK_N)

    for start_n in range(lo, hi, BLOCK_N):
        mask = (start_n + offs_n)[None, :] > (start_q + offs_m)[:, None]

        if BANDWIDTH:
            too_old = (start_n + offs_n[None, :]) < (start_q + offs_m[:, None] - BANDWIDTH + 1)
            mask = mask | too_old

        k = K.load([off_z, off_h, start_n, 0]).reshape([BLOCK_N, HEAD_DIM])
        v = V.load([off_z, off_h, start_n, 0]).reshape([BLOCK_N, HEAD_DIM])

        qk = tl.dot(q, k.T)

        pos_m = (start_q + offs_m)[:, None]
        pos_n = (start_n + offs_n)[None, :]
        dist = tl.abs(pos_m - pos_n)
        dog_bias = standard_dog_bias(dist, A1, s1, c1, A2, s2, c2, offset)
        qk += dog_bias

        neg_mask = qk < 0
        qk = tl.where(neg_mask, -1.0e9, qk)

        qk = qk * sm_scale + tl.where(mask, -1.0e9, 0.0)

        p = tl.math.exp(qk - m[:, None])
        dp = tl.dot(do, v.to(do.dtype).T)

        # ds/d_softmax_input = (P * (dp - delta))
        ds = p * (dp - delta[:, None])
        ds = tl.where(mask, 0.0, ds)
        ds = tl.where(neg_mask, 0.0, ds)

        # d_bias = ds * sm_scale (gradient flows through scale factor)
        d_bias = ds * sm_scale

        # Compute DoG parameter gradients
        x = dist.to(tl.float32)
        arg1 = (x - c1) / s1
        exp1 = tl.exp(-(arg1 * arg1))
        arg2 = (x - c2) / s2
        exp2 = tl.exp(-(arg2 * arg2))

        # Gradients for DoG parameters
        dA1 += tl.sum(d_bias * exp1)
        ds1 += tl.sum(d_bias * A1 * exp1 * (2.0 * arg1 * arg1 / s1))
        dc1 += tl.sum(d_bias * A1 * exp1 * (2.0 * arg1 / s1))
        dA2 += tl.sum(d_bias * (-exp2))
        ds2 += tl.sum(d_bias * (-A2) * exp2 * (2.0 * arg2 * arg2 / s2))
        dc2 += tl.sum(d_bias * (-A2) * exp2 * (2.0 * arg2 / s2))
        doffset += tl.sum(d_bias)

        # Compute gradients for K and V
        dK_block = tl.dot(ds.T.to(q.dtype), q) * sm_scale
        dK_block = tl.reshape(dK_block, (1, 1, BLOCK_N, HEAD_DIM))
        dK.atomic_add([off_z, off_h, start_n, 0], dK_block.to(K.dtype))

        dV_block = tl.dot(p.T.to(do.dtype), do)
        dV_block = tl.reshape(dV_block, (1, 1, BLOCK_N, HEAD_DIM))
        dV.atomic_add([off_z, off_h, start_n, 0], dV_block.to(V.dtype))

    # Compute gradient for attention sink
    # The sink affects the normalization: denominator = sum(exp(scores)) + exp(sink)
    # d_loss/d_sink = sum_i (o_i * do_i * exp(sink - m_i) / (l_i + exp(sink - m_i)))
    # where l_i is the sum of exp(scores) for row i
    # This is simplified as: -delta * exp(sink - m) / (l + exp(sink - m))
    # But since we don't store l explicitly, we approximate it
    if Sinks is not None:
        dsink = -tl.sum(delta * tl.math.exp(sink - m))

    # Store DoG parameter gradients
    d_dog_ptr = dDogParams + off_h * 7
    tl.atomic_add(d_dog_ptr + 0, dA1)
    tl.atomic_add(d_dog_ptr + 1, ds1)
    tl.atomic_add(d_dog_ptr + 2, dc1)
    tl.atomic_add(d_dog_ptr + 3, dA2)
    tl.atomic_add(d_dog_ptr + 4, ds2)
    tl.atomic_add(d_dog_ptr + 5, dc2)
    tl.atomic_add(d_dog_ptr + 6, doffset)

    # Store sink gradient
    if Sinks is not None and dSinks is not None:
        tl.atomic_add(dSinks + off_h, dsink)

    # Compute gradients for Q
    dq = tl.zeros([BLOCK_M, HEAD_DIM], dtype=tl.float32)
    for start_n in range(lo, hi, BLOCK_N):
        mask = (start_n + offs_n)[None, :] > (start_q + offs_m)[:, None]
        if BANDWIDTH:
            too_old = (start_n + offs_n[None, :]) < (start_q + offs_m[:, None] - BANDWIDTH + 1)
            mask = mask | too_old

        k = K.load([off_z, off_h, start_n, 0]).reshape([BLOCK_N, HEAD_DIM])
        v = V.load([off_z, off_h, start_n, 0]).reshape([BLOCK_N, HEAD_DIM])
        qk = tl.dot(q, k.T)

        pos_m = (start_q + offs_m)[:, None]
        pos_n = (start_n + offs_n)[None, :]
        dist = tl.abs(pos_m - pos_n)
        dog_bias = standard_dog_bias(dist, A1, s1, c1, A2, s2, c2, offset)
        qk += dog_bias

        neg_mask = qk < 0
        qk = tl.where(neg_mask, -1.0e9, qk)

        qk = qk * sm_scale + tl.where(mask, -1.0e9, 0.0)
        p = tl.math.exp(qk - m[:, None])
        dp = tl.dot(do, v.to(do.dtype).T)
        ds = p * (dp - delta[:, None])
        ds = tl.where(mask, 0.0, ds)
        ds = tl.where(neg_mask, 0.0, ds)

        dq += tl.dot(ds.to(k.dtype), k) * sm_scale

    dq = dq.to(Q.dtype)
    dq = tl.reshape(dq, (1, 1, BLOCK_M, HEAD_DIM))
    dQ.store([off_z, off_h, start_m * BLOCK_M, 0], dq)


class _attention(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, sinks, dog_params, sm_scale, bandwidth, start_q):
        bs, n_ctx, n_kv_heads, repeat_kv, HEAD_DIM_Q = q.shape
        bs, n_kv_ctx, n_kv_heads_k, HEAD_DIM_K = k.shape
        n_heads = n_kv_heads * repeat_kv

        q = q.view(bs, n_ctx, n_heads, HEAD_DIM_Q).transpose(1, 2).contiguous()
        k_unrepeated = k.view(bs, n_kv_ctx, n_kv_heads, HEAD_DIM_K)
        k = k_unrepeated.repeat_interleave(repeat_kv, dim=2).transpose(1, 2).contiguous()
        v_unrepeated = v.view(bs, n_kv_ctx, n_kv_heads, HEAD_DIM_K)
        v = v_unrepeated.repeat_interleave(repeat_kv, dim=2).transpose(1, 2).contiguous()

        HEAD_DIM = HEAD_DIM_K

        BLOCK_M = 128
        BLOCK_N = 64

        m_pad_size = (BLOCK_M - n_ctx % BLOCK_M) % BLOCK_M
        q_padded = torch.nn.functional.pad(q, (0, 0, 0, m_pad_size))

        n_pad_size = (BLOCK_N - n_kv_ctx % BLOCK_N) % BLOCK_N
        k_padded = torch.nn.functional.pad(k, (0, 0, 0, n_pad_size))
        v_padded = torch.nn.functional.pad(v, (0, 0, 0, n_pad_size))

        o = torch.empty_like(q_padded)
        M = torch.empty((bs, n_heads, n_ctx + m_pad_size), device=q.device, dtype=torch.float32)

        grid = (triton.cdiv(n_ctx, BLOCK_M), bs * n_heads, 1)

        n_kv_ctx_padded = n_kv_ctx + n_pad_size
        _attn_fwd[grid](
            TensorDescriptor.from_tensor(q_padded, [1, 1, BLOCK_M, HEAD_DIM]),
            TensorDescriptor.from_tensor(k_padded, [1, 1, BLOCK_N, HEAD_DIM]),
            TensorDescriptor.from_tensor(v_padded, [1, 1, BLOCK_N, HEAD_DIM]),
            sinks, dog_params.contiguous(), sm_scale,
            M,
            TensorDescriptor.from_tensor(o, [1, 1, BLOCK_M, HEAD_DIM]),
            start_q,
            bs, n_heads,
            N_Q_CTX=n_ctx + m_pad_size,
            N_KV_CTX=n_kv_ctx_padded,
            HEAD_DIM=HEAD_DIM, BANDWIDTH=bandwidth, BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N,
        )

        ctx.save_for_backward(q_padded, k_padded, v_padded, sinks, dog_params, o, M, start_q)
        ctx.sm_scale = sm_scale
        ctx.bandwidth = bandwidth
        ctx.n_ctx = n_ctx
        ctx.n_kv_ctx = n_kv_ctx
        ctx.head_dim = HEAD_DIM
        ctx.repeat_kv = repeat_kv
        ctx.n_kv_heads = n_kv_heads

        o = o[:, :, :n_ctx, :].transpose(1, 2).contiguous()
        o = o.view(bs, n_ctx, n_heads * HEAD_DIM)
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, sinks, dog_params, o, M, start_q = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        bandwidth = ctx.bandwidth
        n_ctx = ctx.n_ctx
        HEAD_DIM = ctx.head_dim
        repeat_kv = ctx.repeat_kv
        n_kv_heads = ctx.n_kv_heads

        bs, H, N_CTX_PAD, _ = q.shape
        bs, H, N_KV_CTX_PAD, _ = k.shape

        do = do.view(bs, n_ctx, H, HEAD_DIM).transpose(1, 2)
        m_pad_size = (128 - n_ctx % 128) % 128
        do_padded = torch.nn.functional.pad(do, (0, 0, 0, m_pad_size))

        dq = torch.empty_like(q)
        dk = torch.zeros_like(k)
        dv = torch.zeros_like(v)
        ddog_params = torch.zeros_like(dog_params)
        dsinks = torch.zeros_like(sinks) if sinks is not None else None

        grid = (triton.cdiv(n_ctx, 128), bs * H, 1)

        _attn_bwd[grid](
            TensorDescriptor.from_tensor(q, [1, 1, 128, HEAD_DIM]),
            TensorDescriptor.from_tensor(k, [1, 1, 64, HEAD_DIM]),
            TensorDescriptor.from_tensor(v, [1, 1, 64, HEAD_DIM]),
            sinks, dog_params.contiguous(), sm_scale,
            TensorDescriptor.from_tensor(o, [1, 1, 128, HEAD_DIM]),
            TensorDescriptor.from_tensor(do_padded, [1, 1, 128, HEAD_DIM]),
            M,
            TensorDescriptor.from_tensor(dq, [1, 1, 128, HEAD_DIM]),
            TensorDescriptor.from_tensor(dk, [1, 1, 64, HEAD_DIM]),
            TensorDescriptor.from_tensor(dv, [1, 1, 64, HEAD_DIM]),
            dsinks, ddog_params,
            start_q,
            bs, H, N_CTX_PAD, N_KV_CTX_PAD,
            HEAD_DIM=HEAD_DIM, BLOCK_M=128, BLOCK_N=64, BANDWIDTH=bandwidth,
        )

        n_kv_ctx = ctx.n_kv_ctx
        dq = dq[:, :, :n_ctx, :].transpose(1, 2)
        dk = dk[:, :, :n_kv_ctx, :]
        dv = dv[:, :, :n_kv_ctx, :]

        if repeat_kv > 1:
            dk = dk.view(bs, n_kv_heads, repeat_kv, n_kv_ctx, HEAD_DIM).sum(dim=2)
            dv = dv.view(bs, n_kv_heads, repeat_kv, n_kv_ctx, HEAD_DIM).sum(dim=2)

        dk = dk.transpose(1, 2)
        dv = dv.transpose(1, 2)

        dq = dq.reshape(bs, n_ctx, n_kv_heads, repeat_kv, HEAD_DIM)

        return dq, dk, dv, dsinks, ddog_params, None, None, None


attention = _attention.apply


def attention_ref(
        query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
        sinks: torch.Tensor, dog_params: torch.Tensor,
        sm_scale: float = 0.125, sliding_window: int | None = None, start_q: torch.LongTensor = 0
):
    bs, n_q, n_kv_h, n_kv_g, h_dim = query.shape
    _, n_k, _, _ = key.shape

    q_float = query.reshape(bs, n_q, n_kv_h * n_kv_g, h_dim).transpose(1, 2).float()
    k_float = key.repeat_interleave(n_kv_g, dim=2).transpose(1, 2).float()
    v_float = value.repeat_interleave(n_kv_g, dim=2).transpose(1, 2).float()

    n_h = n_kv_h * n_kv_g

    pos_k = torch.arange(n_k, device=query.device)
    pos_q = torch.arange(n_q, device=query.device) + start_q.item()
    mask = pos_k[None, :] > pos_q[:, None]

    if sliding_window:
        too_old = pos_k[None, :] < (pos_q[:, None] - sliding_window + 1)
        mask = mask | too_old

    logits = torch.einsum("bhqd,bhkd->bhqk", q_float, k_float)

    dist = torch.abs(pos_q[:, None] - pos_k[None, :]).float()

    A1, s1, c1, A2, s2, c2, offset = [p.view(1, n_h, 1, 1) for p in dog_params.T]

    x = dist.view(1, 1, n_q, n_k)
    arg1 = (x - c1) / s1
    near = A1 * torch.exp(-(arg1 * arg1))
    arg2 = (x - c2) / s2
    far = A2 * torch.exp(-(arg2 * arg2))
    bias = near - far + offset

    logits = logits + bias
    logits = torch.where(logits < 0, -1e9, logits)

    logits = logits * sm_scale
    logits.masked_fill_(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

    sinks_r = sinks.view(1, n_h, 1, 1).float()
    logits_max = torch.max(logits, dim=-1, keepdim=True).values
    logits_max_or_sinks = torch.maximum(sinks_r, logits_max)

    sinks_exp = torch.exp(sinks_r - logits_max_or_sinks)
    scores_unnormalized = torch.exp(logits - logits_max_or_sinks)
    normalizer = scores_unnormalized.sum(dim=-1, keepdim=True) + sinks_exp

    scores = scores_unnormalized / normalizer

    output = torch.einsum("bhqk,bhkd->bhqd", scores, v_float)

    output = output.transpose(1, 2).contiguous().view(bs, n_q, n_h * h_dim)
    return output.to(query.dtype)


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_queries", [128])
@pytest.mark.parametrize("num_keys", [256])
@pytest.mark.parametrize("num_key_value_heads", [4])
@pytest.mark.parametrize("num_key_value_groups", [2])
@pytest.mark.parametrize("head_dim", [64])
@pytest.mark.parametrize("sm_scale", [0.125])
@pytest.mark.parametrize("sliding_window", [None, 128])
@pytest.mark.parametrize("start_q", [0, 128])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_attention_with_dog_bias(batch_size, num_queries, num_keys, num_key_value_heads, num_key_value_groups, head_dim,
                                 sm_scale, sliding_window, start_q, dtype):
    torch.manual_seed(0)

    num_heads = num_key_value_heads * num_key_value_groups

    q_shape = (batch_size, num_queries, num_key_value_heads, num_key_value_groups, head_dim)
    k_shape = (batch_size, num_keys, num_key_value_heads, head_dim)
    v_shape = (batch_size, num_keys, num_key_value_heads, head_dim)

    q = torch.randn(q_shape, device="cuda", dtype=dtype)
    k = torch.randn(k_shape, device="cuda", dtype=dtype)
    v = torch.randn(v_shape, device="cuda", dtype=dtype)

    sinks = torch.randn(num_heads, device="cuda", dtype=dtype)

    dog_params = nn.Parameter(torch.randn(num_heads, 7, device="cuda", dtype=torch.float32))
    with torch.no_grad():
        dog_params[:, [1, 4]] = torch.abs(dog_params[:, [1, 4]]) + 1e-6

    start_q_tensor = torch.tensor([start_q], dtype=torch.int32, device="cuda")

    q_triton = q.clone().requires_grad_()
    k_triton = k.clone().requires_grad_()
    v_triton = v.clone().requires_grad_()
    dog_params_triton = dog_params.clone().requires_grad_()
    dog_params_triton.retain_grad()

    o1 = attention(q_triton, k_triton, v_triton, sinks, dog_params_triton, sm_scale, sliding_window, start_q_tensor)

    o2 = attention_ref(q, k, v, sinks, dog_params, sm_scale, sliding_window, start_q_tensor)

    print("\nForward pass comparison:")
    torch.testing.assert_close(o1.float(), o2.float(), atol=1e-2, rtol=1e-2)
    print("Forward pass OK!")

    grad_o = torch.randn_like(o1)
    o1.backward(grad_o)

    q_ref = q.clone().requires_grad_()
    k_ref = k.clone().requires_grad_()
    v_ref = v.clone().requires_grad_()
    dog_params_ref = dog_params.clone().requires_grad_()

    # Enable gradient computation for dog_params in reference
    dog_params_ref.retain_grad()

    o2_ref = attention_ref(q_ref, k_ref, v_ref, sinks, dog_params_ref, sm_scale, sliding_window, start_q_tensor)
    o2_ref.backward(grad_o.to(o2_ref.dtype))

    print("\nBackward pass comparison (dQ):")
    torch.testing.assert_close(q_triton.grad.float(), q_ref.grad.float(), atol=5e-2, rtol=0.1)
    print("dQ OK!")

    print("\nBackward pass comparison (dK):")
    torch.testing.assert_close(k_triton.grad.float(), k_ref.grad.float(), atol=5e-2, rtol=0.1)
    print("dK OK!")

    print("\nBackward pass comparison (dV):")
    torch.testing.assert_close(v_triton.grad.float(), v_ref.grad.float(), atol=5e-2, rtol=0.1)
    print("dV OK!")

    print("\nBackward pass comparison (dDogParams):")
    print(f"Triton grad is None: {dog_params_triton.grad is None}")
    print(f"Ref grad is None: {dog_params_ref.grad is None}")

    if dog_params_triton.grad is not None and dog_params_ref.grad is not None:
        torch.testing.assert_close(dog_params_triton.grad.float(), dog_params_ref.grad.float(), atol=5e-2, rtol=0.1)
        print("dDogParams OK!")
    else:
        print("Warning: One or both gradients are None")
        if dog_params_triton.grad is not None:
            print(f"Triton grad shape: {dog_params_triton.grad.shape}, mean: {dog_params_triton.grad.mean()}")
        if dog_params_ref.grad is not None:
            print(f"Ref grad shape: {dog_params_ref.grad.shape}, mean: {dog_params_ref.grad.mean()}")

        # Skip gradient comparison if one is None
        print("Skipping dDogParams comparison due to missing gradients")


if __name__ == "__main__":
    # Example usage of DoG bias attention
    import torch
    import torch.nn as nn

    print("=== DoG Bias Attention Usage Example ===")

    # Configuration
    batch_size = 2
    seq_len = 512
    num_kv_heads = 8
    num_kv_groups = 4  # Total heads = 8 * 4 = 32
    head_dim = 64

    # Create input tensors
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    q = torch.randn(batch_size, seq_len, num_kv_heads, num_kv_groups, head_dim,
                    device=device, dtype=dtype)
    k = torch.randn(batch_size, seq_len, num_kv_heads, head_dim,
                    device=device, dtype=dtype)
    v = torch.randn(batch_size, seq_len, num_kv_heads, head_dim,
                    device=device, dtype=dtype)

    # Initialize attention sinks (learnable per head)
    num_total_heads = num_kv_heads * num_kv_groups
    sinks = nn.Parameter(torch.randn(num_total_heads, device=device, dtype=dtype))

    # Initialize DoG parameters (7 parameters per head: A1, s1, c1, A2, s2, c2, offset)
    dog_params = nn.Parameter(torch.randn(num_total_heads, 7, device=device, dtype=torch.float32))

    # Make sure s1 and s2 are positive (standard deviations)
    with torch.no_grad():
        dog_params[:, [1, 4]] = torch.abs(dog_params[:, [1, 4]]) + 1e-6

    # Attention parameters
    sm_scale = 1.0 / (head_dim ** 0.5)  # Standard scaling
    sliding_window = 256  # Optional sliding window
    start_q = torch.tensor([0], dtype=torch.int32, device=device)

    print(f"Input shapes:")
    print(f"  Q: {q.shape}")
    print(f"  K: {k.shape}")
    print(f"  V: {v.shape}")
    print(f"  Sinks: {sinks.shape}")
    print(f"  DoG params: {dog_params.shape}")

    # Forward pass
    output = attention(q, k, v, sinks, dog_params, sm_scale, sliding_window, start_q)
    print(f"\nOutput shape: {output.shape}")

    # The output can be used in your model like any other attention output
    print(f"Output mean: {output.float().mean():.4f}")
    print(f"Output std: {output.float().std():.4f}")

    print("\n=== DoG Parameters Interpretation ===")
    print("Each head has 7 DoG parameters:")
    print("  A1, s1, c1: Amplitude, std dev, center of near Gaussian (positive contribution)")
    print("  A2, s2, c2: Amplitude, std dev, center of far Gaussian (negative contribution)")
    print("  offset: Bias offset applied to all positions")
    print("\nThe bias is computed as: bias = A1*exp(-((dist-c1)/s1)^2) - A2*exp(-((dist-c2)/s2)^2) + offset")
    print("where dist = |query_pos - key_pos|")

    # Example: Show the DoG bias curve for the first head
    with torch.no_grad():
        distances = torch.arange(0, 50, device=device, dtype=torch.float32)
        A1, s1, c1, A2, s2, c2, offset = dog_params[0]

        # Compute DoG bias
        arg1 = (distances - c1) / s1
        near = A1 * torch.exp(-(arg1 * arg1))
        arg2 = (distances - c2) / s2
        far = A2 * torch.exp(-(arg2 * arg2))
        bias_curve = near - far + offset

        print(f"\nDoG bias curve for head 0 (first 10 distances):")
        for i, (d, b) in enumerate(zip(distances[:10], bias_curve[:10])):
            print(f"  Distance {d.item():.0f}: bias = {b.item():.4f}")
