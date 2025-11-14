# -*- coding: utf-8 -*-
# Copyright (c) 2023-2025, Tri Dao, Yu Zhang, Songlin Yang.

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

from fla.ops.utils.op import exp, log
from fla.utils import autocast_custom_bwd, autocast_custom_fwd, input_guard, is_amd

try:
    from torch.distributed.tensor import DTensor
except (ImportError, AttributeError):
    DTensor = None

NUM_WARPS_AUTOTUNE = [1, 2, 4, 8, 16] if is_amd else [1, 2, 4, 8, 16, 32]


@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D']
)
@triton.jit(do_not_specialize=['T'])
def sigmoid_fwd_kernel(
    x, y,
    T,
    B: tl.constexpr,
    D: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < T
    x_val = tl.load(x + offs, mask=mask, other=0.).to(tl.float32)
    y_val = 1.0 / (1.0 + exp(-x_val))
    tl.store(y + offs, y_val.to(y.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D']
)
@triton.jit(do_not_specialize=['T'])
def sigmoid_bwd_kernel(
    x, dy, dx,
    T,
    B: tl.constexpr,
    D: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < T
    x_val = tl.load(x + offs, mask=mask, other=0.).to(tl.float32)
    g_val = tl.load(dy + offs, mask=mask, other=0.).to(tl.float32)
    s = 1.0 / (1.0 + exp(-x_val))
    dx_val = g_val * s * (1.0 - s)
    tl.store(dx + offs, dx_val.to(dx.dtype.element_ty), mask=mask)


def sigmoid_fwd(x: torch.Tensor) -> torch.Tensor:
    T, D = x.numel(), x.shape[-1]
    y = torch.empty_like(x)
    sigmoid_fwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](x, y, T=T, D=D)
    return y


def sigmoid_bwd(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    T, D = x.numel(), x.shape[-1]
    dx = torch.empty_like(x)
    sigmoid_bwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](x, dy, dx, T=T, D=D)
    return dx


class SigmoidFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return sigmoid_fwd(x)

    @staticmethod
    def backward(ctx, dout):
        x, = ctx.saved_tensors
        return sigmoid_bwd(x, dout)


sigmoid = SigmoidFunction.apply


@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D']
)
@triton.jit(do_not_specialize=['T'])
def logsigmoid_fwd_kernel(
    x,
    y,
    temperature,
    T,
    B: tl.constexpr,
    D: tl.constexpr
):
    i = tl.program_id(0)
    o_i = i * B + tl.arange(0, B)
    m_i = o_i < T

    b_x = tl.load(x + o_i, mask=m_i, other=0.).to(tl.float32)
    b_m = tl.minimum(0., b_x)
    b_z = 1. + exp(-tl.abs(b_x))
    b_y = (b_m - log(b_z)) / temperature
    tl.store(y + o_i, b_y.to(y.dtype.element_ty), mask=m_i)


@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D']
)
@triton.jit(do_not_specialize=['T'])
def logsigmoid_bwd_kernel(
    x,
    dx,
    dy,
    temperature,
    T,
    B: tl.constexpr,
    D: tl.constexpr
):
    i = tl.program_id(0)
    o_i = i * B + tl.arange(0, B)
    m_i = o_i < T

    b_x = tl.load(x + o_i, mask=m_i, other=0.).to(tl.float32)
    b_dy = tl.load(dy + o_i, mask=m_i, other=0.).to(tl.float32)
    b_dx = b_dy * ((1. - tl.sigmoid(b_x)) / temperature)
    tl.store(dx + o_i, b_dx.to(dx.dtype.element_ty), mask=m_i)


def logsigmoid_fwd(x: torch.Tensor, temperature: float = 1.) -> torch.Tensor:
    T, D = x.numel(), x.shape[-1]
    y = torch.empty_like(x)
    logsigmoid_fwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](
        x=x,
        y=y,
        temperature=temperature,
        T=T,
        D=D,
    )
    return y


def logsigmoid_bwd(x: torch.Tensor, dy: torch.Tensor, temperature: float = 1.) -> torch.Tensor:
    T, D = x.numel(), x.shape[-1]
    dx = torch.empty_like(x)
    logsigmoid_bwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](
        x=x,
        dx=dx,
        dy=dy,
        temperature=temperature,
        T=T,
        D=D,
    )
    return dx


class LogSigmoidFunction(torch.autograd.Function):

    @staticmethod
    @input_guard
    def forward(ctx, x, temperature):
        ctx.save_for_backward(x,)
        ctx.temperature = temperature
        return logsigmoid_fwd(x, temperature)

    @staticmethod
    @input_guard
    def backward(ctx, dy):
        x, = ctx.saved_tensors
        return logsigmoid_bwd(x, dy, ctx.temperature), None


def logsigmoid(x: torch.Tensor, temperature: float = 1.) -> torch.Tensor:
    return LogSigmoidFunction.apply(x, temperature)


@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D']
)
@triton.jit(do_not_specialize=['T'])
def swish_fwd_kernel(
    x, y,
    T,
    B: tl.constexpr,
    D: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < T
    x_val = tl.load(x + offs, mask=mask, other=0.).to(tl.float32)
    s = 1.0 / (1.0 + exp(-x_val))
    y_val = x_val * s
    tl.store(y + offs, y_val.to(y.dtype.element_ty), mask=mask)


@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D']
)
@triton.jit(do_not_specialize=['T'])
def swish_bwd_kernel(
    x, dy, dx,
    T,
    B: tl.constexpr,
    D: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < T
    x_val = tl.load(x + offs, mask=mask, other=0.).to(tl.float32)
    g_val = tl.load(dy + offs, mask=mask, other=0.).to(tl.float32)
    s = 1.0 / (1.0 + exp(-x_val))
    dx_val = g_val * s * (1.0 + x_val * (1.0 - s))
    tl.store(dx + offs, dx_val.to(dx.dtype.element_ty), mask=mask)


def swish_fwd(x: torch.Tensor) -> torch.Tensor:
    T, D = x.numel(), x.shape[-1]
    y = torch.empty_like(x)
    swish_fwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](x, y, T=T, D=D)
    return y


def swish_bwd(x: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
    T, D = x.numel(), x.shape[-1]
    dx = torch.empty_like(x)
    swish_bwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](x, dy, dx, T=T, D=D)
    return dx


class SwishFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return swish_fwd(x)

    @staticmethod
    def backward(ctx, dout):
        x, = ctx.saved_tensors
        return swish_bwd(x, dout)


swish = SwishFunction.apply

# 1/sqrt(2*pi)-> 0.3989423
# 1/sqrt(2)   -> 0.70710678
# sqrt(2/pi)  -> 0.79788456


# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + torch.erf(x * 0.70710678))
@torch.compile
def bias_gelu(y, bias):
    x = bias + y
    return (x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))).to(dtype=y.dtype)


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.compile
def bias_gelu_bwd(g, y, bias):
    """Assume that y has shape (B, D=D) and bias has shape (D)"""
    x = bias + y
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (
        1 + tanh_out
    )
    grad_y = ff * g
    return grad_y.to(dtype=y.dtype), grad_y.sum(dim=(0), dtype=bias.dtype)


class GeLUFunction(torch.autograd.Function):

    @staticmethod
    # bias is an optional argument
    def forward(ctx, input, bias):
        ctx.save_for_backward(input, bias)
        return bias_gelu(input, bias)

    @staticmethod
    def backward(ctx, grad_output):
        input, bias = ctx.saved_tensors
        tmp = bias_gelu_bwd(grad_output, input, bias)
        return tmp, tmp


bias_gelu_impl = GeLUFunction.apply


# this function is tanh approximation of gelu
# actual gelu is:
# x * 0.5 * (1.0 + torch.erf(x * 0.70710678))
@torch.compile
def gelu_fwd(x):
    return (x * 0.5 * (1.0 + torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))).to(dtype=x.dtype)


# gradient of tanh approximation of gelu
# gradient of actual gelu is:
# 0.5 * (1. + torch.erf(x * 0.70710678)) + 0.3989423 * x * torch.exp(-0.5 * x * x)
@torch.compile
def gelu_bwd(g, x):
    tanh_out = torch.tanh(0.79788456 * x * (1 + 0.044715 * x * x))
    # sqrt(2/pi) * 3 * 0.044715 -> 0.1070322243
    ff = 0.5 * x * ((1 - tanh_out * tanh_out) * (0.79788456 + 0.1070322243 * x * x)) + 0.5 * (
        1 + tanh_out
    )
    return (ff * g).to(dtype=x.dtype)


class FastGeLUFunction(torch.autograd.Function):
    @staticmethod
    # bias is an optional argument
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return gelu_fwd(input)

    @staticmethod
    def backward(ctx, grad_output):
        (input,) = ctx.saved_tensors
        tmp = gelu_bwd(grad_output, input)
        return tmp


fast_gelu_impl = FastGeLUFunction.apply


@torch.compile
def relu_bwd(g, x):
    return torch.where(x >= 0, g, 0.0).to(dtype=x.dtype)


@torch.compile
def sqrelu_fwd(x):
    r = F.relu(x.float())
    return (r * r).to(dtype=x.dtype)


@torch.compile
def sqrelu_bwd(g, x):
    return (2.0 * g * F.relu(x.float())).to(dtype=x.dtype)


class SquaredReLUFunction(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        return sqrelu_fwd(input)

    @staticmethod
    def backward(ctx, grad_output):
        input, = ctx.saved_tensors
        return sqrelu_bwd(grad_output, input)


sqrelu = SquaredReLUFunction.apply


@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D']
)
@triton.jit(do_not_specialize=['T'])
def swiglu_fwd_kernel(
    x, y, z,
    T,
    B: tl.constexpr,
    D: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < T
    x_val = tl.load(x + offs, mask=mask, other=0.).to(tl.float32)
    y_val = tl.load(y + offs, mask=mask, other=0.).to(tl.float32)
    s = 1.0 / (1.0 + exp(-x_val))
    z_val = x_val * s * y_val
    tl.store(z + offs, z_val.to(z.dtype.element_ty), mask=mask)


@triton.heuristics({
    'HAS_WEIGHT': lambda args: args['z'] is not None,
})
@triton.autotune(
    configs=[
        triton.Config({'B': bs}, num_warps=num_warps)
        for bs in [512, 1024, 2048, 4096, 8192]
        for num_warps in NUM_WARPS_AUTOTUNE
    ],
    key=['D']
)
@triton.jit(do_not_specialize=['T'])
def swiglu_fwdbwd_kernel(
    x, y, g, dx, dy, z,
    T,
    B: tl.constexpr,
    D: tl.constexpr,
    HAS_WEIGHT: tl.constexpr
):
    pid = tl.program_id(0)
    offs = pid * B + tl.arange(0, B)
    mask = offs < T
    x_val = tl.load(x + offs, mask=mask, other=0.).to(tl.float32)
    y_val = tl.load(y + offs, mask=mask, other=0.).to(tl.float32)
    g_val = tl.load(g + offs, mask=mask, other=0.).to(tl.float32)

    s = 1.0 / (1.0 + exp(-x_val))
    x_s = x_val * s
    dx_val = g_val * s * (1.0 + x_val * (1.0 - s)) * y_val
    dy_val = g_val * x_s

    tl.store(dx + offs, dx_val.to(dx.dtype.element_ty), mask=mask)
    tl.store(dy + offs, dy_val.to(dy.dtype.element_ty), mask=mask)
    if HAS_WEIGHT:
        z_val = x_s * y_val
        tl.store(z + offs, z_val.to(z.dtype.element_ty),  mask=mask)


def swiglu_fwd(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    T, D = x.numel(), x.shape[-1]
    z = torch.empty_like(x)
    swiglu_fwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](x, y, z, T=T, D=D)
    return z


def swiglu_fwdbwd(x: torch.Tensor, y: torch.Tensor, g: torch.Tensor, use_weight: bool = False):
    T, D = x.numel(), x.shape[-1]
    dx = torch.empty_like(x)
    dy = torch.empty_like(x)
    if use_weight:
        # recomputed for weight grad
        z = torch.empty_like(x)
    else:
        z = None
    swiglu_fwdbwd_kernel[lambda meta: (triton.cdiv(T, meta['B']),)](x, y, g, dx, dy, z, T=T, D=D)
    if use_weight:
        return dx, dy, z
    return dx, dy


class SwiGLUFunction(torch.autograd.Function):
    r"""
    Swish-Gated Linear Unit (SwiGLU) function.

    .. math::
        \text{SwiGLU}(x, y) = swish(x) * y = \frac{x}{1 + \exp(-x)} * y
    """

    @staticmethod
    def forward(ctx, x, y):
        ctx.save_for_backward(x, y)
        return swiglu_fwd(x, y)

    @staticmethod
    def backward(ctx, dout):
        x, y = ctx.saved_tensors
        return swiglu_fwdbwd(x, y, dout)


class SwiGLULinearFunction(torch.autograd.Function):
    r"""
    Swish-Gated Linear Unit (SwiGLU) function followed by a linear transformation.

    .. math::
        \text{SwiGLULinear}(x, y, W, b) = (swish(x) * y) W + b

    This simple wrap discards the intermediate results of SwiGLU(x, y) to save memory.
    """

    @staticmethod
    @autocast_custom_fwd
    def forward(ctx, x, y, weight, bias):
        z = swiglu_fwd(x, y)
        out = F.linear(z, weight, bias)
        # We don't store z, will be recomputed in the backward pass to save memory
        ctx.save_for_backward(x, y, weight)
        ctx.linear_bias_is_none = bias is None
        return out

    @staticmethod
    @autocast_custom_bwd
    def backward(ctx, dout, *args):
        x, y, weight = ctx.saved_tensors
        dout = dout.reshape(-1, dout.shape[-1])
        dz = F.linear(dout, weight.t()).view_as(x)
        dx, dy, z = swiglu_fwdbwd(x, y, dz, use_weight=True)
        dlinear_weight = torch.einsum("bo,bi->oi", dout, z.reshape(-1, z.shape[-1]))
        dlinear_bias = None if ctx.linear_bias_is_none else dout.sum(0)
        return dx, dy, dlinear_weight, dlinear_bias


swiglu = SwiGLUFunction.apply


swiglu_linear = SwiGLULinearFunction.apply


def _make_ix_like(x, dim):
    """Create an index tensor along the specified dimension."""
    d = x.size(dim)
    rho = torch.arange(1, d + 1, device=x.device, dtype=x.dtype)
    view = [1] * x.dim()
    view[0] = -1
    return rho.view(view).transpose(0, dim)


def _threshold_and_support(z, dim=-1):
    """
    Compute the threshold and support size for sparsemax.

    This is the core computation for sparsemax: we sort the input,
    compute cumulative sums, and find the threshold value.
    """
    sorted_z, _ = torch.sort(z, descending=True, dim=dim)
    z_cumsum = torch.cumsum(sorted_z, dim=dim) - 1
    k = _make_ix_like(z, dim)
    support = k * sorted_z > z_cumsum

    k_z = support.sum(dim=dim, keepdim=True)
    tau_z = z_cumsum.gather(dim, k_z - 1) / k_z.to(z.dtype)
    return tau_z, k_z


def sparsemax_fwd(x, dim=-1):
    """
    Sparsemax activation function (forward pass).

    Sparsemax is a sparse alternative to softmax:
    - Returns sparse probability distributions (many zeros)
    - Equivalent to projection onto the probability simplex
    - Can be seen as entmax with α=2

    Reference: "From Softmax to Sparsemax: A Sparse Model of Attention and Multi-Label Classification"
    https://arxiv.org/abs/1602.02068
    """
    tau_z, _ = _threshold_and_support(x, dim=dim)
    output = torch.clamp(x - tau_z, min=0)
    return output


def sparsemax_bwd(output, grad_output, dim=-1):
    """
    Sparsemax backward pass.

    The gradient is computed based on the support of the output
    (i.e., which elements are non-zero).
    """
    support = output > 0
    v_hat = (grad_output * support.to(grad_output.dtype)).sum(dim=dim, keepdim=True) / support.sum(dim=dim, keepdim=True).to(grad_output.dtype)
    grad_input = support.to(grad_output.dtype) * (grad_output - v_hat)
    return grad_input


class SparsemaxFunction(torch.autograd.Function):
    """
    Sparsemax activation function with autograd support.
    """

    @staticmethod
    def forward(ctx, x, dim=-1):
        output = sparsemax_fwd(x, dim=dim)
        ctx.save_for_backward(output)
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        output, = ctx.saved_tensors
        grad_input = sparsemax_bwd(output, grad_output, dim=ctx.dim)
        return grad_input, None


sparsemax = SparsemaxFunction.apply


def _entmax_threshold_and_support(x, alpha=1.5, dim=-1):
    """
    Core computation for entmax: computing the threshold via bisection.

    Entmax is a family of sparse attention mechanisms parameterized by α:
    - α = 1: softmax (dense)
    - α = 1.5: entmax-1.5 (moderately sparse)
    - α = 2: sparsemax (very sparse)
    """
    def p_alpha(x, alpha):
        """Compute p(x) = max(x, 0)^(1/(alpha-1))"""
        return torch.pow(torch.clamp(x, min=0), 1.0 / (alpha - 1.0))

    x_sorted, _ = torch.sort(x, descending=True, dim=dim)

    rho = _make_ix_like(x, dim)
    mean = x_sorted.cumsum(dim=dim) / rho
    mass = rho * torch.pow(torch.clamp(x_sorted - mean, min=0), alpha - 1.0)

    found = (mass.cumsum(dim=dim) - 1.0) < torch.pow(torch.clamp(x_sorted - mean, min=0), alpha - 1.0) * rho

    rho_star = found.sum(dim=dim, keepdim=True)
    threshold = x_sorted.gather(dim, rho_star - 1)

    return threshold, rho_star


def entmax_fwd(x, alpha=1.5, dim=-1):
    """
    Entmax activation function (forward pass).

    Entmax is a family of normalizations that generalizes softmax and sparsemax:
    - α = 1.0: equivalent to softmax
    - α = 1.5: entmax-1.5 (moderately sparse, good balance)
    - α = 2.0: equivalent to sparsemax (very sparse)

    The parameter α controls the sparsity of the output distribution.

    Reference: "Adaptively Sparse Transformers"
    https://arxiv.org/abs/1909.00015
    """
    if alpha == 1.0:
        return F.softmax(x, dim=dim)

    if alpha == 2.0:
        return sparsemax_fwd(x, dim=dim)

    # General case: 1 < alpha < 2
    threshold, _ = _entmax_threshold_and_support(x, alpha=alpha, dim=dim)
    output = torch.pow(torch.clamp(x - threshold, min=0), 1.0 / (alpha - 1.0))

    # Normalize to sum to 1
    output = output / output.sum(dim=dim, keepdim=True)

    return output


def entmax_bwd(x, output, grad_output, alpha=1.5, dim=-1):
    """
    Entmax backward pass.
    """
    if alpha == 1.0:
        # Softmax gradient
        gppr = grad_output * output
        grad_input = gppr - output * gppr.sum(dim=dim, keepdim=True)
        return grad_input

    if alpha == 2.0:
        return sparsemax_bwd(output, grad_output, dim=dim)

    # General case
    support = output > 0
    d_output = torch.where(
        support,
        grad_output * torch.pow(output, 2.0 - alpha),
        torch.zeros_like(grad_output)
    )

    val = d_output.sum(dim=dim, keepdim=True) / support.sum(dim=dim, keepdim=True).to(d_output.dtype)
    grad_input = torch.where(support, d_output - val, torch.zeros_like(d_output))

    return grad_input


class EntmaxFunction(torch.autograd.Function):
    """
    Entmax activation function with autograd support.
    """

    @staticmethod
    def forward(ctx, x, alpha=1.5, dim=-1):
        output = entmax_fwd(x, alpha=alpha, dim=dim)
        ctx.save_for_backward(x, output)
        ctx.alpha = alpha
        ctx.dim = dim
        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, output = ctx.saved_tensors
        grad_input = entmax_bwd(x, output, grad_output, alpha=ctx.alpha, dim=ctx.dim)
        return grad_input, None, None


def entmax(x, alpha=1.5, dim=-1):
    """
    Entmax activation function.

    Args:
        x: Input tensor
        alpha: Sparsity parameter (1.0=softmax, 1.5=entmax-1.5, 2.0=sparsemax)
        dim: Dimension along which to apply entmax

    Returns:
        Tensor with same shape as input, normalized along dim
    """
    return EntmaxFunction.apply(x, alpha, dim)


ACT2FN = {
    'relu': F.relu,
    'sigmoid': sigmoid,
    'logsigmoid': logsigmoid,
    'silu': swish,
    'swish': swish,
    'sqrelu': sqrelu,
    'gelu': fast_gelu_impl,
    'bias_gelu': bias_gelu_impl,
}
