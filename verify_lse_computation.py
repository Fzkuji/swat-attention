#!/usr/bin/env python3
"""验证 LSE 计算是否正确"""
import torch
import torch.nn.functional as F
from adasplash.lazy_attention_triton import _lazy_attention_forward_return_lse

print("="*80)
print("验证 LSE 计算")
print("="*80)

torch.manual_seed(42)

B, H, L, D = 1, 1, 32, 32
device = 'cuda'

# 创建输入
q = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
k = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
v = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16)

tau = torch.zeros(H, device=device, dtype=torch.bfloat16)
bias = torch.zeros(H, 512, device=device, dtype=torch.bfloat16)

print(f"输入: B={B}, H={H}, L={L}, D={D}")

# ============================================================================
# 1. Triton 计算的 LSE
# ============================================================================
print(f"\n" + "="*80)
print("1. Triton 计算的 LSE")
print("="*80)

out_triton, lse_triton = _lazy_attention_forward_return_lse(q, k, v, bias, tau, window_size=512, varlen=None)

print(f"LSE shape: {lse_triton.shape}")
print(f"LSE[0,0,:10]: {lse_triton[0,0,:10]}")
print(f"LSE mean: {lse_triton.mean().item():.6f}")

# ============================================================================
# 2. PyTorch 计算的 LSE（标准方法）
# ============================================================================
print(f"\n" + "="*80)
print("2. PyTorch 计算的 LSE")
print("="*80)

# QK^T
sm_scale = 1.0 / (D ** 0.5)
scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale  # [B, H, L, L]

# Causal mask
causal_mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.bfloat16))
scores = scores.masked_fill(causal_mask[None, None, :, :] == 0, float('-inf'))

# 计算 LSE（PyTorch 标准方法）
# lse[i] = log(sum(exp(scores[i, j]))) for j <= i
lse_pytorch = torch.zeros(B, H, L, device=device, dtype=torch.float32)

for b in range(B):
    for h in range(H):
        for i in range(L):
            # scores[b, h, i, :i+1] 是这个 query 能看到的所有 keys
            valid_scores = scores[b, h, i, :i+1].float()

            # LSE = log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
            m = valid_scores.max()
            lse_pytorch[b, h, i] = m + torch.log(torch.sum(torch.exp(valid_scores - m)))

print(f"LSE shape: {lse_pytorch.shape}")
print(f"LSE[0,0,:10]: {lse_pytorch[0,0,:10]}")
print(f"LSE mean: {lse_pytorch.mean().item():.6f}")

# ============================================================================
# 3. 对比 LSE
# ============================================================================
print(f"\n" + "="*80)
print("3. 对比 LSE")
print("="*80)

lse_diff = (lse_triton - lse_pytorch).abs()
print(f"LSE 差异:")
print(f"  max: {lse_diff.max().item():.10f}")
print(f"  mean: {lse_diff.mean().item():.10f}")
print(f"  diff[0,0,:10]: {lse_diff[0,0,:10]}")

# ============================================================================
# 4. 验证 exp(s - lse) == softmax(s)
# ============================================================================
print(f"\n" + "="*80)
print("4. 验证 exp(s - lse) == softmax(s)")
print("="*80)

# 使用 Triton 的 LSE
print("\n使用 Triton LSE:")
for i in [0, 1, 5, 10]:
    scores_i = scores[0, 0, i, :i+1].float()

    # 方法1: 标准 softmax
    softmax_std = F.softmax(scores_i, dim=0)

    # 方法2: exp(s - lse)
    p_norm = torch.exp(scores_i - lse_triton[0, 0, i].float())

    diff = (softmax_std - p_norm).abs().max().item()
    print(f"  位置 {i}: softmax vs exp(s-lse) 最大差异 = {diff:.10f}")

# 使用 PyTorch 的 LSE
print("\n使用 PyTorch LSE:")
for i in [0, 1, 5, 10]:
    scores_i = scores[0, 0, i, :i+1].float()

    # 方法1: 标准 softmax
    softmax_std = F.softmax(scores_i, dim=0)

    # 方法2: exp(s - lse)
    p_norm = torch.exp(scores_i - lse_pytorch[0, 0, i])

    diff = (softmax_std - p_norm).abs().max().item()
    print(f"  位置 {i}: softmax vs exp(s-lse) 最大差异 = {diff:.10f}")

# ============================================================================
# 5. 检查 softmax 权重和
# ============================================================================
print(f"\n" + "="*80)
print("5. 检查 softmax 权重和（应该为 1）")
print("="*80)

print("\n使用 Triton LSE 计算的 p_norm:")
for i in [0, 1, 5, 10, 20, 31]:
    scores_i = scores[0, 0, i, :i+1].float()
    p_norm = torch.exp(scores_i - lse_triton[0, 0, i].float())
    sum_p = p_norm.sum().item()
    print(f"  位置 {i}: sum(p_norm) = {sum_p:.10f}, 误差 = {abs(sum_p - 1.0):.10e}")

# ============================================================================
# 结论
# ============================================================================
print(f"\n" + "="*80)
print("结论")
print("="*80)

lse_max_diff = lse_diff.max().item()

if lse_max_diff < 1e-5:
    print("✅ LSE 计算正确")
    print("   Triton 和 PyTorch 的 LSE 一致")
    print("   问题不在 LSE，而在其他地方")
elif lse_max_diff < 0.01:
    print("⚠️ LSE 有小误差")
    print(f"   最大差异: {lse_max_diff:.10f}")
    print("   可能是 bf16 精度问题")
else:
    print("❌ LSE 计算错误！")
    print(f"   最大差异: {lse_max_diff:.10f}")
    print("   这会导致 p_norm 不等于 softmax")
    print("   需要修复 _get_lse_kernel_batch")

print("="*80)
