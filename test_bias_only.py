#!/usr/bin/env python3
"""测试只有 bias，不加 tau 的情况"""
import torch
import torch.nn.functional as F
from adasplash import lazy_attention_triton

print("="*80)
print("测试 Bias (tau=0)")
print("="*80)

torch.manual_seed(42)

B, H, L, D = 1, 2, 32, 32
device = 'cuda'

# 创建输入
q = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
k = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
v = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16)

print(f"输入: B={B}, H={H}, L={L}, D={D}")

# 创建 bias
bias_data = torch.randn(H, 512, device=device, dtype=torch.bfloat16) * 0.02

# ============================================================================
# PyTorch 实现 (bias, tau=0)
# ============================================================================
tau_pt = torch.nn.Parameter(torch.zeros(H, device=device, dtype=torch.bfloat16))
bias_pt = torch.nn.Parameter(bias_data.clone())

sm_scale = 1.0 / (D ** 0.5)
scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
causal_mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.bfloat16))
scores = scores.masked_fill(causal_mask[None, None, :, :] == 0, float('-inf'))

# Apply bias
for b in range(B):
    for h in range(H):
        for i in range(L):
            for j in range(i + 1):
                dist = i - j
                if dist < 512:
                    scores[b, h, i, j] += bias_pt[h, dist]

attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
out_pt = torch.matmul(attn_weights, v)

print(f"\nPyTorch:")
print(f"  out mean: {out_pt.mean().item():.10f}")
print(f"  out std: {out_pt.std().item():.10f}")

# ============================================================================
# Triton 实现 (bias, tau=0)
# ============================================================================
tau_triton = torch.nn.Parameter(torch.zeros(H, device=device, dtype=torch.bfloat16))
bias_triton = torch.nn.Parameter(bias_data.clone())

out_triton = lazy_attention_triton(q, k, v, bias_triton, tau_triton)

print(f"\nTriton:")
print(f"  out mean: {out_triton.mean().item():.10f}")
print(f"  out std: {out_triton.std().item():.10f}")

# 对比
diff = (out_pt - out_triton).abs()
rel_diff = (diff / (out_pt.abs() + 1e-8)).mean().item()

print(f"\n" + "="*80)
print(f"差异:")
print(f"  最大差异: {diff.max().item():.10f}")
print(f"  平均差异: {diff.mean().item():.10f}")
print(f"  相对差异: {rel_diff * 100:.4f}%")

if rel_diff < 0.01:
    print("✅ Bias 处理正确 (tau=0)")
else:
    print("❌ Bias 处理有问题!")
    print(f"\n可能原因:")
    print(f"  1. dist_clamped 计算错误")
    print(f"  2. in_window mask 不对")
    print(f"  3. bias 被加了错误的次数")

print("="*80)
