#!/usr/bin/env python3
"""测试 Triton kernel 的 softmax 计算是否正确"""
import torch
import torch.nn.functional as F
from adasplash import lazy_attention_triton

print("="*80)
print("测试 Triton kernel Softmax 计算")
print("="*80)

torch.manual_seed(42)

B, H, L, D = 1, 1, 4, 8  # 非常小的尺寸
device = 'cuda'

# 创建输入
q = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.1
k = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.1
v = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16)

print(f"输入: B={B}, H={H}, L={L}, D={D}")

# PyTorch 实现（只到 softmax，不加 tau）
sm_scale = 1.0 / (D ** 0.5)
scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
print(f"\nScores (QK^T/sqrt(d)):")
print(scores[0, 0])

# Causal mask
causal_mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.bfloat16))
scores = scores.masked_fill(causal_mask[None, None, :, :] == 0, float('-inf'))
print(f"\nScores (after causal mask):")
print(scores[0, 0])

# Softmax
attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
print(f"\nSoftmax weights (PyTorch):")
print(attn_weights[0, 0])

# Output (without tau)
out_pt = torch.matmul(attn_weights, v)
print(f"\nOutput (PyTorch): {out_pt[0, 0]}")

# Triton 实现 (设置 tau=0，相当于只有 softmax)
tau_zero = torch.nn.Parameter(torch.zeros(H, device=device, dtype=torch.bfloat16))
bias_zero = torch.nn.Parameter(torch.zeros(H, 512, device=device, dtype=torch.bfloat16))

out_triton = lazy_attention_triton(q, k, v, bias_zero, tau_zero)
print(f"\nOutput (Triton, tau=0, bias=0): {out_triton[0, 0]}")

# 对比
diff = (out_pt - out_triton).abs()
print(f"\n" + "="*80)
print(f"Output 差异: max={diff.max().item():.10f}, mean={diff.mean().item():.10f}")
if diff.max().item() < 0.01:
    print("✅ Softmax 计算基本正确")
else:
    print("❌ Softmax 计算有问题!")
    print(f"   相对差异: {(diff / (out_pt.abs() + 1e-8)).mean().item() * 100:.2f}%")

print("="*80)
