#!/usr/bin/env python3
"""验证 tau 处理修复是否成功"""
import torch
import torch.nn.functional as F
from adasplash import lazy_attention_triton

print("="*80)
print("验证 Tau 处理修复")
print("="*80)

torch.manual_seed(42)

B, H, L, D = 1, 2, 32, 32
device = 'cuda'
tau_val = -1.0

# 创建输入
q = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
k = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
v = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16)

print(f"输入: B={B}, H={H}, L={L}, D={D}")
print(f"  tau: {tau_val}")

# ============================================================================
# PyTorch 参考实现
# ============================================================================
tau_pt = torch.nn.Parameter(torch.full((H,), tau_val, device=device, dtype=torch.bfloat16))
bias_pt = torch.nn.Parameter(torch.zeros(H, 512, device=device, dtype=torch.bfloat16))

sm_scale = 1.0 / (D ** 0.5)
scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
causal_mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.bfloat16))
scores = scores.masked_fill(causal_mask[None, None, :, :] == 0, float('-inf'))

attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)

# Elastic-Softmax
tau_expanded = tau_pt.view(1, H, 1, 1)
i_positions = torch.arange(1, L + 1, device=device, dtype=torch.bfloat16).view(1, 1, L, 1)
elastic_attn = F.relu(attn_weights + tau_expanded / i_positions)

out_pt = torch.matmul(elastic_attn, v)

print(f"\nPyTorch 参考:")
print(f"  elastic_attn 非零比例: {(elastic_attn > 1e-6).float().mean().item() * 100:.2f}%")
print(f"  out mean: {out_pt.mean().item():.10f}")
print(f"  out std: {out_pt.std().item():.10f}")

# ============================================================================
# Triton 实现
# ============================================================================
tau_triton = torch.nn.Parameter(torch.full((H,), tau_val, device=device, dtype=torch.bfloat16))
bias_triton = torch.nn.Parameter(torch.zeros(H, 512, device=device, dtype=torch.bfloat16))

out_triton = lazy_attention_triton(q, k, v, bias_triton, tau_triton)

print(f"\nTriton 实现:")
print(f"  out mean: {out_triton.mean().item():.10f}")
print(f"  out std: {out_triton.std().item():.10f}")

# ============================================================================
# 对比
# ============================================================================
diff = (out_pt - out_triton).abs()
rel_diff = (diff / (out_pt.abs() + 1e-8)).mean().item()

print(f"\n" + "="*80)
print(f"差异分析:")
print(f"  最大差异: {diff.max().item():.10f}")
print(f"  平均差异: {diff.mean().item():.10f}")
print(f"  相对差异: {rel_diff * 100:.4f}%")

# ============================================================================
# 结论
# ============================================================================
print("\n" + "="*80)

# 当 tau=-1 时，期望输出应该全为0或接近0
expected_near_zero = out_pt.abs().max().item() < 0.1

if expected_near_zero:
    # 输出应该接近0
    if out_triton.abs().max().item() < 0.001:
        print("✅ 修复成功!")
        print(f"   tau={tau_val} 时，Triton 输出正确（接近0）")
    else:
        print("❌ 修复失败!")
        print(f"   tau={tau_val} 时，Triton 输出应该接近0，但实际为 {out_triton.abs().max().item():.6f}")
else:
    # 输出不为0，比较相对差异
    if rel_diff < 0.01:  # <1%
        print("✅ 修复成功!")
        print(f"   Triton 与 PyTorch 输出一致 (相对差异 {rel_diff*100:.4f}%)")
    else:
        print("❌ 修复失败!")
        print(f"   Triton 与 PyTorch 差异过大 (相对差异 {rel_diff*100:.4f}%)")

print("="*80)
