#!/usr/bin/env python3
"""精确对比 Triton kernel vs PyTorch 的输出和梯度"""
import torch
import torch.nn.functional as F
from adasplash import lazy_attention_triton

print("="*80)
print("精确对比 Triton vs PyTorch 的输出")
print("="*80)

torch.manual_seed(42)

# 较小的输入方便调试
B, H, L, D = 1, 2, 16, 32
device = 'cuda'
tau_init = -1.0

# 创建相同的输入
q = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
k = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
v = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16)

print(f"输入: B={B}, H={H}, L={L}, D={D}")
print(f"  dtype: {q.dtype}")

# 创建相同的参数
tau_init_tensor = torch.full((H,), tau_init, device=device, dtype=torch.bfloat16)
bias_init = torch.randn(H, 512, device=device, dtype=torch.bfloat16) * 0.02

# ============================================================================
# PyTorch 实现
# ============================================================================
print("\n" + "="*80)
print("1. PyTorch 实现")
print("="*80)

tau_pt = torch.nn.Parameter(tau_init_tensor.clone())
bias_pt = torch.nn.Parameter(bias_init.clone())

# 计算 attention（模拟 scratch 分支）
sm_scale = 1.0 / (D ** 0.5)
scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale  # [B, H, L, L]

# Causal mask
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

# Softmax
attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

# Elastic-Softmax
tau_expanded = tau_pt.view(1, H, 1, 1)
i_positions = torch.arange(1, L + 1, device=device, dtype=torch.bfloat16).view(1, 1, L, 1)
elastic_attn = F.relu(attn_weights + tau_expanded / i_positions)

# Output
out_pt = torch.matmul(elastic_attn, v)  # [B, H, L, D]

print(f"输出统计:")
print(f"  attn_weights[0, 0, 0, :5]: {attn_weights[0, 0, 0, :5]}")
print(f"  elastic_attn[0, 0, 0, :5]: {elastic_attn[0, 0, 0, :5]}")
print(f"  out_pt[0, 0, 0, :5]: {out_pt[0, 0, 0, :5]}")
print(f"  out_pt mean: {out_pt.mean().item():.10f}")
print(f"  out_pt std: {out_pt.std().item():.10f}")

# Loss
loss_pt = out_pt.sum()
print(f"\nLoss: {loss_pt.item():.10f}")

# 反向
loss_pt.backward()

print(f"\n梯度:")
print(f"  tau.grad: {tau_pt.grad}")
print(f"  tau.grad sum: {tau_pt.grad.sum().item():.10f}")
print(f"  bias.grad[0, :5]: {bias_pt.grad[0, :5]}")
print(f"  bias.grad sum: {bias_pt.grad.sum().item():.10f}")

# ============================================================================
# Triton 实现
# ============================================================================
print("\n" + "="*80)
print("2. Triton 实现")
print("="*80)

tau_triton = torch.nn.Parameter(tau_init_tensor.clone())
bias_triton = torch.nn.Parameter(bias_init.clone())

# 调用 Triton kernel
out_triton = lazy_attention_triton(q, k, v, bias_triton, tau_triton)

print(f"输出统计:")
print(f"  out_triton[0, 0, 0, :5]: {out_triton[0, 0, 0, :5]}")
print(f"  out_triton mean: {out_triton.mean().item():.10f}")
print(f"  out_triton std: {out_triton.std().item():.10f}")

# Loss
loss_triton = out_triton.sum()
print(f"\nLoss: {loss_triton.item():.10f}")

# 反向
loss_triton.backward()

print(f"\n梯度:")
print(f"  tau.grad: {tau_triton.grad}")
print(f"  tau.grad sum: {tau_triton.grad.sum().item():.10f}")
print(f"  bias.grad[0, :5]: {bias_triton.grad[0, :5]}")
print(f"  bias.grad sum: {bias_triton.grad.sum().item():.10f}")

# ============================================================================
# 对比
# ============================================================================
print("\n" + "="*80)
print("3. 对比分析")
print("="*80)

# 前向差异
out_diff = (out_pt - out_triton).abs()
print(f"\n前向输出差异:")
print(f"  最大差异: {out_diff.max().item():.10f}")
print(f"  平均差异: {out_diff.mean().item():.10f}")
print(f"  相对差异: {(out_diff / (out_pt.abs() + 1e-8)).mean().item() * 100:.4f}%")

loss_diff = abs(loss_pt.item() - loss_triton.item())
print(f"\nLoss 差异: {loss_diff:.10f}")
print(f"  PyTorch: {loss_pt.item():.10f}")
print(f"  Triton:  {loss_triton.item():.10f}")
print(f"  相对差异: {loss_diff / max(abs(loss_pt.item()), 1e-10) * 100:.4f}%")

# 梯度差异
tau_grad_diff = (tau_pt.grad - tau_triton.grad).abs()
print(f"\nTau 梯度差异:")
print(f"  最大差异: {tau_grad_diff.max().item():.10f}")
print(f"  平均差异: {tau_grad_diff.mean().item():.10f}")
if tau_pt.grad.abs().max().item() > 1e-10:
    print(f"  相对差异: {(tau_grad_diff / (tau_pt.grad.abs() + 1e-10)).mean().item() * 100:.4f}%")

bias_grad_diff = (bias_pt.grad - bias_triton.grad).abs()
print(f"\nBias 梯度差异:")
print(f"  最大差异: {bias_grad_diff.max().item():.10f}")
print(f"  平均差异: {bias_grad_diff.mean().item():.10f}")
if bias_pt.grad.abs().max().item() > 1e-10:
    print(f"  相对差异: {(bias_grad_diff / (bias_pt.grad.abs() + 1e-10)).mean().item() * 100:.4f}%")

# 结论
print("\n" + "="*80)
print("结论:")
print("="*80)

out_rel_diff = (out_diff / (out_pt.abs() + 1e-8)).mean().item()
tau_rel_diff = (tau_grad_diff / (tau_pt.grad.abs() + 1e-10)).mean().item() if tau_pt.grad.abs().max().item() > 1e-10 else 0
bias_rel_diff = (bias_grad_diff / (bias_pt.grad.abs() + 1e-10)).mean().item() if bias_pt.grad.abs().max().item() > 1e-10 else 0

if out_rel_diff > 0.01:  # >1%
    print("❌ 前向输出有明显差异 (>1%)")
    print("   可能原因: Triton kernel 前向实现有误")
elif tau_rel_diff > 0.1:  # >10%
    print("❌ Tau 梯度有明显差异 (>10%)")
    print("   可能原因: dtau 计算或累加有问题")
elif bias_rel_diff > 0.1:  # >10%
    print("❌ Bias 梯度有明显差异 (>10%)")
    print("   可能原因: dbias 计算或累加有问题")
else:
    print("✅ 单步前向和梯度基本一致")
    print(f"   前向相对差异: {out_rel_diff*100:.4f}%")
    print(f"   Tau梯度相对差异: {tau_rel_diff*100:.4f}%")
    print(f"   Bias梯度相对差异: {bias_rel_diff*100:.4f}%")
    print("\n如果训练效果仍然差，可能是:")
    print("   1. 多步累积的微小数值误差")
    print("   2. 某些特殊情况（如 varlen）处理不同")
    print("   3. 需要调整超参数（学习率、warmup等）")

print("="*80)
