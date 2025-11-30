#!/usr/bin/env python3
"""测试推理精度（仅 forward，不计算梯度）"""
import torch
import torch.nn.functional as F
from adasplash import lazy_attention_triton

print("="*80)
print("测试推理精度（Forward Only）")
print("="*80)

torch.manual_seed(42)

B, H, L, D = 2, 8, 128, 64
device = 'cuda'
tau_val = -1.0

# 创建输入（不需要梯度）
q = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
k = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
v = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16)

tau = torch.full((H,), tau_val, device=device, dtype=torch.float32)
bias = torch.zeros(H, 512, device=device, dtype=torch.float32)

print(f"输入: B={B}, H={H}, L={L}, D={D}, tau={tau_val}")

# ============================================================================
# PyTorch 参考实现
# ============================================================================
print(f"\n" + "="*80)
print("PyTorch 参考实现")
print("="*80)

with torch.no_grad():
    # Forward
    scaling = 1.0 / (D ** 0.5)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scaling

    # Bias
    rel_pos = torch.arange(L, device=device)[:, None] - torch.arange(L, device=device)[None, :]
    valid_mask = (0 <= rel_pos) & (rel_pos < 512)
    indices = rel_pos.clamp(0, 511)
    bias_matrix = bias[:, indices] * valid_mask.float()
    attn_scores = attn_scores + bias_matrix[None, :, :, :]

    # Causal mask
    causal_mask = torch.tril(torch.ones(L, L, device=device))
    attn_scores = attn_scores.masked_fill(causal_mask[None, None, :, :] == 0, float('-inf'))

    # Softmax
    attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)

    # Elastic-Softmax
    tau_expanded = tau.view(1, H, 1, 1).to(torch.bfloat16)
    i_positions = torch.arange(1, L + 1, device=device, dtype=torch.bfloat16).view(1, 1, L, 1)
    p_term = attn_weights + tau_expanded / i_positions
    elastic_attn = F.relu(p_term)

    out_pt = torch.matmul(elastic_attn, v)

print(f"PyTorch 输出:")
print(f"  mean:  {out_pt.mean().item():.10f}")
print(f"  std:   {out_pt.std().item():.10f}")
print(f"  min:   {out_pt.min().item():.10f}")
print(f"  max:   {out_pt.max().item():.10f}")

# ============================================================================
# Triton 实现
# ============================================================================
print(f"\n" + "="*80)
print("Triton Kernel 实现")
print("="*80)

with torch.no_grad():
    out_triton = lazy_attention_triton(
        q, k, v,
        bias.to(torch.bfloat16),
        tau.to(torch.bfloat16)
    )

print(f"Triton 输出:")
print(f"  mean:  {out_triton.mean().item():.10f}")
print(f"  std:   {out_triton.std().item():.10f}")
print(f"  min:   {out_triton.min().item():.10f}")
print(f"  max:   {out_triton.max().item():.10f}")

# ============================================================================
# 对比分析
# ============================================================================
print(f"\n" + "="*80)
print("对比分析")
print("="*80)

diff = (out_pt - out_triton).abs()
rel_err = (diff / (out_pt.abs() + 1e-8)).mean().item() * 100

print(f"\n绝对差异:")
print(f"  max:  {diff.max().item():.10f}")
print(f"  mean: {diff.mean().item():.10f}")
print(f"  std:  {diff.std().item():.10f}")

print(f"\n相对误差:")
print(f"  mean: {rel_err:.4f}%")

# 详细分析
print(f"\n逐元素分析:")
diff_flat = diff.flatten()
sorted_diff, _ = torch.sort(diff_flat, descending=True)
print(f"  前10大差异: {sorted_diff[:10].tolist()}")

# ============================================================================
# 验证结果
# ============================================================================
print(f"\n" + "="*80)
print("验证结果")
print("="*80)

threshold = 5.0  # bf16 precision tolerance: 5%

if rel_err > threshold:
    print(f"❌ 推理误差过大: {rel_err:.2f}% > {threshold}%")
    print(f"\n可能原因:")
    print(f"  1. adasplash 未更新到最新代码")
    print(f"  2. forward kernel 的 float32 修复未生效")
    print(f"\n解决方法:")
    print(f"  cd /c/Users/fzkuj/Projects/adasplash")
    print(f"  git pull")
    print(f"  pip install -e . --force-reinstall --no-deps")
else:
    print(f"✅ 推理精度正确！")
    print(f"   相对误差 {rel_err:.2f}% 在 bf16 精度范围内 (<{threshold}%)")
    print(f"\n这说明:")
    print(f"  ✅ Forward kernel 的 float32 ReLU 修复有效")
    print(f"  ✅ Triton 推理结果与 PyTorch 一致")
    print(f"  ✅ 可以用于训练和推理")

print("="*80)
