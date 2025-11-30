#!/usr/bin/env python3
"""完整测试 forward 和 backward 精度修复（tau=-1.0）"""
import torch
import torch.nn.functional as F
from adasplash import lazy_attention_triton

print("="*80)
print("完整测试 Forward + Backward 精度修复")
print("="*80)

torch.manual_seed(42)

B, H, L, D = 2, 4, 64, 32
device = 'cuda'
tau_val = -1.0

# 创建输入
q = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
k = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
v = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16)
target = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.1

print(f"输入: B={B}, H={H}, L={L}, D={D}, tau={tau_val}")

# ============================================================================
# PyTorch 参考实现
# ============================================================================
print(f"\n" + "="*80)
print("PyTorch 参考实现")
print("="*80)

q_pt = q.clone().requires_grad_(True)
k_pt = k.clone().requires_grad_(True)
v_pt = v.clone().requires_grad_(True)
tau_pt = torch.nn.Parameter(torch.full((H,), tau_val, device=device, dtype=torch.float32))
bias_pt = torch.nn.Parameter(torch.zeros(H, 512, device=device, dtype=torch.float32))

# Forward
scaling = 1.0 / (D ** 0.5)
attn_scores = torch.matmul(q_pt, k_pt.transpose(-2, -1)) * scaling

# Bias
rel_pos = torch.arange(L, device=device)[:, None] - torch.arange(L, device=device)[None, :]
valid_mask = (0 <= rel_pos) & (rel_pos < 512)
indices = rel_pos.clamp(0, 511)
bias_matrix = bias_pt[:, indices] * valid_mask.float()
attn_scores = attn_scores + bias_matrix[None, :, :, :]

# Causal mask
causal_mask = torch.tril(torch.ones(L, L, device=device))
attn_scores = attn_scores.masked_fill(causal_mask[None, None, :, :] == 0, float('-inf'))

# Softmax
attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)

# Elastic-Softmax
tau_expanded = tau_pt.view(1, H, 1, 1).to(torch.bfloat16)
i_positions = torch.arange(1, L + 1, device=device, dtype=torch.bfloat16).view(1, 1, L, 1)
p_term = attn_weights + tau_expanded / i_positions
elastic_attn = F.relu(p_term)

out_pt = torch.matmul(elastic_attn, v_pt)

# Loss and backward
loss_pt = F.mse_loss(out_pt, target)
loss_pt.backward()

print(f"PyTorch:")
print(f"  output mean: {out_pt.mean().item():.10f}")
print(f"  loss: {loss_pt.item():.10f}")
print(f"  tau.grad[0]: {tau_pt.grad[0].item():.10f}")
print(f"  bias.grad mean: {bias_pt.grad.mean().item():.10f}")
print(f"  q.grad mean: {q_pt.grad.mean().item():.10f}")
print(f"  k.grad mean: {k_pt.grad.mean().item():.10f}")
print(f"  v.grad mean: {v_pt.grad.mean().item():.10f}")

# ============================================================================
# Triton 实现
# ============================================================================
print(f"\n" + "="*80)
print("Triton Kernel 实现")
print("="*80)

q_triton = q.clone().requires_grad_(True)
k_triton = k.clone().requires_grad_(True)
v_triton = v.clone().requires_grad_(True)
tau_triton = torch.nn.Parameter(torch.full((H,), tau_val, device=device, dtype=torch.float32))
bias_triton = torch.nn.Parameter(torch.zeros(H, 512, device=device, dtype=torch.float32))

out_triton = lazy_attention_triton(
    q_triton, k_triton, v_triton,
    bias_triton.to(torch.bfloat16),
    tau_triton.to(torch.bfloat16)
)

loss_triton = F.mse_loss(out_triton, target)
loss_triton.backward()

print(f"Triton:")
print(f"  output mean: {out_triton.mean().item():.10f}")
print(f"  loss: {loss_triton.item():.10f}")
print(f"  tau.grad[0]: {tau_triton.grad[0].item():.10f}")
print(f"  bias.grad mean: {bias_triton.grad.mean().item():.10f}")
print(f"  q.grad mean: {q_triton.grad.mean().item():.10f}")
print(f"  k.grad mean: {k_triton.grad.mean().item():.10f}")
print(f"  v.grad mean: {v_triton.grad.mean().item():.10f}")

# ============================================================================
# 对比分析
# ============================================================================
print(f"\n" + "="*80)
print("对比分析")
print("="*80)

# Forward 差异
out_diff = (out_pt - out_triton).abs()
out_rel_err = (out_diff / (out_pt.abs() + 1e-8)).mean().item() * 100

print(f"\nForward 输出差异:")
print(f"  绝对差异 max:  {out_diff.max().item():.10f}")
print(f"  绝对差异 mean: {out_diff.mean().item():.10f}")
print(f"  相对误差:      {out_rel_err:.4f}%")

# Loss 差异
loss_diff = abs(loss_pt.item() - loss_triton.item())
loss_rel_err = loss_diff / (loss_pt.item() + 1e-10) * 100
print(f"\nLoss 差异:")
print(f"  绝对差异: {loss_diff:.15f}")
print(f"  相对误差: {loss_rel_err:.4f}%")

# Backward 差异
print(f"\nBackward 梯度差异:")

def compare_grad(name, grad_pt, grad_triton):
    diff = (grad_pt - grad_triton).abs()
    rel_err = (diff / (grad_pt.abs() + 1e-10)).mean().item() * 100
    print(f"  {name}:")
    print(f"    绝对差异 mean: {diff.mean().item():.10f}")
    print(f"    相对误差:      {rel_err:.4f}%")
    return rel_err

tau_rel = compare_grad("tau", tau_pt.grad, tau_triton.grad)
bias_rel = compare_grad("bias", bias_pt.grad, bias_triton.grad)
q_rel = compare_grad("q", q_pt.grad, q_triton.grad)
k_rel = compare_grad("k", k_pt.grad, k_triton.grad)
v_rel = compare_grad("v", v_pt.grad, v_triton.grad)

# ============================================================================
# 验证结果
# ============================================================================
print(f"\n" + "="*80)
print("验证结果")
print("="*80)

threshold = 5.0  # bf16 precision tolerance: 5%

all_passed = True

if out_rel_err > threshold:
    print(f"❌ Forward 误差过大: {out_rel_err:.2f}% > {threshold}%")
    all_passed = False
else:
    print(f"✅ Forward 正确: 相对误差 {out_rel_err:.2f}% < {threshold}%")

if tau_rel > threshold:
    print(f"❌ Tau 梯度误差过大: {tau_rel:.2f}% > {threshold}%")
    all_passed = False
else:
    print(f"✅ Tau 梯度正确: 相对误差 {tau_rel:.2f}% < {threshold}%")

if bias_rel > threshold:
    print(f"❌ Bias 梯度误差过大: {bias_rel:.2f}% > {threshold}%")
    all_passed = False
else:
    print(f"✅ Bias 梯度正确: 相对误差 {bias_rel:.2f}% < {threshold}%")

if q_rel > threshold:
    print(f"❌ Q 梯度误差过大: {q_rel:.2f}% > {threshold}%")
    all_passed = False
else:
    print(f"✅ Q 梯度正确: 相对误差 {q_rel:.2f}% < {threshold}%")

if k_rel > threshold:
    print(f"❌ K 梯度误差过大: {k_rel:.2f}% > {threshold}%")
    all_passed = False
else:
    print(f"✅ K 梯度正确: 相对误差 {k_rel:.2f}% < {threshold}%")

if v_rel > threshold:
    print(f"❌ V 梯度误差过大: {v_rel:.2f}% > {threshold}%")
    all_passed = False
else:
    print(f"✅ V 梯度正确: 相对误差 {v_rel:.2f}% < {threshold}%")

print(f"\n" + "="*80)
if all_passed:
    print("🎉 所有测试通过！Triton 实现与 PyTorch 一致")
    print("\n下一步:")
    print("  1. 重新安装 adasplash: cd /c/Users/fzkuj/Projects/adasplash && pip install -e . --force-reinstall --no-deps")
    print("  2. 重新运行完整训练测试")
else:
    print("❌ 部分测试失败，需要进一步调试")
print("="*80)
