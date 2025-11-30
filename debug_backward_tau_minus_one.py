#!/usr/bin/env python3
"""详细调试 tau=-1.0 时的反向传播"""
import torch
import torch.nn.functional as F
from adasplash import lazy_attention_triton

print("="*80)
print("调试 tau=-1.0 的反向传播")
print("="*80)

torch.manual_seed(42)

B, H, L, D = 1, 1, 16, 16  # 小尺寸便于调试
device = 'cuda'
tau_val = -1.0

# 创建输入
q = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
k = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
v = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16)
target = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.1

print(f"输入: B={B}, H={H}, L={L}, D={D}, tau={tau_val}")

# ============================================================================
# PyTorch 实现 (autograd)
# ============================================================================
print(f"\n" + "="*80)
print("PyTorch Autograd")
print("="*80)

tau_pt = torch.nn.Parameter(torch.full((H,), tau_val, device=device, dtype=torch.float32))
bias_pt = torch.nn.Parameter(torch.zeros(H, 512, device=device, dtype=torch.float32))

# Forward
scaling = 1.0 / (D ** 0.5)
attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scaling

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

# 保存中间值用于分析
p_norm = attn_weights  # [B, H, L, L]
tau_over_i = tau_expanded / i_positions  # [1, H, L, 1]
p_term = p_norm + tau_over_i  # [B, H, L, L]
mask_relu = p_term > 0

print(f"\n中间值统计:")
print(f"  p_norm mean: {p_norm.mean().item():.6f}")
print(f"  tau/i mean: {tau_over_i.mean().item():.6f}")
print(f"  p_term mean: {p_term.mean().item():.6f}")
print(f"  mask_relu 非零比例: {mask_relu.float().mean().item() * 100:.2f}%")

elastic_attn = F.relu(p_term)
out_pt = torch.matmul(elastic_attn, v)

# Loss and backward
loss_pt = F.mse_loss(out_pt, target)
loss_pt.backward()

print(f"\nPyTorch:")
print(f"  loss: {loss_pt.item():.10f}")
print(f"  tau.grad: {tau_pt.grad}")
print(f"  tau.grad 详细: {tau_pt.grad[0].item():.10f}")

# ============================================================================
# Triton 实现
# ============================================================================
print(f"\n" + "="*80)
print("Triton Kernel")
print("="*80)

tau_triton = torch.nn.Parameter(torch.full((H,), tau_val, device=device, dtype=torch.float32))
bias_triton = torch.nn.Parameter(torch.zeros(H, 512, device=device, dtype=torch.float32))

out_triton = lazy_attention_triton(
    q, k, v,
    bias_triton.to(torch.bfloat16),
    tau_triton.to(torch.bfloat16)
)

loss_triton = F.mse_loss(out_triton, target)
loss_triton.backward()

print(f"\nTriton:")
print(f"  loss: {loss_triton.item():.10f}")
print(f"  tau.grad: {tau_triton.grad}")
print(f"  tau.grad 详细: {tau_triton.grad[0].item():.10f}")

# ============================================================================
# 对比分析
# ============================================================================
print(f"\n" + "="*80)
print("对比分析")
print("="*80)

print(f"\nLoss 差异:")
loss_diff = abs(loss_pt.item() - loss_triton.item())
print(f"  {loss_diff:.15f}")

print(f"\n输出差异:")
out_diff = (out_pt - out_triton).abs()
print(f"  max: {out_diff.max().item():.10f}")
print(f"  mean: {out_diff.mean().item():.10f}")

print(f"\nTau 梯度对比:")
tau_grad_diff = abs(tau_pt.grad[0].item() - tau_triton.grad[0].item())
tau_grad_rel = tau_grad_diff / (abs(tau_pt.grad[0].item()) + 1e-10) * 100

print(f"  PyTorch: {tau_pt.grad[0].item():.10f}")
print(f"  Triton:  {tau_triton.grad[0].item():.10f}")
print(f"  绝对差异: {tau_grad_diff:.10f}")
print(f"  相对差异: {tau_grad_rel:.2f}%")

# ============================================================================
# 手动计算梯度验证
# ============================================================================
print(f"\n" + "="*80)
print("手动计算 tau 梯度验证")
print("="*80)

# 根据公式：d_tau = sum(d_out/d_p * d_p/d_tau * mask_relu)
# 其中 d_p/d_tau = 1/i

# 模拟 dL/d_elastic_attn
d_loss_d_out = 2 * (out_pt - target) / (B * H * L * D)
d_out_d_elastic = v.transpose(-2, -1)  # [B, H, D, L]
d_loss_d_elastic = torch.matmul(d_loss_d_out, d_out_d_elastic)  # [B, H, L, L]

print(f"d_loss_d_elastic:")
print(f"  shape: {d_loss_d_elastic.shape}")
print(f"  mean: {d_loss_d_elastic.mean().item():.10f}")

# d_elastic/d_tau = 1/i (only where mask_relu=True)
d_tau_manual = 0.0
for i in range(L):
    for j in range(i + 1):  # causal
        if mask_relu[0, 0, i, j]:
            d_tau_manual += d_loss_d_elastic[0, 0, i, j].item() / (i + 1)

print(f"\n手动计算的 tau 梯度: {d_tau_manual:.10f}")
print(f"PyTorch autograd:     {tau_pt.grad[0].item():.10f}")
print(f"Triton kernel:        {tau_triton.grad[0].item():.10f}")

# ============================================================================
# 结论
# ============================================================================
print(f"\n" + "="*80)
print("结论")
print("="*80)

if tau_grad_rel > 10:
    print(f"❌ Triton backward 有明显误差 ({tau_grad_rel:.1f}%)")
    print(f"   可能原因:")
    print(f"   1. mask_relu 判断不准确（p_term 接近 0 时）")
    print(f"   2. idx_i_float 计算有误")
    print(f"   3. 累积 dtau_acc 有精度问题")
    print(f"   4. ReLU 边界情况处理不当")
elif tau_grad_rel > 1:
    print(f"⚠️ Triton backward 有小误差 ({tau_grad_rel:.1f}%)")
    print(f"   可能是 bf16 精度问题")
else:
    print(f"✅ Triton backward 正确 ({tau_grad_rel:.1f}%)")

print("="*80)
