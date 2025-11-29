#!/usr/bin/env python3
"""诊断梯度和注意力问题"""
import torch
import torch.nn.functional as F
from adasplash import lazy_attention_triton

print("="*80)
print("诊断梯度和注意力问题")
print("="*80)

torch.manual_seed(42)

B, H, L, D = 1, 2, 32, 32
device = 'cuda'

# 创建输入
q = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16, requires_grad=True) * 0.02
k = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16, requires_grad=True) * 0.02
v = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16, requires_grad=True)

print(f"输入: B={B}, H={H}, L={L}, D={D}")

# ============================================================================
# 测试不同的 tau 值
# ============================================================================
for tau_val in [-2.0, -1.0, -0.5, 0.0, 0.5]:
    print(f"\n" + "="*80)
    print(f"测试 tau = {tau_val}")
    print("="*80)

    tau = torch.nn.Parameter(torch.full((H,), tau_val, device=device, dtype=torch.float32))
    bias = torch.nn.Parameter(torch.zeros(H, 512, device=device, dtype=torch.float32))

    # 前向
    out = lazy_attention_triton(q, k, v, bias.to(q.dtype), tau.to(q.dtype))

    # 简单的损失：让输出接近 0.1
    target = torch.full_like(out, 0.1)
    loss = F.mse_loss(out, target)

    # 反向
    loss.backward(retain_graph=True)

    print(f"  输出统计:")
    print(f"    mean: {out.mean().item():.6f}")
    print(f"    std: {out.std().item():.6f}")
    print(f"    非零比例: {(out.abs() > 1e-6).float().mean().item() * 100:.2f}%")

    print(f"  损失: {loss.item():.6f}")

    print(f"  梯度:")
    print(f"    tau.grad: {tau.grad}")
    print(f"    |tau.grad|: {tau.grad.abs().sum().item():.6f}")
    print(f"    |bias.grad|: {bias.grad.abs().sum().item():.6f}")

    # 计算理论更新量（lr=0.01）
    lr = 0.01
    tau_update = -lr * tau.grad[0].item()
    print(f"  理论更新量 (lr={lr}):")
    print(f"    tau[0]: {tau_update:.8f}")

    # 清理梯度
    if q.grad is not None:
        q.grad.zero_()
    if k.grad is not None:
        k.grad.zero_()
    if v.grad is not None:
        v.grad.zero_()

# ============================================================================
# 对比 PyTorch 实现
# ============================================================================
print(f"\n" + "="*80)
print("对比 PyTorch 实现 (tau=-1.0)")
print("="*80)

tau_val = -1.0
tau_pt = torch.nn.Parameter(torch.full((H,), tau_val, device=device, dtype=torch.float32))
bias_pt = torch.nn.Parameter(torch.zeros(H, 512, device=device, dtype=torch.float32))

# PyTorch 实现
q_pt = q.detach().clone().requires_grad_(True)
k_pt = k.detach().clone().requires_grad_(True)
v_pt = v.detach().clone().requires_grad_(True)

sm_scale = 1.0 / (D ** 0.5)
scores = torch.matmul(q_pt, k_pt.transpose(-2, -1)) * sm_scale
causal_mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.bfloat16))
scores = scores.masked_fill(causal_mask[None, None, :, :] == 0, float('-inf'))

attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)

# Elastic-Softmax
tau_expanded = tau_pt.view(1, H, 1, 1)
i_positions = torch.arange(1, L + 1, device=device, dtype=torch.bfloat16).view(1, 1, L, 1)
elastic_attn = F.relu(attn_weights + tau_expanded.to(torch.bfloat16) / i_positions)

out_pt = torch.matmul(elastic_attn, v_pt)

print(f"PyTorch:")
print(f"  elastic_attn 非零比例: {(elastic_attn > 1e-6).float().mean().item() * 100:.2f}%")
print(f"  out mean: {out_pt.mean().item():.6f}")
print(f"  out std: {out_pt.std().item():.6f}")

# 损失和梯度
target_pt = torch.full_like(out_pt, 0.1)
loss_pt = F.mse_loss(out_pt, target_pt)
loss_pt.backward()

print(f"  loss: {loss_pt.item():.6f}")
print(f"  tau.grad: {tau_pt.grad}")
print(f"  |tau.grad|: {tau_pt.grad.abs().sum().item():.6f}")

# ============================================================================
# 结论
# ============================================================================
print(f"\n" + "="*80)
print("诊断结论")
print("="*80)

print("\n1. tau=-1.0 时，elastic_attn 几乎全为 0（被 ReLU 掉）")
print("   这导致输出接近 0，梯度很小")
print("\n2. 解决方案：")
print("   - 增大学习率（如 0.1 或 1.0）")
print("   - 或者初始化 tau 为更小的负值（如 -0.5）")
print("   - 或者初始化 bias 为正值，抵消 tau 的负面影响")
print("="*80)
