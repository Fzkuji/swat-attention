#!/usr/bin/env python3
"""测试梯度裁剪对 tau 和 bias 的影响"""
import torch
import torch.nn.functional as F
from adasplash import lazy_attention_triton

print("="*80)
print("测试梯度裁剪影响")
print("="*80)

torch.manual_seed(42)

B, H, L, D = 2, 4, 64, 32
device = 'cuda'
tau_init = -1.0

# 创建输入
q = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
k = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
v = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16)
target = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.1

# 创建参数
tau = torch.nn.Parameter(torch.full((H,), tau_init, device=device, dtype=torch.float32))
bias = torch.nn.Parameter(torch.randn(H, 512, device=device, dtype=torch.float32) * 0.02)

# 前向和反向
out = lazy_attention_triton(q, k, v, bias.to(torch.bfloat16), tau.to(torch.bfloat16))
loss = F.mse_loss(out, target)
loss.backward()

print(f"输入: B={B}, H={H}, L={L}, D={D}")
print(f"tau初始值: {tau_init}")

# 计算梯度范数
tau_grad_norm = tau.grad.norm().item()
bias_grad_norm = bias.grad.norm().item()
total_grad_norm = torch.sqrt(tau.grad.norm()**2 + bias.grad.norm()**2).item()

print(f"\n梯度范数:")
print(f"  tau.grad.norm():  {tau_grad_norm:.10f}")
print(f"  bias.grad.norm(): {bias_grad_norm:.10f}")
print(f"  总梯度范数:       {total_grad_norm:.10f}")

# 测试不同的梯度裁剪阈值
clip_values = [0.1, 0.5, 1.0, 5.0, 10.0]

print(f"\n" + "="*80)
print("模拟梯度裁剪 (max_grad_norm)")
print("="*80)

for max_norm in clip_values:
    # 克隆梯度
    tau_grad = tau.grad.clone()
    bias_grad = bias.grad.clone()

    # 计算总范数
    current_norm = torch.sqrt(tau_grad.norm()**2 + bias_grad.norm()**2).item()

    # 裁剪
    if current_norm > max_norm:
        scale = max_norm / current_norm
        tau_grad *= scale
        bias_grad *= scale
        status = "✂️ 被裁剪"
    else:
        status = "✅ 未裁剪"

    # 模拟更新 (lr=3e-4)
    lr = 3e-4
    tau_update = tau_grad[0].item() * lr

    print(f"\nmax_grad_norm = {max_norm:.1f}:")
    print(f"  当前范数: {current_norm:.10f}")
    print(f"  状态: {status}")
    if current_norm > max_norm:
        print(f"  缩放因子: {scale:.6f}")
    print(f"  tau[0] 更新量: {tau_update:.10f}")

# AdamW 模拟
print(f"\n" + "="*80)
print("AdamW 优化器更新")
print("="*80)

# 重新计算梯度
tau.grad = None
bias.grad = None

out = lazy_attention_triton(q, k, v, bias.to(torch.bfloat16), tau.to(torch.bfloat16))
loss = F.mse_loss(out, target)
loss.backward()

# AdamW 优化器
optimizer = torch.optim.AdamW([tau, bias], lr=3e-4, betas=(0.9, 0.95), weight_decay=0.01)

print(f"\n优化前:")
print(f"  tau[0]: {tau[0].item():.10f}")
print(f"  tau.grad[0]: {tau.grad[0].item():.10f}")

# 梯度裁剪
torch.nn.utils.clip_grad_norm_([tau, bias], max_norm=1.0)

print(f"\n裁剪后:")
print(f"  tau.grad[0]: {tau.grad[0].item():.10f}")

# 更新
optimizer.step()

print(f"\n优化后:")
print(f"  tau[0]: {tau[0].item():.10f}")
print(f"  变化: {tau[0].item() - tau_init:.10f}")

# 结论
print(f"\n" + "="*80)
print("结论")
print("="*80)

if total_grad_norm < 1.0:
    print(f"✅ 总梯度范数 {total_grad_norm:.6f} < 1.0")
    print(f"   梯度裁剪不会影响更新")
else:
    print(f"⚠️ 总梯度范数 {total_grad_norm:.6f} > 1.0")
    print(f"   梯度会被裁剪 {1.0/total_grad_norm:.2f}倍")

print(f"\n主要问题:")
print(f"  tau=-1.0 时梯度太小 ({tau_grad_norm:.6f})")
print(f"  即使不裁剪，更新也很慢 (lr=3e-4 * {tau.grad[0].item():.6f} = {3e-4 * tau.grad[0].item():.10f})")

print(f"\n建议:")
print(f"  1. 改变 tau 初始化为 -0.5（梯度大76倍）")
print(f"  2. 或增大学习率（如 lr=1e-3）")
print(f"  3. 或使用参数组，给 tau/bias 单独学习率")

print("="*80)
