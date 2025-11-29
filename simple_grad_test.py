#!/usr/bin/env python3
"""最简单的梯度测试 - 直接调用 kernel"""
import torch
from adasplash import lazy_attention_triton

print("="*80)
print("直接测试 Triton Kernel 的梯度")
print("="*80)

# 简单的输入
B, H, L, D = 2, 4, 16, 64
device = 'cuda'

q = torch.randn(B, H, L, D, device=device, requires_grad=False)
k = torch.randn(B, H, L, D, device=device, requires_grad=False)
v = torch.randn(B, H, L, D, device=device, requires_grad=False)

# 可学习参数
bias = torch.zeros(H, 512, device=device, requires_grad=True)
tau = torch.full((H,), -1.0, device=device, requires_grad=True)

print(f"输入:")
print(f"  Q, K, V shape: {q.shape}")
print(f"  bias shape: {bias.shape}, requires_grad: {bias.requires_grad}")
print(f"  tau shape: {tau.shape}, requires_grad: {tau.requires_grad}")
print(f"  tau values: {tau}")

# 前向传播
print("\n前向传播...")
out = lazy_attention_triton(q, k, v, bias, tau)
print(f"  out shape: {out.shape}")

# 计算简单的 loss
loss = out.sum()
print(f"  loss: {loss.item():.6f}")

# 反向传播
print("\n反向传播...")
loss.backward()

# 检查梯度
print(f"\n梯度检查:")
print(f"  bias.grad is not None: {bias.grad is not None}")
if bias.grad is not None:
    print(f"  bias.grad shape: {bias.grad.shape}")
    print(f"  bias.grad sum: {bias.grad.sum().item():.6f}")
    print(f"  bias.grad abs sum: {bias.grad.abs().sum().item():.6f}")
    print(f"  bias.grad 非零元素: {(bias.grad.abs() > 1e-8).sum().item()}/{bias.grad.numel()}")

print(f"\n  tau.grad is not None: {tau.grad is not None}")
if tau.grad is not None:
    print(f"  tau.grad: {tau.grad}")
    print(f"  tau.grad sum: {tau.grad.sum().item():.6f}")
    print(f"  tau.grad abs sum: {tau.grad.abs().sum().item():.6f}")

# 结论
print("\n" + "="*80)
if bias.grad is not None and tau.grad is not None:
    if bias.grad.abs().sum() > 1e-6 and tau.grad.abs().sum() > 1e-6:
        print("✅ Kernel 本身可以计算梯度!")
        print("   问题可能在模型的其他部分")
    elif tau.grad.abs().sum() < 1e-6:
        print("❌ tau 梯度为零!")
        print("   可能是 mask_relu 的问题")

        # 进一步分析
        print("\n分析: 计算 p_norm + tau/i 的大致范围")
        # Softmax 后的值大约是 1/L (均匀分布)
        approx_p_norm = 1.0 / L
        for i in [1, L//4, L//2, L]:
            p_term = approx_p_norm + tau[0].item() / i
            print(f"  位置 {i}: p_norm≈{approx_p_norm:.4f} + tau/i={tau[0].item()/i:.4f} = {p_term:.4f} {'> 0 ✓' if p_term > 0 else '< 0 ✗'}")
    else:
        print("❌ bias 梯度为零!")
else:
    print("❌ 梯度为 None!")
print("="*80)
