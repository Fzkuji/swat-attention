#!/usr/bin/env python3
"""测试不同学习率的效果"""
import torch
import torch.nn.functional as F
from adasplash import lazy_attention_triton

print("="*80)
print("测试不同学习率")
print("="*80)

torch.manual_seed(42)

B, H, L, D = 2, 4, 64, 32
device = 'cuda'
tau_init = -1.0
num_steps = 100

# 创建输入和目标
q = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
k = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
v = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16)
target = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.1

print(f"输入: B={B}, H={H}, L={L}, D={D}")
print(f"  tau_init: {tau_init}")
print(f"  训练步数: {num_steps}")

# 测试不同的学习率
learning_rates = [0.001, 0.01, 0.1, 1.0]

results = []

for lr in learning_rates:
    print(f"\n" + "="*80)
    print(f"学习率: {lr}")
    print("="*80)

    # 重新初始化参数
    tau = torch.nn.Parameter(torch.full((H,), tau_init, device=device, dtype=torch.float32))
    bias = torch.nn.Parameter(torch.randn(H, 512, device=device, dtype=torch.float32) * 0.02)

    optimizer = torch.optim.SGD([tau, bias], lr=lr)

    losses = []
    tau_values = []

    for step in range(num_steps):
        optimizer.zero_grad()

        out = lazy_attention_triton(q, k, v, bias.to(q.dtype), tau.to(q.dtype))
        loss = F.mse_loss(out, target)

        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        tau_values.append(tau[0].item())

        if step % 50 == 0 or step == num_steps - 1:
            print(f"  Step {step:3d}: loss={loss.item():.6f}, tau[0]={tau[0].item():.6f}")

    initial_loss = losses[0]
    final_loss = losses[-1]
    loss_reduction = (initial_loss - final_loss) / initial_loss * 100
    tau_change = tau_values[-1] - tau_values[0]

    print(f"\n  结果:")
    print(f"    损失降低: {loss_reduction:.2f}%")
    print(f"    tau 变化: {tau_change:.6f}")

    results.append({
        'lr': lr,
        'loss_reduction': loss_reduction,
        'tau_change': tau_change,
        'final_loss': final_loss
    })

# ============================================================================
# 总结
# ============================================================================
print(f"\n" + "="*80)
print("总结")
print("="*80)

print(f"\n{'学习率':<10} {'损失降低%':<12} {'Tau变化':<12} {'最终损失':<12}")
print("-" * 50)
for r in results:
    print(f"{r['lr']:<10.3f} {r['loss_reduction']:<12.2f} {r['tau_change']:<12.6f} {r['final_loss']:<12.6f}")

print(f"\n推荐:")
best_result = max(results, key=lambda x: x['loss_reduction'])
print(f"  最佳学习率: {best_result['lr']}")
print(f"  损失降低: {best_result['loss_reduction']:.2f}%")

print("="*80)
