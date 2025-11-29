#!/usr/bin/env python3
"""测试使用 float32 参数后的训练收敛性"""
import torch
import torch.nn.functional as F
from adasplash import lazy_attention_triton

print("="*80)
print("测试训练收敛性 (float32 参数)")
print("="*80)

torch.manual_seed(42)

B, H, L, D = 2, 4, 64, 32
device = 'cuda'
tau_init = -1.0
num_steps = 100
lr = 0.01

# 创建输入和目标
q = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
k = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
v = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16)
target = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.1

print(f"输入: B={B}, H={H}, L={L}, D={D}")
print(f"  tau_init: {tau_init}")
print(f"  学习率: {lr}")
print(f"  训练步数: {num_steps}")

# ============================================================================
# Triton 实现训练 (使用 float32 参数)
# ============================================================================
print(f"\n" + "="*80)
print("Triton 实现训练 (float32 参数)")
print("="*80)

# 关键修改：使用 float32 存储参数
tau_triton = torch.nn.Parameter(torch.full((H,), tau_init, device=device, dtype=torch.float32))
bias_triton = torch.nn.Parameter(torch.randn(H, 512, device=device, dtype=torch.float32) * 0.02)

optimizer = torch.optim.SGD([tau_triton, bias_triton], lr=lr)

losses = []
tau_values = []

for step in range(num_steps):
    optimizer.zero_grad()

    # 传给 kernel 前转换为 bf16
    out = lazy_attention_triton(
        q, k, v,
        bias_triton.to(q.dtype),
        tau_triton.to(q.dtype)
    )
    loss = F.mse_loss(out, target)

    loss.backward()
    optimizer.step()

    losses.append(loss.item())
    tau_values.append(tau_triton[0].item())

    if step % 20 == 0 or step == num_steps - 1:
        print(f"Step {step:3d}: loss={loss.item():.6f}, "
              f"tau[0]={tau_triton[0].item():.6f}, "
              f"|tau.grad|={tau_triton.grad.abs().sum().item():.6f}, "
              f"|bias.grad|={bias_triton.grad.abs().sum().item():.6f}")

# ============================================================================
# 分析结果
# ============================================================================
print(f"\n" + "="*80)
print("训练结果分析")
print("="*80)

initial_loss = losses[0]
final_loss = losses[-1]
loss_reduction = (initial_loss - final_loss) / initial_loss * 100

print(f"\n损失变化:")
print(f"  初始损失: {initial_loss:.6f}")
print(f"  最终损失: {final_loss:.6f}")
print(f"  降低比例: {loss_reduction:.2f}%")

tau_initial = tau_values[0]
tau_final = tau_values[-1]
tau_change = tau_final - tau_initial

print(f"\nTau 变化:")
print(f"  初始值: {tau_initial:.6f}")
print(f"  最终值: {tau_final:.6f}")
print(f"  变化量: {tau_change:.6f}")

print(f"\nBias 统计:")
print(f"  初始 std: 0.02 (设定)")
print(f"  最终 std: {bias_triton.std().item():.6f}")
print(f"  最终 mean: {bias_triton.mean().item():.6f}")

# ============================================================================
# 结论
# ============================================================================
print(f"\n" + "="*80)
print("结论")
print("="*80)

if loss_reduction > 10:
    print("✅ 损失正常下降 (>10%)")
else:
    print(f"❌ 损失下降不足 ({loss_reduction:.2f}%)")

if abs(tau_change) > 0.01:
    print("✅ Tau 正常更新 (变化 >0.01)")
else:
    print(f"❌ Tau 几乎不变 (变化 {abs(tau_change):.6f})")

if bias_triton.grad.abs().sum().item() > 0.001:
    print("✅ Bias 梯度正常")
else:
    print("❌ Bias 梯度过小")

print(f"\n整体评估:")
if loss_reduction > 10 and abs(tau_change) > 0.01:
    print("✅ 训练正常，float32 参数修复有效！")
    print("   可以开始完整模型训练")
else:
    print("❌ 训练仍有问题，需要进一步调查")

print("="*80)
