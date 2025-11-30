#!/usr/bin/env python3
"""精细测试不同 tau 值的梯度大小"""
import torch
import torch.nn.functional as F
from adasplash import lazy_attention_triton

print("="*80)
print("精细测试 tau 值对梯度的影响")
print("="*80)

torch.manual_seed(42)

B, H, L, D = 2, 4, 64, 32
device = 'cuda'

# 创建输入
q = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
k = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
v = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16)
target = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.1

print(f"输入: B={B}, H={H}, L={L}, D={D}")

# 测试更精细的 tau 值范围
tau_values = [
    -2.0, -1.5, -1.2, -1.1, -1.0, -0.9, -0.8, -0.7, -0.6, -0.5,
    -0.4, -0.3, -0.2, -0.1, 0.0
]

print(f"\n" + "="*80)
print("{'Tau':<8} {'输出mean':<14} {'Loss':<12} {'|tau.grad|':<14} {'更新量(3e-4)':<14} {'相对-0.5':<10}")
print("-" * 90)

results = []
grad_at_minus_half = None

for tau_val in tau_values:
    # 创建参数
    tau = torch.nn.Parameter(torch.full((H,), tau_val, device=device, dtype=torch.float32))
    bias = torch.nn.Parameter(torch.zeros(H, 512, device=device, dtype=torch.float32))

    # 前向
    out = lazy_attention_triton(q, k, v, bias.to(torch.bfloat16), tau.to(torch.bfloat16))
    loss = F.mse_loss(out, target)

    # 反向
    loss.backward()

    # 统计
    tau_grad_norm = tau.grad.abs().sum().item()
    update_amount = 3e-4 * tau_grad_norm  # lr=3e-4

    if tau_val == -0.5:
        grad_at_minus_half = tau_grad_norm

    relative = (tau_grad_norm / grad_at_minus_half * 100) if grad_at_minus_half else 0

    results.append({
        'tau': tau_val,
        'out_mean': out.mean().item(),
        'loss': loss.item(),
        'grad_norm': tau_grad_norm,
        'update': update_amount,
        'relative': relative
    })

    print(f"{tau_val:<8.2f} {out.mean().item():<14.6f} {loss.item():<12.6f} "
          f"{tau_grad_norm:<14.6f} {update_amount:<14.10f} {relative:<10.1f}%")

# 分析
print(f"\n" + "="*80)
print("分析")
print("="*80)

print(f"\n关键观察:")

# 找到梯度接近0的临界点
threshold = 0.001
critical_tau = None
for r in results:
    if r['grad_norm'] > threshold:
        critical_tau = r['tau']
        break

if critical_tau:
    print(f"  梯度显著 (>{threshold}) 的临界值: tau ≈ {critical_tau:.2f}")
else:
    print(f"  所有测试值梯度都 ≤ {threshold}")

# -1.1 的具体情况
tau_1_1 = next((r for r in results if abs(r['tau'] - (-1.1)) < 0.01), None)
if tau_1_1:
    print(f"\n  tau = -1.1:")
    print(f"    梯度: {tau_1_1['grad_norm']:.6f}")
    print(f"    相对 -0.5: {tau_1_1['relative']:.1f}%")
    print(f"    每步更新: {tau_1_1['update']:.10f}")

    if tau_1_1['grad_norm'] < 0.001:
        print(f"    ❌ 梯度太小，训练会很慢")
    elif tau_1_1['grad_norm'] < 0.01:
        print(f"    ⚠️ 梯度偏小，训练较慢")
    else:
        print(f"    ✅ 梯度正常")

# 推荐
print(f"\n推荐:")
good_tau = [r for r in results if r['grad_norm'] > 0.01]
if good_tau:
    best = min(good_tau, key=lambda x: abs(x['tau'] + 0.5))  # 最接近-0.5的好梯度值
    print(f"  推荐初始化: tau = {best['tau']:.2f}")
    print(f"    梯度: {best['grad_norm']:.6f}")
    print(f"    相对 -0.5: {best['relative']:.1f}%")

print(f"\n如果必须从负值开始:")
negative_good = [r for r in good_tau if r['tau'] < 0]
if negative_good:
    best_neg = max(negative_good, key=lambda x: x['tau'])  # 最大的负值（最接近0）
    print(f"  最好的负值: tau = {best_neg['tau']:.2f}")
    print(f"    梯度: {best_neg['grad_norm']:.6f}")
    print(f"    相对 -0.5: {best_neg['relative']:.1f}%")

print("="*80)
