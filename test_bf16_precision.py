#!/usr/bin/env python3
"""测试 bf16 精度限制"""
import torch

print("="*80)
print("测试 bf16 精度限制")
print("="*80)

# 测试 -1.0 附近的最小可分辨差异
val = torch.tensor(-1.0, dtype=torch.bfloat16)
print(f"\n原始值: {val.item():.10f}")

# 尝试加上不同大小的增量
deltas = [0.000001, 0.00001, 0.0001, 0.001, 0.01]

for delta in deltas:
    new_val = val + delta
    diff = (new_val - val).item()
    print(f"  尝试加 {delta:.6f}, 实际差异: {diff:.10f}, {'✅ 可表示' if abs(diff) > 0 else '❌ 被舍入为0'}")

# 测试 SGD 更新
print(f"\n" + "="*80)
print("模拟 SGD 更新")
print("="*80)

tau = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.bfloat16))
optimizer = torch.optim.SGD([tau], lr=0.01)

print(f"初始 tau: {tau.item():.10f}")

# 模拟一个小梯度
tau.grad = torch.tensor(0.001389, dtype=torch.float32)
print(f"梯度: {tau.grad.item():.10f}")
print(f"理论更新量: {-0.01 * tau.grad.item():.10f}")

optimizer.step()
print(f"更新后 tau: {tau.item():.10f}")
print(f"实际变化: {0.0:.10f}")  # 应该是 0

# 解决方案：使用更大的学习率
print(f"\n" + "="*80)
print("解决方案：增大学习率")
print("="*80)

tau2 = torch.nn.Parameter(torch.tensor(-1.0, dtype=torch.bfloat16))
optimizer2 = torch.optim.SGD([tau2], lr=1.0)  # 100倍学习率

tau2.grad = torch.tensor(0.001389, dtype=torch.float32)
print(f"初始 tau: {tau2.item():.10f}")
print(f"学习率: 1.0")
print(f"理论更新量: {-1.0 * tau2.grad.item():.10f}")

optimizer2.step()
print(f"更新后 tau: {tau2.item():.10f}")
print(f"实际变化: {tau2.item() - (-1.0):.10f}")

print(f"\n" + "="*80)
print("结论")
print("="*80)
print("bf16 精度限制导致小更新被舍入为 0")
print("解决方案:")
print("  1. 增大学习率（至少 100 倍）")
print("  2. 使用 float32 参数（但会增加内存）")
print("  3. 使用混合精度优化器（如 AdamW）")
print("="*80)
