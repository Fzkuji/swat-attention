#!/usr/bin/env python3
"""详细分析每个位置的 elastic attention 值"""
import torch
import torch.nn.functional as F

print("="*80)
print("分析 Elastic Attention 为何不为 0")
print("="*80)

torch.manual_seed(42)

B, H, L, D = 1, 1, 8, 32  # 更小的 L 便于分析
device = 'cuda'
tau_val = -1.0

# 创建输入
q = torch.randn(B, H, L, D, device=device, dtype=torch.float32) * 0.02
k = torch.randn(B, H, L, D, device=device, dtype=torch.float32) * 0.02

print(f"输入: L={L}, tau={tau_val}")

# QK^T
scaling = 1.0 / (D ** 0.5)
attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scaling

# Causal mask
causal_mask = torch.tril(torch.ones(L, L, device=device))
attn_scores = attn_scores.masked_fill(causal_mask[None, None, :, :] == 0, float('-inf'))

# Softmax
attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32)

print(f"\n" + "="*80)
print("逐位置分析")
print("="*80)

# 逐个 query 位置分析
for i in range(L):
    print(f"\n位置 i={i} (query {i}, 能看到 {i+1} 个 tokens):")

    # 这个 query 看到的 softmax 权重
    weights = attn_weights[0, 0, i, :i+1]
    print(f"  Softmax 权重: {weights}")
    print(f"  Softmax 和: {weights.sum().item():.10f}")

    # tau/i
    tau_over_i = tau_val / (i + 1)
    print(f"  tau/(i+1) = {tau_val}/{i+1} = {tau_over_i:.6f}")

    # 加上 tau/i
    weights_with_tau = weights + tau_over_i
    print(f"  Softmax + tau/i: {weights_with_tau}")
    print(f"  最大值: {weights_with_tau.max().item():.10f}")
    print(f"  最小值: {weights_with_tau.min().item():.10f}")

    # ReLU 后
    elastic = F.relu(weights_with_tau)
    print(f"  ReLU 后: {elastic}")
    print(f"  非零个数: {(elastic > 0).sum().item()}/{i+1}")

    # 理论分析
    if i == 0:
        print(f"  理论: weight=1.0, tau/1=-1.0, sum=0 ✅")
    else:
        avg_weight = 1.0 / (i + 1)
        print(f"  理论（如果均匀分布）: avg_weight={avg_weight:.6f}, tau/{i+1}={tau_over_i:.6f}")
        print(f"                        sum={avg_weight + tau_over_i:.6f}")

print(f"\n" + "="*80)
print("关键发现")
print("="*80)

print("\n问题：Softmax 权重不是均匀分布的！")
print("\n示例：位置 i=1 (能看到 2 个 tokens)")
i = 1
weights = attn_weights[0, 0, i, :i+1]
print(f"  实际 softmax: {weights}")
print(f"  理论均匀分布: [0.5, 0.5]")
print(f"  差异: {weights - 0.5}")

print(f"\n如果不均匀:")
print(f"  假设 softmax = [0.6, 0.4]")
print(f"  加 tau/2 = -0.5:")
print(f"    [0.6, 0.4] + [-0.5, -0.5] = [0.1, -0.1]")
print(f"  ReLU 后: [0.1, 0]  ← 不为 0！")

print(f"\n" + "="*80)
print("结论")
print("="*80)

print("\n理论上 elastic_attn 全为 0 的前提条件：")
print("  1. tau = -1.0")
print("  2. bias = 0")
print("  3. Softmax 权重均匀分布（每个位置 j 的权重 = 1/(i+1)）")

print("\n实际情况：")
print("  ✅ tau = -1.0")
print("  ✅ bias = 0")
print("  ❌ Softmax 权重不均匀（因为 QK^T 的随机性）")

print("\n所以：")
print("  - 当某个 key 的 attention 权重 > 1/(i+1) 时")
print("  - 加上 tau/i 后可能 > 0")
print("  - ReLU 后不为 0")
print("  - 这是**正确的行为**！")

print("\n真正的问题：")
print("  Scratch 和 Flash 的 elastic_attn 应该完全一致")
print("  但实际有 1.58% 的差异")
print("  说明中间计算有差异（可能是精度或逻辑）")

print("="*80)
