#!/usr/bin/env python3
"""调试 tau_term 的维度问题"""
import torch

print("="*80)
print("验证 tau_term 的维度和广播")
print("="*80)

# 模拟 Triton kernel 的计算
BLOCK_M, BLOCK_N, HEAD_DIM = 64, 64, 32
tau = torch.tensor(-1.0)

# 模拟 query 位置
offs_m = torch.arange(BLOCK_M)  # [0, 1, 2, ..., 63]
idx_i = offs_m + 1  # [1, 2, 3, ..., 64]

print(f"idx_i shape: {idx_i.shape}")
print(f"idx_i[:5]: {idx_i[:5]}")

# 计算 tau_term
tau_term = tau / idx_i
print(f"\ntau_term shape: {tau_term.shape}")
print(f"tau_term[:5]: {tau_term[:5]}")

# 广播
tau_term_expanded = tau_term[:, None]
print(f"\ntau_term[:, None] shape: {tau_term_expanded.shape}")

# 模拟 p_norm
p_norm = torch.randn(BLOCK_M, BLOCK_N) * 0.01  # 模拟 softmax 后的小值

# 计算 p_elastic
p_term = p_norm + tau_term_expanded
p_elastic = torch.clamp(p_term, min=0.0)

print(f"\np_norm shape: {p_norm.shape}")
print(f"p_term shape: {p_term.shape}")
print(f"p_elastic shape: {p_elastic.shape}")

# 统计
print(f"\n统计:")
print(f"  p_term < 0 的比例: {(p_term < 0).float().mean().item() * 100:.2f}%")
print(f"  p_elastic > 0 的比例: {(p_elastic > 0).float().mean().item() * 100:.2f}%")
print(f"  p_elastic 非零元素数: {(p_elastic > 1e-8).sum().item()} / {p_elastic.numel()}")

# 检查每个 query 位置
print(f"\n每个 query 位置的 tau/i:")
for i in [0, 15, 31, 63]:
    print(f"  位置 {i+1}: tau/i = {tau_term[i].item():.6f}, 非零比例 = {(p_elastic[i] > 1e-8).float().mean().item() * 100:.2f}%")

print("="*80)
