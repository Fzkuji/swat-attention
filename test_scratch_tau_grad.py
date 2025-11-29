#!/usr/bin/env python3
"""测试 scratch 分支在 tau=-1 时的梯度情况"""
import torch
import torch.nn as nn
import torch.nn.functional as F

print("="*80)
print("模拟 Scratch 分支的 Elastic-Softmax 梯度计算")
print("="*80)

# 参数
batch_size = 2
num_heads = 16
seq_len = 64
tau_init = -1.0

# 创建随机的 softmax 权重（模拟 attention weights）
torch.manual_seed(42)
attn_weights = F.softmax(torch.randn(batch_size, num_heads, seq_len, seq_len), dim=-1)

print(f"attn_weights shape: {attn_weights.shape}")
print(f"attn_weights min/max/mean: {attn_weights.min().item():.6f} / {attn_weights.max().item():.6f} / {attn_weights.mean().item():.6f}")

# tau 参数
tau = nn.Parameter(torch.full((num_heads,), tau_init))
print(f"\ntau: {tau}")

# i_positions
i_positions = torch.arange(1, seq_len + 1, dtype=torch.float32).view(1, 1, -1, 1)

# Elastic-Softmax 公式: ReLU(Softmax + tau/i)
tau_expanded = tau.view(1, num_heads, 1, 1)
elastic_attn = F.relu(attn_weights + tau_expanded / i_positions)

print(f"\nelastic_attn shape: {elastic_attn.shape}")
print(f"elastic_attn 非零比例: {(elastic_attn > 0).float().mean().item() * 100:.2f}%")

# 简单的 loss (就像 attention output 的 sum)
loss = elastic_attn.sum()

print(f"\nloss: {loss.item():.6f}")

# 反向传播
print("\n反向传播...")
loss.backward()

# 检查梯度
print(f"\ntau.grad: {tau.grad}")
print(f"tau.grad sum: {tau.grad.sum().item():.6f}")
print(f"tau.grad abs sum: {tau.grad.abs().sum().item():.6f}")

# 分析每个 head 的梯度
print(f"\n每个 head 的 tau 梯度:")
for h in range(num_heads):
    print(f"  Head {h}: {tau.grad[h].item():.6f}")

# 详细分析：为什么会有梯度？
print("\n" + "="*80)
print("分析: 为什么 scratch 分支有梯度?")
print("="*80)

# 计算 p_term = attn_weights + tau/i 的统计
p_term = attn_weights + tau_expanded / i_positions
mask_positive = p_term > 0
positive_ratio = mask_positive.float().mean().item()

print(f"\np_term = attn_weights + tau/i 的统计:")
print(f"  min: {p_term.min().item():.6f}")
print(f"  max: {p_term.max().item():.6f}")
print(f"  mean: {p_term.mean().item():.6f}")
print(f"  正数比例: {positive_ratio * 100:.2f}%")

# 采样几个位置
print(f"\n不同位置的 p_term 值 (第一个样本，第一个头):")
for i_pos in [1, 16, 32, 64]:
    idx = i_pos - 1
    p_val = p_term[0, 0, idx, :i_pos]
    positive_count = (p_val > 0).sum().item()
    print(f"  位置 {i_pos}: 正数 {positive_count}/{i_pos} ({positive_count/i_pos*100:.1f}%), mean={p_val.mean().item():.6f}")

print(f"\n关键发现:")
print(f"  - 即使 tau=-1.0，仍有 {positive_ratio*100:.2f}% 的位置满足 p_term > 0")
print(f"  - 这些位置贡献了 tau 的梯度")
print(f"  - 梯度 = Σ (∂L/∂elastic_attn) * (1/i) * 𝟙[p_term > 0]")

# 手动计算期望的梯度（简化）
print(f"\n手动验证梯度计算:")
# 对于 loss = elastic_attn.sum()，∂L/∂elastic_attn = 1
# ∂elastic_attn/∂tau = ReLU'(p_term) * (1/i) = 𝟙[p_term > 0] * (1/i)

manual_grad = torch.zeros_like(tau)
for b in range(batch_size):
    for h in range(num_heads):
        grad_h = 0.0
        for i in range(seq_len):
            for j in range(i + 1):  # causal attention
                i_pos = i + 1
                if p_term[b, h, i, j] > 0:
                    grad_h += 1.0 / i_pos
        manual_grad[h] += grad_h

print(f"手动计算的梯度: {manual_grad}")
print(f"PyTorch autograd 的梯度: {tau.grad}")
print(f"差异: {(manual_grad - tau.grad).abs().max().item():.6f}")

print("="*80)
