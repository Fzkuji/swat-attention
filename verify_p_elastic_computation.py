#!/usr/bin/env python3
"""手动验证 p_elastic 的计算是否正确"""
import torch
import torch.nn.functional as F

print("="*80)
print("手动验证 p_elastic 计算")
print("="*80)

torch.manual_seed(42)

B, H, L, D = 1, 1, 8, 16  # 小尺寸，方便验证
device = 'cuda'
tau_val = -1.0

# 创建输入
q = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
k = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02

# 计算 scores 和 softmax
sm_scale = 1.0 / (D ** 0.5)
scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale  # [B, H, L, L]
print(f"Scores (QK^T / sqrt(d)):")
print(scores[0, 0])

# Causal mask
causal_mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.bfloat16))
scores = scores.masked_fill(causal_mask[None, None, :, :] == 0, float('-inf'))

# Softmax
attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)
print(f"\nSoftmax weights:")
print(attn_weights[0, 0])

# Elastic-Softmax
tau = torch.tensor(tau_val, dtype=torch.bfloat16, device=device)
i_positions = torch.arange(1, L + 1, device=device, dtype=torch.bfloat16).view(1, 1, L, 1)

tau_term = tau / i_positions.squeeze()  # [L]
print(f"\ntau / i:")
print(tau_term)

# 对于每个 query 位置
for q_pos in range(L):
    p_norm = attn_weights[0, 0, q_pos, :q_pos+1]  # 只看 causal 部分
    tau_i = tau_term[q_pos]

    p_term = p_norm + tau_i
    p_elastic = F.relu(p_term)

    print(f"\n位置 {q_pos+1}:")
    print(f"  p_norm: {p_norm}")
    print(f"  tau/{q_pos+1} = {tau_i.item():.6f}")
    print(f"  p_term = p_norm + tau/{q_pos+1}: {p_term}")
    print(f"  p_elastic (ReLU): {p_elastic}")
    print(f"  非零元素: {(p_elastic > 1e-6).sum().item()} / {p_elastic.numel()}")

# 总结
tau_expanded = tau.view(1, 1, 1, 1)
elastic_attn_full = F.relu(attn_weights + tau_expanded / i_positions)

print(f"\n" + "="*80)
print(f"完整 elastic_attn:")
print(elastic_attn_full[0, 0])
print(f"\n非零比例: {(elastic_attn_full > 1e-6).float().mean().item() * 100:.2f}%")
print(f"非零元素数: {(elastic_attn_full > 1e-6).sum().item()} / {elastic_attn_full.numel()}")

if (elastic_attn_full > 1e-6).sum().item() == 0:
    print("\n✅ 所有元素都 <= 0，被 ReLU 截断为 0")
    print("   这是预期的，因为 tau=-1 太负了")
else:
    print("\n⚠️ 有非零元素！")
    non_zero_indices = torch.where(elastic_attn_full[0, 0] > 1e-6)
    print(f"   非零位置: {list(zip(non_zero_indices[0].tolist(), non_zero_indices[1].tolist()))}")

print("="*80)
