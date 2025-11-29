#!/usr/bin/env python3
"""详细对比 Triton 和 PyTorch 的中间值"""
import torch
import torch.nn.functional as F
from adasplash import lazy_attention_triton

print("="*80)
print("详细对比 Triton vs PyTorch 中间计算")
print("="*80)

torch.manual_seed(42)

B, H, L, D = 1, 1, 16, 32  # 较小的尺寸便于调试
device = 'cuda'
tau_val = -1.0

# 创建相同的输入
q = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
k = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
v = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16)

print(f"输入: B={B}, H={H}, L={L}, D={D}")
print(f"tau: {tau_val}")

# ============================================================================
# PyTorch 参考实现（逐步打印中间值）
# ============================================================================
print(f"\n" + "="*80)
print("PyTorch 实现")
print("="*80)

tau_pt = torch.tensor(tau_val, dtype=torch.bfloat16, device=device)

# 1. QK^T
sm_scale = 1.0 / (D ** 0.5)
scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale  # [B, H, L, L]
print(f"\n1. Scores (QK^T * scale):")
print(f"   shape: {scores.shape}")
print(f"   mean: {scores.mean().item():.6f}, std: {scores.std().item():.6f}")
print(f"   [0,0,0,:5]: {scores[0,0,0,:5]}")

# 2. Causal mask
causal_mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.bfloat16))
scores_masked = scores.masked_fill(causal_mask[None, None, :, :] == 0, float('-inf'))
print(f"\n2. Scores (after causal mask):")
print(f"   [0,0,0,:5]: {scores_masked[0,0,0,:5]}")

# 3. Softmax
attn_weights = F.softmax(scores_masked, dim=-1, dtype=torch.float32).to(torch.bfloat16)
print(f"\n3. Attention weights (softmax):")
print(f"   mean: {attn_weights.mean().item():.6f}")
print(f"   [0,0,0,:5]: {attn_weights[0,0,0,:5]}")
print(f"   sum per row: {attn_weights[0,0,0,:].sum().item():.6f}")

# 4. Elastic-Softmax
i_positions = torch.arange(1, L + 1, device=device, dtype=torch.bfloat16)  # [L]
tau_over_i = tau_pt / i_positions  # [L]
print(f"\n4. tau/i:")
print(f"   tau_over_i[:5]: {tau_over_i[:5]}")

# 为每个 query 位置 i 加上 tau/i
elastic_attn = torch.zeros_like(attn_weights)
for i in range(L):
    # query i 看到的 key 0...i，都加上 tau/(i+1)
    elastic_attn[0, 0, i, :i+1] = attn_weights[0, 0, i, :i+1] + tau_over_i[i]

print(f"\n5. Attention + tau/i (before ReLU):")
print(f"   [0,0,0,:5]: {elastic_attn[0,0,0,:5]}")
print(f"   min: {elastic_attn.min().item():.6f}, max: {elastic_attn.max().item():.6f}")

# ReLU
elastic_attn = F.relu(elastic_attn)
print(f"\n6. Elastic attention (after ReLU):")
print(f"   [0,0,0,:5]: {elastic_attn[0,0,0,:5]}")
print(f"   非零比例: {(elastic_attn > 0).float().mean().item() * 100:.2f}%")

# Output
out_pt = torch.matmul(elastic_attn, v)
print(f"\n7. Output:")
print(f"   mean: {out_pt.mean().item():.6f}, std: {out_pt.std().item():.6f}")
print(f"   [0,0,0,:5]: {out_pt[0,0,0,:5]}")

# ============================================================================
# Triton 实现
# ============================================================================
print(f"\n" + "="*80)
print("Triton 实现")
print("="*80)

tau_triton = torch.tensor(tau_val, dtype=torch.bfloat16, device=device).view(H)
bias_triton = torch.zeros(H, 512, device=device, dtype=torch.bfloat16)

out_triton = lazy_attention_triton(q, k, v, bias_triton, tau_triton)

print(f"\nOutput:")
print(f"   mean: {out_triton.mean().item():.6f}, std: {out_triton.std().item():.6f}")
print(f"   [0,0,0,:5]: {out_triton[0,0,0,:5]}")

# ============================================================================
# 对比
# ============================================================================
print(f"\n" + "="*80)
print("对比")
print("="*80)

diff = (out_pt - out_triton).abs()
print(f"\n输出差异:")
print(f"   max: {diff.max().item():.10f}")
print(f"   mean: {diff.mean().item():.10f}")

if diff.max().item() > 0.001:
    print("\n❌ Triton 和 PyTorch 输出有显著差异!")
    print("   可能原因:")
    print("   1. LSE 计算不一致")
    print("   2. tau/i 的计算精度不同")
    print("   3. ReLU 前的值计算有误")
else:
    print("\n✅ Triton 和 PyTorch 输出一致")

# ============================================================================
# 检查为什么 PyTorch elastic_attn 全为 0
# ============================================================================
print(f"\n" + "="*80)
print("分析：为什么 elastic_attn 全为 0?")
print("="*80)

print(f"\n示例：query 位置 i=0 (第1个token)")
i = 0
print(f"   attn_weights[0,0,{i},:{i+1}]: {attn_weights[0,0,i,:i+1]}")
print(f"   tau/(i+1) = {tau_val}/{i+1} = {tau_over_i[i].item():.6f}")
print(f"   attn + tau/i: {attn_weights[0,0,i,:i+1] + tau_over_i[i]}")
print(f"   ReLU后: {F.relu(attn_weights[0,0,i,:i+1] + tau_over_i[i])}")

print(f"\n示例：query 位置 i=1 (第2个token)")
i = 1
print(f"   attn_weights[0,0,{i},:{i+1}]: {attn_weights[0,0,i,:i+1]}")
print(f"   tau/(i+1) = {tau_val}/{i+1} = {tau_over_i[i].item():.6f}")
print(f"   attn + tau/i: {attn_weights[0,0,i,:i+1] + tau_over_i[i]}")
print(f"   ReLU后: {F.relu(attn_weights[0,0,i,:i+1] + tau_over_i[i])}")

print(f"\n结论:")
print(f"   当 tau=-1.0 时:")
print(f"   - i=0: tau/1=-1.0, attn≈1.0, 相加=0, ReLU=0")
print(f"   - i=1: tau/2=-0.5, attn≈0.5, 相加=0, ReLU=0")
print(f"   - i=2: tau/3≈-0.33, attn≈0.33, 相加=0, ReLU=0")
print(f"   所以理论上 elastic_attn 应该全为 0")
print(f"\n   如果 Triton 得到非零值，说明:")
print(f"   - LSE 计算不准确")
print(f"   - 或 p_norm = exp(s - lse) 不等于 softmax(s)")
print(f"   - 或 tau/i 计算有误")

print("="*80)
