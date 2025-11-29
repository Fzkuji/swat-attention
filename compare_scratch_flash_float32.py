#!/usr/bin/env python3
"""用 float32 精确对比 scratch 和 flash 实现"""
import torch
import torch.nn.functional as F
from adasplash import lazy_attention_triton

print("="*80)
print("Scratch vs Flash 精确对比 (float32)")
print("="*80)

torch.manual_seed(42)

B, H, L, D = 1, 2, 32, 32
device = 'cuda'
tau_val = -1.0

# 创建相同的输入（使用 float32！）
q = torch.randn(B, H, L, D, device=device, dtype=torch.float32) * 0.02
k = torch.randn(B, H, L, D, device=device, dtype=torch.float32) * 0.02
v = torch.randn(B, H, L, D, device=device, dtype=torch.float32)

print(f"输入: B={B}, H={H}, L={L}, D={D}")
print(f"dtype: {q.dtype}")
print(f"tau: {tau_val}")

# ============================================================================
# Scratch 实现 (float32)
# ============================================================================
print(f"\n" + "="*80)
print("Scratch 实现 (PyTorch, float32)")
print("="*80)

tau_scratch = torch.full((H,), tau_val, device=device, dtype=torch.float32)
bias_scratch = torch.zeros(H, 512, device=device, dtype=torch.float32)

# 1. QK^T
scaling = 1.0 / (D ** 0.5)
attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scaling
print(f"\n1. Attention scores (QK^T * scale):")
print(f"   shape: {attn_scores.shape}")
print(f"   [0,0,0,:5]: {attn_scores[0,0,0,:5]}")
print(f"   mean: {attn_scores.mean().item():.10f}")

# 2. Apply bias (距离相关)
rel_pos = torch.arange(L, device=device)[:, None] - torch.arange(L, device=device)[None, :]
valid_mask = (0 <= rel_pos) & (rel_pos < 512)
indices = rel_pos.clamp(0, 511)
bias_matrix = bias_scratch[:, indices] * valid_mask.float()  # [H, L, L]
attn_scores = attn_scores + bias_matrix[None, :, :, :]  # [B, H, L, L]

print(f"\n2. After bias:")
print(f"   bias_matrix[0,0,:5]: {bias_matrix[0,0,:5]}")
print(f"   attn_scores[0,0,0,:5]: {attn_scores[0,0,0,:5]}")

# 3. Causal mask
causal_mask = torch.tril(torch.ones(L, L, device=device))
attn_scores = attn_scores.masked_fill(causal_mask[None, None, :, :] == 0, float('-inf'))

print(f"\n3. After causal mask:")
print(f"   attn_scores[0,0,0,:5]: {attn_scores[0,0,0,:5]}")

# 4. Softmax (float32)
attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32)

print(f"\n4. Softmax weights:")
print(f"   [0,0,0,:5]: {attn_weights[0,0,0,:5]}")
print(f"   sum per row (应该=1): {attn_weights[0,0,0,:].sum().item():.10f}")

# 5. Elastic-Softmax: ReLU(Softmax + τ/i)
tau_expanded = tau_scratch.view(1, H, 1, 1)
i_positions = torch.arange(1, L + 1, device=device, dtype=torch.float32).view(1, 1, L, 1)

print(f"\n5. tau/i:")
print(f"   tau_expanded: {tau_expanded[0,:,0,0]}")
print(f"   i_positions[0,0,:5,0]: {i_positions[0,0,:5,0]}")
print(f"   tau/i[0,0,:5,0]: {(tau_expanded / i_positions)[0,0,:5,0]}")

# 加上 tau/i
attn_with_tau = attn_weights + tau_expanded / i_positions

print(f"\n6. Softmax + tau/i (before ReLU):")
print(f"   [0,0,0,:5]: {attn_with_tau[0,0,0,:5]}")
print(f"   min: {attn_with_tau.min().item():.10f}")
print(f"   max: {attn_with_tau.max().item():.10f}")

# ReLU
elastic_attn = F.relu(attn_with_tau)

print(f"\n7. After ReLU:")
print(f"   [0,0,0,:5]: {elastic_attn[0,0,0,:5]}")
print(f"   非零比例: {(elastic_attn > 0).float().mean().item() * 100:.2f}%")
print(f"   sum per row: {elastic_attn[0,0,0,:].sum().item():.10f}")

# 8. Output
out_scratch = torch.matmul(elastic_attn, v)

print(f"\n8. Output:")
print(f"   [0,0,0,:5]: {out_scratch[0,0,0,:5]}")
print(f"   mean: {out_scratch.mean().item():.10f}")
print(f"   std: {out_scratch.std().item():.10f}")

# ============================================================================
# Flash 实现 (Triton, 转成 float32)
# ============================================================================
print(f"\n" + "="*80)
print("Flash 实现 (Triton, float32)")
print("="*80)

# 转成 bf16 给 Triton（因为 kernel 目前只支持 bf16）
q_bf16 = q.to(torch.bfloat16)
k_bf16 = k.to(torch.bfloat16)
v_bf16 = v.to(torch.bfloat16)
tau_flash = torch.full((H,), tau_val, device=device, dtype=torch.bfloat16)
bias_flash = torch.zeros(H, 512, device=device, dtype=torch.bfloat16)

out_flash_bf16 = lazy_attention_triton(q_bf16, k_bf16, v_bf16, bias_flash, tau_flash)

# 转回 float32 比较
out_flash = out_flash_bf16.to(torch.float32)

print(f"\nOutput:")
print(f"   [0,0,0,:5]: {out_flash[0,0,0,:5]}")
print(f"   mean: {out_flash.mean().item():.10f}")
print(f"   std: {out_flash.std().item():.10f}")

# ============================================================================
# 对比
# ============================================================================
print(f"\n" + "="*80)
print("对比分析")
print("="*80)

diff = (out_scratch - out_flash).abs()

print(f"\n输出差异:")
print(f"   max: {diff.max().item():.10f}")
print(f"   mean: {diff.mean().item():.10f}")
print(f"   [0,0,0,:5]: {diff[0,0,0,:5]}")

rel_diff = (diff / (out_scratch.abs() + 1e-10)).mean().item()
print(f"   相对差异: {rel_diff * 100:.4f}%")

# ============================================================================
# 分析差异来源
# ============================================================================
print(f"\n" + "="*80)
print("差异来源分析")
print("="*80)

print(f"\nScratch 实现:")
print(f"   elastic_attn 非零比例: {(elastic_attn > 0).float().mean().item() * 100:.2f}%")
print(f"   elastic_attn 均值: {elastic_attn.mean().item():.10f}")
print(f"   输出均值: {out_scratch.mean().item():.10f}")

print(f"\nFlash 实现:")
print(f"   输出均值: {out_flash.mean().item():.10f}")

# 理论值（tau=-1.0 时应该全为 0）
print(f"\n理论分析:")
print(f"   当 tau=-1.0, bias=0 时:")
print(f"   - i=1: softmax≈1.0, tau/1=-1.0, sum=0, ReLU=0")
print(f"   - i=2: softmax≈0.5, tau/2=-0.5, sum=0, ReLU=0")
print(f"   - 理论上 elastic_attn 应该全为 0")
print(f"   - 输出也应该全为 0")

scratch_close_to_zero = out_scratch.abs().max().item() < 1e-6
flash_close_to_zero = out_flash.abs().max().item() < 1e-6

print(f"\n实际结果:")
print(f"   Scratch 接近 0: {scratch_close_to_zero} (max={out_scratch.abs().max().item():.10f})")
print(f"   Flash 接近 0: {flash_close_to_zero} (max={out_flash.abs().max().item():.10f})")

# ============================================================================
# 结论
# ============================================================================
print(f"\n" + "="*80)
print("结论")
print("="*80)

if scratch_close_to_zero and flash_close_to_zero:
    print("✅ 两者都接近 0（正确）")
    print("   问题可能不在 forward，而在其他地方")
elif scratch_close_to_zero and not flash_close_to_zero:
    print("❌ Scratch 正确（≈0），Flash 错误（≠0）")
    print("   Flash kernel 有 bug!")
    print(f"   Flash 输出: {out_flash.abs().max().item():.10f}")
elif not scratch_close_to_zero and flash_close_to_zero:
    print("❌ Scratch 错误（≠0），Flash 正确（≈0）")
    print("   Scratch 实现有 bug!")
    print(f"   Scratch 输出: {out_scratch.abs().max().item():.10f}")
else:
    print("❌ 两者都不为 0")
    if rel_diff < 0.01:
        print("   但两者一致（相对差异 <1%）")
        print("   可能都有同样的数值精度问题")
    else:
        print(f"   且不一致（相对差异 {rel_diff*100:.2f}%）")
        print("   两者实现逻辑不同!")

print("="*80)
