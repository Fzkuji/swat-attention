#!/usr/bin/env python3
"""找出 Scratch 和 Flash 在哪一步开始产生差异"""
import torch
import torch.nn.functional as F
from adasplash.lazy_attention_triton import _lazy_attention_forward_return_lse

print("="*80)
print("找出差异点")
print("="*80)

torch.manual_seed(42)

B, H, L, D = 1, 1, 16, 32  # 较小的尺寸
device = 'cuda'
tau_val = -1.0

# 创建相同的输入
q_f32 = torch.randn(B, H, L, D, device=device, dtype=torch.float32) * 0.02
k_f32 = torch.randn(B, H, L, D, device=device, dtype=torch.float32) * 0.02
v_f32 = torch.randn(B, H, L, D, device=device, dtype=torch.float32)

# 转成 bf16 给 Flash
q_bf16 = q_f32.to(torch.bfloat16)
k_bf16 = k_f32.to(torch.bfloat16)
v_bf16 = v_f32.to(torch.bfloat16)

print(f"输入: B={B}, H={H}, L={L}, D={D}")

# ============================================================================
# Scratch: 计算中间值
# ============================================================================
print(f"\n" + "="*80)
print("Scratch 中间值")
print("="*80)

# 1. QK^T
scaling = 1.0 / (D ** 0.5)
scores_scratch = torch.matmul(q_f32, k_f32.transpose(-2, -1)) * scaling
print(f"\n1. Scores (QK^T * scale):")
print(f"   [0,0,0,:5]: {scores_scratch[0,0,0,:5]}")

# 2. Causal mask
causal_mask = torch.tril(torch.ones(L, L, device=device))
scores_scratch = scores_scratch.masked_fill(causal_mask[None, None, :, :] == 0, float('-inf'))

# 3. Softmax
attn_scratch = F.softmax(scores_scratch, dim=-1, dtype=torch.float32)
print(f"\n2. Softmax:")
print(f"   [0,0,0,:5]: {attn_scratch[0,0,0,:5]}")
print(f"   row 0 sum: {attn_scratch[0,0,0,:].sum().item():.10f}")

# 4. Elastic-Softmax
tau = torch.tensor(tau_val, dtype=torch.float32, device=device)
i_positions = torch.arange(1, L + 1, device=device, dtype=torch.float32).view(1, 1, L, 1)
tau_over_i = tau / i_positions  # [1, 1, L, 1]

elastic_scratch = torch.zeros_like(attn_scratch)
for i in range(L):
    elastic_scratch[0, 0, i, :i+1] = F.relu(attn_scratch[0, 0, i, :i+1] + tau_over_i[0, 0, i, 0])

print(f"\n3. Elastic attention (ReLU(softmax + tau/i)):")
print(f"   [0,0,0,:5]: {elastic_scratch[0,0,0,:5]}")
print(f"   非零比例: {(elastic_scratch > 0).float().mean().item() * 100:.2f}%")

# 5. Output
out_scratch = torch.matmul(elastic_scratch, v_f32)
print(f"\n4. Output:")
print(f"   [0,0,0,:5]: {out_scratch[0,0,0,:5]}")
print(f"   mean: {out_scratch.mean().item():.10f}")

# ============================================================================
# Flash: 获取中间值（从 LSE）
# ============================================================================
print(f"\n" + "="*80)
print("Flash 中间值")
print("="*80)

tau_bf16 = torch.tensor(tau_val, dtype=torch.bfloat16, device=device).view(H)
bias_bf16 = torch.zeros(H, 512, device=device, dtype=torch.bfloat16)

out_flash, lse_flash = _lazy_attention_forward_return_lse(
    q_bf16, k_bf16, v_bf16, bias_bf16, tau_bf16, window_size=512, varlen=None
)

print(f"\n1. LSE:")
print(f"   [0,0,:5]: {lse_flash[0,0,:5]}")

# 用 LSE 反推 softmax
print(f"\n2. 通过 LSE 计算 Softmax (exp(s - lse)):")

# 重新计算 scores（用 bf16）
scores_flash_bf16 = torch.matmul(q_bf16, k_bf16.transpose(-2, -1)) * scaling
scores_flash_bf16 = scores_flash_bf16.masked_fill(causal_mask[None, None, :, :] == 0, float('-inf'))

# 用 LSE 计算 p_norm
attn_flash_from_lse = torch.zeros_like(scores_flash_bf16)
for i in range(L):
    scores_i = scores_flash_bf16[0, 0, i, :i+1].float()
    p_norm = torch.exp(scores_i - lse_flash[0, 0, i])
    attn_flash_from_lse[0, 0, i, :i+1] = p_norm.to(torch.bfloat16)

print(f"   [0,0,0,:5]: {attn_flash_from_lse[0,0,0,:5]}")
print(f"   row 0 sum: {attn_flash_from_lse[0,0,0,:].sum().item():.10f}")

# 3. Output
print(f"\n3. Output:")
print(f"   [0,0,0,:5]: {out_flash[0,0,0,:5]}")
print(f"   mean: {out_flash.mean().item():.10f}")

# ============================================================================
# 对比每一步
# ============================================================================
print(f"\n" + "="*80)
print("逐步对比")
print("="*80)

# 对比 softmax
attn_diff = (attn_scratch.to(torch.bfloat16) - attn_flash_from_lse).abs()
print(f"\nSoftmax 差异:")
print(f"   max: {attn_diff.max().item():.10f}")
print(f"   mean: {attn_diff.mean().item():.10f}")
print(f"   row 0 sum 差异: {abs(attn_scratch[0,0,0,:].sum().item() - attn_flash_from_lse[0,0,0,:].sum().item()):.10f}")

# 对比输出
out_diff = (out_scratch.to(torch.bfloat16) - out_flash).abs()
print(f"\n输出差异:")
print(f"   max: {out_diff.max().item():.10f}")
print(f"   mean: {out_diff.mean().item():.10f}")

# ============================================================================
# 关键测试：LSE 是否准确
# ============================================================================
print(f"\n" + "="*80)
print("LSE 准确性测试")
print("="*80)

# 用 PyTorch 计算 LSE
lse_pytorch = torch.zeros(B, H, L, device=device, dtype=torch.float32)
for i in range(L):
    scores_i = scores_scratch[0, 0, i, :i+1]
    m = scores_i.max()
    lse_pytorch[0, 0, i] = m + torch.log(torch.sum(torch.exp(scores_i - m)))

lse_diff = (lse_pytorch - lse_flash.float()).abs()
print(f"\nLSE 差异 (PyTorch vs Triton):")
print(f"   max: {lse_diff.max().item():.10f}")
print(f"   mean: {lse_diff.mean().item():.10f}")
print(f"   [0,0,:5]: {lse_diff[0,0,:5]}")

# ============================================================================
# 结论
# ============================================================================
print(f"\n" + "="*80)
print("结论")
print("="*80)

if lse_diff.max().item() > 0.01:
    print("❌ LSE 计算有明显误差")
    print("   这会导致 softmax 不准确")
    print("   需要修复 _get_lse_kernel_batch")
elif attn_diff.max().item() > 0.01:
    print("❌ Softmax 有明显差异（但 LSE 正确）")
    print("   可能是 exp(s - lse) 的精度问题")
elif out_diff.max().item() > 0.001:
    print("❌ 输出有明显差异（但 softmax 正确）")
    print("   问题在 elastic-softmax 或 matmul(p, v)")
else:
    print("✅ 各步骤都基本一致")
    print(f"   最大差异 < 0.001 (bf16 精度范围内)")
    print(f"   Scratch 和 Flash 实现逻辑一致")

print("="*80)
