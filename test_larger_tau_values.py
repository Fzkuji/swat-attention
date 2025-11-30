#!/usr/bin/env python3
"""测试更大范围的 tau 值，看 scratch 和 flash 的差异"""
import torch
import torch.nn.functional as F
from adasplash import lazy_attention_triton

print("="*80)
print("测试不同 tau 值下的差异")
print("="*80)

torch.manual_seed(42)

B, H, L, D = 1, 2, 64, 32
device = 'cuda'

# 创建输入
q = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
k = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
v = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16)

print(f"输入: B={B}, H={H}, L={L}, D={D}")

# 测试不同的 tau 值
tau_values = [-2.0, -1.5, -1.0, -0.5, -0.1, 0.0, 0.1, 0.5]

print(f"\n" + "="*80)
print("对比不同 tau 值")
print("="*80)

results = []

for tau_val in tau_values:
    # Scratch 实现
    tau_scratch = torch.full((H,), tau_val, device=device, dtype=torch.bfloat16)
    bias_scratch = torch.zeros(H, 512, device=device, dtype=torch.bfloat16)

    scaling = 1.0 / (D ** 0.5)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scaling

    # Causal mask
    causal_mask = torch.tril(torch.ones(L, L, device=device, dtype=torch.bfloat16))
    attn_scores = attn_scores.masked_fill(causal_mask[None, None, :, :] == 0, float('-inf'))

    # Softmax
    attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)

    # Elastic-Softmax
    tau_expanded = tau_scratch.view(1, H, 1, 1)
    i_positions = torch.arange(1, L + 1, device=device, dtype=torch.bfloat16).view(1, 1, L, 1)
    elastic_attn = F.relu(attn_weights + tau_expanded / i_positions)

    out_scratch = torch.matmul(elastic_attn, v)

    # Flash 实现
    tau_flash = torch.full((H,), tau_val, device=device, dtype=torch.bfloat16)
    bias_flash = torch.zeros(H, 512, device=device, dtype=torch.bfloat16)

    out_flash = lazy_attention_triton(q, k, v, bias_flash, tau_flash)

    # 对比
    diff = (out_scratch - out_flash).abs()
    rel_diff = (diff / (out_scratch.abs() + 1e-8)).mean().item()

    results.append({
        'tau': tau_val,
        'scratch_mean': out_scratch.mean().item(),
        'flash_mean': out_flash.mean().item(),
        'abs_diff': diff.mean().item(),
        'rel_diff': rel_diff * 100,
        'max_diff': diff.max().item()
    })

    print(f"\ntau = {tau_val:>6.2f}:")
    print(f"  Scratch mean: {out_scratch.mean().item():>12.6f}")
    print(f"  Flash   mean: {out_flash.mean().item():>12.6f}")
    print(f"  平均差异:     {diff.mean().item():>12.6f}")
    print(f"  相对差异:     {rel_diff:>12.4f}%")

# 总结
print(f"\n" + "="*80)
print("总结")
print("="*80)

print(f"\n{'Tau':<8} {'Scratch':<14} {'Flash':<14} {'平均差异':<12} {'相对差异%':<10}")
print("-" * 70)
for r in results:
    print(f"{r['tau']:<8.2f} {r['scratch_mean']:<14.6f} {r['flash_mean']:<14.6f} "
          f"{r['abs_diff']:<12.6f} {r['rel_diff']:<10.4f}")

# 找出差异最大的情况
max_rel_diff = max(results, key=lambda x: x['rel_diff'])
print(f"\n最大相对差异:")
print(f"  tau = {max_rel_diff['tau']}")
print(f"  相对差异: {max_rel_diff['rel_diff']:.4f}%")

# 找出差异最小的情况
min_rel_diff = min(results, key=lambda x: x['rel_diff'])
print(f"\n最小相对差异:")
print(f"  tau = {min_rel_diff['tau']}")
print(f"  相对差异: {min_rel_diff['rel_diff']:.4f}%")

print(f"\n观察:")
if max_rel_diff['rel_diff'] > 10:
    print(f"  ⚠️ 某些 tau 值下差异很大 (>{max_rel_diff['rel_diff']:.1f}%)")
    print(f"  这说明实现有显著差异，不仅仅是精度问题")
elif max_rel_diff['rel_diff'] > 5:
    print(f"  ⚠️ 有中等差异 (~{max_rel_diff['rel_diff']:.1f}%)")
    print(f"  可能需要调查")
else:
    print(f"  ✅ 差异较小 (<{max_rel_diff['rel_diff']:.1f}%)")
    print(f"  主要是 bf16 精度导致")

print("="*80)
