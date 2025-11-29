#!/usr/bin/env python3
"""诊断为什么 tau <= -1 时梯度为零"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from adasplash import lazy_attention_triton

torch.manual_seed(42)

B, H, L, D = 2, 4, 32, 64
device = 'cuda'

# 创建输入
q = torch.randn(B, H, L, D, device=device) * 0.02
k = torch.randn(B, H, L, D, device=device) * 0.02
v = torch.randn(B, H, L, D, device=device)

# 测试不同的 tau 值
tau_values = [-2.0, -1.0, -0.5, 0.0]

print("="*80)
print("测试不同 tau 值的梯度")
print("="*80)

for tau_val in tau_values:
    print(f"\n{'='*80}")
    print(f"测试 tau = {tau_val}")
    print(f"{'='*80}")

    # Flash 分支
    tau_flash = nn.Parameter(torch.full((H,), tau_val, device=device))
    bias_flash = nn.Parameter(torch.zeros(H, 512, device=device))
    nn.init.normal_(bias_flash, mean=0.0, std=1e-3)

    out_flash = lazy_attention_triton(q, k, v, bias_flash, tau_flash)
    loss_flash = out_flash.sum()
    loss_flash.backward()

    print(f"Flash 分支:")
    print(f"  loss: {loss_flash.item():.6f}")
    print(f"  tau.grad: {tau_flash.grad}")
    print(f"  tau.grad abs sum: {tau_flash.grad.abs().sum().item():.10f}")

    # 估算 mask_relu 的比例
    # 通过反向计算 p_norm 的大致范围
    sm_scale = 1.0 / (D ** 0.5)
    scores_approx = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
    causal_mask = torch.tril(torch.ones(L, L, device=device))
    scores_approx = scores_approx.masked_fill(causal_mask[None, None, :, :] == 0, float('-inf'))

    # 加上 bias（简化）
    for i in range(min(L, 20)):  # 只看前20个位置
        for j in range(i + 1):
            dist = i - j
            if dist < 512:
                scores_approx[:, :, i, j] += bias_flash[:, dist].detach()

    p_norm_approx = F.softmax(scores_approx, dim=-1, dtype=torch.float32)

    # 计算 p_term 和 mask_relu
    tau_expanded = tau_flash.detach().view(1, H, 1, 1)
    i_positions = torch.arange(1, L + 1, device=device, dtype=torch.float32).view(1, 1, L, 1)
    p_term = p_norm_approx + tau_expanded / i_positions
    mask_relu = p_term > 0

    positive_ratio = mask_relu.float().mean().item()
    print(f"  估算的 mask_relu 正数比例: {positive_ratio * 100:.2f}%")
    print(f"  p_norm 范围: [{p_norm_approx.min().item():.6f}, {p_norm_approx.max().item():.6f}], mean={p_norm_approx.mean().item():.6f}")
    print(f"  p_term 范围: [{p_term.min().item():.6f}, {p_term.max().item():.6f}], mean={p_term.mean().item():.6f}")

    # 分析每个位置
    for i in [0, 7, 15, 31]:
        if i < L:
            p_term_at_i = p_term[0, 0, i, :i+1]
            positive_count = (p_term_at_i > 0).sum().item()
            print(f"  位置 {i+1}: {positive_count}/{i+1} = {positive_count/(i+1)*100:.1f}% 满足 p_term > 0")

print("\n" + "="*80)
print("结论:")
print("="*80)
print("如果 tau <= -1 时，mask_relu 正数比例 < 1%，说明：")
print("  1. 几乎所有位置的 p_term = p_norm + tau/i <= 0")
print("  2. ReLU 输出全为 0")
print("  3. 反向传播时，mask_relu 屏蔽了梯度，导致 dtau = 0")
print("\n可能的解决方案:")
print("  1. 修改 tau 的初始化（例如从 -1.0 改为 -0.5）")
print("  2. 修改梯度计算，考虑 ReLU 边界处的梯度")
print("  3. 使用 Softplus 或 ELU 替代 ReLU")
print("="*80)
