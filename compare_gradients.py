#!/usr/bin/env python3
"""对比 Scratch 和 Flash 的梯度"""
import torch
import torch.nn.functional as F
from adasplash import lazy_attention_triton

print("="*80)
print("对比 Scratch vs Flash 的梯度")
print("="*80)

torch.manual_seed(42)

B, H, L, D = 2, 4, 64, 32
device = 'cuda'

# 创建输入
q = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
k = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
v = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16)
target = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.1

print(f"输入: B={B}, H={H}, L={L}, D={D}")

# 测试不同的 tau 值
tau_values = [-1.0, -0.5, -0.1, 0.0]

for tau_val in tau_values:
    print(f"\n" + "="*80)
    print(f"tau = {tau_val}")
    print("="*80)

    # ========================================================================
    # Scratch 实现
    # ========================================================================
    tau_scratch = torch.nn.Parameter(torch.full((H,), tau_val, device=device, dtype=torch.float32))
    bias_scratch = torch.nn.Parameter(torch.randn(H, 512, device=device, dtype=torch.float32) * 0.02)

    scaling = 1.0 / (D ** 0.5)
    attn_scores = torch.matmul(q, k.transpose(-2, -1)) * scaling

    # Apply bias (scratch 实现方式)
    rel_pos = torch.arange(L, device=device)[:, None] - torch.arange(L, device=device)[None, :]
    valid_mask = (0 <= rel_pos) & (rel_pos < 512)
    indices = rel_pos.clamp(0, 511)
    bias_matrix = bias_scratch[:, indices] * valid_mask.float()  # [H, L, L]
    attn_scores = attn_scores + bias_matrix[None, :, :, :]  # [B, H, L, L]

    # Causal mask
    causal_mask = torch.tril(torch.ones(L, L, device=device))
    attn_scores = attn_scores.masked_fill(causal_mask[None, None, :, :] == 0, float('-inf'))

    # Softmax
    attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(torch.bfloat16)

    # Elastic-Softmax
    tau_expanded = tau_scratch.view(1, H, 1, 1).to(torch.bfloat16)
    i_positions = torch.arange(1, L + 1, device=device, dtype=torch.bfloat16).view(1, 1, L, 1)
    elastic_attn = F.relu(attn_weights + tau_expanded / i_positions)

    out_scratch = torch.matmul(elastic_attn, v)

    # Loss
    loss_scratch = F.mse_loss(out_scratch, target)
    loss_scratch.backward()

    print(f"\nScratch:")
    print(f"  输出 mean: {out_scratch.mean().item():.10f}")
    print(f"  损失:      {loss_scratch.item():.10f}")
    print(f"  tau.grad:  {tau_scratch.grad}")
    print(f"  |tau.grad|: {tau_scratch.grad.abs().sum().item():.10f}")
    if bias_scratch.grad is not None:
        print(f"  |bias.grad|: {bias_scratch.grad.abs().sum().item():.10f}")
    else:
        print(f"  |bias.grad|: None (没有梯度！)")

    # ========================================================================
    # Flash 实现
    # ========================================================================
    tau_flash = torch.nn.Parameter(torch.full((H,), tau_val, device=device, dtype=torch.float32))
    bias_flash = torch.nn.Parameter(torch.randn(H, 512, device=device, dtype=torch.float32) * 0.02)

    # 使用相同的 bias 初始化
    with torch.no_grad():
        bias_flash.copy_(bias_scratch.data)

    out_flash = lazy_attention_triton(
        q, k, v,
        bias_flash.to(torch.bfloat16),
        tau_flash.to(torch.bfloat16)
    )

    # Loss
    loss_flash = F.mse_loss(out_flash, target)
    loss_flash.backward()

    print(f"\nFlash:")
    print(f"  输出 mean: {out_flash.mean().item():.10f}")
    print(f"  损失:      {loss_flash.item():.10f}")
    print(f"  tau.grad:  {tau_flash.grad}")
    print(f"  |tau.grad|: {tau_flash.grad.abs().sum().item():.10f}")
    if bias_flash.grad is not None:
        print(f"  |bias.grad|: {bias_flash.grad.abs().sum().item():.10f}")
    else:
        print(f"  |bias.grad|: None (没有梯度！)")

    # ========================================================================
    # 对比
    # ========================================================================
    print(f"\n对比:")

    # 输出差异
    out_diff = (out_scratch - out_flash).abs().mean().item()
    print(f"  输出差异:   {out_diff:.10f}")

    # Loss 差异
    loss_diff = abs(loss_scratch.item() - loss_flash.item())
    print(f"  Loss 差异:  {loss_diff:.10f}")

    # Tau 梯度差异
    tau_grad_diff = (tau_scratch.grad - tau_flash.grad).abs()
    print(f"  Tau 梯度差异:")
    print(f"    绝对: {tau_grad_diff.mean().item():.10f}")
    if tau_scratch.grad.abs().max() > 1e-8:
        rel = (tau_grad_diff / (tau_scratch.grad.abs() + 1e-10)).mean().item()
        print(f"    相对: {rel * 100:.2f}%")

    # Bias 梯度差异
    if bias_scratch.grad is not None and bias_flash.grad is not None:
        bias_grad_diff = (bias_scratch.grad - bias_flash.grad).abs()
        print(f"  Bias 梯度差异:")
        print(f"    绝对: {bias_grad_diff.mean().item():.10f}")
        if bias_scratch.grad.abs().max() > 1e-8:
            rel = (bias_grad_diff / (bias_scratch.grad.abs() + 1e-10)).mean().item()
            print(f"    相对: {rel * 100:.2f}%")
    else:
        print(f"  Bias 梯度差异: 无法对比（某个为 None）")

# ============================================================================
# 总结
# ============================================================================
print(f"\n" + "="*80)
print("总结")
print("="*80)

print("\n如果梯度差异很大（>10%），说明反向传播有问题")
print("如果梯度差异很小（<1%），说明训练配置或超参数有问题")

print("="*80)
