#!/usr/bin/env python3
"""详细对比 scratch vs flash，找出差异"""
import torch
import torch.nn.functional as F
from adasplash import lazy_attention_triton

torch.manual_seed(42)

B, H, L, D = 2, 4, 32, 64
device = 'cuda'
tau_init = -1.0

# 创建相同的输入
q = torch.randn(B, H, L, D, device=device, requires_grad=False) * 0.02
k = torch.randn(B, H, L, D, device=device, requires_grad=False) * 0.02
v = torch.randn(B, H, L, D, device=device)

print("="*80)
print("详细对比 Scratch vs Flash")
print("="*80)

# ============================================================================
# Scratch 分支实现
# ============================================================================
print("\n1. Scratch 分支 (PyTorch native)")
print("-"*80)

tau_scratch = torch.nn.Parameter(torch.full((H,), tau_init, device=device))
bias_scratch = torch.nn.Parameter(torch.zeros(H, 512, device=device))
torch.nn.init.normal_(bias_scratch, mean=0.0, std=1e-3)

# 前向
sm_scale = 1.0 / (D ** 0.5)
scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale
causal_mask = torch.tril(torch.ones(L, L, device=device))
scores = scores.masked_fill(causal_mask[None, None, :, :] == 0, float('-inf'))

# Apply bias
for i in range(L):
    for j in range(i + 1):
        dist = i - j
        if dist < 512:
            scores[:, :, i, j] += bias_scratch[:, dist]

attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)

# Elastic-Softmax
tau_expanded = tau_scratch.view(1, H, 1, 1)
i_positions = torch.arange(1, L + 1, device=device, dtype=torch.float32).view(1, 1, L, 1)
p_term_scratch = attn_weights + tau_expanded / i_positions
elastic_attn_scratch = F.relu(p_term_scratch)

print(f"p_term 统计:")
print(f"  min: {p_term_scratch.min().item():.6f}")
print(f"  max: {p_term_scratch.max().item():.6f}")
print(f"  mean: {p_term_scratch.mean().item():.6f}")
print(f"  正数比例: {(p_term_scratch > 0).float().mean().item() * 100:.2f}%")

print(f"\nelastic_attn 统计:")
print(f"  非零比例: {(elastic_attn_scratch > 1e-8).float().mean().item() * 100:.2f}%")
print(f"  mean: {elastic_attn_scratch.mean().item():.6f}")

out_scratch = torch.matmul(elastic_attn_scratch, v)
loss_scratch = out_scratch.sum()

print(f"\nloss: {loss_scratch.item():.6f}")

# 反向
loss_scratch.backward()

print(f"\n梯度:")
print(f"  tau.grad: {tau_scratch.grad}")
print(f"  tau.grad abs sum: {tau_scratch.grad.abs().sum().item():.10f}")
print(f"  bias.grad abs sum: {bias_scratch.grad.abs().sum().item():.10f}")

# ============================================================================
# Flash 分支实现
# ============================================================================
print("\n" + "="*80)
print("2. Flash 分支 (Triton kernel)")
print("-"*80)

tau_flash = torch.nn.Parameter(torch.full((H,), tau_init, device=device))
bias_flash = torch.nn.Parameter(bias_scratch.data.clone())  # 使用相同的 bias！

out_flash = lazy_attention_triton(q, k, v, bias_flash, tau_flash)
loss_flash = out_flash.sum()

print(f"loss: {loss_flash.item():.6f}")

# 反向
loss_flash.backward()

print(f"\n梯度:")
print(f"  tau.grad: {tau_flash.grad}")
print(f"  tau.grad abs sum: {tau_flash.grad.abs().sum().item():.10f}")
print(f"  bias.grad abs sum: {bias_flash.grad.abs().sum().item():.10f}")

# ============================================================================
# 对比
# ============================================================================
print("\n" + "="*80)
print("3. 对比分析")
print("="*80)

print(f"\nLoss 差异: {abs(loss_scratch.item() - loss_flash.item()):.10f}")
print(f"  Scratch: {loss_scratch.item():.10f}")
print(f"  Flash:   {loss_flash.item():.10f}")

print(f"\nTau 梯度对比:")
print(f"  Scratch abs sum: {tau_scratch.grad.abs().sum().item():.10f}")
print(f"  Flash abs sum:   {tau_flash.grad.abs().sum().item():.10f}")
print(f"  差异: {abs(tau_scratch.grad.abs().sum().item() - tau_flash.grad.abs().sum().item()):.10f}")

for h in range(H):
    print(f"  Head {h}: Scratch={tau_scratch.grad[h].item():.10f}, Flash={tau_flash.grad[h].item():.10f}")

print(f"\nBias 梯度对比:")
print(f"  Scratch abs sum: {bias_scratch.grad.abs().sum().item():.10f}")
print(f"  Flash abs sum:   {bias_flash.grad.abs().sum().item():.10f}")
print(f"  差异: {abs(bias_scratch.grad.abs().sum().item() - bias_flash.grad.abs().sum().item()):.10f}")

# 结论
print("\n" + "="*80)
print("结论:")
print("="*80)

if tau_flash.grad.abs().sum().item() < 1e-8:
    print("❌ Flash 的 tau 梯度为零!")
    if tau_scratch.grad.abs().sum().item() > 1e-6:
        print("   但 Scratch 有梯度，说明 Triton kernel 的反向传播有问题")
        print("\n   可能的原因:")
        print("   1. mask_relu 计算不正确")
        print("   2. dp_elastic 计算有误")
        print("   3. dtau 的累加逻辑有 bug")
        print("   4. 数值精度问题（bfloat16 vs float32）")
    else:
        print("   Scratch 也没有梯度，说明确实是 tau=-1 太负导致的")
        print("   需要修改 tau 的初始化值")
else:
    if abs(tau_scratch.grad.abs().sum().item() - tau_flash.grad.abs().sum().item()) < 0.01:
        print("✅ 两者梯度基本一致")
    else:
        print("⚠️ 两者梯度有差异，但都不为零")
        print(f"   差异比例: {abs(tau_scratch.grad.abs().sum().item() - tau_flash.grad.abs().sum().item()) / max(tau_scratch.grad.abs().sum().item(), 1e-10) * 100:.2f}%")

print("="*80)
