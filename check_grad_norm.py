#!/usr/bin/env python3
"""检查梯度范数和更新"""
import torch
import torch.nn.functional as F
from adasplash import lazy_attention_triton

print("="*80)
print("检查梯度范数")
print("="*80)

torch.manual_seed(42)

B, H, L, D = 2, 4, 64, 32
device = 'cuda'
tau_init = -1.0

# 创建输入和目标
q = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
k = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
v = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16)
target = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.1

print(f"输入: B={B}, H={H}, L={L}, D={D}")

# ============================================================================
# 测试 1: 检查原始梯度
# ============================================================================
print(f"\n" + "="*80)
print("测试 1: 原始梯度范数")
print("="*80)

tau = torch.nn.Parameter(torch.full((H,), tau_init, device=device, dtype=torch.float32))
bias = torch.nn.Parameter(torch.randn(H, 512, device=device, dtype=torch.float32) * 0.02)

# 前向和反向
out = lazy_attention_triton(q, k, v, bias.to(q.dtype), tau.to(q.dtype))
loss = F.mse_loss(out, target)
loss.backward()

# 计算梯度范数
tau_grad_norm = tau.grad.norm().item()
bias_grad_norm = bias.grad.norm().item()
total_grad_norm = torch.sqrt(tau.grad.norm()**2 + bias.grad.norm()**2).item()

print(f"梯度范数:")
print(f"  tau.grad.norm():  {tau_grad_norm:.6f}")
print(f"  bias.grad.norm(): {bias_grad_norm:.6f}")
print(f"  total grad norm:  {total_grad_norm:.6f}")

print(f"\n梯度统计:")
print(f"  tau.grad:  mean={tau.grad.mean().item():.6f}, max={tau.grad.max().item():.6f}, min={tau.grad.min().item():.6f}")
print(f"  bias.grad: mean={bias.grad.mean().item():.6f}, max={bias.grad.abs().max().item():.6f}")

# ============================================================================
# 测试 2: 模拟梯度裁剪
# ============================================================================
print(f"\n" + "="*80)
print("测试 2: 模拟不同的梯度裁剪阈值")
print("="*80)

clip_values = [0.001, 0.01, 0.1, 1.0, 10.0, None]

for clip_value in clip_values:
    # 重置梯度
    tau_grad_clipped = tau.grad.clone()
    bias_grad_clipped = bias.grad.clone()

    if clip_value is not None:
        # 计算当前梯度范数
        current_norm = torch.sqrt(tau_grad_clipped.norm()**2 + bias_grad_clipped.norm()**2).item()

        if current_norm > clip_value:
            # 裁剪梯度
            scale = clip_value / current_norm
            tau_grad_clipped *= scale
            bias_grad_clipped *= scale
            clipped_norm = clip_value
            status = "✂️ 被裁剪"
        else:
            clipped_norm = current_norm
            status = "✅ 未裁剪"
    else:
        clipped_norm = total_grad_norm
        status = "🚫 无裁剪"

    # 计算更新量（lr=0.01）
    lr = 0.01
    tau_update = tau_grad_clipped[0].item() * lr

    if clip_value is not None:
        print(f"  clip={clip_value:<6.3f}: norm={clipped_norm:.6f}, tau更新={tau_update:.8f}, {status}")
    else:
        print(f"  无裁剪      : norm={clipped_norm:.6f}, tau更新={tau_update:.8f}, {status}")

# ============================================================================
# 测试 3: 检查是否需要 gradient scaling
# ============================================================================
print(f"\n" + "="*80)
print("测试 3: 梯度缩放")
print("="*80)

scales = [0.1, 1.0, 10.0, 100.0]

for scale in scales:
    # 缩放后的更新
    tau_update_scaled = tau.grad[0].item() * 0.01 * scale
    print(f"  scale={scale:<6.1f}: tau更新={tau_update_scaled:.8f}")

# ============================================================================
# 测试 4: 对比 AdamW
# ============================================================================
print(f"\n" + "="*80)
print("测试 4: SGD vs AdamW")
print("="*80)

num_steps = 10

for opt_name, opt_class, opt_kwargs in [
    ('SGD lr=0.01', torch.optim.SGD, {'lr': 0.01}),
    ('SGD lr=0.1', torch.optim.SGD, {'lr': 0.1}),
    ('AdamW lr=0.001', torch.optim.AdamW, {'lr': 0.001}),
    ('AdamW lr=0.01', torch.optim.AdamW, {'lr': 0.01}),
]:
    # 重新初始化
    tau_test = torch.nn.Parameter(torch.full((H,), tau_init, device=device, dtype=torch.float32))
    bias_test = torch.nn.Parameter(torch.randn(H, 512, device=device, dtype=torch.float32) * 0.02)

    optimizer = opt_class([tau_test, bias_test], **opt_kwargs)

    losses = []
    for step in range(num_steps):
        optimizer.zero_grad()
        out = lazy_attention_triton(q, k, v, bias_test.to(q.dtype), tau_test.to(q.dtype))
        loss = F.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    loss_reduction = (losses[0] - losses[-1]) / losses[0] * 100
    tau_change = tau_test[0].item() - tau_init

    print(f"  {opt_name:<20}: loss减少={loss_reduction:>6.2f}%, tau变化={tau_change:>10.6f}")

# ============================================================================
# 结论
# ============================================================================
print(f"\n" + "="*80)
print("结论")
print("="*80)

print(f"\n1. 原始梯度范数: {total_grad_norm:.6f}")
print(f"   - 这个范数不算小，不应该被常规的梯度裁剪（如 1.0）影响")

print(f"\n2. 如果你的训练使用了梯度裁剪（如 DeepSpeed），检查:")
print(f"   - 裁剪阈值是否过小（< {total_grad_norm:.6f}）")
print(f"   - 是否有自动梯度缩放（AMP）")

print(f"\n3. 学习率可能需要增大:")
print(f"   - 当前 lr=0.01 时，tau 每步变化约 {tau.grad[0].item() * 0.01:.8f}")
print(f"   - 建议尝试 lr=0.1 或使用 AdamW")

print("="*80)
