#!/usr/bin/env python3
"""手动推导并验证 Elastic-Softmax 的梯度公式"""
import torch
import torch.nn.functional as F

print("="*80)
print("验证 Elastic-Softmax 梯度公式")
print("="*80)

torch.manual_seed(42)
L, D = 8, 4
tau_val = -1.0

# 创建简单的输入（单个 query）
scores = torch.randn(L, requires_grad=True, dtype=torch.float32) * 0.5  # [L]
v = torch.randn(L, D, dtype=torch.float32)  # [L, D]
tau = torch.tensor(tau_val, requires_grad=True, dtype=torch.float32)

# 位置 i（这个 query 能看到多少个 token）
i = L

print(f"输入:")
print(f"  scores shape: {scores.shape}")
print(f"  v shape: {v.shape}")
print(f"  tau: {tau.item()}")
print(f"  位置 i: {i}")

# 前向传播
print("\n" + "-"*80)
print("前向传播")
print("-"*80)

# 1. Softmax
p = F.softmax(scores, dim=0)  # [L]
print(f"  p (softmax) = {p}")

# 2. Elastic-Softmax
p_elastic = F.relu(p + tau / i)  # [L]
print(f"  tau/i = {tau.item()/i:.6f}")
print(f"  p + tau/i = {(p + tau/i)}")
print(f"  p_elastic (ReLU) = {p_elastic}")

# 3. Output
out = torch.matmul(p_elastic, v)  # [D]
print(f"  out = {out}")

# 4. Loss (简单的 sum)
loss = out.sum()
print(f"  loss = {loss.item():.6f}")

# 反向传播（PyTorch autograd）
print("\n" + "-"*80)
print("反向传播 (PyTorch autograd)")
print("-"*80)

loss.backward()

print(f"  scores.grad = {scores.grad}")
print(f"  tau.grad = {tau.grad.item():.10f}")

# 手动计算梯度
print("\n" + "-"*80)
print("手动计算梯度")
print("-"*80)

# 重置
scores2 = scores.detach().clone()
p2 = F.softmax(scores2, dim=0)
p_elastic2 = F.relu(p2 + tau.detach() / i)
out2 = torch.matmul(p_elastic2, v)

# dL/dout = 1 (因为 loss = out.sum())
d_out = torch.ones_like(out2)  # [D]

# dout/dp_elastic = v
d_p_elastic = torch.matmul(d_out, v.T)  # [D] @ [D, L] = [L]
print(f"  d_p_elastic = {d_p_elastic}")

# dp_elastic/dp = ReLU'(p + tau/i)
mask_relu = (p2 + tau.detach() / i) > 0
d_p = d_p_elastic * mask_relu.float()
print(f"  mask_relu = {mask_relu}")
print(f"  d_p = {d_p}")

# dp/dscores = softmax 的 Jacobian
# dscores_j = Σ_k (dp_k/dscores_j) * d_p_k
#           = Σ_k [(δ_jk * p_k - p_j * p_k)] * d_p_k
#           = p_j * d_p_j - p_j * Σ_k (p_k * d_p_k)
#           = p_j * (d_p_j - Σ_k (p_k * d_p_k))

delta = (p2 * d_p).sum()
d_scores_manual = p2 * (d_p - delta)
print(f"  delta = {delta.item():.6f}")
print(f"  d_scores_manual = {d_scores_manual}")

# 对比
print("\n" + "-"*80)
print("对比 autograd vs 手动计算")
print("-"*80)

diff_scores = (scores.grad - d_scores_manual).abs().max().item()
print(f"  scores.grad 差异: {diff_scores:.10f}")
if diff_scores < 1e-6:
    print(f"  ✅ scores 梯度一致")
else:
    print(f"  ❌ scores 梯度不一致!")
    print(f"     autograd: {scores.grad}")
    print(f"     manual:   {d_scores_manual}")

# 手动计算 tau 梯度
# dL/dtau = Σ_k (dL/dp_elastic_k) * (dp_elastic_k/dtau)
#         = Σ_k d_p_elastic_k * mask_relu_k * (1/i)
d_tau_manual = (d_p_elastic * mask_relu.float() * (1.0 / i)).sum()
print(f"\n  tau.grad 对比:")
print(f"     autograd: {tau.grad.item():.10f}")
print(f"     manual:   {d_tau_manual.item():.10f}")
diff_tau = abs(tau.grad.item() - d_tau_manual.item())
if diff_tau < 1e-6:
    print(f"  ✅ tau 梯度一致")
else:
    print(f"  ❌ tau 梯度不一致! 差异: {diff_tau:.10f}")

# 关键检查：mask_relu 的比例
print("\n" + "="*80)
print("关键统计")
print("="*80)

positive_ratio = mask_relu.float().mean().item()
print(f"  mask_relu 正数比例: {positive_ratio * 100:.2f}%")
print(f"  tau / i = {tau.item() / i:.6f}")
print(f"  p 的范围: [{p.min().item():.6f}, {p.max().item():.6f}]")
print(f"  p + tau/i 的范围: [{(p + tau/i).min().item():.6f}, {(p + tau/i).max().item():.6f}]")

if positive_ratio < 0.1:
    print(f"\n  ⚠️ 警告: mask_relu 正数比例很低 (<10%)!")
    print(f"     这会导致 tau 梯度很小，训练缓慢")
    print(f"     解决方法: 增大 bias 初始化方差，让 scores 更分散")

print("="*80)
