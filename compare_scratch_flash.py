#!/usr/bin/env python3
"""比较 scratch 分支和 flash 分支的梯度"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# 模拟 Elastic-Softmax 的梯度计算
def elastic_softmax_scratch(attn_weights, tau, i_positions):
    """scratch 分支的实现 (PyTorch native)"""
    # attn_weights 已经是 softmax 后的结果
    result = F.relu(attn_weights + tau / i_positions)
    return result

def elastic_softmax_triton_forward(p_norm, tau, idx_i):
    """Triton kernel 的前向传播"""
    tau_term = tau / idx_i
    p_elastic = torch.clamp(p_norm + tau_term, min=0.0)
    return p_elastic

def elastic_softmax_triton_backward(do, p_norm, tau, idx_i, v):
    """Triton kernel 的反向传播（简化版）"""
    tau_term = tau / idx_i
    p_term = p_norm + tau_term
    mask_relu = p_term > 0

    # dp_elastic 简化为 do（假设后续只是简单的矩阵乘法）
    dp_elastic = do

    # dTau 的计算
    term_tau = dp_elastic * (1.0 / idx_i)
    term_tau = torch.where(mask_relu, term_tau, torch.zeros_like(term_tau))
    dtau = term_tau.sum()

    return dtau, mask_relu

print("="*80)
print("比较 Scratch vs Flash 的梯度计算")
print("="*80)

# 测试参数
batch_size = 1
num_heads = 4
seq_len = 16
tau_init = -1.0

# 创建测试数据
torch.manual_seed(42)
attn_weights = F.softmax(torch.randn(batch_size, num_heads, seq_len, seq_len), dim=-1)
i_positions = torch.arange(1, seq_len + 1, dtype=torch.float32).view(1, 1, -1, 1)

# Scratch 分支方法
print("\n" + "-"*80)
print("测试 Scratch 分支 (PyTorch native)")
print("-"*80)

tau_scratch = nn.Parameter(torch.full((num_heads,), tau_init))
tau_expanded = tau_scratch.view(1, num_heads, 1, 1)

# 前向
result_scratch = elastic_softmax_scratch(attn_weights, tau_expanded, i_positions)
print(f"Result scratch shape: {result_scratch.shape}")
print(f"Result scratch 非零元素: {(result_scratch > 0).sum().item()}/{result_scratch.numel()}")
print(f"Result scratch mean: {result_scratch.mean().item():.6f}")

# 反向
loss_scratch = result_scratch.sum()
loss_scratch.backward()

print(f"\ntau_scratch.grad: {tau_scratch.grad}")
print(f"tau_scratch.grad sum: {tau_scratch.grad.sum().item():.6f}")

# Flash 分支方法 (Triton kernel 逻辑)
print("\n" + "-"*80)
print("测试 Flash 分支 (Triton kernel 逻辑)")
print("-"*80)

tau_flash = nn.Parameter(torch.full((num_heads,), tau_init))

# 简化：每个 query 位置的 p_norm 就是 attn_weights 的一行
# 这里我们取第一个样本，每个头的第一个 query 位置
p_norm = attn_weights[0, :, 0, :seq_len]  # [num_heads, seq_len]
idx_i = torch.arange(1, seq_len + 1, dtype=torch.float32)  # [seq_len]

print(f"p_norm shape: {p_norm.shape}")
print(f"p_norm[0, :5]: {p_norm[0, :5]}")

# 计算梯度
do = torch.ones_like(p_norm)  # 假设上游梯度全为1
dtau_list = []
mask_relu_stats = []

for h in range(num_heads):
    dtau_h, mask_h = elastic_softmax_triton_backward(
        do[h], p_norm[h], tau_flash[h], idx_i, None
    )
    dtau_list.append(dtau_h)
    mask_relu_stats.append(mask_h.sum().item())

dtau_flash = torch.stack(dtau_list)
print(f"\ndtau_flash: {dtau_flash}")
print(f"dtau_flash sum: {dtau_flash.sum().item():.6f}")
print(f"mask_relu 为 True 的比例: {[f'{s}/{seq_len}' for s in mask_relu_stats]}")

# 分析
print("\n" + "="*80)
print("分析")
print("="*80)

print(f"\n1. Scratch 梯度 sum: {tau_scratch.grad.sum().item():.6f}")
print(f"2. Flash 梯度 sum: {dtau_flash.sum().item():.6f}")

if abs(tau_scratch.grad.sum().item()) > 1e-6 and abs(dtau_flash.sum().item()) < 1e-6:
    print("\n❌ 发现问题: Flash 分支梯度为零，但 Scratch 分支有梯度!")
    print("\n可能原因:")
    print(f"  - mask_relu 几乎全为 False (tau={tau_init:.1f} 太负)")
    print(f"  - p_norm + tau/i 几乎全为负数")

    # 检查具体值
    for h in range(num_heads):
        tau_term = tau_init / idx_i
        p_term = p_norm[h] + tau_term
        positive_count = (p_term > 0).sum().item()
        print(f"  - Head {h}: p_term > 0 的数量: {positive_count}/{seq_len}")
        if positive_count > 0:
            print(f"    - p_term max: {p_term.max().item():.6f}")
            print(f"    - p_term[p_term>0]: {p_term[p_term>0]}")

elif abs(tau_scratch.grad.sum().item() - dtau_flash.sum().item()) < 1e-4:
    print("\n✅ 两个分支的梯度基本一致")
else:
    print(f"\n⚠️ 梯度有差异:")
    print(f"  - Scratch: {tau_scratch.grad.sum().item():.6f}")
    print(f"  - Flash: {dtau_flash.sum().item():.6f}")
    print(f"  - 差值: {abs(tau_scratch.grad.sum().item() - dtau_flash.sum().item()):.6f}")

print("="*80)
