#!/usr/bin/env python3
"""监控训练过程中参数的变化"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM
import fla  # noqa

print("="*80)
print("监控参数变化")
print("="*80)

# 加载模型
config = AutoConfig.from_pretrained("legacy/training/configs/swat_340M.json")
model = AutoModelForCausalLM.from_config(config).cuda()

# 打印初始参数
print("\n初始参数值:")
for i in [0, 11, 23]:  # 采样几层
    tau = model.model.layers[i].attn.tau
    bias = model.model.layers[i].attn.learnable_bias_diagonals
    print(f"Layer {i}:")
    print(f"  tau: {tau.data}")
    print(f"  bias mean: {bias.data.mean().item():.6f}, std: {bias.data.std().item():.6f}")

# 模拟一个训练步骤
print("\n" + "-"*80)
print("执行一个训练步骤...")
print("-"*80)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
input_ids = torch.randint(0, config.vocab_size, (2, 128), device='cuda')
labels = input_ids.clone()

# 前向
outputs = model(input_ids, labels=labels)
loss = outputs.loss

print(f"Loss: {loss.item():.4f}")

# 反向
loss.backward()

# 检查梯度
print("\n梯度统计:")
for i in [0, 11, 23]:
    tau = model.model.layers[i].attn.tau
    bias = model.model.layers[i].attn.learnable_bias_diagonals

    tau_grad_sum = tau.grad.sum().item() if tau.grad is not None else 0.0
    bias_grad_sum = bias.grad.abs().sum().item() if bias.grad is not None else 0.0

    print(f"Layer {i}:")
    print(f"  tau.grad sum: {tau_grad_sum:.10f}")
    print(f"  bias.grad abs sum: {bias_grad_sum:.10f}")

# 优化器步骤
optimizer.step()

# 检查更新后的参数
print("\n更新后的参数值:")
for i in [0, 11, 23]:
    tau = model.model.layers[i].attn.tau
    bias = model.model.layers[i].attn.learnable_bias_diagonals
    print(f"Layer {i}:")
    print(f"  tau: {tau.data}")
    print(f"  bias mean: {bias.data.mean().item():.6f}, std: {bias.data.std().item():.6f}")

# 分析变化
print("\n" + "="*80)
print("结论:")
print("="*80)

all_tau_changed = True
all_bias_changed = True

for i in range(config.num_hidden_layers):
    tau = model.model.layers[i].attn.tau
    bias = model.model.layers[i].attn.learnable_bias_diagonals

    # 检查是否所有 tau 还是 -1.0
    if torch.allclose(tau.data, torch.full_like(tau.data, -1.0), atol=1e-6):
        all_tau_changed = False
        break

    # 检查 bias 是否变化
    if bias.data.abs().sum() < 1e-6:
        all_bias_changed = False
        break

if all_tau_changed:
    print("✅ Tau 参数已经开始更新!")
else:
    print("❌ Tau 参数还是 -1.0，没有更新")

if all_bias_changed:
    print("✅ Bias 参数已经开始更新!")
else:
    print("❌ Bias 参数没有更新")

print("="*80)
