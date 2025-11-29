#!/usr/bin/env python3
"""详细诊断梯度流问题"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM
import fla  # noqa

# 添加钩子来跟踪梯度
from adasplash import lazy_attention_triton

backward_info = {
    'called': False,
    'dbias_sum': None,
    'dtau_sum': None,
}

original_backward = lazy_attention_triton.LazyAttentionTritonFunc.backward

def patched_backward(ctx, do):
    backward_info['called'] = True
    print("🔍 LazyAttentionTritonFunc.backward() 被调用!")
    result = original_backward(ctx, do)
    # result = (dq, dk, dv, dbias, dtau, None, None)
    dbias = result[3]
    dtau = result[4]
    if dbias is not None:
        backward_info['dbias_sum'] = dbias.abs().sum().item()
        print(f"   dbias.abs().sum() = {backward_info['dbias_sum']:.6f}")
    if dtau is not None:
        backward_info['dtau_sum'] = dtau.abs().sum().item()
        print(f"   dtau.abs().sum() = {backward_info['dtau_sum']:.6f}")
    return result

lazy_attention_triton.LazyAttentionTritonFunc.backward = staticmethod(patched_backward)

print("="*80)
print("详细梯度流诊断")
print("="*80)

# 加载配置
config = AutoConfig.from_pretrained("legacy/training/configs/swat_340M.json")
print(f"\n配置: fuse_cross_entropy={config.fuse_cross_entropy}, fuse_norm={config.fuse_norm}")

model = AutoModelForCausalLM.from_config(config)

# 找到第一层的attention参数
first_attn_bias = None
first_attn_tau = None

for name, param in model.named_parameters():
    if 'layers.0.attn.learnable_bias' in name:
        first_attn_bias = param
        print(f"找到 bias: {name}, shape={param.shape}, dtype={param.dtype}")
    if 'layers.0.attn.tau' in name:
        first_attn_tau = param
        print(f"找到 tau: {name}, shape={param.shape}, dtype={param.dtype}")
    if first_attn_bias is not None and first_attn_tau is not None:
        break

# 添加 hook 到参数上
def make_hook(name):
    def hook(grad):
        if grad is not None:
            print(f"   {name}.grad hook: sum={grad.sum().item():.6f}, abs_sum={grad.abs().sum().item():.6f}")
        else:
            print(f"   {name}.grad hook: grad is None!")
        return grad
    return hook

first_attn_bias.register_hook(make_hook("bias"))
first_attn_tau.register_hook(make_hook("tau"))

print("\n" + "-"*80)
print("前向+反向传播")
print("-"*80)

model = model.cuda()
input_ids = torch.randint(0, config.vocab_size, (2, 64), device='cuda')
labels = input_ids.clone()

print(f"\n初始值:")
print(f"  bias[0, 0] = {first_attn_bias[0, 0].item():.6f}")
print(f"  tau[0] = {first_attn_tau[0].item():.6f}")
print(f"  bias.requires_grad = {first_attn_bias.requires_grad}")
print(f"  tau.requires_grad = {first_attn_tau.requires_grad}")

# 前向传播
print("\n1. 前向传播...")
outputs = model(input_ids, labels=labels)
print(f"   Loss: {outputs.loss.item():.4f}")

# 反向传播
print("\n2. 反向传播...")
outputs.loss.backward()

# 检查结果
print("\n" + "="*80)
print("诊断结果:")
print("="*80)

print(f"\n1. Triton kernel backward 是否被调用: {backward_info['called']}")
if backward_info['called']:
    print(f"   - kernel 计算的 dbias sum: {backward_info['dbias_sum']:.6f}")
    print(f"   - kernel 计算的 dtau sum: {backward_info['dtau_sum']:.6f}")

print(f"\n2. 参数梯度:")
if first_attn_bias.grad is not None:
    print(f"   - bias.grad sum: {first_attn_bias.grad.sum().item():.6f}")
    print(f"   - bias.grad abs sum: {first_attn_bias.grad.abs().sum().item():.6f}")
    print(f"   - bias.grad 非零元素: {(first_attn_bias.grad != 0).sum().item()}/{first_attn_bias.grad.numel()}")
else:
    print(f"   - bias.grad: None")

if first_attn_tau.grad is not None:
    print(f"   - tau.grad sum: {first_attn_tau.grad.sum().item():.6f}")
    print(f"   - tau.grad abs sum: {first_attn_tau.grad.abs().sum().item():.6f}")
    print(f"   - tau.grad 非零元素: {(first_attn_tau.grad != 0).sum().item()}/{first_attn_tau.grad.numel()}")
else:
    print(f"   - tau.grad: None")

# 结论
print("\n" + "="*80)
print("结论:")
print("="*80)

if not backward_info['called']:
    print("❌ Triton kernel backward 没有被调用!")
    print("   原因: attention output 可能被 detach 或计算图断裂")
elif backward_info['dbias_sum'] == 0 and backward_info['dtau_sum'] == 0:
    print("❌ Kernel backward 被调用了，但计算的梯度为零!")
    print("   原因: ")
    print("   1. 可能 do (输出梯度) 为零")
    print("   2. 可能 kernel 的梯度计算逻辑有bug")
    print("   3. 可能所有的 mask_relu 都是 False (p_term <= 0)")
elif first_attn_bias.grad is None or first_attn_tau.grad is None:
    print("❌ Kernel 计算了非零梯度，但参数梯度为 None!")
    print("   原因: 梯度没有正确传播到参数")
elif first_attn_bias.grad.abs().sum() == 0 and first_attn_tau.grad.abs().sum() == 0:
    print("❌ Kernel 计算了非零梯度，但参数梯度为零!")
    print("   原因: 梯度在传播过程中变成了零")
else:
    print("✅ 梯度正常!")
    print(f"   bias 梯度: {first_attn_bias.grad.abs().sum().item():.6f}")
    print(f"   tau 梯度: {first_attn_tau.grad.abs().sum().item():.6f}")

print("="*80)
