#!/usr/bin/env python3
"""Test if the Triton kernel backward is being called at all"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM
import fla  # noqa

# Monkey-patch to detect if backward is called
from adasplash import lazy_attention_triton
original_backward = lazy_attention_triton.LazyAttentionTritonFunc.backward

backward_called = [False]

def patched_backward(ctx, do):
    backward_called[0] = True
    print("🔍 LazyAttentionTritonFunc.backward() WAS CALLED!")
    return original_backward(ctx, do)

lazy_attention_triton.LazyAttentionTritonFunc.backward = staticmethod(patched_backward)

print("="*80)
print("Testing if Triton Kernel Backward is Called")
print("="*80)

# Load config and create model
config = AutoConfig.from_pretrained("legacy/training/configs/swat_340M.json")
model = AutoModelForCausalLM.from_config(config)

# Find first attention layer's parameters
first_attn_bias = None
first_attn_tau = None

for name, param in model.named_parameters():
    if 'layers.0.attn.learnable_bias' in name:
        first_attn_bias = param
        print(f"Found bias: {name}, shape={param.shape}")
    if 'layers.0.attn.tau' in name:
        first_attn_tau = param
        print(f"Found tau: {name}, shape={param.shape}")
    if first_attn_bias is not None and first_attn_tau is not None:
        break

print("\n" + "-"*80)
print("Running forward + backward")
print("-"*80)

model = model.cuda()

# Create input
input_ids = torch.randint(0, config.vocab_size, (2, 64), device='cuda')
labels = input_ids.clone()

# Forward
print("\n1. Forward pass...")
outputs = model(input_ids, labels=labels)
print(f"   Loss: {outputs.loss.item():.4f}")

# Backward
print("\n2. Backward pass...")
outputs.loss.backward()

# Check if backward was called
print("\n" + "="*80)
if backward_called[0]:
    print("✅ Triton kernel backward WAS called!")
    print("\nBut gradients are still zero, which means:")
    print("  - The backward pass runs")
    print("  - But computed gradients might be zero")
    print("  - OR gradients are not being propagated correctly")
else:
    print("❌ Triton kernel backward was NEVER called!")
    print("\nThis means:")
    print("  - The attention output is likely detached")
    print("  - OR the computation graph is broken")
    print("  - OR autograd skips the backward for some reason")

# Print gradients
print(f"\ntau.grad: {first_attn_tau.grad}")
print(f"bias.grad[0, :5]: {first_attn_bias.grad[0, :5] if first_attn_bias.grad is not None else None}")
print("="*80)
