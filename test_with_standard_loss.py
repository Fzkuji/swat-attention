#!/usr/bin/env python3
"""Test if standard CrossEntropyLoss fixes the gradient issue"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM
import fla  # noqa

print("="*80)
print("Testing with STANDARD CrossEntropyLoss (fuse_cross_entropy=False)")
print("="*80)

# Load config and DISABLE fused cross entropy
config = AutoConfig.from_pretrained("legacy/training/configs/swat_340M.json")
print(f"\nOriginal config.fuse_cross_entropy: {config.fuse_cross_entropy}")

# DISABLE fused cross entropy
config.fuse_cross_entropy = False
print(f"Modified config.fuse_cross_entropy: {config.fuse_cross_entropy}")

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

if first_attn_bias is None or first_attn_tau is None:
    print("❌ Could not find attention parameters!")
    exit(1)

print("\n" + "-"*80)
print("Running forward + backward with STANDARD CrossEntropyLoss")
print("-"*80)

model = model.cuda()

# Create input
input_ids = torch.randint(0, config.vocab_size, (2, 64), device='cuda')
labels = input_ids.clone()

print(f"\nInitial values:")
print(f"  bias[0, 0] = {first_attn_bias[0, 0].item():.6f}")
print(f"  tau[0] = {first_attn_tau[0].item():.6f}")

# Forward with labels (compute loss)
print("\nForward with labels...")
outputs = model(input_ids, labels=labels)

print(f"Loss: {outputs.loss.item():.4f}")

# Backward
print("Running backward...")
outputs.loss.backward()

print(f"\nChecking gradients:")
print(f"  tau.grad: {first_attn_tau.grad}")
print(f"  bias.grad[0, :5]: {first_attn_bias.grad[0, :5]}")

# Final verdict
print("\n" + "="*80)
bias_has_grad = first_attn_bias.grad is not None and first_attn_bias.grad.abs().sum() > 1e-10
tau_has_grad = first_attn_tau.grad is not None and first_attn_tau.grad.abs().sum() > 1e-10

if bias_has_grad and tau_has_grad:
    print("✅ SUCCESS with standard CrossEntropyLoss!")
    print("   This confirms FusedCrossEntropyLoss is BREAKING gradients!")
else:
    print("❌ Still zero gradients even with standard loss!")
    print("   The problem is elsewhere (not the loss function)")
print("="*80)
