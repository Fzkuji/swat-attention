#!/usr/bin/env python3
"""Test if attention parameters receive gradients in full model"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM
import fla  # noqa

print("="*80)
print("Testing Attention Parameter Gradients in Full Model")
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

if first_attn_bias is None or first_attn_tau is None:
    print("❌ Could not find attention parameters!")
    exit(1)

# Hook to check if parameters are used in forward
bias_used = [False]
tau_used = [False]

def hook_fn(grad):
    return grad  # Just check if hook is called

first_attn_bias.register_hook(lambda grad: print(f"Bias grad hook called! grad sum={grad.sum().item():.6f}") or grad)
first_attn_tau.register_hook(lambda grad: print(f"Tau grad hook called! grad sum={grad.sum().item():.6f}") or grad)

print("\n" + "-"*80)
print("Testing Forward + Backward")
print("-"*80)

model = model.cuda()

# Create input
input_ids = torch.randint(0, config.vocab_size, (1, 64), device='cuda')

print(f"\nInitial values:")
print(f"  bias[0, 0] = {first_attn_bias[0, 0].item():.6f}")
print(f"  tau[0] = {first_attn_tau[0].item():.6f}")

# Forward pass WITHOUT labels (just get logits)
print("\n1. Forward without labels...")
with torch.no_grad():
    outputs_no_labels = model(input_ids)
    print(f"  Logits shape: {outputs_no_labels.logits.shape}")

# Forward pass WITH labels (compute loss)
print("\n2. Forward with labels (compute loss)...")
labels = input_ids.clone()
outputs = model(input_ids, labels=labels)

print(f"  Loss: {outputs.loss.item():.4f}")

# Backward
print("\n3. Running backward...")
outputs.loss.backward()

print(f"\n4. Checking gradients:")
print(f"  bias.grad exists: {first_attn_bias.grad is not None}")
print(f"  tau.grad exists: {first_attn_tau.grad is not None}")

if first_attn_bias.grad is not None:
    print(f"  bias.grad[0, 0] = {first_attn_bias.grad[0, 0].item():.6f}")
    print(f"  bias.grad sum = {first_attn_bias.grad.sum().item():.6f}")
    print(f"  bias.grad nonzero = {(first_attn_bias.grad != 0).sum().item()}")
else:
    print(f"  bias.grad = None")

if first_attn_tau.grad is not None:
    print(f"  tau.grad[0] = {first_attn_tau.grad[0].item():.6f}")
    print(f"  tau.grad sum = {first_attn_tau.grad.sum().item():.6f}")
    print(f"  tau.grad nonzero = {(first_attn_tau.grad != 0).sum().item()}")
else:
    print(f"  tau.grad = None")

# Final verdict
print("\n" + "="*80)
bias_has_grad = first_attn_bias.grad is not None and first_attn_bias.grad.abs().sum() > 1e-10
tau_has_grad = first_attn_tau.grad is not None and first_attn_tau.grad.abs().sum() > 1e-10

if bias_has_grad and tau_has_grad:
    print("✅ SUCCESS: Both bias and tau receive gradients!")
else:
    print("❌ PROBLEM: Gradients are ZERO or None!")
    if not bias_has_grad:
        print("  - bias gradient is zero or None")
    if not tau_has_grad:
        print("  - tau gradient is zero or None")
    print("\nThis confirms the bug: attention parameters don't receive gradients!")
    print("Possible reasons:")
    print("  1. lazy_attention_triton is not being called")
    print("  2. lazy_attention_triton output is detached")
    print("  3. FusedCrossEntropyLoss breaks the gradient flow")
print("="*80)
