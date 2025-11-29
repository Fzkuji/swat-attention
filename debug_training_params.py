#!/usr/bin/env python3
"""Debug script to monitor parameter updates during training"""
import torch
from transformers import AutoConfig, AutoModelForCausalLM
import fla  # noqa

# Load config
config_path = "legacy/training/configs/swat_340M.json"
config = AutoConfig.from_pretrained(config_path)

print("="*80)
print(f"Config loaded from: {config_path}")
print("="*80)
print(f"  model_type: {config.model_type}")
print(f"  num_heads: {config.num_heads}")
print(f"  hidden_size: {config.hidden_size}")
print(f"  max_bias_length: {config.max_bias_length}")

# Initialize model from config (same as training script with from_config=True)
print("\nInitializing model from config...")
model = AutoModelForCausalLM.from_config(config)

# Check parameters
print("\n" + "="*80)
print("Checking tau and bias parameters:")
print("="*80)

found_tau = False
found_bias = False

for name, param in model.named_parameters():
    if 'tau' in name.lower():
        found_tau = True
        print(f"\n✅ Found: {name}")
        print(f"   Shape: {param.shape}")
        print(f"   dtype: {param.dtype}")
        print(f"   requires_grad: {param.requires_grad}")
        print(f"   Values: {param.data}")

    if 'learnable_bias' in name.lower() or 'bias_diagonal' in name.lower():
        found_bias = True
        print(f"\n✅ Found: {name}")
        print(f"   Shape: {param.shape}")
        print(f"   dtype: {param.dtype}")
        print(f"   requires_grad: {param.requires_grad}")
        print(f"   First values: {param.data[0, :5]}")

if not found_tau:
    print("\n❌ ERROR: No tau parameter found in model!")
if not found_bias:
    print("\n❌ ERROR: No learnable_bias parameter found in model!")

print("\n" + "="*80)
print("Simulating training step...")
print("="*80)

# Get initial values
tau_params = [p for n, p in model.named_parameters() if 'tau' in n.lower()]
bias_params = [p for n, p in model.named_parameters() if 'learnable_bias' in n.lower()]

if tau_params and bias_params:
    tau_param = tau_params[0]
    bias_param = bias_params[0]

    initial_tau = tau_param.data.clone()
    initial_bias = bias_param.data.clone()

    print(f"Initial tau values: {initial_tau}")
    print(f"Initial bias[0, :5]: {initial_bias[0, :5]}")

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)

    # Simple forward pass
    model = model.cuda()
    input_ids = torch.randint(0, config.vocab_size, (1, 512), device='cuda')

    print("\nRunning forward + backward...")
    outputs = model(input_ids, labels=input_ids)
    loss = outputs.loss
    loss.backward()

    print(f"Loss: {loss.item():.4f}")
    print(f"tau.grad: {tau_param.grad}")
    print(f"bias.grad[0, :5]: {bias_param.grad[0, :5]}")

    # Optimizer step
    optimizer.step()
    optimizer.zero_grad()

    print(f"\nAfter optimizer step:")
    print(f"tau values: {tau_param.data}")
    print(f"bias[0, :5]: {bias_param.data[0, :5]}")

    tau_changed = not torch.allclose(tau_param.data, initial_tau, atol=1e-8)
    bias_changed = not torch.allclose(bias_param.data[0, :5], initial_bias[0, :5], atol=1e-8)

    print("\n" + "="*80)
    if tau_changed and bias_changed:
        print("✅ SUCCESS: Parameters updated correctly!")
    else:
        print("❌ PROBLEM:")
        if not tau_changed:
            print("  - tau did not change")
        if not bias_changed:
            print("  - bias did not change")
    print("="*80)
else:
    print("\n❌ Could not find tau or bias parameters for testing")
