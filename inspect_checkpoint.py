#!/usr/bin/env python3
"""Inspect a training checkpoint to check tau and bias values"""
import sys
import torch
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python inspect_checkpoint.py <checkpoint_path>")
    print("Example: python inspect_checkpoint.py exp/swat-340M-10B-512-test/checkpoint-XXX")
    sys.exit(1)

checkpoint_path = Path(sys.argv[1])

print("="*80)
print(f"Inspecting checkpoint: {checkpoint_path}")
print("="*80)

# Try to load model state dict
model_file = checkpoint_path / "model.safetensors"
if not model_file.exists():
    model_file = checkpoint_path / "pytorch_model.bin"

if not model_file.exists():
    print(f"❌ Error: Could not find model file in {checkpoint_path}")
    sys.exit(1)

print(f"\nLoading from: {model_file}")

if model_file.suffix == '.safetensors':
    from safetensors import safe_open
    tensors = {}
    with safe_open(model_file, framework="pt", device="cpu") as f:
        for key in f.keys():
            tensors[key] = f.get_tensor(key)
else:
    state_dict = torch.load(model_file, map_location='cpu')
    if 'model' in state_dict:
        tensors = state_dict['model']
    elif 'state_dict' in state_dict:
        tensors = state_dict['state_dict']
    else:
        tensors = state_dict

print(f"\nTotal parameters: {len(tensors)}")

# Find tau and bias parameters
print("\n" + "="*80)
print("Searching for tau and bias parameters:")
print("="*80)

tau_keys = [k for k in tensors.keys() if 'tau' in k.lower()]
bias_keys = [k for k in tensors.keys() if 'learnable_bias' in k.lower() or 'bias_diagonal' in k.lower()]

if tau_keys:
    print(f"\n✅ Found {len(tau_keys)} tau parameter(s):")
    for key in tau_keys:
        param = tensors[key]
        print(f"\n  {key}")
        print(f"    Shape: {param.shape}")
        print(f"    dtype: {param.dtype}")
        print(f"    Values: {param}")

        # Check if all values are -1.0
        if torch.all(param == -1.0):
            print(f"    ⚠️  WARNING: All values are exactly -1.0 (initial value)")
        else:
            print(f"    ✅ Values have changed from initial -1.0")
else:
    print("\n❌ No tau parameters found!")

if bias_keys:
    print(f"\n✅ Found {len(bias_keys)} bias parameter(s):")
    for key in bias_keys:
        param = tensors[key]
        print(f"\n  {key}")
        print(f"    Shape: {param.shape}")
        print(f"    dtype: {param.dtype}")
        print(f"    First row (first 10 values): {param[0, :10]}")
        print(f"    Mean: {param.mean().item():.6f}, Std: {param.std().item():.6f}")

        # Check if close to initial (zeros with small noise)
        if abs(param.mean().item()) < 1e-3 and param.std().item() < 0.01:
            print(f"    ⚠️  WARNING: Values are close to initial (mean≈0, std≈1e-3)")
        else:
            print(f"    ✅ Values have changed from initialization")
else:
    print("\n❌ No learnable_bias parameters found!")

# Summary
print("\n" + "="*80)
print("Summary:")
print("="*80)

if not tau_keys and not bias_keys:
    print("❌ PROBLEM: No tau or bias parameters found in checkpoint!")
    print("   This might mean:")
    print("   1. The model was trained on a different branch (scratch)")
    print("   2. Parameters were not registered correctly")
    print("   3. Checkpoint is from an incompatible model version")
else:
    tau_trained = False
    bias_trained = False

    if tau_keys:
        tau_param = tensors[tau_keys[0]]
        tau_trained = not torch.all(tau_param == -1.0)

    if bias_keys:
        bias_param = tensors[bias_keys[0]]
        bias_trained = not (abs(bias_param.mean().item()) < 1e-3 and bias_param.std().item() < 0.01)

    if tau_trained and bias_trained:
        print("✅ Both tau and bias appear to have been trained")
    else:
        print("❌ Parameters appear NOT to have been trained:")
        if not tau_trained:
            print("   - tau is still at initial value (-1.0)")
        if not bias_trained:
            print("   - bias is still close to initial value (zeros)")

print("="*80)
