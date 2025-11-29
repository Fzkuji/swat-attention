#!/usr/bin/env python3
"""Diagnostic script to check SWAttention parameters in the model"""
import torch
import sys

print("="*80)
print("Model Parameter Diagnostic")
print("="*80)

try:
    from fla.layers.swattn import SWAttention

    # Create a test layer
    layer = SWAttention(
        hidden_size=128,
        num_heads=4,
        max_bias_length=512
    )

    print("\n1. Parameter Registration Check:")
    print("-" * 80)
    param_names = [name for name, _ in layer.named_parameters()]
    print(f"Total parameters: {len(param_names)}")

    has_bias = 'learnable_bias_diagonals' in param_names
    has_tau = 'tau' in param_names

    print(f"  learnable_bias_diagonals: {'✅ Found' if has_bias else '❌ Missing'}")
    print(f"  tau: {'✅ Found' if has_tau else '❌ Missing'}")

    print("\n2. Parameter Properties:")
    print("-" * 80)
    for name, param in layer.named_parameters():
        if 'bias' in name.lower() or 'tau' in name.lower():
            print(f"\n  {name}:")
            print(f"    Shape: {param.shape}")
            print(f"    dtype: {param.dtype}")
            print(f"    requires_grad: {param.requires_grad}")
            print(f"    device: {param.device}")
            print(f"    Initial value (first element): {param.flatten()[0].item():.6f}")

    print("\n3. Quick Training Test:")
    print("-" * 80)

    # Move to GPU if available
    if torch.cuda.is_available():
        layer = layer.cuda()
        device = 'cuda'
    else:
        device = 'cpu'

    # Create optimizer
    optimizer = torch.optim.AdamW(layer.parameters(), lr=0.001)

    # Get initial values
    initial_tau = layer.tau.clone().detach()
    initial_bias = layer.learnable_bias_diagonals.clone().detach()

    print(f"  Initial tau[0]: {initial_tau[0].item():.6f}")
    print(f"  Initial bias[0,0]: {initial_bias[0,0].item():.6f}")

    # Training step
    hidden = torch.randn(1, 16, 128, device=device)
    out, _, _ = layer(hidden)
    loss = out.sum()
    loss.backward()

    print(f"\n  Gradients:")
    print(f"    tau.grad[0]: {layer.tau.grad[0].item():.6f}")
    print(f"    bias.grad[0,0]: {layer.learnable_bias_diagonals.grad[0,0].item():.6f}")

    optimizer.step()

    print(f"\n  After optimizer step:")
    print(f"    tau[0]: {layer.tau[0].item():.6f} (changed: {abs(layer.tau[0].item() - initial_tau[0].item()) > 1e-8})")
    print(f"    bias[0,0]: {layer.learnable_bias_diagonals[0,0].item():.6f} (changed: {abs(layer.learnable_bias_diagonals[0,0].item() - initial_bias[0,0].item()) > 1e-8})")

    print("\n" + "="*80)
    tau_changed = abs(layer.tau[0].item() - initial_tau[0].item()) > 1e-8
    bias_changed = abs(layer.learnable_bias_diagonals[0,0].item() - initial_bias[0,0].item()) > 1e-8

    if tau_changed and bias_changed:
        print("✅ SUCCESS: Both tau and bias parameters can be trained!")
    else:
        print("❌ PROBLEM DETECTED:")
        if not tau_changed:
            print("  - tau parameter did not update")
        if not bias_changed:
            print("  - bias parameter did not update")
    print("="*80)

except Exception as e:
    print(f"\n❌ Error during test: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("\n4. Recommendations for Training:")
print("-" * 80)
print("If parameters are not updating in your actual training:")
print("  1. Check if you're loading a checkpoint that overwrites these parameters")
print("  2. Verify optimizer is created AFTER model initialization")
print("  3. Check if any parameters are frozen with .requires_grad = False")
print("  4. Ensure learning rate is not too small")
print("  5. Run: python test_tau_training.py (in adasplash repo) to verify kernel")
