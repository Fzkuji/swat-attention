#!/usr/bin/env python3
"""Test if FusedCrossEntropyLoss breaks gradients"""
import torch
import torch.nn as nn
from fla.modules import FusedCrossEntropyLoss

print("="*80)
print("Testing FusedCrossEntropyLoss Gradient Flow")
print("="*80)

# Simple test: hidden -> linear -> logits -> loss
hidden_size = 128
vocab_size = 32000
batch_size = 2
seq_len = 16

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Create fake attention layer with learnable parameters
class FakeAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.randn(16, 512))
        self.tau = nn.Parameter(torch.full((16,), -1.0))
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        # Ensure bias and tau are used in forward
        x = x + self.bias.mean() * 0.0  # Add without changing value
        x = x + self.tau.mean() * 0.0
        return self.proj(x)

# Create model
attn = FakeAttention().to(device)
lm_head = nn.Linear(hidden_size, vocab_size, bias=False).to(device)

print(f"\nInitial values:")
print(f"  attn.bias[0, 0] = {attn.bias[0, 0].item():.6f}")
print(f"  attn.tau[0] = {attn.tau[0].item():.6f}")

# Forward pass
hidden = torch.randn(batch_size, seq_len, hidden_size, device=device, requires_grad=True)
hidden = attn(hidden)
logits = lm_head(hidden)
labels = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)

# Test with FusedCrossEntropyLoss
print("\n" + "-"*80)
print("Testing with FusedCrossEntropyLoss(inplace_backward=True)")
print("-"*80)

criterion = FusedCrossEntropyLoss(inplace_backward=True)
loss = criterion(logits.view(-1, vocab_size), labels.view(-1))

print(f"Loss: {loss.item():.4f}")

# Backward
loss.backward()

print(f"\nGradients:")
print(f"  attn.bias.grad[0, 0] = {attn.bias.grad[0, 0].item() if attn.bias.grad is not None else None}")
print(f"  attn.tau.grad[0] = {attn.tau.grad[0].item() if attn.tau.grad is not None else None}")
print(f"  lm_head.weight.grad sum = {lm_head.weight.grad.sum().item() if lm_head.weight.grad is not None else None}")

# Check if gradients exist
print("\n" + "="*80)
if attn.bias.grad is not None and attn.tau.grad is not None and \
   attn.bias.grad.abs().sum() > 0 and attn.tau.grad.abs().sum() > 0:
    print("✅ Gradients flow correctly through FusedCrossEntropyLoss!")
else:
    print("❌ Gradients are blocked!")
    if attn.bias.grad is None:
        print("  - attn.bias.grad is None")
    elif attn.bias.grad.abs().sum() == 0:
        print("  - attn.bias.grad exists but all zeros")

    if attn.tau.grad is None:
        print("  - attn.tau.grad is None")
    elif attn.tau.grad.abs().sum() == 0:
        print("  - attn.tau.grad exists but all zeros")
print("="*80)

# Compare with standard CrossEntropyLoss
print("\n" + "-"*80)
print("Testing with standard nn.CrossEntropyLoss for comparison")
print("-"*80)

# Reset
attn.zero_grad()
lm_head.zero_grad()

# Forward again
hidden2 = torch.randn(batch_size, seq_len, hidden_size, device=device, requires_grad=True)
hidden2 = attn(hidden2)
logits2 = lm_head(hidden2)

criterion2 = nn.CrossEntropyLoss()
loss2 = criterion2(logits2.view(-1, vocab_size), labels.view(-1))

print(f"Loss: {loss2.item():.4f}")

# Backward
loss2.backward()

print(f"\nGradients:")
print(f"  attn.bias.grad[0, 0] = {attn.bias.grad[0, 0].item() if attn.bias.grad is not None else None}")
print(f"  attn.tau.grad[0] = {attn.tau.grad[0].item() if attn.tau.grad is not None else None}")

print("\n" + "="*80)
if attn.bias.grad is not None and attn.tau.grad is not None and \
   attn.bias.grad.abs().sum() > 0 and attn.tau.grad.abs().sum() > 0:
    print("✅ Gradients flow correctly with standard CrossEntropyLoss!")
else:
    print("❌ Gradients blocked with standard loss too!")
print("="*80)
