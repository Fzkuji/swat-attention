#!/usr/bin/env python3
"""检查 mask_relu 的统计信息，看是否因为全为 False 导致梯度为零"""
import torch
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM
import fla  # noqa

print("="*80)
print("检查 mask_relu 统计信息")
print("="*80)

# 加载配置
config = AutoConfig.from_pretrained("legacy/training/configs/swat_340M.json")
model = AutoModelForCausalLM.from_config(config).cuda()

# 创建输入
input_ids = torch.randint(0, config.vocab_size, (2, 64), device='cuda')

# Hook 来捕获 attention layer 的中间值
captured_data = {}

def capture_swattn_forward(module, input, output):
    """捕获 SWAttention 的输入输出"""
    # 这个hook在 layer.attn(hidden_states) 被调用时触发
    # input[0] 是 hidden_states
    captured_data['hidden_states'] = input[0].detach()
    captured_data['output'] = output[0].detach()  # output = (o, attentions, past_key_values)

# 注册hook到第一层的attention
hook_handle = model.model.layers[0].attn.register_forward_hook(capture_swattn_forward)

# 前向传播
print("\n运行前向传播...")
with torch.no_grad():
    outputs = model(input_ids)

hook_handle.remove()

# 手动计算 attention，分析 mask_relu 的统计
print("\n手动模拟 attention 计算...")
attn_layer = model.model.layers[0].attn

# 获取参数
tau = attn_layer.tau
bias = attn_layer.learnable_bias_diagonals

print(f"\ntau: {tau}")
print(f"tau mean: {tau.mean().item():.6f}")
print(f"bias shape: {bias.shape}")
print(f"bias mean: {bias.mean().item():.6f}, std: {bias.std().item():.6f}")

# 模拟一个简化的 attention 计算
batch_size = 2
seq_len = 64
num_heads = config.num_heads
head_dim = config.hidden_size // num_heads

# 创建随机的 Q, K, V (简化，只用来分析 scores)
torch.manual_seed(42)
q = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda') * 0.1
k = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda') * 0.1
v = torch.randn(batch_size, num_heads, seq_len, head_dim, device='cuda')

# 计算 attention scores
sm_scale = 1.0 / (head_dim ** 0.5)
scores = torch.matmul(q, k.transpose(-2, -1)) * sm_scale  # [B, H, L, L]

# 应用 causal mask
causal_mask = torch.tril(torch.ones(seq_len, seq_len, device='cuda'))
scores = scores.masked_fill(causal_mask[None, None, :, :] == 0, float('-inf'))

# 应用 bias (简化版本，只处理主对角线附近)
for i in range(seq_len):
    for j in range(max(0, i - bias.shape[1] + 1), i + 1):
        dist = i - j
        if dist < bias.shape[1]:
            scores[:, :, i, j] += bias[:, dist]

# Softmax
attn_weights = F.softmax(scores, dim=-1, dtype=torch.float32)  # [B, H, L, L]

print(f"\nattn_weights shape: {attn_weights.shape}")
print(f"attn_weights[0, 0, 0, :10]: {attn_weights[0, 0, 0, :10]}")
print(f"attn_weights mean: {attn_weights.mean().item():.6f}")

# 计算 p_norm + tau / i，检查有多少是正数
tau_expanded = tau.view(1, num_heads, 1, 1)  # [1, H, 1, 1]
i_positions = torch.arange(1, seq_len + 1, device='cuda', dtype=torch.float32).view(1, 1, seq_len, 1)  # [1, 1, L, 1]

# 计算 p_term (对应 Triton kernel 中的 p_norm + tau_term)
# 注意：这里的 attn_weights 对应 p_norm
p_term = attn_weights + tau_expanded / i_positions  # [B, H, L, L]

# 统计 mask_relu (p_term > 0) 的比例
mask_relu = p_term > 0
positive_ratio = mask_relu.float().mean().item()

print(f"\n" + "-"*80)
print("Elastic-Softmax 分析:")
print("-"*80)
print(f"tau / i 的范围:")
print(f"  - tau[0] / 1 = {tau[0].item() / 1:.6f}")
print(f"  - tau[0] / {seq_len} = {tau[0].item() / seq_len:.6f}")

print(f"\np_norm (attn_weights after softmax) 的范围:")
print(f"  - min: {attn_weights.min().item():.6f}")
print(f"  - max: {attn_weights.max().item():.6f}")
print(f"  - mean: {attn_weights.mean().item():.6f}")

print(f"\np_term = p_norm + tau/i 的范围:")
print(f"  - min: {p_term.min().item():.6f}")
print(f"  - max: {p_term.max().item():.6f}")
print(f"  - mean: {p_term.mean().item():.6f}")

print(f"\nmask_relu (p_term > 0) 统计:")
print(f"  - 正数比例: {positive_ratio * 100:.2f}%")
print(f"  - 正数数量: {mask_relu.sum().item()} / {mask_relu.numel()}")

# 分析每个位置的情况
for i in [0, 10, 30, 63]:  # 采样几个位置
    p_term_at_i = p_term[0, 0, i, :i+1]  # 第一个样本，第一个头，位置i
    positive_count = (p_term_at_i > 0).sum().item()
    total = i + 1
    print(f"  - 位置 {i+1}: {positive_count}/{total} = {positive_count/total*100:.1f}% 为正")

print("\n" + "="*80)
print("结论:")
print("="*80)

if positive_ratio < 0.01:
    print("❌ 发现问题: mask_relu 几乎全为 False (<1%)!")
    print(f"   - tau={tau[0].item():.2f} 太负，导致 p_term = p_norm + tau/i 几乎全为负")
    print(f"   - 反向传播时，term_tau = where(mask_relu, dp_elastic * (1/i), 0)")
    print(f"   - 因为 mask_relu 几乎全 False，所以 dtau ≈ 0")
    print(f"\n解决方案:")
    print(f"   1. 修改 tau 的初始化值（例如从 -1.0 改为 0.0 或更小的负数）")
    print(f"   2. 或者修改梯度计算逻辑，不用 mask_relu 屏蔽")
elif positive_ratio > 0.5:
    print("✅ mask_relu 的正数比例正常 (>50%)")
    print("   问题可能在其他地方")
else:
    print(f"⚠️ mask_relu 正数比例较低 ({positive_ratio*100:.1f}%)")
    print("   可能导致梯度偏小，但不至于完全为零")

print("="*80)
