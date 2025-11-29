#!/usr/bin/env python3
"""用 PyTorch 模拟 Triton kernel 的逻辑，逐步验证"""
import torch
import torch.nn.functional as F

print("="*80)
print("模拟 Triton Kernel 逻辑")
print("="*80)

torch.manual_seed(42)

B, H, L, D = 1, 1, 32, 32
BLOCK_M = 64
BLOCK_N = 64
device = 'cuda'
tau_val = -1.0

# 创建输入
q = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
k = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16) * 0.02
v = torch.randn(B, H, L, D, device=device, dtype=torch.bfloat16)
tau = torch.tensor(tau_val, dtype=torch.bfloat16, device=device)
bias = torch.zeros(H, 512, device=device, dtype=torch.bfloat16)

print(f"输入: B={B}, H={H}, L={L}, D={D}")
print(f"  BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}")
print(f"  tau={tau.item()}")

# ============================================================================
# 步骤 1: 计算 LSE (模拟 _get_lse_kernel_batch)
# ============================================================================
print("\n" + "-"*80)
print("步骤 1: 计算 LSE")
print("-"*80)

lse = torch.empty(B, H, L, dtype=torch.float32, device=device)

for b in range(B):
    for h in range(H):
        for m_block_idx in range((L + BLOCK_M - 1) // BLOCK_M):
            offs_m = torch.arange(BLOCK_M, device=device) + m_block_idx * BLOCK_M
            offs_m = offs_m[offs_m < L]  # 只取有效的

            if len(offs_m) == 0:
                continue

            q_block = q[b, h, offs_m, :]  # [len(offs_m), D]

            m_i = torch.full((len(offs_m),), float('-inf'), dtype=torch.float32, device=device)
            l_i = torch.zeros(len(offs_m), dtype=torch.float32, device=device)

            # Causal: 只处理 <= offs_m 的部分
            n_end = (offs_m.max().item() + 1)

            for n_start in range(0, n_end, BLOCK_N):
                offs_n = torch.arange(BLOCK_N, device=device) + n_start
                offs_n = offs_n[offs_n < L]  # 只取有效的

                if len(offs_n) == 0:
                    continue

                k_block = k[b, h, offs_n, :]  # [len(offs_n), D]

                # QK^T
                sm_scale = 1.0 / (D ** 0.5)
                s = torch.matmul(q_block, k_block.t()) * sm_scale  # [len(offs_m), len(offs_n)]

                # Causal mask
                dist = offs_m[:, None] - offs_n[None, :]
                s = torch.where(dist >= 0, s.float(), torch.tensor(float('-inf'), device=device))

                # Online softmax (numerical stability)
                m_block = s.max(dim=1)[0]  # [len(offs_m)]
                new_m_i = torch.maximum(m_i, m_block)
                alpha = torch.exp(m_i - new_m_i)
                l_i = l_i * alpha + torch.exp(s - new_m_i[:, None]).sum(dim=1)
                m_i = new_m_i

            # Final LSE
            lse[b, h, offs_m] = m_i + torch.log(l_i)

print(f"LSE shape: {lse.shape}")
print(f"LSE[0, 0, :5]: {lse[0, 0, :5]}")

# ============================================================================
# 步骤 2: 前向传播 (模拟 _lazy_fwd_kernel_batch)
# ============================================================================
print("\n" + "-"*80)
print("步骤 2: 前向传播")
print("-"*80)

out = torch.zeros(B, H, L, D, dtype=torch.bfloat16, device=device)

for b in range(B):
    for h in range(H):
        for m_block_idx in range((L + BLOCK_M - 1) // BLOCK_M):
            offs_m = torch.arange(BLOCK_M, device=device) + m_block_idx * BLOCK_M
            offs_m = offs_m[offs_m < L]

            if len(offs_m) == 0:
                continue

            q_block = q[b, h, offs_m, :]
            lse_block = lse[b, h, offs_m]
            idx_i = offs_m + 1  # Position indices
            tau_term = tau / idx_i.float()  # [len(offs_m)]

            acc = torch.zeros(len(offs_m), D, dtype=torch.float32, device=device)

            n_end = (offs_m.max().item() + 1)

            for n_start in range(0, n_end, BLOCK_N):
                offs_n = torch.arange(BLOCK_N, device=device) + n_start
                offs_n = offs_n[offs_n < L]

                if len(offs_n) == 0:
                    continue

                k_block = k[b, h, offs_n, :]
                v_block = v[b, h, offs_n, :]

                # QK^T
                sm_scale = 1.0 / (D ** 0.5)
                s = torch.matmul(q_block, k_block.t()) * sm_scale

                # Causal mask
                dist = offs_m[:, None] - offs_n[None, :]
                s = torch.where(dist >= 0, s.float(), torch.tensor(float('-inf'), device=device))

                # p_norm = exp(s - lse)
                p_norm = torch.exp(s - lse_block[:, None])  # [len(offs_m), len(offs_n)]

                # Elastic-Softmax: p_elastic = ReLU(p_norm + tau/i)
                p_term = p_norm + tau_term[:, None]  # [len(offs_m), len(offs_n)]
                p_elastic = torch.clamp(p_term, min=0.0)

                # Accumulate
                acc += torch.matmul(p_elastic.to(v.dtype), v_block)

            out[b, h, offs_m, :] = acc.to(torch.bfloat16)

print(f"Out shape: {out.shape}")
print(f"Out[0, 0, 0, :5]: {out[0, 0, 0, :5]}")
print(f"Out mean: {out.mean().item():.10f}")
print(f"Out std: {out.std().item():.10f}")

# ============================================================================
# 对比 Triton
# ============================================================================
print("\n" + "="*80)
print("对比")
print("="*80)

from adasplash import lazy_attention_triton

tau_triton = torch.nn.Parameter(torch.full((H,), tau_val, device=device, dtype=torch.bfloat16))
bias_triton = torch.nn.Parameter(torch.zeros(H, 512, device=device, dtype=torch.bfloat16))

out_triton = lazy_attention_triton(q, k, v, bias_triton, tau_triton)

diff = (out - out_triton).abs()
print(f"Simulated vs Triton:")
print(f"  最大差异: {diff.max().item():.10f}")
print(f"  平均差异: {diff.mean().item():.10f}")
print(f"  Triton mean: {out_triton.mean().item():.10f}")
print(f"  Simulated mean: {out.mean().item():.10f}")

if diff.max().item() < 0.001:
    print("\n✅ 模拟逻辑与 Triton 一致")
    print("   问题不在 kernel 逻辑，可能在其他地方")
else:
    print("\n❌ 模拟逻辑与 Triton 不一致")
    print("   可能的原因:")
    print("   1. 模拟的循环逻辑与 Triton 不同")
    print("   2. 数值精度处理不同")
    print("   3. Triton kernel 有额外的隐藏逻辑")

print("="*80)
