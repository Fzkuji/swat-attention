#!/usr/bin/env python3
"""对比 scratch 和 flash 训练的模型"""
import torch
import sys
import os

print("="*80)
print("对比 Scratch vs Flash 训练的模型")
print("="*80)

# 检查是否提供了 checkpoint 路径
if len(sys.argv) < 3:
    print("\n用法:")
    print("  python compare_trained_models.py <scratch_checkpoint> <flash_checkpoint>")
    print("\n示例:")
    print("  python compare_trained_models.py \\")
    print("    checkpoints/scratch/step_1000 \\")
    print("    checkpoints/flash/step_1000")
    print("\n说明:")
    print("  - 加载两个模型")
    print("  - 用相同输入计算 loss")
    print("  - 对比模型参数（tau, bias）")
    print("  - 分析差异来源")
    sys.exit(0)

scratch_ckpt = sys.argv[1]
flash_ckpt = sys.argv[2]

if not os.path.exists(scratch_ckpt):
    print(f"❌ Scratch checkpoint 不存在: {scratch_ckpt}")
    sys.exit(1)

if not os.path.exists(flash_ckpt):
    print(f"❌ Flash checkpoint 不存在: {flash_ckpt}")
    sys.exit(1)

print(f"\nCheckpoints:")
print(f"  Scratch: {scratch_ckpt}")
print(f"  Flash:   {flash_ckpt}")

# TODO: 加载模型和配置
# 这需要知道具体的模型结构和 checkpoint 格式

print("\n" + "="*80)
print("对比参数")
print("="*80)

# 对比 tau
print("\nTau 参数:")
# scratch_tau = ...
# flash_tau = ...
# print(f"  Scratch: {scratch_tau}")
# print(f"  Flash:   {flash_tau}")
# print(f"  差异:    {(scratch_tau - flash_tau).abs()}")

# 对比 bias
print("\nBias 参数:")
# ...

print("\n" + "="*80)
print("对比前向输出")
print("="*80)

# 创建相同的测试输入
# 运行两个模型
# 对比输出和 loss

print("\n注意：此脚本需要根据实际的模型结构补充代码")
print("="*80)
