#!/usr/bin/env python3
"""检查 adasplash 是否使用了最新代码"""
import os

print("="*80)
print("检查 adasplash 版本")
print("="*80)

# 1. 检查导入路径
import adasplash
print(f"\nadasplash 导入路径:")
print(f"  {adasplash.__file__}")

# 2. 找到 lazy_attention_triton.py 文件
adasplash_dir = os.path.dirname(adasplash.__file__)
source_file = os.path.join(adasplash_dir, 'lazy_attention_triton.py')
print(f"\nlazy_attention_triton.py 路径:")
print(f"  {source_file}")

if not os.path.exists(source_file):
    print(f"  ❌ 文件不存在！")
    exit(1)

# 3. 检查源代码中是否有 p_norm_f32
with open(source_file, 'r') as f:
    source = f.read()

has_fix = 'p_norm_f32' in source

print(f"\n是否包含修复 (p_norm_f32):")
if has_fix:
    print("  ✅ 是 - 已包含修复")
    # 打印相关代码行
    lines = source.split('\n')
    for i, line in enumerate(lines):
        if 'p_norm_f32' in line or 'p_term_f32' in line or 'mask_relu' in line:
            print(f"    {i+1}: {line}")
else:
    print("  ❌ 否 - 仍是旧版本")
    print("\n需要执行:")
    print("  cd /c/Users/fzkuj/Projects/adasplash")
    print("  git pull")
    print("  pip install -e . --force-reinstall --no-deps")

print("\n" + "="*80)
