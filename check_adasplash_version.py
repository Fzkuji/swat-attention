#!/usr/bin/env python3
"""检查 adasplash 是否使用了最新代码"""
import adasplash
import inspect

print("="*80)
print("检查 adasplash 版本")
print("="*80)

# 1. 检查导入路径
print(f"\nadasplash 导入路径:")
print(f"  {adasplash.__file__}")

# 2. 读取源代码检查是否有修复
from adasplash import lazy_attention_triton as lat_module
source_file = inspect.getsourcefile(lat_module._lazy_bwd_preprocess_kernel)
print(f"\n_lazy_bwd_preprocess_kernel 源文件:")
print(f"  {source_file}")

# 3. 检查源代码中是否有 p_norm_f32
source = inspect.getsource(lat_module._lazy_bwd_preprocess_kernel)
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
