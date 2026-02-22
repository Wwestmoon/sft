#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import sys

# 启用详细警告
warnings.simplefilter('always', SyntaxWarning)
warnings.simplefilter('always', UserWarning)

# 重定向警告到标准输出并显示来源
def warn_with_traceback(message, category, filename, lineno, file=None, line=None):
    sys.stderr.write(warnings.formatwarning(message, category, filename, lineno, line))
    print("-" * 60, file=sys.stderr)
    print("Stack trace for warning:", file=sys.stderr)
    import traceback
    traceback.print_stack(file=sys.stderr)
    print("\n" + "-" * 60, file=sys.stderr)

warnings.showwarning = warn_with_traceback

print("=== Testing DataFrame Conversion ===\n")

# 创建一个包含日期列的表格
test_table = """| Date       | Title                          |
| 1854-08-15 | Ineffabilis Deus               |
| 1864-09-08 | Apostolicae Sedis Moderationi  |
| 1867-10-26 | Quanta Cura                    |
"""

from sft_generator.base import TableConverter

print("Step 1: Converting markdown to DataFrame...")
try:
    df = TableConverter.markdown_to_dataframe(test_table)
    print("✓ Conversion successful")
    print(f"DataFrame shape: {df.shape}")

    print("\nStep 2: Checking DataFrame info...")
    df.info()

    print("\nStep 3: Checking column dtypes...")
    for col in df.columns:
        print(f"{col}: {df[col].dtype}")

    print("\nStep 4: First 5 rows...")
    print(df.head())

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Test completed ===\n")
