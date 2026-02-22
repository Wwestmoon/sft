#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import sys
import pandas as pd
import io

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

# 创建一个包含日期的简单表格
test_table = """| Date       | Title                          |
| 1854-08-15 | Ineffabilis Deus               |
| 1864-09-08 | Apostolicae Sedis Moderationi  |
| 1867-10-26 | Quanta Cura                    |
"""

from sft_generator.base import TableConverter

print("Testing TableConverter...")
try:
    df = TableConverter.markdown_to_dataframe(test_table)
    print("Successfully converted to DataFrame")
    print("\nDataFrame:")
    print(df)
    print("\nDataFrame info:")
    df.info()

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# 测试LLM生成的代码
print("\n" + "=" * 60)
print("Testing LLM-generated code...")
llm_code = """filtered_encyclicals = df[(df['Date'] >= '1854-08-15') & (df['Date'] <= '1867-10-26')]
count_encyclicals = filtered_encyclicals.shape[0]
print(count_encyclicals)"""

try:
    local_vars = {"df": df}
    safe_globals = {
        "__builtins__": __import__("builtins"),
        "pd": __import__("pandas"),
        "numpy": __import__("numpy"),
        "re": __import__("re"),
        "datetime": __import__("datetime"),
        "math": __import__("math")
    }
    exec(llm_code, safe_globals, local_vars)
    print("Code executed successfully")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
