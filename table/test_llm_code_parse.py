#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
import sys
import re

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

print("Testing LLM code parsing...")

# 模拟 LLM 返回的响应
llm_response = """```python
filtered_encyclicals = df[(df['Date'] >= '1854-08-15') & (df['Date'] <= '1867-10-26')]
count_encyclicals = filtered_encyclicals.shape[0]
print(count_encyclicals)
```"""

from sft_generator.sub_question_solver import SubQuestionSolver

print("\nTesting code block extraction...")
extractor = SubQuestionSolver(None)

try:
    # 尝试提取代码块
    code = extractor._extract_code_block(llm_response)
    print("Successfully extracted code")
    print(f"Code length: {len(code)}")
    print("\nExtracted code:")
    print(code)
    print("\nCode repr:")
    print(repr(code))

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

# 测试正则表达式模式匹配
print("\n" + "=" * 60)
print("Testing regex pattern compilation...")
try:
    patterns = [
        '```python\n(.*?)```',
        r'```python\n(.*?)```',
        '```\n(.*?)```',
        r'```\n(.*?)```',
        'code:(.*)'
    ]

    print("Testing patterns:")
    for i, pattern in enumerate(patterns):
        print(f"\nPattern {i}: {repr(pattern)}")
        try:
            compiled = re.compile(pattern)
            match = compiled.search(llm_response)
            if match:
                print(f"Found match")
                print(f"Groups: {match.groups()}")
        except Exception as e:
            print(f"Error: {e}")

except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
