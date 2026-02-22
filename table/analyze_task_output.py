#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
任务输出分析脚本
自动分析任务输出文件，生成详细的统计报告
"""

import re
import json
import argparse


def analyze_task_output(file_path):
    """
    分析任务输出文件，生成统计报告
    """
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # 统计信息
    stats = {
        'total_samples': 0,
        'success_samples': 0,
        'fail_samples': 0,
        'llm_calls': 0,
        'python_failures': 0,
        'code_execution_failures': 0,
        'answer_mismatches': 0,
        'total_attempts': 0,
        'found_matching_answers': 0
    }

    # 统计总处理样本数
    stats['total_samples'] = len(re.findall(r'已处理样本', content))

    # 统计成功匹配样本数
    stats['found_matching_answers'] = len(re.findall(r'Found matching answer', content))
    stats['success_samples'] = stats['found_matching_answers']
    stats['fail_samples'] = stats['total_samples'] - stats['success_samples']

    # 统计LLM调用次数
    stats['llm_calls'] = len(re.findall(r'正在调用 LLM', content))

    # 统计Python求解失败次数
    stats['python_failures'] = len(re.findall(r'Python 求解失败', content))

    # 统计代码执行失败次数
    stats['code_execution_failures'] = len(re.findall(r'代码执行失败', content))

    # 统计答案不匹配次数
    stats['answer_mismatches'] = len(re.findall(r'Answer does not match', content))

    # 统计总尝试次数
    # 使用更准确的正则表达式匹配尝试次数
    stats['total_attempts'] = len(re.findall(r'Attempt \d+', content)) + len(re.findall(r'attempt \d+', content.lower()))

    # 计算平均尝试次数
    avg_attempts = 0
    if stats['total_samples'] > 0:
        avg_attempts = round(stats['total_attempts'] / stats['total_samples'], 2)

    # 统计不同尝试次数的成功样本数
    one_attempt_success = len(re.findall(r'Attempt 1.*Found matching answer', content, re.DOTALL))
    two_attempts_success = len(re.findall(r'Attempt 2.*Found matching answer', content, re.DOTALL))
    three_attempts_failure = stats['fail_samples']

    # 生成报告
    report = f"""
## 数据生成统计报告

### 总体统计
- **总处理样本数**: {stats['total_samples']}
- **成功匹配样本数**: {stats['success_samples']}
- **失败样本数**: {stats['fail_samples']}
- **成功率**: {round(stats['success_samples'] / stats['total_samples'] * 100, 1)}%

### LLM调用统计
- **LLM总调用次数**: {stats['llm_calls']}次

### 尝试次数统计
- **总尝试次数**: {stats['total_attempts']}次
- **平均尝试次数**: {avg_attempts}次/样本
- **1次尝试成功样本数**: {one_attempt_success}个
- **2次尝试成功样本数**: {two_attempts_success}个
- **3次尝试失败样本数**: {three_attempts_failure}个

### Python代码执行统计
- **Python求解失败次数**: {stats['python_failures']}次
- **代码执行失败次数**: {stats['code_execution_failures']}次
- **LLM作为备用策略次数**: 多次（Python失败后使用LLM）

### 答案质量统计
- **答案不匹配次数**: {stats['answer_mismatches']}次
- **找到匹配答案次数**: {stats['found_matching_answers']}次

### 失败样本详情
"""

    # 查找失败样本ID
    fail_sample_ids = []

    # 从训练数据错误文件中获取失败样本详情
    try:
        with open('/Users/westmoon/mycode/table/training_data/training_data_errors.jsonl', 'r', encoding='utf-8') as f:
            errors = [json.loads(line) for line in f if line.strip()]
            for error in errors:
                fail_sample_ids.append(error['sample_id'])
                report += f"1. **{error['sample_id']}**: {error['question']} - {error['total_attempts']}次尝试均失败\n"
    except Exception as e:
        print(f"读取错误文件失败: {e}")

    return report


def main():
    parser = argparse.ArgumentParser(description='任务输出分析脚本')
    parser.add_argument('file_path', help='任务输出文件路径')
    parser.add_argument('-o', '--output', help='输出报告文件路径', default='task_analysis_report.md')

    args = parser.parse_args()

    try:
        report = analyze_task_output(args.file_path)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"报告已生成: {args.output}")
    except Exception as e:
        print(f"分析失败: {e}")


if __name__ == '__main__':
    main()
