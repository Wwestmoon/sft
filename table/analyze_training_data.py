#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接解析训练数据文件的分析脚本
"""

import json
import os


def analyze_training_data(data_file, error_file):
    """
    分析训练数据文件，生成统计报告
    """
    # 读取成功样本数据
    with open(data_file, 'r', encoding='utf-8') as f:
        data_lines = [line.strip() for line in f if line.strip()]
        data_samples = [json.loads(line) for line in data_lines]

    # 读取失败样本数据
    if os.path.exists(error_file):
        with open(error_file, 'r', encoding='utf-8') as f:
            error_lines = [line.strip() for line in f if line.strip()]
            error_samples = [json.loads(line) for line in error_lines]
    else:
        error_samples = []

    # 统计信息
    stats = {
        'total_samples': len(data_samples) + len(error_samples),
        'success_samples': len(data_samples),
        'fail_samples': len(error_samples),
        'total_attempts': 0,
        'successful_attempts': 0,
        'python_uses': 0,
        'critic_uses': 0,
        'avg_sub_questions': 0,
        'python_strategy_rate': 0
    }

    # 统计尝试次数和策略使用情况
    for sample in data_samples:
        stats['total_attempts'] += sample['total_attempts']
        stats['successful_attempts'] += 1  # 每个成功样本至少有一个成功尝试

        for attempt in sample['attempts']:
            if 'response' in attempt and 'sub_question_results' in attempt['response']:
                # 统计 Python 求解次数
                python_uses = sum(1 for result in attempt['response']['sub_question_results'] if result['strategy'] == 'Python')
                stats['python_uses'] += python_uses

            if 'error_analysis' in attempt and attempt['error_analysis']:
                stats['critic_uses'] += 1

    for sample in error_samples:
        stats['total_attempts'] += sample['total_attempts']

        for attempt in sample['attempts']:
            if 'response' in attempt and 'sub_question_results' in attempt['response']:
                # 统计 Python 求解次数
                python_uses = sum(1 for result in attempt['response']['sub_question_results'] if result['strategy'] == 'Python')
                stats['python_uses'] += python_uses

            if 'error_analysis' in attempt and attempt['error_analysis']:
                stats['critic_uses'] += 1

    # 计算平均子问题数量
    total_sub_questions = 0
    for sample in data_samples:
        for attempt in sample['attempts']:
            if 'response' in attempt and 'sub_question_results' in attempt['response']:
                total_sub_questions += len(attempt['response']['sub_question_results'])

    for sample in error_samples:
        for attempt in sample['attempts']:
            if 'response' in attempt and 'sub_question_results' in attempt['response']:
                total_sub_questions += len(attempt['response']['sub_question_results'])

    if stats['total_attempts'] > 0:
        stats['avg_sub_questions'] = round(total_sub_questions / stats['total_attempts'], 1)

    # 计算 Python 策略使用率
    if total_sub_questions > 0:
        stats['python_strategy_rate'] = round(stats['python_uses'] / total_sub_questions * 100, 1)

    # 计算平均尝试次数
    avg_attempts = 0
    if stats['total_samples'] > 0:
        avg_attempts = round(stats['total_attempts'] / stats['total_samples'], 1)

    # 计算成功尝试的平均次数
    avg_successful_attempts = 0
    if stats['success_samples'] > 0:
        successful_attempts_count = 0
        for sample in data_samples:
            successful_attempts_count += sum(1 for attempt in sample['attempts'] if attempt['match'])
        avg_successful_attempts = round(successful_attempts_count / stats['success_samples'], 1)

    # 计算成功率
    success_rate = 0
    if stats['total_samples'] > 0:
        success_rate = round(stats['success_samples'] / stats['total_samples'] * 100, 1)

    generation_success_rate = 0
    if stats['total_attempts'] > 0:
        generation_success_rate = round(stats['successful_attempts'] / stats['total_attempts'] * 100, 1)

    # 生成报告
    report = f"""
## SFT 数据生成器测试报告

### 总体统计
- **总样本数**: {stats['total_samples']}
- **成功样本数**: {stats['success_samples']}
- **失败样本数**: {stats['fail_samples']}
- **样本级成功率**: {success_rate}%
- **总尝试次数**: {stats['total_attempts']}
- **成功尝试次数**: {stats['successful_attempts']}
- **生成级成功率**: {generation_success_rate}%

### 尝试次数统计
- **平均尝试次数**: {avg_attempts}次/样本
- **平均成功尝试次数**: {avg_successful_attempts}次/成功样本

### 子问题求解统计
- **平均子问题数量**: {stats['avg_sub_questions']}个/尝试
- **Python 求解总次数**: {stats['python_uses']}
- **LLM 求解总次数**: {total_sub_questions - stats['python_uses']}
- **Python 求解占比**: {stats['python_strategy_rate']}%
- **LLM 求解占比**: {100 - stats['python_strategy_rate']}%

### Critic 使用统计
- **Critic 使用次数**: {stats['critic_uses']}
- **Critic 使用频率**: {round(stats['critic_uses'] / stats['total_attempts'] * 100, 1)}%

### 失败样本详情
"""

    if stats['fail_samples'] > 0:
        for sample in error_samples:
            report += f"1. **{sample['sample_id']}**: {sample['question']} - {sample['total_attempts']}次尝试均失败\n"
    else:
        report += "无失败样本"

    return report, stats


def main():
    data_file = '/Users/westmoon/mycode/table/training_data/training_data.jsonl'
    error_file = '/Users/westmoon/mycode/table/training_data/training_data_errors.jsonl'

    report, stats = analyze_training_data(data_file, error_file)

    # 保存报告
    output_file = 'training_data_analysis_report.md'
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(report)

    print("测试报告已生成:", output_file)
    print("\n=== 关键统计信息 ===")
    print(f"总样本数: {stats['total_samples']}")
    print(f"成功样本数: {stats['success_samples']}")
    print(f"失败样本数: {stats['fail_samples']}")
    print(f"样本级成功率: {round(stats['success_samples'] / stats['total_samples'] * 100, 1)}%")
    print(f"总尝试次数: {stats['total_attempts']}")
    print(f"平均尝试次数: {round(stats['total_attempts'] / stats['total_samples'], 1)}次/样本")
    print(f"平均子问题数量: {stats['avg_sub_questions']}个/尝试")
    print(f"Python 求解占比: {stats['python_strategy_rate']}%")
    print(f"Critic 使用次数: {stats['critic_uses']}")


if __name__ == '__main__':
    main()
