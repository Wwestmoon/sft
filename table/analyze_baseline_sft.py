#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
分析基线实验结果的脚本
"""

import json
import os


def analyze_baseline_results(data_file, error_file):
    """
    分析基线实验结果，生成统计报告
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
        'successful_attempts': 0
    }

    # 统计尝试次数
    for sample in data_samples:
        stats['total_attempts'] += sample['total_attempts']
        stats['successful_attempts'] += 1

    for sample in error_samples:
        stats['total_attempts'] += sample['total_attempts']

    # 计算成功率
    success_rate = 0
    if stats['total_samples'] > 0:
        success_rate = round(stats['success_samples'] / stats['total_samples'] * 100, 1)

    generation_success_rate = 0
    if stats['total_attempts'] > 0:
        generation_success_rate = round(stats['successful_attempts'] / stats['total_attempts'] * 100, 1)

    # 生成报告
    report = f"""## 基线实验分析报告

### 总体统计
- **总样本数**: {stats['total_samples']}
- **成功样本数**: {stats['success_samples']}
- **失败样本数**: {stats['fail_samples']}
- **样本级成功率**: {success_rate}%
- **总尝试次数**: {stats['total_attempts']}
- **成功尝试次数**: {stats['successful_attempts']}
- **生成级成功率**: {generation_success_rate}%

### 失败样本详情
"""

    if stats['fail_samples'] > 0:
        for sample in error_samples:
            report += f"1. **{sample['sample_id']}**: {sample['question']} - {sample['total_attempts']}次尝试均失败\n"
    else:
        report += "无失败样本"

    return report, stats


def main():
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="分析基线实验结果")
    parser.add_argument("--data_file", help="成功样本文件路径")
    parser.add_argument("--error_file", help="错误样本文件路径")
    parser.add_argument("--output", default="baseline_sft_analysis_report.md", help="报告输出文件路径")
    args = parser.parse_args()

    if not args.data_file:
        print("请指定成功样本文件路径")
        return

    if not args.error_file:
        # 自动推断错误文件路径
        args.error_file = args.data_file.replace('.jsonl', '_errors.jsonl')

    # 分析结果
    report, stats = analyze_baseline_results(args.data_file, args.error_file)

    # 保存报告
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write(report)

    print(f"分析报告已生成: {args.output}")
    print("\n=== 关键统计信息 ===")
    print(f"总样本数: {stats['total_samples']}")
    print(f"成功样本数: {stats['success_samples']}")
    print(f"失败样本数: {stats['fail_samples']}")
    print(f"样本级成功率: {round(stats['success_samples'] / stats['total_samples'] * 100, 1)}%")
    print(f"总尝试次数: {stats['total_attempts']}")
    print(f"平均尝试次数: {round(stats['total_attempts'] / stats['total_samples'], 1)}次/样本")


if __name__ == '__main__':
    main()
