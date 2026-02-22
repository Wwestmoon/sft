#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT 数据生成器的主入口脚本
"""

import argparse
import os
import warnings

# 方案一：直接继续（忽略警告）
warnings.filterwarnings('ignore', category=SyntaxWarning)
warnings.filterwarnings('ignore', category=UserWarning)

from sft_generator import SFTDataGenerator
from sft_generator.base import SimpleAnswerExtractor


def main():
    parser = argparse.ArgumentParser(description='SFT 数据生成器')

    # 输入输出参数
    parser.add_argument('--input_file', type=str,
                        default='/Users/westmoon/mycode/table/processed_data/training_processed.jsonl',
                        help='输入文件路径')
    parser.add_argument('--output_file', type=str,
                        default='/Users/westmoon/mycode/table/training_data/training_data',
                        help='输出文件路径')
    parser.add_argument('--num_samples', type=int, default=100,
                        help='处理的样本数量')
    parser.add_argument('--sample_id', type=str, default=None,
                        help='指定要测试的样本 ID')
    parser.add_argument('--start_index', type=int, default=0,
                        help='起始样本索引')
    parser.add_argument('--end_index', type=int, default=None,
                        help='结束样本索引')
    parser.add_argument('--max_attempts', type=int, default=3,
                        help='每个样本的最大尝试次数')

    # 模型参数
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                        help='模型名称')
    parser.add_argument('--base_url', type=str, default='https://yunwu.ai/v1',
                        help='API 基础 URL')
    parser.add_argument('--api_key', type=str,
                        default='sk-mBv9A5UrRCQ7OzJDPLKZkjIRzoywwVcu4wvStR4P6E7K9Kw9',
                        help='API 密钥')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='采样温度')
    parser.add_argument('--top_p', type=float, default=0.9,
                        help='核采样参数')

    # 操作模式
    parser.add_argument('--test', action='store_true',
                        help='运行小规模测试')
    parser.add_argument('--generate', default=True,
                        help='生成训练数据')
    parser.add_argument('--verbose', action='store_true',
                        help='显示详细信息')

    args = parser.parse_args()

    # 模型配置
    model_config = {
        "model": args.model,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "temperature": args.temperature,
        "top_p": args.top_p
    }

    # 初始化生成器
    print("初始化数据生成器...")
    generator = SFTDataGenerator(model_config)

    # 根据操作模式执行相应的功能
    if args.test:
        print("开始小规模测试...")
        stats, test_data, all_attempts = generator.test_small_scale(args.input_file, args.num_samples, args.sample_id)

        print("\n=== 测试报告 ===")
        print(f"总样本数: {stats['total_samples']}")
        print(f"成功样本数: {stats['successful_samples']}")
        print(f"样本级匹配率: {stats['sample_match_rate']:.2%}")
        print(f"总尝试次数: {stats['total_attempts']}")
        print(f"成功尝试次数: {stats['successful_attempts']}")
        print(f"生成级匹配率: {stats['attempt_match_rate']:.2%}")
        print(f"平均子问题数量: {stats['average_sub_questions']:.1f}")
        print(f"Python 求解占比: {stats['python_solver_ratio']:.2%}")
        print(f"LLM 求解占比: {stats['llm_solver_ratio']:.2%}")
    elif args.generate:
        print("开始生成训练数据...")
        # 根据起始和结束索引生成输出文件名
        if args.start_index is not None or args.end_index is not None:
            output_file = f"{args.output_file}_{args.start_index}_{args.end_index}.jsonl"
        else:
            output_file = f"{args.output_file}.jsonl"

        # 调用生成训练数据的方法
        results = generator.generate_training_data(
            args.input_file,
            output_file,
            args.num_samples,
            args.max_attempts,
            args.sample_id,
            args.start_index,
            args.end_index
        )

        print(f"\n生成完成！")
        print(f"处理样本数: {len(results)}")
        print(f"成功生成的训练条数: {len(results)}")
        print(f"结果已保存到: {output_file}")
    else:
        print("请指定操作模式（--test 或 --generate）")
        parser.print_help()


if __name__ == "__main__":
    main()
