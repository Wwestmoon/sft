#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用 AnswerSynthesizer.refine_answer 方法优化训练数据中的 response
"""

import json
import jsonlines
import argparse
from sft_generator.base import OpenAICompatibleAPI
from sft_generator.answer_synthesizer import AnswerSynthesizer


def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='使用 refine_answer 方法优化训练数据中的 response')
    parser.add_argument('--input', '-i',
                        default='/Users/westmoon/mycode/table/training_data/training_data_debug_0_None.jsonl.messages.jsonl',
                        help='输入的 messages.jsonl 文件路径')
    parser.add_argument('--original', '-o',
                        default='/Users/westmoon/mycode/table/training_data/training_data_debug_0_None.jsonl',
                        help='原始 training_data 文件路径')
    parser.add_argument('--output', '-p',
                        default='/Users/westmoon/mycode/table/training_data/training_data_debug_0_None_refined_messages.jsonl',
                        help='输出的优化后文件路径')
    parser.add_argument('--start', '-s', type=int, default=1,
                        help='起始样本索引（可选）')
    parser.add_argument('--end', '-e', type=int, default=10,
                        help='结束样本索引（可选）')

    args = parser.parse_args()

    # 配置 LLM API
    model_config = {
        "base_url": "https://yunwu.ai/v1",
        "api_key": "sk-mBv9A5UrRCQ7OzJDPLKZkjIRzoywwVcu4wvStR4P6E7K9Kw9",
        "model": "gpt-4o-mini"
    }

    # 初始化 LLM API 和 AnswerSynthesizer
    llm_api = OpenAICompatibleAPI(model_config)
    synthesizer = AnswerSynthesizer(llm_api)

    # 输入和输出文件路径
    messages_file = args.input
    original_file = args.original
    output_file = args.output

    print(f"开始读取文件: {messages_file}")

    # 读取原始训练数据以获取 table 和 question
    original_data = {}
    with open(original_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                original_data[data['id']] = {
                    'table': data['table'],
                    'question': data['question']
                }

    # 读取 messages.jsonl 文件
    all_messages = []
    with open(messages_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                all_messages.append(json.loads(line))

    # 处理起始和结束索引
    if args.start is not None or args.end is not None:
        start_idx = args.start if args.start is not None else 0
        end_idx = args.end if args.end is not None else len(all_messages)
        messages_to_process = all_messages[start_idx:end_idx]
        print(f"处理样本范围: {start_idx} 到 {end_idx}（共 {len(messages_to_process)} 个样本）")
    else:
        messages_to_process = all_messages
        print(f"处理所有样本（共 {len(messages_to_process)} 个）")

    refined_results = []

    # 处理样本
    for i, data in enumerate(messages_to_process):
        # 提取所需字段
        sample_id = data['id']
        messages = data['messages']

        print(f"\n处理样本 {i+1}/{len(messages_to_process)}: {sample_id}")

        # 找到用户和助手消息
        user_message = None
        assistant_message = None
        for msg in messages:
            if msg['role'] == 'user':
                user_message = msg['content']
            elif msg['role'] == 'assistant':
                assistant_message = msg['content']

        if not user_message or not assistant_message:
            print(f"  警告: 未找到用户或助手消息，跳过")
            continue

        # 从原始数据中获取 table 和 question
        if sample_id not in original_data:
            print(f"  警告: 在原始数据中未找到样本 {sample_id}，跳过")
            continue

        table = original_data[sample_id]['table']
        original_question = original_data[sample_id]['question']

        print(f"  优化 response...")

        # 调用 refine_answer 方法进行优化
        try:
            refined_response = synthesizer.refine_answer(
                assistant_message,
                table,
                original_question
            )

            # 创建优化后的消息
            refined_messages = data.copy()
            # 添加原始 response 记录
            refined_messages['original_response'] = assistant_message
            # 更新为优化后的 response
            refined_messages['messages'] = [
                {'role': 'user', 'content': user_message},
                {'role': 'assistant', 'content': refined_response}
            ]

            refined_results.append(refined_messages)

            print(f"  优化成功!")

        except Exception as e:
            print(f"  优化失败: {e}")
            # 如果优化失败，保留原始消息
            refined_results.append(data.copy())

    # 保存优化后的结果
    print(f"\n优化完成，保存结果到: {output_file}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for result in refined_results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"成功优化了 {len(refined_results)} 个样本")


if __name__ == "__main__":
    main()
