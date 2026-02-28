#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
重新合成答案的脚本：从 training_data_0_None.jsonl 中提取 match 为 true 的样本，
重新调用 AnswerSynthesizer 来合成最终答案，并保存到新的 jsonl 文件中
"""

import json
import os
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor
from sft_generator.base import OpenAICompatibleAPI
from sft_generator.answer_synthesizer import AnswerSynthesizer

# 线程锁，确保文件写入安全性
file_write_lock = threading.Lock()


def load_baseline_responses(baseline_file):
    """
    加载 baseline_sft_critic_0_all_messages.jsonl 文件，创建 id 到 initial response 的映射
    """
    print(f"正在加载 baseline 文件: {baseline_file}")
    id_to_response = {}

    if not os.path.exists(baseline_file):
        print(f"Baseline 文件不存在: {baseline_file}")
        return id_to_response

    try:
        with open(baseline_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if 'id' in data and 'messages' in data:
                            # 提取助手的响应作为 initial response（通常是 messages 数组中的偶数索引之后的消息）
                            # 查找最后一个助手响应（role: assistant）
                            initial_response = ""
                            for msg in data['messages']:
                                if msg.get('role') == 'assistant':
                                    initial_response = msg.get('content', '')

                            if initial_response:
                                id_to_response[data['id']] = initial_response

                    except Exception as e:
                        print(f"解析 baseline 文件数据失败: {e}")
                        continue

        print(f"成功加载 {len(id_to_response)} 个 baseline 响应")

    except Exception as e:
        print(f"读取 baseline 文件失败: {e}")

    return id_to_response


def load_training_data(input_file):
    """
    加载训练数据并提取 match 为 true 的样本
    """
    print(f"正在加载数据文件: {input_file}")
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    # 检查是否有成功匹配的尝试
                    if 'attempts' in data:
                        for attempt in data['attempts']:
                            if attempt.get('match', False):
                                samples.append({
                                    'id': data['id'],
                                    'table': data['table'],
                                    'question': data['question'],
                                    'expected_answer': data['expected_answer'],
                                    'decomposition': attempt['response']['decomposition'],
                                    'sub_question_results': attempt['response']['sub_question_results'],
                                    'original_final_response': attempt['response']['final_response']
                                })
                                break
                except Exception as e:
                    print(f"解析数据失败: {e}")
                    continue

    print(f"成功提取 {len(samples)} 个匹配的样本")
    return samples


def process_single_sample(sample, llm_api):
    """
    处理单个样本的答案合成（用于并行处理，共享API实例）
    """
    synthesizer = AnswerSynthesizer(llm_api)

    try:
        # 重新合成答案
        new_final_response = synthesizer.synthesize_answer(
            sample['table'],
            sample['sub_question_results'],
            # sample['initial_response'],
            sample['question']
        )

        return {
            'id': sample['id'],
            'table': sample['table'],
            'question': sample['question'],
            'expected_answer': sample['expected_answer'],
            'decomposition': sample['decomposition'],
            'sub_question_results': sample['sub_question_results'],
            # 'initial_response':sample['initial_response'],
            'original_final_response': sample['original_final_response'],
            'new_final_response': new_final_response
        }

    except Exception as e:
        print(f"✗ 合成样本 {sample['id']} 的答案失败: {e}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")
        return None


def re_synthesize_answers(samples, model_config, output_file, max_workers=10):
    """
    重新合成答案（支持并行处理，处理完一个写入一个）
    """
    print(f"正在初始化答案合成器...")
    print(f"使用 {max_workers} 个并发任务")

    print(f"开始重新合成 {len(samples)} 个样本的答案...")

    # 确保输出目录存在
    if not os.path.exists(os.path.dirname(output_file)):
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # 清空输出文件
    if os.path.exists(output_file):
        try:
            os.remove(output_file)
        except Exception as e:
            print(f"删除已存在文件失败: {e}")
            return []

    results = []

    # 创建单一的 API 实例，所有线程共享
    llm_api = OpenAICompatibleAPI(model_config)

    try:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_sample = {
                executor.submit(process_single_sample, sample, llm_api): sample
                for sample in samples
            }

            # 获取结果
            for i, future in enumerate(future_to_sample):
                sample = future_to_sample[future]
                try:
                    result = future.result(timeout=300)  # 设置超时时间
                    if result:
                        results.append(result)
                        print(f"✓ 处理样本 {i+1}/{len(samples)}: {sample['id']}")

                        # 处理完一个立即写入到文件（使用锁机制确保安全性）
                        with file_write_lock:
                            with open(output_file, 'a', encoding='utf-8') as f:
                                f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    else:
                        print(f"✗ 处理样本 {i+1}/{len(samples)}: {sample['id']}")
                except Exception as e:
                    print(f"✗ 处理样本 {i+1}/{len(samples)}: {sample['id']} 失败: {e}")
                    import traceback
                    print(f"详细错误信息: {traceback.format_exc()}")
    except Exception as e:
        print(f"执行任务时发生严重错误: {e}")
        import traceback
        print(f"详细错误信息: {traceback.format_exc()}")

    print(f"\n重新合成完成！成功处理 {len(results)} 个样本")

    # 资源清理，确保API实例被正确释放
    if 'llm_api' in locals():
        del llm_api

    return results


def save_results(results, output_file):
    """
    保存结果到 jsonl 文件
    """
    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print(f"正在保存结果到: {output_file}")

    with open(output_file, 'w', encoding='utf-8') as f:
        for result in results:
            f.write(json.dumps(result, ensure_ascii=False) + '\n')

    print(f"结果保存成功！共 {len(results)} 条记录")


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='重新合成答案脚本')

    # 输入输出参数
    parser.add_argument('--input_file', type=str,
                        default='/Users/westmoon/mycode/table/training_data/training_data_0_None.jsonl',
                        help='输入文件路径')
    parser.add_argument('--baseline_file', type=str,
                        default='/Users/westmoon/mycode/table/baseline_results/baseline_sft_critic_0_all_messages.jsonl',
                        help='baseline 文件路径')
    parser.add_argument('--output_file', type=str,
                        default='/Users/westmoon/mycode/table/training_data/re_synthesized_answers—1.jsonl',
                        help='输出文件路径')

    # 处理范围参数
    parser.add_argument('--start_index', type=int, default=0,
                        help='起始样本索引')
    parser.add_argument('--end_index', type=int, default=None,
                        help='结束样本索引')

    # 并发参数
    parser.add_argument('--max_workers', type=int, default=200,
                        help='最大并发任务数')

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

    args = parser.parse_args()

    # 模型配置
    model_config = {
        "model": args.model,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "temperature": args.temperature,
        "top_p": args.top_p
    }

    # 加载 baseline 响应
    baseline_responses = load_baseline_responses(args.baseline_file)

    # 执行流程
    samples = load_training_data(args.input_file)

    # # 过滤掉在 baseline 文件中找不到对应响应的样本
    # filtered_samples = []
    # skipped_samples = 0

    # for sample in samples:
    #     if sample['id'] in baseline_responses:
    #         # 为样本添加 initial response
    #         sample['initial_response'] = baseline_responses[sample['id']]
    #         filtered_samples.append(sample)
    #     else:
    #         skipped_samples += 1

    # print(f"过滤掉 {skipped_samples} 个未找到 initial response 的样本")

    # if not filtered_samples:
    #     print("未找到任何匹配的样本，程序结束")
    #     return

    # samples = filtered_samples

    # 处理起始和结束索引
    if args.start_index is not None or args.end_index is not None:
        if args.end_index is None:
            args.end_index = len(samples)

        # 确保索引在有效范围内
        args.start_index = max(0, args.start_index)
        args.end_index = min(len(samples), args.end_index)

        samples = samples[args.start_index:args.end_index]
        print(f"处理范围: 样本 {args.start_index} 到 {args.end_index}，共 {len(samples)} 个样本")

    if not samples:
        print("未找到匹配的样本，程序结束")
        return

    results = re_synthesize_answers(samples, model_config, args.output_file, args.max_workers)


if __name__ == "__main__":
    main()
