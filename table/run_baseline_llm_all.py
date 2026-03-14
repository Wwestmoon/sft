#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-Shot Baseline 脚本，支持并发处理、准确率计算、重试机制和循环测试
读取 JSONL 格式的数据集
"""

import os
import json
import requests
import sys
import re
import csv
import time
import random
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


# Add WikiTableQuestions directory to Python path (insert at front to avoid local evaluator.py shadowing)
wiki_table_questions_path = '/Users/westmoon/mycode/table/WikiTableQuestions'
if wiki_table_questions_path not in sys.path:
    sys.path.insert(0, wiki_table_questions_path)

from evaluator import to_value, to_value_list, check_denotation

# Restore sys.path to avoid polluting imports for other modules
sys.path.remove(wiki_table_questions_path)


class OpenAICompatibleAPI:
    """
    OpenAI 兼容的 API 接口
    支持多种 LLM 提供商，包括 OpenAI、DeepSeek、Qwen、Llama 等
    """

    def __init__(self, model, base_url, api_key, temperature=0.0, top_p=1.0):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.temperature = temperature
        self.top_p = top_p
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def call(self, prompt, max_retries=5):
        """
        调用 OpenAI 兼容的 API，支持重试机制
        """
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 8192,
            "temperature": self.temperature,
            "top_p": self.top_p
        }

        for retry in range(1, max_retries + 1):
            try:
                # 构建完整的 API 端点
                url = f"{self.base_url.rstrip('/')}/chat/completions"
                response = requests.post(url, headers=self.headers, json=data, timeout=60)
                response.raise_for_status()

                result = response.json()
                return result["choices"][0]["message"]["content"].strip()

            except Exception as e:
                print(f"API 调用失败 (重试 {retry}/{max_retries}): {type(e).__name__}: {str(e)}")
                if hasattr(e, 'response') and e.response:
                    try:
                        error_details = e.response.json()
                        print(f"API 错误详情: {json.dumps(error_details, ensure_ascii=False, indent=2)}")
                    except:
                        print(f"API 响应内容: {e.response.text}")

                if retry < max_retries:
                    print(f"等待 {2 ** retry} 秒后重试...")
                    time.sleep(2 ** retry)

        return "error"


def detect_task_type(filepath):
    """根据文件名自动检测任务类型"""
    basename = os.path.basename(filepath).lower()
    if 'finqa' in basename:
        return 'finqa'
    elif 'scitab' in basename:
        return 'scitab'
    elif 'pubhealth' in basename:
        return 'pubhealth'
    elif 'tabmwp' in basename:
        return 'tabmwp'
    else:
        return 'wtq'


def normalize_sample(sample, task_type):
    """统一不同数据集的 key 格式"""
    normalized = dict(sample)
    # scitab/pubhealth 用 index 而非 id
    if 'id' not in normalized and 'index' in normalized:
        normalized['id'] = normalized['index']
    # scitab/pubhealth 用 claim 而非 question
    if 'question' not in normalized and 'claim' in normalized:
        normalized['question'] = normalized['claim']
    return normalized


def build_prompt(markdown_table, question, task_type='wtq', context=None):
    """
    Build prompt based on task type
    """
    if task_type in ('scitab', 'pubhealth'):
        prompt = (
            f"Table:\n{markdown_table}\n\n"
            f"Claim: {question}\n\n"
            f"Based on the table, determine whether the claim is supported or refuted. "
            f"Please reason step by step and output the final answer in the format 'Final Answer: SUPPORTS' or 'Final Answer: REFUTES'."
        )
    elif task_type == 'finqa' and context:
        prompt = (
            f"Context:\n{context}\n\n"
            f"Table:\n{markdown_table}\n\n"
            f"Question: {question}\n\n"
            f"Please reason step by step and output the final answer in the specified format. The final answer should be in the format 'Final Answer: [answer]'."
        )
    else:
        prompt = (
            f"Table:\n{markdown_table}\n\n"
            f"Question: {question}\n\n"
            f"Please reason step by step and output the final answer in the specified format. The final answer should be in the format 'Final Answer: [answer]'."
        )
    return prompt


class SimpleAnswerExtractor:
    """
    简单的答案提取器，支持从多种格式中提取答案（与 sft_baseline_critic 相同）
    """

    def extract(self, response):
        """
        从响应中提取答案，支持多种格式
        """
        if not response or response == "error":
            return None

        # 1. 首先尝试提取 "Final Answer: " 格式的答案
        pattern1 = r'Final Answer: ([^\n]+)'
        match = re.search(pattern1, response)
        if match:
            return match.group(1).strip()

        # 2. 尝试提取直接回答格式的答案（如 "The answer is X"）
        pattern2 = r'The answer is ([^\n.]+)'
        match = re.search(pattern2, response)
        if match:
            return match.group(1).strip()

        # 3. 尝试提取 "Answer: " 格式的答案
        pattern3 = r'Answer: ([^\n]+)'
        match = re.search(pattern3, response)
        if match:
            return match.group(1).strip()

        # 4. 如果没有找到特定格式，尝试提取第一个可能是答案的短语
        # 去除常见的开头短语
        cleaned_response = response.strip()
        for prefix in ["The answer is ", "Answer: ", "Final Answer: ", "The correct answer is "]:
            if cleaned_response.startswith(prefix):
                cleaned_response = cleaned_response[len(prefix):].strip()

        # 尝试提取直到第一个句号或换行符的内容
        if '.' in cleaned_response:
            return cleaned_response.split('.', 1)[0].strip()
        elif '\n' in cleaned_response:
            return cleaned_response.split('\n', 1)[0].strip()
        else:
            return cleaned_response.strip()


class QualityController:
    """
    质量控制器类，用于答案匹配
    """

    def __init__(self, extractor):
        """
        初始化质量控制器

        Args:
            extractor: 答案提取器
        """
        self.extractor = extractor

    def _match_answer(self, extracted_answer, expected_answer):
        """
        Match answers using the official evaluation script's logic

        Args:
            extracted_answer: 提取的答案
            expected_answer: 预期答案

        Returns:
            是否匹配
        """
        if not extracted_answer or not expected_answer:
            return False

        # 将答案转换为官方评估脚本支持的 Value 类型
        target_values = to_value_list([str(expected_answer)])
        predicted_values = to_value_list([str(extracted_answer)])

        # 使用官方评估脚本的 check_denotation 方法进行匹配
        return check_denotation(target_values, predicted_values)


def process_sample(sample, model_config, task_type='wtq'):
    """
    处理单���样本，为每个任务创建独立的 API 实例以确保内存独立
    """
    print(f"正在处理样本: {sample['id']}")

    # 为每个样本创建独立的 API 实例，确保内存独立
    api = OpenAICompatibleAPI(
        model=model_config["model"],
        base_url=model_config["base_url"],
        api_key=model_config["api_key"],
        temperature=model_config.get("temperature", 0.1),
        top_p=model_config.get("top_p", 0.9)
    )

    extractor = SimpleAnswerExtractor()
    quality_controller = QualityController(extractor)

    try:
        # 构建 prompt（根据任务类型）
        context = sample.get('context', None)
        prompt = build_prompt(sample['table'], sample['question'], task_type=task_type, context=context)

        # 调用 API（支持重试）
        answer = api.call(prompt)

        # 提取答案并检查匹配
        extracted_answer = extractor.extract(answer)
        match = quality_controller._match_answer(extracted_answer, sample['answer'])

        result = {
            'id': sample['id'],
            'prediction': answer,
            'extracted_answer': extracted_answer,
            'expected_answer': sample['answer'],
            'match': match
        }

        print(f"样本 {sample['id']} 处理完成，匹配结果: {match}")
        return ('success', result)

    except Exception as e:
        print(f"样本 {sample['id']} 处理失败: {e}")
        error_info = {
            'sample_id': sample['id'],
            'question': sample['question'],
            'expected_answer': sample['answer'],
            'error': str(e)
        }
        return ('error', error_info)


def run_baseline(input_file, output_file, model_config, num_samples=10, max_workers=10, task_type='wtq'):
    """运行 Zero‑Shot Baseline，支持并发处理，只处理 `num_samples` 条样本。"""
    # 1️⃣ 加载并统一 key 格式
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                raw = json.loads(line)
                samples.append(normalize_sample(raw, task_type))

    # 2️⃣ 若需要，仅截取前 `num_samples` 条
    if num_samples > 0 and num_samples < len(samples):
        samples = samples[:num_samples]

    print(f"总样本数: {len(samples)}")
    print(f"并发工作线程数: {max_workers}")

    # 3️⃣ 读取已存在的输出文件，过滤已处理的 id（断点续传）
    processed_sample_ids = set()
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        data = json.loads(line)
                        if 'id' in data:
                            processed_sample_ids.add(data['id'])
                    except Exception as e:
                        print(f"解析已处理样本数据失败: {e}")

    # 4️⃣ 只保留未处理的样本进行计算
    samples_to_process = [s for s in samples if s['id'] not in processed_sample_ids]
    print(f"需要处理的新样本数: {len(samples_to_process)}")

    # 5️⃣ 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    successful_results = []
    errors = []

    # 6️⃣ 并发执行
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_sample = {executor.submit(process_sample, sample, model_config, task_type): sample for sample in samples_to_process}
        for future in as_completed(future_to_sample):
            try:
                result_type, result = future.result()
                if result_type == 'success':
                    successful_results.append(result)
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                else:
                    errors.append(result)
                    error_file = output_file.replace('.jsonl', '_errors.jsonl')
                    with open(error_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
            except Exception as e:
                print(f"任务执行失败: {e}")

    return successful_results, errors


def run_cyclic_test(input_file, model_config, num_samples=10, max_workers=10, num_cycles=1):
    """
    运行循环测试，计算平均准确率
    """
    cycle_results = []

    for cycle in range(1, num_cycles + 1):
        print(f"\n=== 循环测试 {cycle}/{num_cycles} ===\n")

        # 为每个循环创建唯一的输出文件
        cycle_output_file = f"cycle_{cycle}_predictions.jsonl"
        # 确保每次循环重新处理所有样本
        if os.path.exists(cycle_output_file):
            os.remove(cycle_output_file)
        if os.path.exists(cycle_output_file.replace('.jsonl', '_errors.jsonl')):
            os.remove(cycle_output_file.replace('.jsonl', '_errors.jsonl'))

        successful_results, errors = run_baseline(
            input_file, cycle_output_file, model_config,
            num_samples=num_samples, max_workers=max_workers
        )

        if successful_results:
            correct_predictions = sum(1 for result in successful_results if result['match'])
            accuracy = correct_predictions / len(successful_results) * 100
        else:
            accuracy = 0.0

        cycle_result = {
            'cycle': cycle,
            'total_samples': num_samples,
            'successful_samples': len(successful_results),
            'errors': len(errors),
            'accuracy': accuracy,
            'correct_predictions': sum(1 for result in successful_results if result['match']) if successful_results else 0,
            'error_predictions': len(successful_results) - sum(1 for result in successful_results if result['match']) if successful_results else 0
        }

        cycle_results.append(cycle_result)
        print(f"\n循环 {cycle} 结果: 准确率 {accuracy:.2f}%")

    return cycle_results


def save_cycle_results_to_csv(cycle_results, output_file):
    """
    将循环测试结果保存到 CSV 文件
    """
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8', newline='') as csvfile:
        fieldnames = ['cycle', 'total_samples', 'successful_samples', 'errors', 'accuracy', 'correct_predictions', 'error_predictions']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        writer.writeheader()
        for result in cycle_results:
            writer.writerow(result)

    print(f"循环测试结果已保存到: {output_file}")


def calculate_average_score(cycle_results):
    """
    计算平均分数
    """
    total_cycles = len(cycle_results)
    if total_cycles == 0:
        return {
            'average_accuracy': 0.0,
            'total_samples': 0,
            'total_successful': 0,
            'total_errors': 0,
            'total_correct': 0,
            'total_incorrect': 0
        }

    total_accuracy = sum(result['accuracy'] for result in cycle_results)
    total_samples = sum(result['total_samples'] for result in cycle_results)
    total_successful = sum(result['successful_samples'] for result in cycle_results)
    total_errors = sum(result['errors'] for result in cycle_results)
    total_correct = sum(result['correct_predictions'] for result in cycle_results)
    total_incorrect = sum(result['error_predictions'] for result in cycle_results)

    average_accuracy = total_accuracy / total_cycles

    print(f"\n=== 平均分数 ===\n")
    print(f"循环次数: {total_cycles}")
    print(f"平均准确率: {average_accuracy:.2f}%")
    print(f"总样本数: {total_samples}")
    print(f"总成功样本: {total_successful}")
    print(f"总错误数: {total_errors}")
    print(f"总正确预测: {total_correct}")
    print(f"总错误预测: {total_incorrect}")

    return {
        'average_accuracy': average_accuracy,
        'total_cycles': total_cycles,
        'total_samples': total_samples,
        'total_successful': total_successful,
        'total_errors': total_errors,
        'total_correct': total_correct,
        'total_incorrect': total_incorrect
    }


def main():
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="运行 LLM 基准测试，支持并发处理、准确率计算、重试机制和循环测试")
    parser.add_argument("--model", default="gpt-4o-mini", help="模型名称")
    parser.add_argument("--base_url", default="https://yunwu.ai/v1", help="API 端点")
    parser.add_argument("--api_key", default="sk-mBv9A5UrRCQ7OzJDPLKZkjIRzoywwVcu4wvStR4P6E7K9Kw9", help="API 密钥")
    parser.add_argument("--num_samples", type=int, default=10, help="测试样本数量")
    parser.add_argument("--output_dir", default='/Users/westmoon/mycode/table/baseline_all', help="输出目录")
    parser.add_argument("--temperature", type=float, default=0.1, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="核采样参数")
    parser.add_argument("--max_workers", type=int, default=10, help="最大并发工作线程数")
    parser.add_argument("--num_cycles", type=int, default=2, help="循环测试次数")
    parser.add_argument("--save_csv", action="store_true", help="是否保存循环结果到 CSV 文件")
    parser.add_argument("--datasets", default="", help="逗号分隔的数据集名称，只测试这些文件；不填则默认全部")



    args = parser.parse_args()

    # 模型配置
    model_config = {
        "model": args.model,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "temperature": args.temperature,
        "top_p": args.top_p
    }

    # CSV 汇总文件（追加模式）
    csv_path = os.path.join(args.output_dir, 'benchmark_summary.csv')
    # 写入表头（如果文件不存在）
    if not os.path.exists(csv_path):
        with open(csv_path, 'w', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['timestamp', 'model', 'file_type', 'accuracy'])
    # 用于后续统计的字典
    all_results = {}

    test_files = [
        '/Users/westmoon/mycode/table/processed_data/test_processed.jsonl',
    ]
    test_dir = '/Users/westmoon/mycode/table/test'
    dataset_to_path = {}
    # 将 processed_data 中的默认文件加入映射
    dataset_to_path[os.path.splitext(os.path.basename(test_files[0]))[0]] = test_files[0]
    if os.path.isdir(test_dir):
        for fname in sorted(os.listdir(test_dir)):
            if fname.endswith('.jsonl'):
                full_path = os.path.join(test_dir, fname)
                test_files.append(full_path)
                dataset_to_path[os.path.splitext(fname)[0]] = full_path
    # 根据用户指定的 datasets 进行过滤
    if args.datasets:
        requested = {name.strip() for name in args.datasets.split(',') if name.strip()}
        filtered = []
        for name in requested:
            if name in dataset_to_path:
                filtered.append(dataset_to_path[name])
            else:
                print(f"⚠ 警告：未找到数据集 '{name}'，将被忽略")
        test_files = filtered
    # 如果未指定 datasets 且 test_files 为空（极端情况），保留原始列表
    if not test_files:
        print("⚠ 警告：未找到任何待测文件，将使用默认的 processed_data 文件")
        test_files = [
            '/Users/westmoon/mycode/table/processed_data/test_processed.jsonl',
        ]

    for input_file in test_files:
        task_type = detect_task_type(input_file)
        task_name = os.path.splitext(os.path.basename(input_file))[0]

        print(f"\n{'='*60}")
        print(f"任务: {task_name} (类型: {task_type})")
        print(f"文件: {input_file}")
        print(f"{'='*60}\n")

        output_file = os.path.join(args.output_dir, f"{model_config['model']}_{task_name}_predictions.jsonl")
        if os.path.exists(output_file):
            os.remove(output_file)
        error_file = output_file.replace('.jsonl', '_errors.jsonl')
        if os.path.exists(error_file):
            os.remove(error_file)

        # 根据循环次数决定是否使用循环测试函数
        if args.num_cycles > 1:
            cycle_results = run_cyclic_test(
                input_file,
                model_config,
                num_samples=args.num_samples,
                max_workers=args.max_workers,
                num_cycles=args.num_cycles
            )
            # 直接打印每个循环的结果，而不是只打印平均值
            for cr in cycle_results:
                print(f"\n循环 {cr['cycle']} 详细结果:")
                print(f"  总样本数: {cr['total_samples']}")
                print(f"  成功样本: {cr['successful_samples']}")
                print(f"  错误样本: {cr['errors']}")
                print(f"  准确率: {cr['accuracy']:.2f}%")
                print(f"  正确预测: {cr['correct_predictions']}")
                print(f"  错误预测: {cr['error_predictions']}")
            # 如果需要汇总 CSV，仍然可以保存每个循环的结果到 CSV（追加模式）
            if args.save_csv:
                csv_file = os.path.join(args.output_dir, f"{model_config['model']}_{task_name}_cycle_results.csv")
                save_cycle_results_to_csv(cycle_results, csv_file)
            # 计算平均分数仅在需要时使用（不再自动打印）
            # avg_res = calculate_average_score(cycle_results)
            # accuracy = avg_res['average_accuracy']
        else:
            successful_results, errors = run_baseline(
                input_file,
                output_file,
                model_config,
                num_samples=args.num_samples,
                max_workers=args.max_workers,
                task_type=task_type
            )
            if successful_results:
                correct_predictions = sum(1 for result in successful_results if result['match'])
                accuracy = correct_predictions / len(successful_results) * 100
                print(f"\n--- {task_name} 结果 ---")
                print(f"成功: {len(successful_results)}, 失败: {len(errors)}, 准确率: {accuracy:.2f}%")
            else:
                accuracy = 0.0
                print(f"\n--- {task_name}: 没有成功的测试结果 ---")
                errors = []


        # 在写入 CSV 前确保 accuracy 已定义
        if args.num_cycles > 1:
            # 计算循环的平均准确率，供 CSV 记录使用
            accuracy = sum(cr['accuracy'] for cr in cycle_results) / len(cycle_results) if cycle_results else 0.0
        # 记录单个任务的统计到 CSV（追加模式）
        with open(csv_path, 'a', encoding='utf-8', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([datetime.now().isoformat(), args.model, task_type, f"{accuracy:.2f}"])

        all_results[task_name] = {
            'task_type': task_type,
            'successful': len(successful_results) if args.num_cycles == 1 else sum(r['successful_samples'] for r in cycle_results),
            'errors': len(errors) if args.num_cycles == 1 else sum(r['errors'] for r in cycle_results),
            'accuracy': accuracy
        }

    # 汇总所有任务结果
    print(f"\n{'='*60}")
    print(f"所有任务汇总")
    print(f"{'='*60}")
    for task_name, result in all_results.items():
        print(f"  {task_name:40s} | 类型: {result['task_type']:10s} | 准确率: {result['accuracy']:.2f}%")
    print(f"\n=== 全部测试完成 ===")


if __name__ == "__main__":
    main()
