#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Stage 4: Quality Control and Testing Module
"""

import random
import re
import sys
import os
import json

# Add WikiTableQuestions directory to Python path
wiki_table_questions_path = '/Users/westmoon/mycode/table/WikiTableQuestions'
if wiki_table_questions_path not in sys.path:
    sys.path.append(wiki_table_questions_path)

from sft_generator.refine_critic import CriticRefine
from evaluator import to_value, to_value_list, check_denotation


class QualityController:
    """
    Quality controller class
    """

    def __init__(self, extractor, llm_api=None):
        """
        Initialize quality controller

        Args:
            extractor: Answer extractor
            llm_api: LLM API instance
        """
        self.extractor = extractor
        self.critic_refine = CriticRefine(llm_api) if llm_api is not None else None

    def quality_control(self, generator, markdown_table, question, expected_answer, max_attempts=3):
        """
        Quality control (multiple sampling and verification) with critic-refine

        Args:
            generator: SFTDataGenerator instance
            markdown_table: Markdown table
            question: Question
            expected_answer: Expected answer
            max_attempts: Maximum number of attempts

        Returns:
            All attempts results
        """
        attempts = []

        for attempt in range(1, max_attempts + 1):
            print(f"Attempt {attempt}...")

            try:
                if attempt == 1:
                    # 第一次尝试，直接生成响应
                    response = generator.generate_single_response(markdown_table, question)
                else:
                    # 后续尝试，使用前一次尝试的优化结果
                    previous_attempt = attempts[-1]
                    if previous_attempt['error_analysis'] and 'response' in previous_attempt['error_analysis']:
                        # 使用前一次尝试的 error analysis 中优化后的 response
                        response = previous_attempt['error_analysis']['response']
                        print(f"Using refined response from attempt {attempt - 1}")
                    else:
                        # 如果前一次没有优化后的 response，重新生成
                        response = generator.generate_single_response(markdown_table, question)

                # 提取答案并检查是否匹配
                extracted_answer = self.extractor.extract(response['final_response'])
                match = self._match_answer(extracted_answer, expected_answer)

                if match:
                    # 匹配成功，停止尝试
                    attempt_result = {
                        "attempt": attempt,
                        "response": response,
                        "extracted_answer": extracted_answer,
                        "match": match,
                        "error_analysis": None
                    }
                    attempts.append(attempt_result)
                    print("Found matching answer, stop generating")
                    break
                else:
                    # 匹配失败，进行refine
                    error_analysis = None
                    if self.critic_refine is not None:
                        error_analysis = self.critic_refine.refine_response(
                            question, markdown_table, response['decomposition'],
                            response['sub_question_results'], response['final_response'], expected_answer
                        )
                    attempt_result = {
                        "attempt": attempt,
                        "response": response,
                        "extracted_answer": extracted_answer,
                        "match": match,
                        "error_analysis": error_analysis
                    }
                    attempts.append(attempt_result)
                    print(f"Answer does not match, attempting attempt {attempt + 1}")

            except Exception as e:
                print(f"Generation failed: {e}")
                attempts.append({
                    "attempt": attempt,
                    "response": None,
                    "extracted_answer": None,
                    "match": False,
                    "error_analysis": None
                })

        return attempts

    def _get_improved_question(self, question, previous_errors, attempt):
        """
        Get improved question based on previous errors

        Args:
            question: Original question
            previous_errors: Previous error analyses
            attempt: Current attempt number

        Returns:
            Improved question
        """
        if attempt == 1 or not previous_errors or self.critic_refine is None:
            return question

        # Generate improved prompt using previous error analysis
        previous_error = previous_errors[-1]
        if previous_error and previous_error.get('analysis'):
            analysis = previous_error['analysis']

            prompt = (
                f"Question: {question}\n\n"
                f"Previous error analysis: {analysis.get('type', 'unknown error')} - {analysis.get('reason', 'no reason specified')}\n"
                f"Suggestion: {analysis.get('suggestion', 'no suggestion')}\n\n"
                f"Based on the above analysis, re-optimize the question description to make it more helpful for the model to answer accurately."
            )

            if self.critic_refine.llm_api is not None:
                try:
                    return self.critic_refine.llm_api.call(prompt)
                except Exception as e:
                    print(f"Failed to generate improved question: {e}")

        return question

    def _match_answer(self, extracted_answer, expected_answer):
        """
        Match answers using the official evaluation script's logic

        Args:
            extracted_answer: Extracted answer
            expected_answer: Expected answer

        Returns:
            Whether the answers match
        """
        if not extracted_answer or not expected_answer:
            return False

        # 将答案转换为官方评估脚本支持的 Value 类型
        target_values = to_value_list([str(expected_answer)])
        predicted_values = to_value_list([str(extracted_answer)])

        # 使用官方评估脚本的 check_denotation 方法进行匹配
        return check_denotation(target_values, predicted_values)


class TestRunner:
    """
    Test runner class
    """

    def __init__(self, generator, controller):
        """
        Initialize test runner

        Args:
            generator: SFTDataGenerator instance
            controller: QualityController instance
        """
        self.generator = generator
        self.controller = controller

    def test_small_scale(self, dataset_file, num_samples=10, sample_id=None, max_workers=1):
        """
        Small scale test

        Args:
            dataset_file: Dataset file path
            num_samples: Number of test samples
            sample_id: Specify sample ID to test
            max_workers: Number of concurrent working threads

        Returns:
            Test results and statistics
        """
        import time
        from concurrent.futures import ThreadPoolExecutor, as_completed

        print(f"Starting small scale test,抽取 {num_samples} samples...")
        test_data = self._load_small_scale_test_data(dataset_file, num_samples, sample_id)

        stats = {
            "total_samples": len(test_data),
            "successful_samples": 0,
            "total_attempts": 0,
            "successful_attempts": 0,
            "python_solver_ratio": 0,
            "llm_solver_ratio": 0,
            "average_sub_questions": 0,
            "error_types": {},
            "average_attempts_per_sample": 0,
            "total_time": 0,
            "average_time_per_sample": 0,
            "success_attempt_counts": []
        }

        all_attempts = []
        errors = []

        # 封装每个样本的测试逻辑
        def process_sample(sample):
            start_time = time.time()
            print(f"\n测试样本: {sample['id']}")
            print(f"问题: {sample['question']}")
            print(f"预期答案: {sample['answer']}")

            attempts = self.controller.quality_control(
                self.generator,
                sample['table'],
                sample['question'],
                sample['answer']
            )

            sample_time = time.time() - start_time
            return sample, attempts, sample_time

        # 使用线程池并发执行测试
        total_start_time = time.time()
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # 提交所有任务
            future_to_sample = {executor.submit(process_sample, sample): sample for sample in test_data}

            # 收集结果
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    sample, attempts, sample_time = future.result()
                    all_attempts.append((sample, attempts))
                    stats['total_time'] += sample_time

                    # 更新统计信息
                    stats['total_attempts'] += len(attempts)
                    successful_attempts = sum(1 for attempt in attempts if attempt['match'])
                    stats['successful_attempts'] += successful_attempts

                    if successful_attempts > 0:
                        stats['successful_samples'] += 1
                        # 记录成功尝试的次数
                        for i, attempt in enumerate(attempts, 1):
                            if attempt['match']:
                                stats['success_attempt_counts'].append(i)
                                break

                    # 统计求解方式和错误类型
                    if len(attempts) > 0 and attempts[0]['response']:
                        sub_questions = attempts[0]['response']['decomposition']
                        stats['average_sub_questions'] += len(sub_questions)

                        python_count = sum(1 for q in sub_questions if q['strategy'] == 'Python')
                        llm_count = sum(1 for q in sub_questions if q['strategy'] == 'LLM')

                        total = python_count + llm_count
                        if total > 0:
                            stats['python_solver_ratio'] += python_count / total
                            stats['llm_solver_ratio'] += llm_count / total

                    # 统计错误类型
                    for attempt in attempts:
                        if 'error_analysis' in attempt and attempt['error_analysis'] and 'analysis' in attempt['error_analysis']:
                            error_type = attempt['error_analysis']['analysis']['type']
                            stats['error_types'][error_type] = stats['error_types'].get(error_type, 0) + 1
                    # 只在所有尝试都失败时记录错误
                    if not any(attempt['match'] for attempt in attempts):
                        error_info = {
                            'sample_id': sample['id'],
                            'question': sample['question'],
                            'expected_answer': sample['answer'],
                            'total_attempts': len(attempts),
                            'attempts': []
                        }
                        for attempt in attempts:
                            attempt_info = {
                                'attempt': attempt['attempt'],
                                'extracted_answer': attempt['extracted_answer'],
                                'response': attempt['response'],
                                'error_analysis': attempt['error_analysis']
                            }
                            error_info['attempts'].append(attempt_info)
                        errors.append(error_info)
                except Exception as e:
                    print(f"处理样本 {sample['id']} 时出错: {e}")

        stats['total_time'] = time.time() - total_start_time

        # 计算平均
        if stats['total_samples'] > 0:
            stats['average_sub_questions'] /= stats['total_samples']
            stats['python_solver_ratio'] /= stats['total_samples']
            stats['llm_solver_ratio'] /= stats['total_samples']
            stats['average_attempts_per_sample'] = stats['total_attempts'] / stats['total_samples']
            stats['average_time_per_sample'] = stats['total_time'] / stats['total_samples']

        # 计算平均成功尝试次数
        if stats['successful_samples'] > 0:
            stats['average_success_attempts'] = sum(stats['success_attempt_counts']) / stats['successful_samples']

        # 计算匹配率
        stats['sample_match_rate'] = stats['successful_samples'] / stats['total_samples']
        stats['attempt_match_rate'] = stats['successful_attempts'] / stats['total_attempts'] if stats['total_attempts'] > 0 else 0

        # 保存错误结果到文件
        if errors:
            error_file = 'test_errors.jsonl'
            with open(error_file, 'w', encoding='utf-8') as f:
                for error in errors:
                    f.write(json.dumps(error, ensure_ascii=False) + '\n')
            print(f"\n错误结果已保存到 {error_file}，共 {len(errors)} 个错误")

        # 打印详细统计信息
        print("\n=== 详细统计信息 ===")
        print(f"总样本数: {stats['total_samples']}")
        print(f"成功样本数: {stats['successful_samples']}")
        print(f"样本级匹配率: {stats['sample_match_rate']:.2%}")
        print(f"总尝试次数: {stats['total_attempts']}")
        print(f"成功尝试次数: {stats['successful_attempts']}")
        print(f"生成级匹配率: {stats['attempt_match_rate']:.2%}")
        print(f"平均子问题数量: {stats['average_sub_questions']:.1f}")
        print(f"Python 求解占比: {stats['python_solver_ratio']:.2%}")
        print(f"LLM 求解占比: {stats['llm_solver_ratio']:.2%}")
        print(f"平均尝试次数: {stats['average_attempts_per_sample']:.1f}")
        print(f"平均成功尝试次数: {stats.get('average_success_attempts', 0):.1f}")
        print(f"总运行时间: {stats['total_time']:.2f} 秒")
        print(f"平均运行时间: {stats['average_time_per_sample']:.2f} 秒")

        # 保存成功结果（包含完整的尝试过程）
        successful_results = []
        for sample, attempts in all_attempts:
            # 检查样本是否有成功的尝试
            has_successful_attempt = any(attempt['match'] for attempt in attempts)
            if has_successful_attempt:
                result = {
                    'id': sample['id'],
                    'table': sample['table'],
                    'question': sample['question'],
                    'expected_answer': sample['answer'],
                    'total_attempts': len(attempts),
                    'attempts': []
                }
                for attempt in attempts:
                    attempt_info = {
                        'attempt': attempt['attempt'],
                        'response': attempt['response'],
                        'extracted_answer': attempt['extracted_answer'],
                        'match': attempt['match']
                    }
                    if 'error_analysis' in attempt and attempt['error_analysis'] is not None:
                        attempt_info['error_analysis'] = attempt['error_analysis']
                    result['attempts'].append(attempt_info)
                successful_results.append(result)

        if successful_results:
            output_file = 'successful_results.jsonl'
            with open(output_file, 'w', encoding='utf-8') as f:
                for result in successful_results:
                    f.write(json.dumps(result, ensure_ascii=False) + '\n')
            print(f"\n成功结果已保存到 {output_file}，共 {len(successful_results)} 个")

        return stats, test_data, all_attempts

    def _load_small_scale_test_data(self, dataset_file, num_samples=10, sample_id=None):
        """
        加载小规模测试数据

        Args:
            dataset_file: 数据集文件路径
            num_samples: 测试样本数量
            sample_id: 指定要测试的样本 ID

        Returns:
            测试数据列表
        """
        import json
        import os

        all_samples = []
        try:
            with open(dataset_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        data = json.loads(line)
                        if 'id' in data and 'question' in data and 'table' in data and 'answer' in data:
                            if sample_id is None or data['id'] == sample_id:
                                all_samples.append(data)
        except Exception as e:
            print(f"加载测试数据失败: {e}")
            return []

        if len(all_samples) > num_samples:
            return random.sample(all_samples, num_samples)

        return all_samples
