#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT 数据生成器的主入口文件
"""

import os
import json
import random
import concurrent.futures
from typing import List, Dict, Any, Optional

from sft_generator.base import LLMAPI, OpenAICompatibleAPI, AnswerExtractor, SimpleAnswerExtractor, TableConverter
from sft_generator.decomposition import QuestionDecomposer
from sft_generator.sub_question_solver import SubQuestionSolver
from sft_generator.answer_synthesizer import AnswerSynthesizer
from sft_generator.quality_control import QualityController, TestRunner
from sft_generator.refine_critic import CriticRefine


class SFTDataGenerator:
    """
    负责生成 SFT 训练数据的主类
    """

    def __init__(self, model_config: Dict[str, Any], llm_api: Optional[LLMAPI] = None,
                 extractor: Optional[AnswerExtractor] = None):
        """
        初始化 SFTDataGenerator

        Args:
            model_config: 模型配置
            llm_api: LLM API 实例
            extractor: 答案提取器实例
        """
        self.model_config = model_config
        self.llm_api = llm_api or OpenAICompatibleAPI(model_config)
        self.extractor = extractor or SimpleAnswerExtractor()

        self.decomposer = QuestionDecomposer(self.llm_api)
        self.solver = SubQuestionSolver(self.llm_api)
        self.synthesizer = AnswerSynthesizer(self.llm_api)
        self.quality_controller = QualityController(self.extractor, self.llm_api)
        self.test_runner = TestRunner(self, self.quality_controller)
        self.critic_refine = CriticRefine(self.llm_api) if llm_api is not None else None

    def generate_single_response(self, markdown_table: str, question: str) -> Dict[str, Any]:
        """
        单次生成完整响应的主流程

        Args:
            markdown_table: Markdown 表格
            question: 问题

        Returns:
            生成的响应
        """
        try:
            # 阶段 1：问题分解与规划
            decomposition = self.decomposer.decompose_and_plan(markdown_table, question)

            # 转换表格为 DataFrame（只执行一次）
            df = TableConverter.markdown_to_dataframe(markdown_table)
            if df is None:
                return {
                    "decomposition": decomposition,
                    "sub_question_results": [],
                    "final_response": "表格转换失败"
                }

            # 阶段 2：子问题求解（串行处理，确保下一个子问题能获得上一个的答案）
            sub_question_results = self._solve_sub_questions_serially(df, decomposition)

            # 阶段 3：答案汇总
            final_response = self.synthesizer.synthesize_answer(decomposition, sub_question_results, question)

            return {
                "decomposition": decomposition,
                "sub_question_results": sub_question_results,
                "final_response": final_response
            }
        except Exception as e:
            print(f"生成过程出错: {e}")
            import traceback
            print(f"详细错误信息: {traceback.format_exc()}")
            return {
                "decomposition": [],
                "sub_question_results": [],
                "final_response": f"生成过程出错: {e}"
            }

    def _solve_sub_questions_serially(self, df: Any, decomposition: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        串行求解子问题（确保下一个子问题能获得所有之前子问题的答案和问题）

        Args:
            df: 数据表
            decomposition: 问题分解结果

        Returns:
            子问题求解结果
        """
        sub_question_results = []
        context_info = []

        for i, item in enumerate(decomposition):
            sub_question = item['sub_question']
            strategy = item['strategy']

            # 如果是后续子问题，将所有之前子问题的答案和问题传递进去
            if i > 0:
                context_str = "Based on the previous information:\n"
                for j in range(i):
                    prev_item = decomposition[j]
                    prev_result = sub_question_results[j]
                    context_str += (
                        f"Previous question {j+1}: \"{prev_item['sub_question']}\" "
                        f"Answer: \"{prev_result['answer']}\"\n"
                    )
                context_sub_question = context_str + f"Now answer the following question: {sub_question}"
            else:
                context_sub_question = sub_question

            try:
                result = self.solver.solve_sub_question(df, context_sub_question, strategy)
                result['sub_question'] = item['sub_question']
                result['strategy'] = item['strategy']
                sub_question_results.append(result)
                # 保存上下文信息
                context_info.append({
                    "sub_question": item['sub_question'],
                    "answer": result['answer']
                })
            except Exception as e:
                print(f"求解子问题 '{item['sub_question']}' 时出错: {e}")
                sub_question_results.append({
                    "sub_question": item['sub_question'],
                    "strategy": item['strategy'],
                    "data_flow": "",
                    "answer": "求解失败"
                })
                context_info.append({
                    "sub_question": item['sub_question'],
                    "answer": "求解失败"
                })

        return sub_question_results

    def quality_control(self, markdown_table: str, question: str, expected_answer: str,
                       max_attempts: int = 3) -> List[Dict[str, Any]]:
        """
        质量控制（多次采样与验证）

        Args:
            markdown_table: Markdown 表格
            question: 问题
            expected_answer: 预期答案
            max_attempts: 最大尝试次数

        Returns:
            所有尝试结果
        """
        return self.quality_controller.quality_control(self, markdown_table, question, expected_answer, max_attempts)

    def test_small_scale(self, dataset_file: str, num_samples: int = 10, sample_id: Optional[str] = None) -> Dict[str, Any]:
        """
        小规模测试

        Args:
            dataset_file: 数据集文件路径
            num_samples: 测试样本数量
            sample_id: 指定要测试的样本 ID

        Returns:
            测试结果
        """
        return self.test_runner.test_small_scale(dataset_file, num_samples, sample_id)

    def generate_training_data(self, input_file: str, output_file: str, num_samples: Optional[int] = None,
                               max_attempts: int = 3, sample_id: Optional[str] = None,
                               start_index: Optional[int] = None, end_index: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        生成训练数据的主函数

        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径
            num_samples: 处理的样本数量（None 表示处理全部）
            max_attempts: 每个样本的最大尝试次数
            sample_id: 指定要处理的样本 ID

        Returns:
            生成的训练数据
        """
        successful_results = []
        errors = []

        # 统计信息
        total_python_uses = 0
        total_critic_uses = 0
        total_attempts = 0
        successful_attempts = 0

        samples = self._load_samples(input_file, num_samples, sample_id, start_index, end_index)

        # 定义错误文件路径
        error_file = output_file.replace('.jsonl', '_errors.jsonl')

        # 检查输出文件和错误文件是否已经存在，如果存在则读取已处理的样本 ID
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
        if os.path.exists(error_file):
            with open(error_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            if 'sample_id' in data:
                                processed_sample_ids.add(data['sample_id'])
                            elif 'id' in data:
                                processed_sample_ids.add(data['id'])
                        except Exception as e:
                            print(f"解析错误文件样本数据失败: {e}")
        print(f"已处理的样本数量: {len(processed_sample_ids)}")

        # 过滤掉已经处理过的样本
        samples = [sample for sample in samples if sample['id'] not in processed_sample_ids]

        print(f"开始生成训练数据，共处理 {len(samples)} 个新样本...")

        # 确保输出目录存在
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

        # 并发处理样本
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(samples), 100)) as executor:
            future_to_sample = {
                executor.submit(self._process_single_sample_with_independent_context, sample, max_attempts): sample
                for sample in samples
            }

            for i, future in enumerate(concurrent.futures.as_completed(future_to_sample)):
                sample = future_to_sample[future]
                try:
                    attempts = future.result()

                    # 更新统计信息
                    total_attempts += len(attempts)
                    sample_successful_attempts = sum(1 for attempt in attempts if attempt['match'])
                    successful_attempts += sample_successful_attempts

                    # 统计 Python 使用次数和 Critic 使用次数
                    for attempt in attempts:
                        if attempt['response'] and 'sub_question_results' in attempt['response']:
                            python_uses = sum(1 for result in attempt['response']['sub_question_results'] if result['strategy'] == 'Python')
                            total_python_uses += python_uses

                        if 'error_analysis' in attempt and attempt['error_analysis']:
                            total_critic_uses += 1

                    # 检查样本是否有成功的尝试
                    has_successful_attempt = any(attempt['match'] for attempt in attempts)
                    if has_successful_attempt:
                        # 保存成功的结果（包含完整的尝试过程）
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
                        # 立即写入到文件
                        with open(output_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    else:
                        # 保存错误结果
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
                                'response': attempt['response'],
                                'extracted_answer': attempt['extracted_answer'],
                                'match': attempt['match']
                            }
                            if 'error_analysis' in attempt and attempt['error_analysis'] is not None:
                                attempt_info['error_analysis'] = attempt['error_analysis']
                            error_info['attempts'].append(attempt_info)
                        errors.append(error_info)
                        # 立即写入到错误文件
                        with open(error_file, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(error_info, ensure_ascii=False) + '\n')
                    print(f"已处理样本 {i + 1}/{len(samples)}: {sample['id']}")
                except Exception as e:
                    print(f"处理样本 {sample['id']} 时出错: {e}")

        # 输出错误信息
        if errors:
            print(f"\n错误结果已保存到 {error_file}，共 {len(errors)} 个错误")

        # 输出统计数据
        print("\n=== 训练数据生成统计 ===\n")
        print(f"总样本数: {len(samples)}")
        print(f"成功样本数: {len(successful_results)}")
        print(f"失败样本数: {len(errors)}")
        print(f"样本级成功率: {len(successful_results) / len(samples) * 100:.2f}%")
        print(f"总尝试次数: {total_attempts}")
        print(f"成功尝试次数: {successful_attempts}")
        print(f"生成级成功率: {successful_attempts / total_attempts * 100:.2f}%" if total_attempts > 0 else "生成级成功率: 0%")
        print(f"平均尝试次数: {total_attempts / len(samples):.1f}" if len(samples) > 0 else "平均尝试次数: 0")
        print(f"Python 求解次数: {total_python_uses}")
        print(f"Critic 使用次数: {total_critic_uses}")
        print(f"平均每个样本 Python 求解次数: {total_python_uses / len(samples):.1f}" if len(samples) > 0 else "平均每个样本 Python 求解次数: 0")
        print(f"平均每个样本 Critic 使用次数: {total_critic_uses / len(samples):.1f}" if len(samples) > 0 else "平均每个样本 Critic 使用次数: 0")

        return successful_results

    def _load_samples(self, input_file: str, num_samples: Optional[int] = None, sample_id: Optional[str] = None, start_index: Optional[int] = None, end_index: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        加载样本

        Args:
            input_file: 输入文件路径
            num_samples: 加载的样本数量
            sample_id: 指定要加载的样本 ID
            start_index: 起始样本索引
            end_index: 结束样本索引

        Returns:
            样本列表
        """
        samples = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    data = json.loads(line)
                    if sample_id is None or data['id'] == sample_id:
                        samples.append(data)

        # 处理起始索引和结束索引
        if start_index is not None or end_index is not None:
            # 确保索引在有效范围内
            if start_index is None:
                start_index = 0
            if end_index is None:
                end_index = len(samples)
            # 截取指定范围的样本
            samples = samples[start_index:end_index]
        elif sample_id is None and num_samples and num_samples < len(samples):
            # 如果没有指定起始和结束索引，但指定了样本数量，则随机采样
            samples = random.sample(samples, num_samples)

        return samples

    def _process_single_sample(self, sample: Dict[str, Any], max_attempts: int) -> List[Dict[str, Any]]:
        """
        处理单个样本

        Args:
            sample: 样本数据
            max_attempts: 最大尝试次数

        Returns:
            该样本的所有处理结果
        """
        attempts = self.quality_control(
            sample['table'],
            sample['question'],
            sample['answer'],
            max_attempts
        )

        # 格式化结果
        results = []
        for attempt in attempts:
            result = {
                'id': sample['id'],
                'table': sample['table'],
                'question': sample['question'],
                'expected_answer': sample['answer'],
                'attempt': attempt['attempt'],
                'response': attempt['response'],
                'extracted_answer': attempt['extracted_answer'],
                'match': attempt['match']
            }

            if 'error_analysis' in attempt and attempt['error_analysis'] is not None:
                result['error_analysis'] = attempt['error_analysis']

            results.append(result)

        return results

    @staticmethod
    def _save_results(results: List[Dict[str, Any]], output_file: str) -> None:
        """
        保存训练数据

        Args:
            results: 训练数据
            output_file: 输出文件路径
        """
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f"结果保存到 {output_file}")

    def _process_single_sample_with_independent_context(self, sample: Dict[str, Any], max_attempts: int) -> List[Dict[str, Any]]:
        """
        使用独立上下文处理单个样本，确保线程安全

        Args:
            sample: 样本数据
            max_attempts: 最大尝试次数

        Returns:
            该样本的所有处理结果
        """
        # 创建独立的数据生成器实例，确保所有组件都是独立的
        # 重用同一个 llm_api 实例，因为它是线程安全的
        independent_generator = SFTDataGenerator(
            self.model_config,
            llm_api=self.llm_api,  # 重用同一个 llm_api 实例
            extractor=self.extractor
        )
        # 使用独立上下文处理样本
        attempts = independent_generator.quality_control(
            sample['table'],
            sample['question'],
            sample['answer'],
            max_attempts
        )

        # 格式化结果
        results = []
        for attempt in attempts:
            result = {
                'id': sample['id'],
                'table': sample['table'],
                'question': sample['question'],
                'expected_answer': sample['answer'],
                'attempt': attempt['attempt'],
                'response': attempt['response'],
                'extracted_answer': attempt['extracted_answer'],
                'match': attempt['match']
            }

            if 'error_analysis' in attempt and attempt['error_analysis'] is not None:
                result['error_analysis'] = attempt['error_analysis']

            results.append(result)

        return results
