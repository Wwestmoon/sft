#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接使用 LLM 进行蒸馏的基线实验脚本，包含 critic refine 功能
"""

import os
import json
import argparse
import re
from typing import Dict, Any
from sft_generator.base import OpenAICompatibleAPI, SimpleAnswerExtractor, LLMAPI
from sft_generator.quality_control import QualityController
from sft_generator.main import SFTDataGenerator


def build_distillation_prompt(markdown_table, question):
    """
    构建用于蒸馏的 prompt，要求详细回答和推理过程
    """
    prompt = (
        f"Please analyze the following table and answer the question in detail:\n\n{markdown_table}\n\n"
        f"Question: {question}\n\n"
        f"Your response should include:\n"
        f"1. A clear answer to the question\n"
        f"2. A detailed reasoning process showing how you arrived at the answer\n"
        f"3. Any relevant calculations or data points from the table\n"
        f"\nPlease provide a comprehensive and thorough response. "
        f"At the very end of your response, please include the final answer in the format: 'Final Answer: [your answer]'."
    )
    return prompt


def direct_question_answering(llm_api, markdown_table, question):
    """
    直接使用 LLM 回答问题，获取详细响应用于蒸馏
    """
    prompt = build_distillation_prompt(markdown_table, question)
    try:
        response = llm_api.call(prompt)
        return {
            "final_response": response
        }
    except Exception as e:
        print(f"LLM 调用失败: {e}")
        return {
            "final_response": "error"
        }


class ErrorAnalyzer:
    """
    错误分析器类，用于分析生成响应中的错误
    """

    def __init__(self, llm_api):
        """
        初始化错误分析器

        Args:
            llm_api: LLM API 实例
        """
        self.llm_api = llm_api
        self.answer_extractor = SimpleAnswerExtractor()

    def analyze_error(self, question: str, table: str, final_response: str, expected_answer: str) -> Dict[str, Any]:
        """
        分析生成响应中的错误

        Args:
            question: 问题
            table: 表格内容
            final_response: 最终响应
            expected_answer: 预期答案

        Returns:
            错误分析结果
        """
        if self.llm_api is not None:
            return self._analyze_with_llm(question, table, final_response, expected_answer)
        else:
            return {
                "reason": "Cannot analyze error (LLM API not configured)",
                "suggestion": "Provide more context or configure LLM API for detailed analysis"
            }

    def _analyze_with_llm(self, question: str, table: str, final_response: str, expected_answer: str) -> Dict[str, Any]:
        """
        使用 LLM 分析错误

        Args:
            question: 问题
            table: 表格内容
            final_response: 最终响应
            expected_answer: 预期答案

        Returns:
            错误分析结果
        """
        prompt = (
            f"Please analyze the error in the generated response for the following question.\n\n"
            f"Question: {question}\n"
            f"Table:\n{table}\n"
            f"Final Response: {final_response}\n"
            f"Expected Answer: {expected_answer}\n\n"
            f"Your analysis should include:\n"
            f"1. The exact reason why the response is incorrect\n"
            f"2. A very specific and actionable suggestion on how to improve the next attempt to avoid this error:\n"
            f"   - Your suggestion should be detailed enough that it can be directly used to improve the next attempt\n"
            f"   - Focus on what the model should do differently in the next attempt\n"
            f"   - For example: 'The model should carefully check the table for the value corresponding to 2019'\n\n"
            f"Format your response as JSON with the following fields:\n"
            f"- reason: detailed error reason\n"
            f"- suggestion: specific improvement suggestion"
        )

        try:
            analysis = self.llm_api.call(prompt)
            return self._parse_analysis(analysis)
        except Exception as e:
            print(f"Error analysis failed: {e}")
            return {
                "reason": "Failed to parse LLM analysis result",
                "suggestion": "Check the response format of the LLM API"
            }

    def _parse_analysis(self, analysis: str) -> Dict[str, Any]:
        """
        解析 LLM 返回的分析结果

        Args:
            analysis: LLM 返回的分析结果

        Returns:
            解析后的分析结果
        """
        try:
            json_match = re.search(r'\{.*\}', analysis, re.DOTALL)
            if json_match:
                analysis_json = json.loads(json_match.group())
                return analysis_json
        except Exception as e:
            print(f"Failed to parse analysis JSON: {e}")

        return {
            "reason": "Failed to parse LLM analysis result",
            "suggestion": "Check the response format of the LLM API"
        }


class ErrorFixer:
    """
    Error fixer class for fixing response generation based on error analysis results
    """

    def __init__(self, llm_api: LLMAPI = None):
        """
        Initialize error fixer

        Args:
            llm_api: LLM API instance
        """
        self.llm_api = llm_api
        self.answer_extractor = SimpleAnswerExtractor()

    def fix_response(self, question: str, table: str, generated_response: str,
                     expected_answer: str, error_analysis: Dict[str, Any]) -> str:
        """
        Fix generated response based on error analysis results

        Args:
            question: Question
            table: Table
            generated_response: Generated response
            expected_answer: Expected answer
            error_analysis: Error analysis result

        Returns:
            Fixed response
        """
        if self.llm_api is not None:
            return self._fix_with_llm(question, table, generated_response, expected_answer, error_analysis)
        else:
            return self._fix_with_rules(question, table, generated_response, expected_answer)

    def _fix_with_rules(self, question: str, table: str, generated_response: str,
                        expected_answer: str, error_analysis: Dict[str, Any]) -> str:
        """
        Fix response using rules

        Args:
            question: Question
            table: Table
            generated_response: Generated response
            expected_answer: Expected answer
            error_analysis: Error analysis result

        Returns:
            Fixed response
        """
        # 直接替换答案，不检查错误类型
        extracted_answer = self.answer_extractor.extract(generated_response)
        if extracted_answer and extracted_answer != expected_answer:
            return generated_response.replace(extracted_answer, expected_answer)
        return generated_response

    def _fix_with_llm(self, question: str, table: str, generated_response: str,
                      expected_answer: str, error_analysis: Dict[str, Any]) -> str:
        """
        使用 LLM 修复响应

        Args:
            question: 问题
            table: 表格内容
            generated_response: 生成的响应
            expected_answer: 预期答案
            error_analysis: 错误分析结果

        Returns:
            修复后的响应
        """
        prompt = (
            f"Please fix the generated response based on the error analysis and provide an improved answer.\n\n"
            f"Question: {question}\n"
            f"Table:\n{table}\n"
            f"Generated Response: {generated_response}\n"
            f"Expected Answer: {expected_answer}\n"
            f"Error Reason: {error_analysis.get('reason', '')}\n"
            f"Improvement Suggestion: {error_analysis.get('suggestion', '')}\n\n"
            f"Your improved response should:\n"
            f"1. Address the error identified in the analysis\n"
            f"2. Provide a clear and correct answer\n"
            f"3. Follow the same structure as the original response\n"
            f"4. Be concise and focused on the question\n"
            f"5. End with the correct final answer in the exact format: 'Final Answer: {expected_answer}'"
        )

        try:
            return self.llm_api.call(prompt)
        except Exception as e:
            print(f"Failed to fix response: {e}")
            return self._fix_with_rules(question, table, generated_response, expected_answer, error_analysis)


class CriticRefine:
    """
    Critic-Refine framework class for optimizing generated responses through criticism and improvement
    """

    def __init__(self, llm_api: LLMAPI = None):
        """
        Initialize Critic-Refine framework

        Args:
            llm_api: LLM API instance
        """
        self.llm_api = llm_api
        self.error_analyzer = ErrorAnalyzer(llm_api)
        self.error_fixer = ErrorFixer(llm_api)

    def refine_response(self, question: str, table: str, final_response: str, expected_answer: str) -> Dict[str, Any]:
        """
        Criticize and improve generated response

        Args:
            question: 问题
            table: 表格内容
            final_response: 最终响应
            expected_answer: 预期答案

        Returns:
            Criticism and improvement result
        """
        error_analysis = self.error_analyzer.analyze_error(
            question, table, final_response, expected_answer
        )

        refined_response = self.error_fixer.fix_response(
            question, table, final_response, expected_answer, error_analysis
        )

        return {
            "original": final_response,
            "analysis": error_analysis,
            "refined": refined_response
        }


def run_baseline_experiment(input_file, output_file, model_config, num_samples=None, start_index=None, end_index=None, max_attempts=3):
    """
    运行基线实验，包含 critic refine 功能
    """
    # 初始化共享的 API 和组件
    llm_api = OpenAICompatibleAPI(model_config)
    extractor = SimpleAnswerExtractor()
    quality_controller = QualityController(extractor)
    critic_refine = CriticRefine(llm_api)

    # 加载样本数据
    samples = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))

    # 处理样本范围
    if start_index is not None or end_index is not None:
        start_idx = start_index if start_index is not None else 0
        end_idx = end_index if end_index is not None else len(samples)
        samples = samples[start_idx:end_idx]
    elif num_samples is not None and num_samples < len(samples):
        samples = samples[:num_samples]

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

    print(f"开始处理 {len(samples)} 个新样本...")

    # 确保输出目录存在
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    successful_results = []
    errors = []

    # 并发处理样本，为每个任务创建独立的上下文
    from concurrent.futures import ThreadPoolExecutor, as_completed

    # 定义任务处理函数
    def process_sample(sample):
        """
        处理单个样本，使用共享的 API 和组件，包含 critic refine 功能
        """
        print(f"正在处理样本: {sample['id']}")

        attempts = []
        final_response = None
        extracted_answer = None
        match = False

        for attempt_num in range(1, max_attempts + 1):
            print(f"  尝试 {attempt_num}...")
            try:
                if attempt_num == 1:
                    # 第一次尝试，直接使用 LLM 回答问题
                    response = direct_question_answering(llm_api, sample['table'], sample['question'])
                else:
                    # 后续尝试，使用 critic refine 方法
                    refined_result = critic_refine.refine_response(
                        sample['question'], sample['table'], final_response, sample['answer']
                    )
                    response = {'final_response': refined_result['refined']}
                    print(f"  使用 critic refine 改进了响应")

                final_response = response['final_response']
                extracted_answer = extractor.extract(final_response)
                match = quality_controller._match_answer(extracted_answer, sample['answer'])

                # 保存尝试结果
                attempt_info = {
                    'attempt': attempt_num,
                    'response': final_response,
                    'extracted_answer': extracted_answer,
                    'match': match,
                    'error_analysis': None
                }

                if attempt_num > 1:
                    # 对于非第一次尝试，保存 error_analysis
                    refined_result = critic_refine.refine_response(
                        sample['question'], sample['table'], final_response, sample['answer']
                    )
                    attempt_info['error_analysis'] = refined_result['analysis']

                attempts.append(attempt_info)

                if match:
                    result = {
                        'id': sample['id'],
                        'table': sample['table'],
                        'question': sample['question'],
                        'expected_answer': sample['answer'],
                        'total_attempts': attempt_num,
                        'attempts': attempts
                    }
                    # 立即写入到成功文件
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(result, ensure_ascii=False) + '\n')
                    return ('success', result)

            except Exception as e:
                print(f"  尝试 {attempt_num} 失败: {e}")
                attempts.append({
                    'attempt': attempt_num,
                    'response': str(e),
                    'extracted_answer': None,
                    'match': False,
                    'error_analysis': None
                })

        # 所有尝试都失败
        error_info = {
            'sample_id': sample['id'],
            'question': sample['question'],
            'expected_answer': sample['answer'],
            'total_attempts': max_attempts,
            'attempts': attempts
        }
        # 立即写入到错误文件
        with open(error_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(error_info, ensure_ascii=False) + '\n')
        return ('error', error_info)

    # 使用线程池并发处理
    max_workers = min(100, len(samples))  # 限制并发数，避免资源耗尽
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # 提交所有任务
        future_to_sample = {executor.submit(process_sample, sample): sample for sample in samples}

        # 收集结果
        for future in as_completed(future_to_sample):
            try:
                result_type, result = future.result()
                if result_type == 'success':
                    successful_results.append(result)
                else:
                    errors.append(result)
            except Exception as e:
                print(f"任务执行失败: {e}")

    # 输出统计信息
    print(f"\n=== 基线实验结果 ===\n")
    print(f"总样本数: {len(samples)}")
    print(f"成功样本数: {len(successful_results)}")
    print(f"失败样本数: {len(errors)}")
    print(f"成功率: {len(successful_results) / len(samples) * 100:.2f}%")
    print(f"成功结果已保存到: {output_file}")
    print(f"错误结果已保存到: {error_file}")

    return successful_results, errors


def main():
    parser = argparse.ArgumentParser(description='直接使用 LLM 进行蒸馏的基线实验，包含 critic refine 功能')

    # 输入输出参数
    parser.add_argument('--input_file', type=str,
                        default='/Users/westmoon/mycode/table/processed_data/training_processed.jsonl',
                        help='输入文件路径')
    parser.add_argument('--output_file', type=str,
                        default='/Users/westmoon/mycode/table/baseline_results/baseline_sft_critic',
                        help='输出文件路径')
    parser.add_argument('--num_samples', type=int, default=None,
                        help='处理的样本数量')
    parser.add_argument('--start_index', type=int, default=0,
                        help='起始样本索引')
    parser.add_argument('--end_index', type=int, default=None,
                        help='结束样本索引')

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
    parser.add_argument('--max_attempts', type=int, default=3,
                        help='每个样本的最大尝试次数')

    args = parser.parse_args()

    # 模型配置
    model_config = {
        "model": args.model,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "temperature": args.temperature,
        "top_p": args.top_p
    }

    # 根据起始和结束索引生成输出文件名
    if args.start_index is not None or args.end_index is not None:
        output_file = f"{args.output_file}_{args.start_index or 0}_{args.end_index or 'all'}.jsonl"
    else:
        output_file = f"{args.output_file}.jsonl"

    # 运行基线实验
    successful_results, errors = run_baseline_experiment(
        args.input_file,
        output_file,
        model_config,
        args.num_samples,
        args.start_index,
        args.end_index,
        args.max_attempts
    )


if __name__ == "__main__":
    main()