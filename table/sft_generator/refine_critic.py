#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Refine-Critic 框架
"""

import json
import re
from typing import Dict, Any, List, Optional
from sft_generator.base import LLMAPI, SimpleAnswerExtractor
from sft_generator.decomposition import QuestionDecomposer
from sft_generator.sub_question_solver import SubQuestionSolver
from sft_generator.answer_synthesizer import AnswerSynthesizer
from sft_generator.base import TableConverter


class ErrorAnalyzer:
    """
    错误分析器类，用于分析生成响应中的错误
    """

    def __init__(self, llm_api: LLMAPI = None):
        """
        初始化错误分析器

        Args:
            llm_api: LLM API 实例
        """
        self.llm_api = llm_api
        self.answer_extractor = SimpleAnswerExtractor()

    def analyze_error(self, question: str, table: str, decomposition: List[Dict], sub_question_results: List[Dict], final_response: str, expected_answer: str) -> Dict[str, Any]:
        """
        分析生成响应中的错误

        Args:
            question: 问题
            table: 表格内容
            decomposition: 问题分解
            sub_question_results: 子问题结果
            final_response: 最终响应
            expected_answer: 预期答案

        Returns:
            错误分析结果
        """
        if self.llm_api is not None:
            return self._analyze_with_llm(question, table, decomposition, sub_question_results, final_response, expected_answer)
        else:
            return {
                "type": "unknown",
                "reason": "Cannot analyze error (LLM API not configured)",
                "suggestion": "Provide more context or configure LLM API for detailed analysis"
            }


    def _analyze_with_llm(self, question: str, table: str, decomposition: List[Dict], sub_question_results: List[Dict], final_response: str, expected_answer: str) -> Dict[str, Any]:
        """
        使用 LLM 分析错误

        Args:
            question: 问题
            table: 表格内容
            decomposition: 问题分解
            sub_question_results: 子问题结果
            final_response: 最终响应
            expected_answer: 预期答案

        Returns:
            错误分析结果
        """
        prompt = (
            f"Please analyze the error in the generated response for the following question.\n\n"
            f"Question: {question}\n"
            f"Table:\n{table}\n"
            f"Problem Decomposition: {json.dumps(decomposition, ensure_ascii=False)}\n"
            f"Sub-Question Results: {json.dumps([{'sub_question': r['sub_question'], 'strategy': r['strategy'], 'answer': r['answer'], 'code': r.get('code', '') if r['strategy'] == 'Python' else '', 'reasoning': r.get('reasoning', '') if r['strategy'] == 'LLM' else ''} for r in sub_question_results], ensure_ascii=False)}\n"
            f"Final Response: {final_response}\n"
            f"Expected Answer: {expected_answer}\n\n"
            f"Your analysis should include:\n"
            f"1. The type of error (must be one of plan_error, subquestion_error, answer_synthesis_error)\n"
            f"2. The exact reason why the response is incorrect:\n"
            f"   - For plan_error: analyze whether the problem decomposition is incorrect\n"
            f"   - For subquestion_error:\n"
            f"      - If Python strategy: analyze the generated code and its execution results to identify issues like code-data mismatch\n"
            f"      - If LLM strategy: analyze the reasoning process to identify flaws\n"
            f"   - For answer_synthesis_error:\n"
            f"      - Check if the final answer format is correct (should be 'Final Answer: [Your answer]')\n"
            f"      - Verify if the answer is complete and consistent with sub-question answers\n"
            f"      - Analyze if the answer has correct semantics but incorrect wording or format\n"
            f"      - Identify if the final answer includes unnecessary information or contradicts the sub-question answers\n"
            f"      - Check if the final answer directly addresses the question (e.g., if the question asks for a year, ensure the final answer is just the year)\n"
            f"3. A very specific and actionable suggestion on how to improve the next attempt to avoid this error:\n"
            f"   - For Python strategy: provide detailed code improvements or corrections, including specific pandas operations or methods to use\n"
            f"   - For LLM strategy: provide specific guidance on what information to focus on or how to adjust the reasoning process\n"
            f"   - For answer_synthesis_error: provide step-by-step guidance on how to correctly synthesize the sub-question answers into the final answer\n"
            f"   - The suggestion should be detailed enough that it can be directly used to improve the next attempt\n"
            f"4. If the error is a subquestion_error, please specify the index of the erroneous sub-question (starting from 0)\n\n"
            f"Format your response as JSON with the following fields:\n"
            f"- type: the type of error (must be one of plan_error, subquestion_error, answer_synthesis_error)\n"
            f"- reason: detailed error reason\n"
            f"- suggestion: specific improvement suggestion\n"
            f"- sub_question_index: the index of the erroneous sub-question (only required for subquestion_error)"
        )

        try:
            analysis = self.llm_api.call(prompt)
            return self._parse_analysis(analysis)
        except Exception as e:
            print(f"Error analysis failed: {e}")
            
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
            "type": "parse_failure",
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

    def fix_response(self, question: str, generated_response: str,
                     expected_answer: str, error_analysis: Dict[str, Any]) -> str:
        """
        Fix generated response based on error analysis results

        Args:
            question: Question
            generated_response: Generated response
            expected_answer: Expected answer
            error_analysis: Error analysis result

        Returns:
            Fixed response
        """
        if self.llm_api is not None:
            return self._fix_with_llm(question, generated_response, expected_answer, error_analysis)
        else:
            return self._fix_with_rules(question, generated_response, expected_answer, error_analysis)

    def _fix_with_rules(self, question: str, generated_response: str,
                        expected_answer: str, error_analysis: Dict[str, Any]) -> str:
        """
        Fix response using rules

        Args:
            question: Question
            generated_response: Generated response
            expected_answer: Expected answer
            error_analysis: Error analysis result

        Returns:
            Fixed response
        """
        if error_analysis.get("type") == "answer_mismatch":
            return generated_response.replace(
                self.answer_extractor.extract(generated_response), expected_answer)
        return generated_response

    def _fix_with_llm(self, question: str, generated_response: str,
                      expected_answer: str, error_analysis: Dict[str, Any]) -> str:
        """
        使用 LLM 修复响应

        Args:
            question: 问题
            generated_response: 生成的响应
            expected_answer: 预期答案
            error_analysis: 错误分析结果

        Returns:
            修复后的响应
        """
        prompt = (
            f"Please fix the generated response based on the error analysis and provide an improved answer.\n\n"
            f"Question: {question}\n"
            f"Generated Response: {generated_response}\n"
            f"Expected Answer: {expected_answer}\n"
            f"Error Reason: {error_analysis.get('reason', '')}\n\n"
            f"Your improved response should:\n"
            f"1. Address the error identified in the analysis\n"
            f"2. Provide a clear and correct answer\n"
            f"3. Follow the same structure as the original response\n"
            f"4. Be concise and focused on the question"
        )

        try:
            return self.llm_api.call(prompt)
        except Exception as e:
            print(f"Failed to fix response: {e}")
            return self._fix_with_rules(question, generated_response, expected_answer, error_analysis)


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
        self.decomposer = QuestionDecomposer(llm_api) if llm_api is not None else None
        self.solver = SubQuestionSolver(llm_api) if llm_api is not None else None
        self.synthesizer = AnswerSynthesizer(llm_api) if llm_api is not None else None
        self.converter = TableConverter() if llm_api is not None else None

    def refine_response(self, question: str, table: str, decomposition: List[Dict],
                       sub_question_results: List[Dict], final_response: str,
                       expected_answer: str) -> Dict[str, Any]:
        """
        Criticize and improve generated response

        Args:
            question: 问题
            table: 表格内容
            decomposition: 问题分解
            sub_question_results: 子问题结果
            final_response: 最终响应
            expected_answer: 预期答案

        Returns:
            Criticism and improvement result
        """
        error_analysis = self.error_analyzer.analyze_error(
            question, table, decomposition, sub_question_results, final_response, expected_answer
        )

        if error_analysis["type"] == "unknown" or error_analysis["type"] == "parse_failure":
            return {
                "original": final_response,
                "analysis": error_analysis,
                "refined": final_response
            }

        # 根据错误类型决定如何修复响应
        if error_analysis["type"] == "plan_error":
            # 问题分解阶段出错，需要重新分解问题
            refined_result = self._refine_decomposition(question, table, decomposition, sub_question_results, final_response, expected_answer, error_analysis)
        elif error_analysis["type"] == "subquestion_error":
            # 子问题求解阶段出错，需要重新求解子问题
            refined_result = self._refine_subquestion_solving(question, table, decomposition, sub_question_results, final_response, expected_answer, error_analysis)
        elif error_analysis["type"] == "answer_synthesis_error":
            # 答案合成阶段出错，需要重新合成答案
            refined_result = self._refine_answer_synthesis(question, table, decomposition, sub_question_results, final_response, expected_answer, error_analysis)
        else:
            # 其他错误类型，使用默认修复方法
            refined_response = self.error_fixer.fix_response(
                question, final_response, expected_answer, error_analysis
            )
            refined_result = {"decomposition": decomposition, "sub_question_results": sub_question_results, "final_response": refined_response}

        return {
            "original": final_response,
            "analysis": error_analysis,
            "refined": refined_result["final_response"],
            "response": refined_result  # 包含优化后的分解和子问题结果
        }

    def _refine_decomposition(self, question: str, table: str, decomposition: List[Dict],
                           sub_question_results: List[Dict], final_response: str,
                           expected_answer: str, error_analysis: Dict[str, Any]) -> Dict:
        """
        重新分解问题

        Args:
            question: 问题
            table: 表格内容
            decomposition: 问题分解
            sub_question_results: 子问题结果
            final_response: 最终响应
            expected_answer: 预期答案
            error_analysis: 错误分析结果

        Returns:
            包含重新分解后的问题、子问题结果和最终响应的字典
        """
        # 根据错误分析结果生成改进后的问题描述
        improved_question = self.generate_improved_prompt(question, table, decomposition,
                                                       sub_question_results, final_response, error_analysis)
        # 生成改进后的问题分解
        improved_decomposition = self.decomposer.decompose_and_plan(table, improved_question)
        # 将表格转换为 DataFrame
        df = self.converter.markdown_to_dataframe(table)
        if df is None:
            return {"decomposition": decomposition, "sub_question_results": sub_question_results, "final_response": "表格转换失败"}
        # 重新求解子问题
        improved_sub_question_results = []
        for _, item in enumerate(improved_decomposition):
            sub_question = item['sub_question']
            strategy = item['strategy']
            try:
                result = self.solver.solve_sub_question(df, sub_question, strategy)
                improved_sub_question_results.append(result)
            except Exception as e:
                improved_sub_question_results.append({
                    "sub_question": sub_question,
                    "strategy": strategy,
                    "data_flow": f"错误: {e}",
                    "answer": "求解失败"
                })
        # 重新合成答案
        final_response = self.synthesizer.synthesize_answer(improved_decomposition, improved_sub_question_results, question)
        return {"decomposition": improved_decomposition, "sub_question_results": improved_sub_question_results, "final_response": final_response}

    def _refine_subquestion_solving(self, question: str, table: str, decomposition: List[Dict],
                                  sub_question_results: List[Dict], final_response: str,
                                  expected_answer: str, error_analysis: Dict[str, Any]) -> Dict:
        """
        重新求解子问题

        Args:
            question: 问题
            table: 表格内容
            decomposition: 问题分解
            sub_question_results: 子问题结果
            final_response: 最终响应
            expected_answer: 预期答案
            error_analysis: 错误分析结果

        Returns:
            包含重新求解后的子问题结果和最终响应的字典
        """
        # 将表格转换为 DataFrame
        df = self.converter.markdown_to_dataframe(table)
        if df is None:
            return {"decomposition": decomposition, "sub_question_results": sub_question_results, "final_response": "表格转换失败"}

        # 重新求解子问题（串行处理，确保下一个子问题能获得上一个的答案）
        improved_sub_question_results = []
        context_info = []

        # 获取出错的子问题索引
        sub_question_index = error_analysis.get('sub_question_index', -1)

        for i, item in enumerate(decomposition):
            sub_question = item['sub_question']
            strategy = item['strategy']
            try:
                # 如果是出错的子问题，根据错误分析结果调整解决方法
                if i == sub_question_index:
                    # 保留原问题，根据错误分析结果调整策略或方法
                    if i > 0:
                        context_str = "Based on the previous information:\n"
                        for j in range(i):
                            prev_item = decomposition[j]
                            prev_result = improved_sub_question_results[j]
                            context_str += (
                                f"Previous question {j+1}: \"{prev_item['sub_question']}\" "
                                f"Answer: \"{prev_result['answer']}\"\n"
                            )
                        context_sub_question = context_str + f"Now answer the following question: {sub_question}"
                    else:
                        context_sub_question = sub_question
                    # 在解决子问题时结合报错信息
                    if strategy == "Python":
                        result = self._solve_python_sub_question_with_error(df, context_sub_question, error_analysis)
                    else:
                        result = self._solve_llm_sub_question_with_error(df, context_sub_question, error_analysis)
                elif i > sub_question_index:
                    # 出错子问题之后的子问题，重新执行但不结合报错信息
                    result = self.solver.solve_sub_question(df, sub_question, strategy, context_info)
                else:
                    # 出错子问题之前的子问题，保持原结果
                    result = sub_question_results[i]

                # 保存原策略信息
                result['sub_question'] = item['sub_question']
                result['strategy'] = strategy
                improved_sub_question_results.append(result)
                # 保存上下文信息
                context_info.append({
                    "sub_question": item['sub_question'],
                    "answer": result['answer']
                })
            except Exception as e:
                improved_sub_question_results.append({
                    "sub_question": sub_question,
                    "strategy": strategy,
                    "data_flow": "",
                    "answer": "求解失败"
                })
                context_info.append({
                    "sub_question": item['sub_question'],
                    "answer": "求解失败"
                })

        # 重新合成答案
        final_response = self.synthesizer.synthesize_answer(decomposition, improved_sub_question_results, question)
        return {"decomposition": decomposition, "sub_question_results": improved_sub_question_results, "final_response": final_response}

    def _solve_python_sub_question_with_error(self, df, sub_question, error_analysis):
        """
        结合报错信息使用 Python 策略解决子问题
        """
        # 检查是否有之前的 Python 代码和报错信息
        previous_code = ""
        previous_error = ""

        if error_analysis.get('response'):
            sub_question_results = error_analysis['response'].get('sub_question_results', [])
            sub_question_index = error_analysis.get('analysis', {}).get('sub_question_index', -1)
            if sub_question_index >= 0 and sub_question_index < len(sub_question_results):
                previous_result = sub_question_results[sub_question_index]
                if previous_result.get('code'):
                    previous_code = previous_result['code']
                if previous_result.get('data_flow'):
                    # 检查是否有报错信息
                    if any('error' in str(item).lower() for item in previous_result['data_flow']):
                        previous_error = str(previous_result['data_flow'])

        # 最多尝试三次代码生成和执行
        max_attempts = 3
        for attempt in range(max_attempts):
            code_prompt = (
                "Generate Python code to solve the sub-question using the existing pandas DataFrame 'df'.\n\n"
                f"Table columns: {list(df.columns)}\n"
                f"DataFrame dtypes: {df.dtypes.to_dict()}\n"
                f"DataFrame content (first 5 rows):\n{df.head().to_string(index=False)}\n\n"
                f"Sub-question: {sub_question}\n\n"
                f"Error Reason: {error_analysis.get('suggestion', '')}\n\n"
                f"Previous Attempt Code: {previous_code}\n\n"
                f"Previous Attempt Error: {previous_error}\n\n"
                "CRITICAL CODE REQUIREMENTS:\n"
                "1. **DO NOT recreate or redefine the DataFrame 'df'** - it is already provided to you\n"
                "2. **DO NOT create any new DataFrames from scratch** using pd.DataFrame() or similar methods\n"
                "3. **Directly operate on the existing DataFrame 'df'** provided as input\n"
                "4. Output all intermediate and final results using print statements\n"
                "5. Do not include any explanations or comments\n"
                "6. Ensure the code can be executed directly\n"
                "7. Address the error identified in the error analysis to ensure the code is correct"
            )

            try:
                print(f"正在调用 LLM 生成代码 (尝试 {attempt + 1}/{max_attempts})...")  # 调试信息
                code = self.llm_api.call(code_prompt)
                print("LLM 返回的代码:", repr(code))  # 调试信息
                code_block = self._extract_code_block(code)
                if not code_block:
                    raise ValueError("无法提取有效的代码块")

                data_flow = self._execute_code_safely(code_block, df)

                answer = self._generate_natural_answer(sub_question, data_flow)
                print("自然语言：",answer)
                return {
                    "sub_question": sub_question,
                    "strategy": "Python",
                    "data_flow": data_flow,
                    "answer": answer,
                    "code": code_block  # 记录生成的代码
                }
            except Exception as e:
                print(f"Python 求解失败 (尝试 {attempt + 1}/{max_attempts}): {e}")
                # 更新之前的错误信息，用于下一次尝试
                previous_code = code_block if 'code_block' in locals() else previous_code
                previous_error = str(e)
                # 如果是最后一次尝试，使用 LLM 求解作为备用方案
                if attempt == max_attempts - 1:
                    print("已达到最大尝试次数，使用 LLM 求解作为备用方案")
                    return self._solve_llm_sub_question_with_error(df, sub_question, error_analysis)

    def _solve_llm_sub_question_with_error(self, df, sub_question, error_analysis):
        """
        结合报错信息使用 LLM 策略解决子问题
        """
        from tabulate import tabulate
        markdown_table = tabulate(df, headers='keys', tablefmt='pipe')

        prompt = (
            f"Please answer the following question about the table. Make sure to explicitly reference specific data "
            f"from the table in your answer.\n\n"
            f"Table:\n{markdown_table}\n\n"
            f"Question: {sub_question}\n\n"
            f"Error Reason: {error_analysis.get('suggestion', '')}\n\n"
            f"Your answer should include:\n"
            f"1. Your reasoning process about how you arrived at the answer\n"
            f"2. A final answer that addresses the error identified in the analysis and retains all key data flow information\n\n"
            f"Format your response as:\n"
            f"Reasoning: [Your reasoning process]\n"
            f"Answer: [Your answer that includes all key data flow information]"
        )

        try:
            response = self.llm_api.call(prompt)
            return {
                "sub_question": sub_question,
                "strategy": "LLM",
                "reasoning": response,
                "answer": self._extract_answer_from_reasoning(response)
            }
        except Exception as e:
            print(f"LLM solving failed: {e}")
            return {
                "sub_question": sub_question,
                "strategy": "LLM",
                "reasoning": "",
                "answer": "LLM solving failed"
            }

    def _extract_code_block(self, response):
        """
        从 LLM 响应中提取代码块
        """
        code_patterns = [
            r'```python\n(.*?)```',
            r'```\n(.*?)```',
            r'code:(.*)'
        ]

        for pattern in code_patterns:
            match = re.search(pattern, response, re.DOTALL)
            if match:
                code = match.group(1).strip()
                if code:
                    return code
        return response

    def _execute_code_safely(self, code, df):
        """
        在安全沙箱中执行代码
        """
        import io
        from contextlib import redirect_stdout

        data_flow = []
        try:
            local_vars = {"df": df}
            f = io.StringIO()
            with redirect_stdout(f):
                # 提供更宽松的安全执行环境
                safe_globals = {
                    "__builtins__": __import__("builtins"),  # 引入完整的内置模块
                    "pd": __import__("pandas"),
                    "numpy": __import__("numpy"),
                    "re": __import__("re"),
                    "datetime": __import__("datetime"),
                    "math": __import__("math")
                }
                exec(code, safe_globals, local_vars)
            output = f.getvalue()

            if output:
                lines = output.strip().split('\n')
                for line in lines:
                    if line.strip():
                        data_flow.append(line.strip())
        except Exception as e:
            print(f"代码执行失败: {e}")
            raise  # 重新抛出异常，让调用方可以捕获到

        return data_flow

    def _generate_natural_answer(self, sub_question, data_flow):
        """
        Generate natural language answer
        """
        prompt = (
            f"Based on the following sub-question and data flow (which contains intermediate and final results), "
            f"generate a natural language answer that does not mention technical terms like 'code' or 'execution', "
            f"but clearly references specific numbers and facts from the data flow.\n\n"
            f"Sub-question: {sub_question[-1]}\n\n"
            f"Data flow:\n"
            f"{chr(10).join(data_flow)}\n\n"
            f"Your answer should focus on the data and results, not how they were obtained."
        )

        try:
            return self.llm_api.call(prompt)
        except Exception as e:
            print(f"Failed to generate natural language answer: {e}")
            return "Failed to generate natural language answer"

    def _extract_answer_from_reasoning(self, reasoning):
        """
        Extract answer from reasoning process
        """
        if "Answer:" in reasoning:
            answer = reasoning.split("Answer:", 1)[1].strip()
            if answer:
                return answer
        return reasoning.strip()

    def _get_improved_sub_question(self, sub_question: str, error_analysis: Dict[str, Any]) -> str:
        """
        根据错误分析结果生成改进后的子问题

        Args:
            sub_question: 原始子问题
            error_analysis: 错误分析结果

        Returns:
            改进后的子问题
        """
        if self.llm_api is None:
            return sub_question

        prompt = (
            f"Please improve the following sub-question based on the error analysis:\n\n"
            f"Original Sub-Question: {sub_question}\n"
            f"Error Reason: {error_analysis.get('suggestion', '')}\n\n"
            f"Your improved sub-question should address the error and be more specific to ensure correct execution. "
            f"Do not change the core meaning of the sub-question, but make it more actionable."
        )

        try:
            return self.llm_api.call(prompt)
        except Exception as e:
            print(f"Failed to improve sub-question: {e}")
            return sub_question

    def _refine_answer_synthesis(self, question: str, table: str, decomposition: List[Dict],
                               sub_question_results: List[Dict], final_response: str,
                               expected_answer: str, error_analysis: Dict[str, Any]) -> Dict:
        """
        重新合成答案

        Args:
            question: 问题
            table: 表格内容
            decomposition: 问题分解
            sub_question_results: 子问题结果
            final_response: 最终响应
            expected_answer: 预期答案
            error_analysis: 错误分析结果

        Returns:
            包含重新合成后的答案的字典，保留原分解和子问题结果
        """
        # 参考原有的合成答案提示词，结合报错信息进行修改
        results_str = []
        for i, result in enumerate(sub_question_results):
            # 处理任意数量的子问题序数
            if i == 0:
                ordinal = "first"
            elif i == 1:
                ordinal = "second"
            elif i == 2:
                ordinal = "third"
            elif i == 3:
                ordinal = "fourth"
            elif i == 4:
                ordinal = "fifth"
            else:
                ordinal = f"{i + 1}th"
            results_str.append(
                f"{ordinal.capitalize()} sub-question: {result['sub_question']}\n"
                f"Strategy: {result['strategy']}\n"
                f"Answer: {result['answer']}"
            )

        prompt = (
            f"Please synthesize the answers to the sub-questions into a comprehensive response to the original question.\n\n"
            f"Original Question: {question}\n\n"
            f"Sub-question Answers:\n"
            f"{chr(10).join(results_str)}\n\n"
            f"Error Reason: {error_analysis.get('suggestion', '')}\n\n"
            f"Your response must strictly adhere to the following requirements:\n"
            f"1. Follow the logical flow from the sub-question answers to the final answer\n"
            f"2. Be coherent and easy to understand\n"
            f"3. Include all relevant information from the sub-question answers\n"
            f"4. End with a clear final answer in the exact format: 'Final Answer: [Your answer]'\n"
            f"5. Avoid mentioning 'sub-questions' or 'strategies'\n"
            f"6. The final answer must be concise and directly address the question\n"
            f"7. If the answer is a number representing a quantity (e.g., 'how many'), only include the numeric value without any additional text or units\n"
            f"8. If the answer is a number representing a duration (e.g., 'how long'), include the numeric value with appropriate units (e.g., 'years', 'days')\n"
            f"9. If the question asks for a name, category, or identifier (e.g., 'which place', 'which team', 'what is the name'), ensure the final answer includes only the name or identifier, not numerical values or other details\n"
            f"10. Verify that the final answer is consistent with all sub-question answers\n"
            f"11. Ensure that your answer is accurate and free of errors\n"
            f"12. If there are conflicting answers in sub-questions, resolve them logically\n"
            f"13. Address the improvement suggestion to ensure the final answer is correct\n\n"
            f"Format your response with the following structure:\n"
            f"1. Start with a clear introductory sentence that explains your approach to answering the question\n"
            f"2. Provide detailed reasoning steps that lead from the available information to the final answer\n"
            f"3. Include relevant calculations, data points, or examples from the table to support your reasoning\n"
            f"4. End with a concise final answer in the required format\n\n"
            f"[Your logical reasoning flow here]\n\n"
            f"Final Answer: [Your answer]"
        )

        try:
            final_response = self.llm_api.call(prompt)
            return {"decomposition": decomposition, "sub_question_results": sub_question_results, "final_response": final_response}
        except Exception as e:
            print(f"Failed to synthesize answer: {e}")
            return {"decomposition": decomposition, "sub_question_results": sub_question_results, "final_response": "无法汇总答案"}

    def generate_improved_prompt(self, question: str, table: str, decomposition: List[Dict],
                               sub_question_results: List[Dict], previous_response: str,
                               error_analysis: Dict[str, Any]) -> str:
        """
        Generate improved prompt to avoid similar errors

        Args:
            question: 问题
            table: 表格内容
            decomposition: 问题分解
            sub_question_results: 子问题结果
            previous_response: 之前生成的响应
            error_analysis: 错误分析结果

        Returns:
            Improved prompt
        """
        if self.llm_api is None:
            return question

        prompt = (
            f"Please improve the prompt for the following question based on the error analysis of the previous response.\n\n"
            f"Question: {question}\n"
            f"Table:\n{table}\n"
            f"Problem Decomposition: {json.dumps(decomposition, ensure_ascii=False)}\n"
            f"Sub-Question Results: {json.dumps([{'sub_question': r['sub_question'], 'strategy': r['strategy'], 'answer': r['answer'], 'code': r.get('code', '') if r['strategy'] == 'Python' else '', 'reasoning': r.get('reasoning', '') if r['strategy'] == 'LLM' else ''} for r in sub_question_results], ensure_ascii=False)}\n"
            f"Previous Response: {previous_response}\n"
            f"Error Reason: {error_analysis.get('suggestion', '')}\n\n"
            f"Your improved prompt should:\n"
            f"1. Address the error type and reason identified in the analysis\n"
            f"2. Provide clear guidance on how to avoid similar errors\n"
            f"3. Include any necessary context or constraints\n"
            f"4. Keep the prompt concise and focused on the question"
        )

        try:
            return self.llm_api.call(prompt)
        except Exception as e:
            print(f"Failed to generate improved prompt: {e}")
            return question
