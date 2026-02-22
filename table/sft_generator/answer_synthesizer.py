#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 3：答案汇总模块
"""


class AnswerSynthesizer:
    """
    答案汇总器类
    """

    def __init__(self, llm_api):
        self.llm_api = llm_api

    def synthesize_answer(self, decomposition, sub_question_results, original_question):
        """
        阶段 3：答案汇总
        """
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
            f"Original Question: {original_question}\n\n"
            f"Sub-question Answers:\n"
            f"{chr(10).join(results_str)}\n\n"
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
            f"12. If there are conflicting answers in sub-questions, resolve them logically\n\n"
            f"Format your response with the following structure:\n"
            f"1. Start with a clear introductory sentence that explains your approach to answering the question\n"
            f"2. Provide detailed reasoning steps that lead from the available information to the final answer\n"
            f"3. Include relevant calculations, data points, or examples from the table to support your reasoning\n"
            f"4. End with a concise final answer in the required format\n\n"
            f"[Your logical reasoning flow here]\n\n"
            f"Final Answer: [Your answer]"
        )

        try:
            print("发送给 LLM 的答案汇总提示:", repr(prompt))  # 调试信息
            response = self.llm_api.call(prompt)
            print("LLM 返回的答案汇总响应:", repr(response))  # 调试信息
            return response
        except Exception as e:
            print(f"答案汇总失败: {e}")
            return "无法汇总答案"
