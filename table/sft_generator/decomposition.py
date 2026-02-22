#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 1：问题分解与规划模块
"""

import json
import re


class QuestionDecomposer:
    """
    问题分解器类
    """

    def __init__(self, llm_api):
        self.llm_api = llm_api

    def decompose_and_plan(self, markdown_table, question):
        """
        问题分解与规划
        """
        prompt = (
            f"Please decompose the following complex question about the table into simple, specific sub-questions and "
            f"determine the solution strategy for each sub-question (either 'Python' for computational problems or 'LLM' "
            f"for semantic understanding problems).\n\n"
            f"Table:\n{markdown_table}\n\n"
            f"Question: {question}\n\n"
            f"Important Notes:\n"
            f"1. The entire table contains information relevant to the current question. Analyze the table structure carefully before answering.\n"
            f"2. Each sub-question should build upon the results of previous sub-questions, and there should be clear information transmission between them.\n"
            f"3. Sub-questions should be as specific as possible, avoiding ambiguity. Do not introduce data, columns, or categories that are not explicitly present in the table.\n"
            f"4. For questions involving dates or sequences, ensure sub-questions clearly define the scope and order.\n"
            f"5. Make sure each sub-question has a single, clear purpose.\n"
            f"6. When dealing with categorical data (such as Surface, Tournament, Playoffs), only consider the categories that are actually present in the table. Do not assume or invent additional categories.\n"
            f"7. Pay close attention to the exact wording of categorical values in the table, as similar but distinct values (e.g., 'Did not qualify' vs. 'No playoff') may have different meanings.\n"
            f"8. For counting tasks, ensure the filtering criteria are precise and match the exact values from the table.\n\n"
            f"Format your response as a JSON array with each item containing 'sub_question' and 'strategy' fields.\n"
            f"Example format for sequence questions:\n"
            f"[\n"
            f"    {{\"sub_question\": \"Identify all games played on or before November 10 and find the one against Las Vegas Legends\", \"strategy\": \"Python\"}},\n"
            f"    {{\"sub_question\": \"Find the game that comes immediately after the Las Vegas Legends game on November 10 by checking the date and opponent columns\", \"strategy\": \"Python\"}},\n"
            f"    {{\"sub_question\": \"Extract the opponent's name from the game identified in the previous step\", \"strategy\": \"LLM\"}}\n"
            f"]"
        )

        try:
            response = self.llm_api.call(prompt)
            # print("LLM 返回的问题分解响应:", repr(response))  # 调试信息
            # 尝试去除代码块标记
            cleaned_response = response.strip()
            if cleaned_response.startswith('```json'):
                cleaned_response = cleaned_response[len('```json'):].strip()
            if cleaned_response.endswith('```'):
                cleaned_response = cleaned_response[:-len('```')].strip()
            if cleaned_response.startswith('```'):
                cleaned_response = cleaned_response[len('```'):].strip()
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-len('```')].strip()
            print("清洗后的响应:", repr(cleaned_response))  # 调试信息
            # 尝试解析响应为 JSON
            try:
                decomposition = json.loads(cleaned_response)
                return decomposition
            except json.JSONDecodeError as e:
                print(f"无法解析问题分解响应为 JSON，错误: {e}")
                return [{"sub_question": question, "strategy": "LLM"}]
        except Exception as e:
            print(f"问题分解失败: {e}")
            return [{"sub_question": question, "strategy": "LLM"}]
