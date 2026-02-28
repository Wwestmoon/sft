#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
阶段 2：子问题求解模块
"""

import re
import io
from contextlib import redirect_stdout


class SubQuestionSolver:
    """
    子问题求解器类
    """

    def __init__(self, llm_api):
        self.llm_api = llm_api

    def solve_sub_question(self, markdown_table, df, sub_question, strategy, context_info=None):
        """
        子问题求解
        """
        if context_info:
            context_str = "Based on the previous information:\n"
            for i, item in enumerate(context_info):
                context_str += (
                    f"Previous question {i+1}: \"{item['sub_question']}\" "
                    f"Answer: \"{item['data_flow'] if item['strategy']=='LLM' else item['answer']}\"\n"
                )
            context_sub_question = context_str + f"Now answer the following question: {sub_question}"
        else:
            context_sub_question = sub_question

        if strategy == "Python":
            return self._solve_python_sub_question(markdown_table,df, sub_question,context_sub_question)
        else:
            return self._solve_llm_sub_question(markdown_table, sub_question,context_sub_question)

    def _solve_python_sub_question(self, markdown_table, df, sub_question,context_sub_question):
        """
        策略 A：Python 求解（用于计算类问题）
        """
        code_prompt = (
            "Generate Python code to solve the sub-question using the existing pandas DataFrame 'df'.\n\n"
            f"Table columns: {list(df.columns)}\n"
            f"DataFrame dtypes: {df.dtypes.to_dict()}\n"
            f"DataFrame content (first 5 rows):\n{df.head().to_string(index=False)}\n\n"
            f"Sub-question: {context_sub_question}\n\n"
            "Important Note: The entire table contains information relevant to the current question. "
            "Analyze the table content carefully and use the appropriate columns to solve the sub-question.\n"
            "Be very careful to only extract data that is explicitly requested by the sub-question.\n"
            "If the sub-question asks for values of specific items that were identified in previous steps,\n"
            "make sure to only include those specific items and exclude others.\n\n"
            "CRITICAL CODE REQUIREMENTS:\n"
            "1. **DO NOT recreate or redefine the DataFrame 'df'** - it is already provided to you\n"
            "2. **DO NOT create any new DataFrames from scratch** using pd.DataFrame() or similar methods\n"
            "3. **Directly operate on the existing DataFrame 'df'** provided as input\n"
            "4. Output all intermediate and final results using print statements\n"
            "5. Do not include any explanations or comments\n"
            "6. Ensure the code can be executed directly"
        )

        # 最多尝试三次代码生成和执行
        max_attempts = 3
        previous_code = ""
        previous_error = ""

        for attempt in range(max_attempts):
            try:
                print(f"正在调用 LLM 生成代码 (尝试 {attempt + 1}/{max_attempts})...")  # 调试信息
                # 构建代码生成提示，包含之前的尝试信息
                current_code_prompt = code_prompt
                if previous_code:
                    current_code_prompt += f"\n\nPrevious Attempt Code:\n{previous_code}"
                if previous_error:
                    current_code_prompt += f"\n\nPrevious Attempt Error:\n{previous_error}"
                    current_code_prompt += "\n\nPlease fix the code based on the previous error and provide a correct solution."

                code = self.llm_api.call(current_code_prompt)
                print("LLM 返回的代码:", repr(code))  # 调试信息
                code_block = self._extract_code_block(code)
                if not code_block:
                    raise ValueError("无法提取有效的代码块")

                data_flow = self._execute_code_safely(code_block, df)

                answer = self._generate_natural_answer(markdown_table,sub_question, data_flow)
                print("自然语言：",answer )
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
                    return self._solve_llm_sub_question(markdown_table, sub_question)

    def _extract_code_block(self, response):
        """
        从 LLM 响应中提取代码块
        """
        # print("LLM 响应:", repr(response))  # 调试信息 
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
                    # print("提取到的代码块:", repr(code))  # 调试信息
                    return code

        print("未提取到代码块，返回完整响应:", repr(response))  # 调试信息
        return response

    def _execute_code_safely(self, code, df):
        """
        在安全沙箱中执行代码
        """
        data_flow = []

        try:
            # print("正在执行的代码:", repr(code))  # 调试信息
            # 直接使用原始 DataFrame，不进行提前的类型转换
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

            print("代码执行结果:", repr(output))  # 调试信息
            if output:
                lines = output.strip().split('\n')
                for line in lines:
                    if line.strip():
                        data_flow.append(line.strip())
        except Exception as e:
            print(f"代码执行失败: {e}")
            raise  # 重新抛出异常，让调用方可以捕获到

        return data_flow

    def _generate_natural_answer(self, markdown_table,sub_question, data_flow):
        """
        Generate natural language answer
        """
        prompt = f"""
You will be given:

1. A structured table
2. A question
3. A verified final answer (computed using Python)

Your task is to generate a clear step-by-step reasoning process based only on the table.

Requirements:

- The reasoning must be based only on the table.
- Show intermediate steps when calculations are needed.
- The logic should naturally lead to the given final answer.
- Do not use external knowledge.
- Do not include any information that is not directly relevant to the question.

Table:{markdown_table}
Question: {sub_question}
Verified final answer:
{chr(10).join(data_flow)}

Think step by step and then provide the final answer.
"""
        try:
             return self.llm_api.call(prompt)
        except Exception as e:
            print(f"Failed to generate natural language answer: {e}")
            return "Failed to generate natural language answer"

    def _solve_llm_sub_question(self, markdown_table, sub_question,context_sub_question):
        """
        Strategy B: LLM solving (for semantic understanding problems)
        """
        prompt = (
            f"Please answer the following question about the table. Make sure to explicitly reference specific data "
            f"from the table in your answer.\n\n"
            f"Table:\n{markdown_table}\n\n"
            f"Question: {sub_question}\n\n"
            f"Format your response strictly as follows:\n"
            f"Answer: [Your detailed reasoning process that includes all key data flow information]"
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

    def _extract_answer_from_reasoning(self, reasoning):
        """
        Extract answer from reasoning process
        """
        if "Answer:" in reasoning:
            answer = reasoning.split("Answer:", 1)[1].strip()
            if answer:
                return answer
        return reasoning.strip()
