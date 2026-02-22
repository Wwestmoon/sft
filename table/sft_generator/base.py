#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SFT 数据生成器的基础接口和核心类
"""

import os
import json
import pandas as pd
import re
import random
import time


class LLMAPI:
    """
    LLM API 的基础接口
    """

    def __init__(self, model_config):
        self.model_config = model_config

    def call(self, prompt):
        """
        调用 LLM API
        """
        raise NotImplementedError("Subclasses must implement this method")


class OpenAICompatibleAPI(LLMAPI):
    """
    OpenAI 兼容的 API 接口
    支持多种 LLM 提供商，包括 OpenAI、DeepSeek、Qwen、Llama 等
    """

    def __init__(self, model_config):
        super().__init__(model_config)
        self.model = model_config["model"]
        self.base_url = model_config["base_url"]
        self.api_key = model_config["api_key"]
        self.temperature = model_config.get("temperature", 0.1)
        self.top_p = model_config.get("top_p", 0.9)
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

    def call(self, prompt):
        """
        调用 OpenAI 兼容的 API
        """
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry

        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 1000,
            "temperature": self.temperature,
            "top_p": self.top_p
        }

        try:
            # 创建会话并设置重试机制
            session = requests.Session()
            retry = Retry(
                total=3,
                backoff_factor=0.5,
                status_forcelist=[429, 500, 502, 503, 504],
                allowed_methods=["HEAD", "GET", "POST"]
            )
            adapter = HTTPAdapter(max_retries=retry)
            session.mount("https://", adapter)
            session.mount("http://", adapter)

            # 构建完整的 API 端点
            url = f"{self.base_url.rstrip('/')}/chat/completions"
            response = session.post(url, headers=self.headers, json=data, timeout=30)
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"API 调用失败: {e}")
            return "error"


class AnswerExtractor:
    """
    答案提取器的基础接口
    """

    def extract(self, response):
        """
        从响应中提取答案
        """
        raise NotImplementedError("Subclasses must implement this method")


class SimpleAnswerExtractor(AnswerExtractor):
    """
    简单的答案提取器，支持从多种格式中提取答案
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


class TableConverter:
    """
    表格转换器类
    """

    @staticmethod
    def _auto_convert_data_types(df):
        """
        自动检测和转换 DataFrame 中的数据类型
        """
        for column in df.columns:
            # 只转换完全是数字的列
            try:
                # 检查列中是否所有值都是数字或空值
                all_numeric = True
                for value in df[column]:
                    if pd.notna(value):
                        # 去除千位分隔符
                        cleaned_value = str(value).replace(',', '').strip()
                        # 检查是否可以转换为数字
                        try:
                            float(cleaned_value)
                        except:
                            all_numeric = False
                            break

                if all_numeric:
                    # 去除千位分隔符
                    df[column] = df[column].replace(',', '', regex=True).str.strip()
                    # 转换为数字类型
                    df[column] = pd.to_numeric(df[column], errors='coerce')
                    continue
            except:
                pass

            # 所有其他列都保持为字符串类型
            df[column] = df[column].astype(str).str.strip()

    @staticmethod
    def markdown_to_dataframe(markdown_table):
        """
        将 Markdown 表格转换为 Pandas DataFrame，并自动检测和转换数据类型
        """
        try:
            # 解析 Markdown 表格
            lines = markdown_table.strip().split('\n')
            if len(lines) < 3:
                raise ValueError("Markdown 表格格式不正确")

            # 提取表头和数据
            header = lines[0].strip('|').split('|')
            header = [h.strip() for h in header if h.strip()]

            data = []
            # 从第 2 行开始解析数据，但要确保跳过表头分隔线
            for i, line in enumerate(lines[1:]):
                original_line = line  # 保存原始行用于调试
                # 跳过表头分隔线（包含 --- 的行）
                if all(c.strip() == '-' for c in line.strip('|').split('|')):
                    continue

                line = line.strip('|')
                if line:
                    # 使用简单的分割方法，确保所有行都能被解析
                    row = line.split('|')
                    row = [cell.strip() for cell in row if cell.strip()]
                    # 确保每行有相同的列数
                    if len(row) < len(header):
                        row += [''] * (len(header) - len(row))
                    elif len(row) > len(header):
                        row = row[:len(header)]
                    data.append(row)

            # 处理重复的列名
            # 首先检查是否有重复的列名
            if len(header) != len(set(header)):
                # 创建一个新的列名列表
                new_header = []
                column_counts = {}
                for col in header:
                    if col in column_counts:
                        column_counts[col] += 1
                        new_header.append(f"{col}_{column_counts[col]}")
                    else:
                        column_counts[col] = 0
                        new_header.append(col)
                header = new_header
            # 创建 DataFrame
            df = pd.DataFrame(data, columns=header)

            # 自动检测和转换数据类型
            TableConverter._auto_convert_data_types(df)

            return df
        except Exception as e:
            print(f"表格转换失败: {e}")
            return None


class ResultSaver:
    """
    结果保存器类
    """

    @staticmethod
    def save_results(results, output_file):
        """
        保存生成的训练数据
        """
        if not os.path.exists(os.path.dirname(output_file)):
            os.makedirs(os.path.dirname(output_file), exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            for result in results:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')

        print(f"结果保存到 {output_file}")
