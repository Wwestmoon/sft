#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Zero-Shot Baseline 脚本，支持多种 LLM 提供商
读取 JSONL 格式的数据集
"""

import os
import json
import requests


class OpenAICompatibleAPI:
    """
    OpenAI 兼容的 API 接口
    支持多种 LLM 提供商，包括 OpenAI、DeepSeek、Qwen、Llama 等
    """

    def __init__(self, model, base_url, api_key):
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }

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

    def call(self, prompt):
        """
        调用 OpenAI 兼容的 API
        """
        data = {
            "model": self.model,
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "max_tokens": 100,
            "temperature": self.temperature,
            "top_p": self.top_p
        }

        try:
            # 构建完整的 API 端点
            url = f"{self.base_url.rstrip('/')}/chat/completions"
            response = requests.post(url, headers=self.headers, json=data)
            response.raise_for_status()

            result = response.json()
            return result["choices"][0]["message"]["content"].strip()
        except Exception as e:
            print(f"API 调用失败: {e}")
            return "error"


def build_prompt(markdown_table, question):
    """
    Build prompt containing Markdown table and question
    """
    prompt = (
        f"Please answer the question based on the following table:\n\n{markdown_table}\n\n"
        f"Question: {question}\n\n"
        f"Please give the answer directly without any explanation."
    )
    return prompt


def run_baseline(input_file, output_file, model_config, num_samples=10):
    """
    运行 Zero-Shot Baseline
    """
    # 初始化 API
    api = OpenAICompatibleAPI(
        model=model_config["model"],
        base_url=model_config["base_url"],
        api_key=model_config["api_key"],
        temperature=model_config.get("temperature", 0.1),
        top_p=model_config.get("top_p", 0.9)
    )

    # 读取预处理后的数据集
    data = []
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))

    # 只处理 num_samples 个样本
    if num_samples > 0 and num_samples < len(data):
        data = data[:num_samples]

    # 处理每个样本
    predictions = []
    for i, sample in enumerate(data):
        # 构建 prompt
        prompt = build_prompt(sample['table'], sample['question'])

        # 调用 API
        answer = api.call(prompt)

        # 保存预测结果
        predictions.append({
            'id': sample['id'],
            'prediction': answer
        })

        print(f"已处理 {i + 1} 个样本")

    # 保存预测结果为 JSONL 格式，以便后续处理
    with open(output_file, 'w', encoding='utf-8') as f:
        for prediction in predictions:
            f.write(json.dumps(prediction, ensure_ascii=False) + '\n')

    print(f"Baseline 运行完成，保存到 {output_file}")
    print(f"共处理 {len(predictions)} 个样本")


def main():
    # 解析命令行参数
    import argparse
    parser = argparse.ArgumentParser(description="运行 LLM 基准测试")
    parser.add_argument("--model", default="gpt4o-mini", help="模型名称")
    parser.add_argument("--base_url", default="https://yunwu.ai/v1", help="API 端点")
    parser.add_argument("--api_key", default="sk-fVYYQ0opk2YfCMR7018370CeD0Cf429aAbA31f01Dd1dBf26", help="API 密钥")
    parser.add_argument("--num_samples", type=int, default=10, help="测试样本数量")
    parser.add_argument("--input_file", default='/Users/westmoon/mycode/table/processed_data/test_processed.jsonl', help="输入文件路径")
    parser.add_argument("--output_dir", default='/Users/westmoon/mycode/table/baseline_results', help="输出目录")
    parser.add_argument("--temperature", type=float, default=0.1, help="采样温度")
    parser.add_argument("--top_p", type=float, default=0.9, help="核采样参数")

    args = parser.parse_args()

    # 模型配置
    model_config = {
        "model": args.model,
        "base_url": args.base_url,
        "api_key": args.api_key,
        "temperature": args.temperature,
        "top_p": args.top_p
    }

    # 输入和输出文件路径
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(args.output_dir, f"{model_config['model']}_predictions.jsonl")

    # 运行 Baseline
    run_baseline(args.input_file, output_file, model_config, num_samples=args.num_samples)


if __name__ == "__main__":
    main()
