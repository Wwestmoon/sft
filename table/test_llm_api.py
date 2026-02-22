#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试 LLM API 配置的脚本
"""

import requests

def test_llm_api():
    """
    测试 LLM API 配置
    """
    # 配置参数
    config = {
        "base_url": "https://yunwu.ai/v1",
        "api_key": "sk-mBv9A5UrRCQ7OzJDPLKZkjIRzoywwVcu4wvStR4P6E7K9Kw9",
        "model": "gpt-4o-mini"
    }

    # API 端点
    url = f"{config['base_url'].rstrip('/')}/chat/completions"

    # 请求头部
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config['api_key']}"
    }

    # 请求数据
    data = {
        "model": config["model"],
        "messages": [
            {"role": "user", "content": "Hello, please say 'API test successful'"}
        ],
        "max_tokens": 100,
        "temperature": 0.1
    }

    print("测试 API 配置...")
    print(f"API URL: {url}")
    print(f"API Key: {config['api_key'][:5]}***")
    print(f"Model: {config['model']}")
    print("-" * 50)

    try:
        response = requests.post(url, headers=headers, json=data)
        print(f"HTTP 状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        print(f"响应内容: {response.text}")

        if response.status_code == 200:
            result = response.json()
            content = result["choices"][0]["message"]["content"].strip()
            print(f"API 响应成功: {content}")
        else:
            print(f"API 调用失败")

    except Exception as e:
        print(f"请求过程中发生错误: {e}")

if __name__ == "__main__":
    test_llm_api()
