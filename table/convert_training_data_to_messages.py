#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
将 training_data_0_None.jsonl 转换为 messages 格式的脚本
"""

import json
import jsonlines
import argparse
import sys
import re


# 添加 WikiTableQuestions 到系统路径
wiki_table_questions_path = '/Users/westmoon/mycode/table/WikiTableQuestions'
if wiki_table_questions_path not in sys.path:
    sys.path.append(wiki_table_questions_path)

from evaluator import to_value, to_value_list, check_denotation


def extract_final_answer(response):
    """
    从响应中提取答案，使用与 SimpleAnswerExtractor 相同的方法
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
    cleaned_response = response.strip()
    for prefix in ["The answer is ", "Answer: ", "Final Answer: ", "The correct answer is "]:
        if cleaned_response.startswith(prefix):
            cleaned_response = cleaned_response[len(prefix):].strip()

    if '.' in cleaned_response:
        return cleaned_response.split('.', 1)[0].strip()
    elif '\n' in cleaned_response:
        return cleaned_response.split('\n', 1)[0].strip()
    else:
        return cleaned_response.strip()


def match_answer(extracted_answer, expected_answer):
    """
    使用与 QualityController._match_answer 相同的方法匹配答案
    """
    if not extracted_answer or not expected_answer:
        return False

    target_values = to_value_list([str(expected_answer)])
    predicted_values = to_value_list([str(extracted_answer)])

    return check_denotation(target_values, predicted_values)


def main():
    # 设置命令行参数
    parser = argparse.ArgumentParser(description='Convert training_data_0_None.jsonl to messages format with answer verification')
    parser.add_argument('--input', '-i',
                        default='/Users/westmoon/mycode/table/training_data/training_data_debug_0_None.jsonl',
                        help='Path to input training_data JSONL file')
    parser.add_argument('--output', '-o',
                        default='/Users/westmoon/mycode/table/training_data/training_data_debug_0_None.jsonl.messages.jsonl',
                        help='Path to output messages JSONL file')
    parser.add_argument('--attempt', '-a',
                        type=int,
                        help='Specify attempt number to extract (e.g., --attempt 1 to extract only attempt=1)')

    args = parser.parse_args()

    messages_list = []
    errors = 0

    with jsonlines.open(args.input) as reader:
        for line in reader:
            target_attempt = None
            if 'attempts' in line and isinstance(line['attempts'], list):
                if args.attempt:
                    # 如果指定了 attempt 号，查找对应的尝试
                    for attempt in line['attempts']:
                        if attempt.get('attempt') == args.attempt:
                            target_attempt = attempt
                            break
                else:
                    # 如果没有指定 attempt 号，查找最佳尝试（首先查找匹配的答案，然后取第一个）
                    for attempt in line['attempts']:
                        if attempt.get('match', False):
                            target_attempt = attempt
                            break

                    # 如果没有匹配的答案，取第一个尝试
                    if not target_attempt and line['attempts']:
                        target_attempt = line['attempts'][0]
            else:
                print(f"Warning: Entry {line['id']} has no attempts")
                continue

            if target_attempt:
                # 从尝试中提取响应内容
                if isinstance(target_attempt, dict):
                    if 'response' in target_attempt:
                        if isinstance(target_attempt['response'], dict) and 'final_response' in target_attempt['response']:
                            assistant_content = target_attempt['response']['final_response']
                        elif isinstance(target_attempt['response'], str):
                            assistant_content = target_attempt['response']
                        else:
                            print(f"Warning: Response is not string or dict with final_response in entry {line['id']}")
                            errors += 1
                            continue
                    elif 'final_response' in target_attempt:
                        assistant_content = target_attempt['final_response']
                    else:
                        print(f"Warning: Attempt has no response or final_response field in entry {line['id']}")
                        errors += 1
                        continue
                else:
                    print(f"Warning: Attempt is not a dictionary in entry {line['id']}")
                    errors += 1
                    continue
                # assistant_content=line['new_final_response']
                # 提取并验证答案（如果有预期答案）
                if 'expected_answer' in line:
                    extracted_answer = extract_final_answer(assistant_content)
                    if not match_answer(extracted_answer, line['expected_answer']):
                        errors += 1
                        print(f"Error: Entry {line['id']} - Extracted answer '{extracted_answer}' does not match expected answer '{line['expected_answer']}'")
                        continue

                # 构建用户内容
                user_content = f"""Table:
{line['table']}

Question: {line['question']}

Please reason step by step and output the final answer in the specified format. The final answer should be in the format "Final Answer: [answer]".
"""

                # 构建消息格式
                messages = {
                    "id": line['id'],
                    "messages": [
                        {
                            "role": "user",
                            "content": user_content
                        },
                        {
                            "role": "assistant",
                            "content": assistant_content
                        }
                    ]
                }

                messages_list.append(messages)

    # 保存为 JSONL 格式
    with jsonlines.open(args.output, mode='w') as writer:
        for item in messages_list:
            writer.write(item)

    print(f"Conversion completed! Saved to {args.output}")
    print(f"Converted {len(messages_list)} entries")
    print(f"Errors found: {errors}")


if __name__ == "__main__":
    main()
