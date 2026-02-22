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
    parser = argparse.ArgumentParser(description='Convert JSONL to messages format with answer verification')
    parser.add_argument('--input', '-i',
                        default='/Users/westmoon/mycode/table/baseline_results/baseline_sft_critic_0_10.jsonl',
                        help='Path to input JSONL file')
    parser.add_argument('--output', '-o',
                        default='/Users/westmoon/mycode/table/baseline_results/baseline_sft_critic_0_10_messages.jsonl',
                        help='Path to output JSONL file')

    args = parser.parse_args()

    messages_list = []
    errors = 0

    with jsonlines.open(args.input) as reader:
        for line in reader:
            best_attempt = None
            for attempt in line['attempts']:
                if attempt['match']:
                    best_attempt = attempt
                    break

            if best_attempt:
                # 提取并验证答案
                extracted_answer = extract_final_answer(best_attempt['response'])
                if not match_answer(extracted_answer, line['expected_answer']):
                    errors += 1
                    print(f"Error: Entry {line['id']} - Extracted answer '{extracted_answer}' does not match expected answer '{line['expected_answer']}'")
                    # 强制使用预期答案
                    extracted_answer = line['expected_answer']

                user_content = f"""Table:
{line['table']}

Question: {line['question']}

Please reason step by step and output the final answer in the specified format. The final answer should be in the format "Final Answer: [answer]".
"""

                assistant_content = best_attempt['response']

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