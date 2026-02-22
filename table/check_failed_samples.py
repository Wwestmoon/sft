import json

def check_failed_samples():
    with open('/Users/westmoon/mycode/table/training_data/training_data_generated.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            try:
                data = json.loads(line)
                # 检查是否有匹配失败的尝试
                has_failed_attempt = any(not attempt['match'] for attempt in data['attempts'])
                if has_failed_attempt:
                    print(f"样本 ID: {data['id']}")
                    print(f"问题: {data['question']}")
                    print(f"预期答案: {data['expected_answer']}")
                    # 打印失败的尝试
                    for attempt in data['attempts']:
                        if not attempt['match']:
                            print(f"尝试 {attempt['attempt']}:")
                            print(f"实际答案: {attempt['extracted_answer']}")
                            print(f"错误分析: {json.dumps(attempt.get('error_analysis', '无'), ensure_ascii=False)}")
                    print("-" * 50)
            except Exception as e:
                print(f"解析错误: {e}")
                continue

if __name__ == "__main__":
    check_failed_samples()
