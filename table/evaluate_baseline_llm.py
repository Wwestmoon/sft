#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
评估脚本，支持指定预测结果文件
调用官方的 evaluator.py 来评估 baseline 结果
"""

import subprocess
import os


def evaluate_baseline(tagged_path, prediction_path):
    """
    调用官方评估脚本评估 baseline 结果
    """
    evaluator_path = '/Users/westmoon/mycode/table/WikiTableQuestions/evaluator.py'

    # 构建评估命令
    command = [
        'python3', evaluator_path,
        '-t', tagged_path,
        prediction_path
    ]

    try:
        # 执行评估命令
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=True
        )

        # 解析输出
        output = result.stdout
        error = result.stderr

        # 打印错误信息
        if error:
            print("评估过程中的错误信息：")
            print(error)

        # 打印评估结果
        print("评估结果：")
        print(output)

        # 提取准确率
        accuracy = None
        for line in error.split('\n'):
            if 'Accuracy:' in line:
                accuracy = float(line.strip().split(': ')[1])

        return accuracy, output, error

    except subprocess.CalledProcessError as e:
        print(f"评估失败：{e}")
        print(f"错误信息：{e.stderr}")
        return None, None, e.stderr


def main(prediction_file=None):
    # 配置路径
    tagged_dataset_path = '/Users/westmoon/mycode/table/WikiTableQuestions/tagged/data'
    baseline_results_dir = '/Users/westmoon/mycode/table/baseline_results'

    # 确定使用的预测结果文件
    if prediction_file is None:
        # 查找最新的预测结果文件
        prediction_files = []
        for filename in os.listdir(baseline_results_dir):
            if filename.endswith('.tsv') and filename != 'mock_api_predictions.tsv':
                file_path = os.path.join(baseline_results_dir, filename)
                prediction_files.append((os.path.getmtime(file_path), file_path))

        if not prediction_files:
            print("未找到预测结果文件")
            return

        # 按修改时间排序，选择最新的文件
        prediction_files.sort(reverse=True, key=lambda x: x[0])
        prediction_path = prediction_files[0][1]
    else:
        prediction_path = os.path.join(baseline_results_dir, prediction_file)
        if not os.path.exists(prediction_path):
            print(f"预测结果文件 {prediction_path} 不存在")
            return

    print(f"使用预测结果文件：{prediction_path}")

    # 执行评估
    accuracy, output, error = evaluate_baseline(tagged_dataset_path, prediction_path)

    if accuracy is not None:
        print(f"\n最终准确率：{accuracy}")
    else:
        print("评估未成功完成")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="评估 LLM 预测结果")
    parser.add_argument("prediction_file", nargs="?", help="预测结果文件名（位于 baseline_results 目录下）")
    args = parser.parse_args()
    main(args.prediction_file)
