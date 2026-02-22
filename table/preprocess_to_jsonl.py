#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
直接从原始数据中读取 .table 文件，将训练集和测试集转换为 JSONL 格式
"""

import os
import csv
import json


def read_tsv_file(file_path):
    """
    读取 TSV 文件并返回数据
    """
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            data.append(row)
    return data


def load_table_file(context_path):
    """
    加载 .table 文件
    将路径从 csv/xxx-csv/yyy.csv 转换为 csv/xxx-csv/yyy.table
    """
    table_path = os.path.splitext(context_path)[0] + '.table'
    table_full_path = '/Users/westmoon/mycode/table/WikiTableQuestions/' + table_path

    if os.path.exists(table_full_path):
        with open(table_full_path, 'r', encoding='utf-8') as f:
            return f.read()
    else:
        print(f"警告：未找到 .table 文件 {table_full_path}")
        return ""


def preprocess_dataset(input_file, output_file):
    """
    预处理数据集
    """
    # 读取原始数据
    data = read_tsv_file(input_file)

    # 处理每个样本
    processed_data = []
    for i, sample in enumerate(data):
        # 加载 .table 文件
        table_content = load_table_file(sample['context'])

        processed_sample = {
            'id': sample['id'],
            'question': sample['utterance'],
            'table': table_content,
            'answer': sample['targetValue']
        }
        processed_data.append(processed_sample)

        if (i + 1) % 1000 == 0:
            print(f"已处理 {i + 1} 个样本")

    # 保存为 JSONL 文件
    with open(output_file, 'w', encoding='utf-8') as f:
        for sample in processed_data:
            f.write(json.dumps(sample, ensure_ascii=False) + '\n')

    print(f"数据集预处理完成，保存到 {output_file}")
    print(f"共处理 {len(processed_data)} 个样本")


def main():
    # 数据集文件路径
    training_file = '/Users/westmoon/mycode/table/WikiTableQuestions/data/training.tsv'
    test_file = '/Users/westmoon/mycode/table/WikiTableQuestions/data/pristine-unseen-tables.tsv'

    # 输出文件路径
    output_dir = '/Users/westmoon/mycode/table/processed_data'
    os.makedirs(output_dir, exist_ok=True)
    training_output = os.path.join(output_dir, 'training_processed.jsonl')
    test_output = os.path.join(output_dir, 'test_processed.jsonl')

    # 预处理训练集
    print("正在预处理训练集...")
    preprocess_dataset(training_file, training_output)

    # 预处理测试集
    print("\n正在预处理测试集...")
    preprocess_dataset(test_file, test_output)


if __name__ == "__main__":
    main()
