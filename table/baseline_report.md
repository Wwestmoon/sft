# WikiTableQuestions 数据集 Baseline 性能报告

## 1. 实验概述

本报告旨在展示 WikiTableQuestions 数据集的最基础 baseline 性能。我们使用了 Zero-Shot 方法，直接将表格和问题输入给大模型，没有进行任何额外的提示工程或推理步骤。

## 2. 实验设置

### 2.1 数据集

- **训练集**：/Users/westmoon/mycode/table/WikiTableQuestions/data/training.tsv
  - 样本数量：14149 个
  - 表格格式：使用原始的 .table 文件（Markdown 格式）

- **测试集**：/Users/westmoon/mycode/table/WikiTableQuestions/data/pristine-unseen-tables.tsv
  - 样本数量：4344 个
  - 表格格式：使用原始的 .table 文件（Markdown 格式）

### 2.2 方法

#### 2.2.1 模型配置

- **模型**：deepseek-v3.2
- **API 端点**：https://api.deepseek.com/chat/completions
- **API Key**：69889e93-1810-4a79-95e1-50843d43f5fb
- **输入格式**：Markdown 表格 + 问题
- **输出格式**：直接返回答案，不进行任何解释

#### 2.2.2 评估指标

使用官方提供的 evaluator.py 进行评估，计算准确率（Accuracy）。

## 3. 实验结果

### 3.1 整体性能

- **测试集样本数量**：10 个
- **正确答案数量**：0 个
- **准确率**：0.0

### 3.2 详细结果分析

所有测试的样本都未能正确回答，原因是 API 调用失败（认证失败），导致返回默认答案 "I don't know"。

### 3.3 典型案例

#### 失败案例 1：
- **ID**：nu-0
- **问题**：which country had the most cyclists finish within the top 10?
- **正确答案**：Italy
- **模型回答**：I don't know

#### 失败案例 2：
- **ID**：nu-1
- **问题**：how many people were murdered in 1940/41?
- **正确答案**：100,000
- **模型回答**：I don't know

#### 失败案例 3：
- **ID**：nu-2
- **问题**：how long did it take for the new york americans to win the national cup after 1936?
- **正确答案**：17 years
- **模型回答**：I don't know

## 4. 失败原因分析

1. **API 认证失败**：使用的 API 密钥或端点可能不正确，导致所有 API 调用都失败。

2. **API 端点问题**：DeepSeek API 的端点可能需要调整，或者有特定的请求格式要求。

3. **零样本学习局限性**：即使使用真实的大模型，直接将原始表格和问题输入而不进行任何处理，可能也会导致较低的准确率，因为表格结构和问题需要特定的处理方法才能更好地被模型理解。

## 5. 改进建议

1. **验证 API 配置**：检查 API 密钥是否正确，确保 API 端点和请求格式符合要求。

2. **提示工程**：设计更有效的提示词，帮助模型理解表格和问题的关系，例如：
   - 明确告诉模型如何使用表格中的信息
   - 提供一些示例
   - 要求模型解释其推理过程

3. **表格预处理**：对表格进行更深入的预处理，如：
   - 提取表格的结构信息
   - 识别表格的表头和数据类型
   - 处理表格中的特殊字符和格式

4. **问题分析**：对问题进行分析和分类，帮助模型更好地理解问题类型，例如：
   - 数值计算问题
   - 比较问题
   - 时间序列问题

5. **多步推理**：对于复杂问题，采用多步推理方法，将问题分解为多个子任务，逐步解决。

## 6. 代码和数据

### 6.1 代码文件位置

- **数据预处理**：/Users/westmoon/mycode/table/preprocess_to_jsonl.py
- **Baseline 运行**：/Users/westmoon/mycode/table/run_baseline_llm.py
- **评估脚本**：/Users/westmoon/mycode/table/evaluate_baseline_llm.py
- **官方评估脚本**：/Users/westmoon/mycode/table/WikiTableQuestions/evaluator.py

### 6.2 数据文件位置

- **原始数据集**：/Users/westmoon/mycode/table/WikiTableQuestions/
- **预处理后的数据集**：/Users/westmoon/mycode/table/processed_data/
- **Baseline 结果**：/Users/westmoon/mycode/table/baseline_results/

### 6.3 支持的 LLM 提供商

脚本架构支持多种 LLM 提供商，包括：
- OpenAI（GPT-4、GPT-3.5）
- DeepSeek（deepseek-v3.2）
- Qwen
- Llama 2
- 其他 OpenAI 兼容的 API

## 7. 使用方法

### 7.1 运行 Baseline

```bash
# 运行 10 个样本
python3 run_baseline_llm.py

# 运行指定数量的样本
python3 run_baseline_llm.py --num_samples 50

# 使用不同的模型配置
python3 run_baseline_llm.py --model "gpt-4o" --base_url "https://api.openai.com" --api_key "your-key" --num_samples 100
```

### 7.2 评估结果

```bash
# 自动找到最新的预测结果文件
python3 evaluate_baseline_llm.py

# 评估指定的预测结果文件
python3 evaluate_baseline_llm.py deepseek-v3.2_predictions.tsv
```

## 8. 结论

本实验使用 DeepSeek API 建立了 WikiTableQuestions 数据集的最基础 baseline，但由于 API 认证失败，未能成功获取模型答案。尽管结果显示准确率为 0，但我们实现了一个通用的脚本架构，支持多种 LLM 提供商，为后续研究奠定了基础。

为了提高准确率，我们需要：
1. 验证 API 配置，确保能够成功调用
2. 使用真实的大模型
3. 设计有效的提示工程
4. 对表格和问题进行更深入的预处理
5. 采用更复杂的方法，如多步推理等

这个 baseline 为后续研究提供了一个参考点，我们可以通过与这个 baseline 的对比来评估我们方法的改进效果。
