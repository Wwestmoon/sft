# SFT 数据生成器

一个用于生成高质量结构化表格问答（SFT）训练数据的工具包。该工具包提供了完整的 pipeline，从问题分解、子问题求解到答案合成和质量控制。

## 架构

该项目采用模块化设计，包含以下主要组件：

### 核心组件

1. **SFTDataGenerator** (main.py): 主生成器类，协调所有功能
2. **LLMAPI** (base.py): LLM API 接口抽象
3. **AnswerExtractor** (base.py): 答案提取器接口
4. **TableConverter** (base.py): 表格转换工具
5. **ResultSaver** (base.py): 结果保存工具

### 功能模块

1. **QuestionDecomposer** (decomposition.py): 问题分解与规划
2. **SubQuestionSolver** (sub_question_solver.py): 子问题求解（支持 Python 和 LLM 策略）
3. **AnswerSynthesizer** (answer_synthesizer.py): 答案合成
4. **QualityController** (quality_control.py): 质量控制
5. **TestRunner** (quality_control.py): 测试运行器
6. **CriticRefine** (refine_critic.py): 批评和改进框架
7. **ConcurrentProcessor** (concurrency.py): 并发处理

## 使用方法

### 命令行界面

```bash
# 运行小规模测试
python3 run_sft_generator.py --test --num_samples 10

# 生成训练数据
python3 run_sft_generator.py --generate --input_file input.jsonl --output_file output.jsonl --num_samples 100

# 自定义配置
python3 run_sft_generator.py --test --model gpt-4o-mini --base_url https://api.example.com/v1 --api_key YOUR_API_KEY --num_samples 5
```

### 编程接口

```python
from sft_generator import SFTDataGenerator
from sft_generator.base import MockLLMAPI

model_config = {
    "model": "gpt-4o-mini",
    "base_url": "https://api.example.com/v1",
    "api_key": "YOUR_API_KEY",
    "temperature": 0.1,
    "top_p": 0.9
}

generator = SFTDataGenerator(model_config)

# 生成单个响应
response = generator.generate_single_response(markdown_table, question)

# 批量生成训练数据
results = generator.generate_training_data(input_file, output_file, num_samples=100)
```

## 工作流程

### 1. 问题分解与规划

- 将复杂问题分解为简单的子问题
- 为每个子问题选择最佳求解策略（Python 或 LLM）

### 2. 子问题求解

#### Python 求解策略

- 生成可执行的 Python 代码
- 在安全沙箱中执行代码
- 捕获数据流并生成自然语言答案

#### LLM 求解策略

- 直接使用 LLM 求解语义理解类问题
- 要求明确引用表格数据
- 捕获推理过程和证据

### 3. 答案合成

- 将子问题答案综合成完整响应
- 使用 LLM 确保响应连贯性和自然性

### 4. 质量控制

- 多次生成和验证
- 提取并匹配答案
- 基于批评进行改进（refine-critic 框架）

### 5. 数据保存

- 保存生成的训练数据为 JSONL 格式
- 包含详细的元数据信息
- 支持增量更新和数据恢复

## 配置选项

### 模型配置

```python
model_config = {
    "model": "gpt-4o-mini",  # 模型名称
    "base_url": "https://api.example.com/v1",  # API 基础 URL
    "api_key": "YOUR_API_KEY",  # API 密钥
    "temperature": 0.1,  # 采样温度
    "top_p": 0.9,  # 核采样参数
    "max_tokens": 2000  # 最大令牌数
}
```

### 执行选项

```bash
--num_samples 10  # 处理的样本数量
--max_attempts 3  # 每个样本的最大尝试次数
--verbose         # 显示详细信息
```

## 输出格式

### 训练数据格式 (JSONL)

```json
{
  "id": "sample-001",
  "table": "| Column 1 | Column 2 |\n|----------|----------|\n| Value 1  | Value 2  |",
  "question": "What is the value in column 2?",
  "expected_answer": "Value 2",
  "attempt": 1,
  "response": {
    "decomposition": [],
    "sub_question_results": [],
    "final_response": "The value in column 2 is Value 2."
  },
  "extracted_answer": "Value 2",
  "match": true,
  "error_analysis": null
}
```

## 质量控制

### 匹配策略

支持多种答案匹配策略：

1. 精确匹配
2. 包含匹配
3. 正则表达式匹配
4. 语义匹配（使用 LLM）

### 错误分析

系统会分析失败的尝试，并提供改进建议：

```json
{
  "type": "answer_mismatch",
  "reason": "Extracted answer '42' does not match expected 'forty-two'",
  "suggestion": "Consider using more context in the answer extraction process.",
  "detailed_analysis": "The model returned '42' instead of 'forty-two' because it was focused on numerical values.",
  "fix_strategy": "Add instructions to prioritize natural language answers."
}
```

## 性能优化

### 并发处理

支持多线程并发处理：

```python
# 自动根据可用资源调整并发数
generator = SFTDataGenerator(model_config)
results = generator.generate_training_data(input_file, output_file, num_samples=100)
```

### 异步处理

支持异步任务处理：

```python
from sft_generator import AsyncProcessor

async def process_sample(sample):
    # 处理单个样本的异步函数
    pass

processor = AsyncProcessor()
results = await processor.process_async(samples, process_sample, max_workers=5)
```

### 速率限制

内置速率限制机制：

```python
from sft_generator import RateLimiter

limiter = RateLimiter(max_requests=10, per_seconds=60)
with limiter:
    # 受限的 API 调用
```

## 测试

### 小规模测试

```bash
python3 run_sft_generator.py --test --num_samples 10 --max_attempts 3
```

### 性能统计

系统会记录详细的性能统计信息：

```json
{
  "total_samples": 10,
  "successful_samples": 8,
  "total_attempts": 22,
  "successful_attempts": 8,
  "sample_match_rate": 0.8,
  "attempt_match_rate": 0.36,
  "python_solver_ratio": 0.7,
  "llm_solver_ratio": 0.3,
  "average_sub_questions": 2.2,
  "error_types": {
    "answer_mismatch": 3,
    "extraction_failure": 1
  },
  "average_attempts_per_sample": 2.2
}
```

## 注意事项

1. **API 密钥安全**：保护您的 API 密钥，避免在代码中硬编码
2. **模型成本**：使用真实模型时注意成本控制
3. **速率限制**：确保遵守 API 提供商的速率限制
4. **错误处理**：实现适当的错误处理和重试机制
5. **数据存储**：确保有足够的磁盘空间存储大量训练数据

## 未来改进

1. 支持更多求解策略
2. 增强的错误分析和改进建议
3. 更好的性能优化
4. 集成更多评估指标
5. 支持模型微调
