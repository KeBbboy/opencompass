# TTFT (Time To First Token) 测量器使用说明

## 概述

TTFT 测量器是一个完全解耦的模块，用于精确测量模型从接收输入到生成第一个 token 的时间。该模块设计遵循以下原则：

- ✅ **完全解耦**：不修改任何核心模型代码（forward.py, patch.py 等）
- ✅ **准确测量**：通过 HuggingFace 的 StoppingCriteria 回调机制精确捕获第一个 token 生成时刻
- ✅ **易于控制**：通过配置参数轻松启用/禁用
- ✅ **独立保存**：每条数据单独保存 TTFT 记录到文件

## 启用 TTFT 测量

### 方法 1：在配置文件中启用

在你的模型配置文件（例如 `.py` 配置文件）中添加以下参数：

```python
models = [
    dict(
        type='QwenAttentionConvert',
        path='your/model/path',
        # ... 其他配置 ...

        # TTFT 测量配置
        enable_ttft=True,                    # 启用 TTFT 测量
        ttft_save_dir='./ttft_logs',        # TTFT 日志保存目录
        ttft_save_to_file=True,             # 是否保存到文件
    )
]
```

### 方法 2：通过环境变量控制

```bash
# 启用 TTFT 测量
export ENABLE_TTFT=1
export TTFT_SAVE_DIR="./my_ttft_logs"

# 运行评测
python run.py configs/your_config.py
```

### 方法 3：代码中动态启用

```python
model = QwenAttentionConvert(
    path='your/model/path',
    enable_ttft=True,
    ttft_save_dir='./ttft_logs',
    ttft_save_to_file=True
)
```

## 输出文件格式

### 文件命名

每条数据生成一个 JSON 文件，命名格式：

```
{timestamp}_{task_name}.json
```

示例：
```
20250128_143025_123456_MMLU_abstract_algebra.json
20250128_143026_234567_LongBench_hotpotqa.json
```

### 文件内容

```json
{
  "ttft_seconds": 0.3245,
  "total_time_seconds": 2.1567,
  "total_tokens": 128,
  "timestamp": "2025-01-28T14:30:25.123456",
  "decode_time_seconds": 1.8322,
  "decode_tokens": 127,
  "avg_decode_time_per_token": 0.0144,
  "decode_throughput_tokens_per_sec": 69.31,
  "input_text": "Question: What is the capital of France?",
  "output_text": "The capital of France is Paris...",
  "task_name": "MMLU_abstract_algebra",
  "extra_info": {
    "batch_size": 1,
    "input_tokens": 15,
    "output_tokens": 128,
    "max_out_len": 256,
    "method": "RQA_sum",
    "model_path": "/path/to/model"
  }
}
```

## 输出说明

### 终端输出

当 TTFT 测量启用时，你会看到：

```
🕐 TTFT: 0.3245s | Task: MMLU_abstract_algebra

======================================================================
TTFT Measurement Summary - MMLU_abstract_algebra
----------------------------------------------------------------------
  Time to First Token (TTFT): 0.3245s
  Total Generation Time:       2.1567s
  Total Tokens Generated:      128
  Avg Decode Time/Token:       0.0144s
  Decode Throughput:           69.31 tokens/s
======================================================================

📁 TTFT metrics saved to: ./ttft_logs/20250128_143025_123456_MMLU_abstract_algebra.json
```

## 性能指标解释

- **TTFT (Time To First Token)**: 从输入到第一个 token 生成的时间，包括：
  - 模型加载和准备（如果需要）
  - Prefill 阶段（处理完整输入序列）
  - 生成第一个 token

- **Total Time**: 完整生成过程的总时间

- **Decode Time**: 生成所有 token（除第一个）的时间

- **Avg Decode Time/Token**: 平均每个 token 的生成时间（解码阶段）

- **Decode Throughput**: 解码阶段的吞吐量（tokens/秒）

## 使用示例

### 示例 1：评测 MMLU 数据集并记录 TTFT

```python
from opencompass.models import QwenAttentionConvert

model = QwenAttentionConvert(
    path='Qwen/Qwen2.5-7B-Instruct',
    enable_ttft=True,
    ttft_save_dir='./ttft_logs/mmlu',
    method='RQA_sum',
    # ... 其他配置
)

# 运行评测，每条数据会自动保存 TTFT
```

### 示例 2：分析 TTFT 日志

```python
import json
import glob
import pandas as pd

# 读取所有 TTFT 日志
log_files = glob.glob('./ttft_logs/*.json')
data = []

for file in log_files:
    with open(file, 'r') as f:
        data.append(json.load(f))

# 转换为 DataFrame 进行分析
df = pd.DataFrame(data)

# 统计分析
print(f"Average TTFT: {df['ttft_seconds'].mean():.4f}s")
print(f"Median TTFT: {df['ttft_seconds'].median():.4f}s")
print(f"P95 TTFT: {df['ttft_seconds'].quantile(0.95):.4f}s")
print(f"P99 TTFT: {df['ttft_seconds'].quantile(0.99):.4f}s")
```

## 禁用 TTFT 测量

如果不需要测量 TTFT，只需：

1. 不设置 `enable_ttft=True`，或
2. 设置 `enable_ttft=False`

这样完全不会影响模型的运行性能。

## 技术实现细节

TTFT 测量器通过以下方式工作：

1. **集成点**：作为 `StoppingCriteria` 添加到 HuggingFace 的 `generate()` 方法中
2. **回调时机**：每生成一个 token，`StoppingCriteria.__call__()` 会被调用
3. **首次检测**：第一次调用时记录 TTFT，后续调用记录每步时间
4. **零侵入**：不修改任何 attention forward 代码或 patch 文件

## 注意事项

1. **批处理**：当前实现主要针对 batch_size=1 的场景，批处理场景下记录第一条数据的信息
2. **文件命名**：确保 `task_info` 参数传递正确，以便生成有意义的文件名
3. **存储空间**：长时间运行可能产生大量日志文件，建议定期清理或归档

## 故障排查

### Q: TTFT 没有被记录？

A: 检查：
1. `enable_ttft=True` 是否设置
2. `ttft_save_dir` 目录是否有写入权限
3. 查看终端是否有错误信息

### Q: 文件名中 task_name 显示为默认值？

A: 确保在调用 `generate()` 时传递了 `task_info` 参数：

```python
outputs = model.generate(
    inputs=inputs,
    max_out_len=256,
    task_info="MMLU_abstract_algebra"  # 添加这个参数
)
```

### Q: TTFT 时间看起来不准确？

A: TTFT 测量包括了整个 prefill 阶段，如果：
- 输入很长：TTFT 会较大（prefill 时间长）
- 首次运行：可能包含模型加载时间
- 建议多次运行取平均值

## 相关文件

- `ttft_measurer.py`: TTFT 测量器核心实现
- `sparity_model.py`: 集成点（generate 方法）
- `TTFT_README.md`: 本文档
