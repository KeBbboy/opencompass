# TTFT 分析工具集

这个文件夹包含用于分析 TTFT (Time To First Token) 测量结果的工具。

## 📁 工具列表

### 1. `analyze_ttft.py` - 整体 TTFT 分析

分析所有 TTFT 数据的整体统计信息，按方法、任务分组。

**基本用法：**

```bash
# 分析所有 TTFT 数据
python ttft_tools/analyze_ttft.py ./ttft_logs

# 分析特定方法的数据
python ttft_tools/analyze_ttft.py ./ttft_logs/RQA_sum_capacity512

# 保存报告到文件
python ttft_tools/analyze_ttft.py ./ttft_logs -o report.txt

# 导出 CSV
python ttft_tools/analyze_ttft.py ./ttft_logs --csv data.csv
```

**输出示例：**

```
================================================================================
TTFT 分析报告
================================================================================

整体统计:
--------------------------------------------------------------------------------
  样本数量: 150
  平均 TTFT: 0.3245s
  中位数 TTFT: 0.3102s
  标准差: 0.0847s
  最小值: 0.2103s
  最大值: 0.6789s
  P50: 0.3102s
  P90: 0.4521s
  P95: 0.5012s
  P99: 0.6234s

按方法分组统计:
--------------------------------------------------------------------------------

  方法: RQA_sum
    样本数: 50
    平均: 0.3123s
    中位数: 0.3045s
    最小值: 0.2103s
    最大值: 0.5234s
```

---

### 2. `compute_dataset_ttft.py` - 按数据集统计平均 TTFT ⭐

**计算每个数据集的平均 TTFT**，这是最常用的工具，用于评估不同数据集的性能。

**基本用法：**

```bash
# 计算数据集平均 TTFT（自动生成 txt 报告）
python ttft_tools/compute_dataset_ttft.py ./ttft_logs

# 保存为指定文件
python ttft_tools/compute_dataset_ttft.py ./ttft_logs -o dataset_results.txt

# 同时生成 CSV 和 JSON
python ttft_tools/compute_dataset_ttft.py ./ttft_logs --csv results.csv --json results.json

# 分析特定方法的数据集性能
python ttft_tools/compute_dataset_ttft.py ./ttft_logs/RQA_sum_capacity512 -o rqa_sum_datasets.txt

# 不打印终端摘要，只保存文件
python ttft_tools/compute_dataset_ttft.py ./ttft_logs --no-summary -o results.txt
```

**输出示例：**

```
====================================================================================================
数据集 TTFT 统计摘要
====================================================================================================

总数据集数量: 18
总样本数量: 200

----------------------------------------------------------------------------------------------------
数据集                           样本数     平均TTFT        中位数TTFT      方法
----------------------------------------------------------------------------------------------------
narrativeqa                      15         0.4523          0.4412          RQA_sum
qasper                           12         0.3891          0.3756          RQA_sum
multifieldqa_en                  10         0.3245          0.3102          RQA_sum
hotpotqa                         18         0.2987          0.2876          RQA_sum
2wikimqa                         11         0.2765          0.2698          RQA_sum
...
----------------------------------------------------------------------------------------------------

整体平均 TTFT: 0.3245s
整体中位数 TTFT: 0.3102s
TTFT 最小值: 0.2103s (数据集: passage_retrieval_en)
TTFT 最大值: 0.4523s (数据集: narrativeqa)
====================================================================================================
```

**详细信息部分：**

```
详细信息:
====================================================================================================

数据集: narrativeqa
  样本数: 15
  平均 TTFT: 0.4523s
  中位数 TTFT: 0.4412s
  TTFT 范围: [0.3876s - 0.5234s]
  标准差: 0.0487s
  平均输入 tokens: 3456
  平均输出 tokens: 128
  方法: RQA_sum

数据集: qasper
  样本数: 12
  平均 TTFT: 0.3891s
  中位数 TTFT: 0.3756s
  TTFT 范围: [0.3102s - 0.4567s]
  标准差: 0.0392s
  平均输入 tokens: 2987
  平均输出 tokens: 156
  方法: RQA_sum
```

**CSV 输出格式：**

| dataset | samples | avg_ttft | median_ttft | min_ttft | max_ttft | std_ttft | avg_input_tokens | avg_output_tokens | method |
|---------|---------|----------|-------------|----------|----------|----------|------------------|-------------------|--------|
| narrativeqa | 15 | 0.4523 | 0.4412 | 0.3876 | 0.5234 | 0.0487 | 3456 | 128 | RQA_sum |
| qasper | 12 | 0.3891 | 0.3756 | 0.3102 | 0.4567 | 0.0392 | 2987 | 156 | RQA_sum |

---

## 🚀 快速开始

### 场景 1: 评估单个方法在所有数据集上的性能

```bash
# 运行评测
export ENABLE_TTFT=True
bash run_bash.sh

# 分析结果（按数据集）
python ttft_tools/compute_dataset_ttft.py ./ttft_logs/RQA_sum_capacity512 \
    -o results/RQA_sum_datasets.txt \
    --csv results/RQA_sum_datasets.csv
```

### 场景 2: 对比不同方法的性能

```bash
# 为每个方法生成数据集报告
python ttft_tools/compute_dataset_ttft.py ./ttft_logs/RQA_sum_capacity512 \
    -o results/RQA_sum.txt --csv results/RQA_sum.csv

python ttft_tools/compute_dataset_ttft.py ./ttft_logs/sum_gqa_capacity512 \
    -o results/sum_gqa.txt --csv results/sum_gqa.csv

python ttft_tools/compute_dataset_ttft.py ./ttft_logs/windowkv_gqa_capacity512 \
    -o results/windowkv_gqa.txt --csv results/windowkv_gqa.csv

# 然后用 Excel 或 pandas 对比 CSV 文件
```

### 场景 3: 批量处理多个配置

```bash
#!/bin/bash

# 为所有方法和配置生成报告
for dir in ./ttft_logs/*/; do
    method=$(basename "$dir")
    python ttft_tools/compute_dataset_ttft.py "$dir" \
        -o "results/${method}_datasets.txt" \
        --csv "results/${method}_datasets.csv"
done

echo "所有报告已生成到 results/ 目录"
```

---

## 📊 输出文件说明

### TXT 格式
- 易于阅读的文本报告
- 包含表格和详细信息
- 适合直接查看和分享

### CSV 格式
- 结构化数据
- 可用 Excel、pandas 等工具分析
- 适合数据对比和可视化

### JSON 格式
- 机器可读格式
- 适合程序化处理
- 保留完整的数据结构

---

## 💡 使用技巧

1. **按数据集分析最常用**：使用 `compute_dataset_ttft.py` 获取每个数据集的性能指标

2. **对比不同容量**：为同一方法的不同 capacity 生成报告，对比性能差异

3. **导出 CSV 进行可视化**：用 Python/pandas 读取 CSV 绘制图表

4. **自动化分析**：编写脚本批量处理多个实验结果

---

## 📌 注意事项

- 确保 TTFT 日志目录存在且包含 JSON 文件
- 数据集名称从 `task_name` 字段自动提取
- 如果没有指定输出文件，默认保存为 `dataset_ttft_summary.txt`
- 所有工具都支持递归搜索子目录中的 JSON 文件

---

## 🔗 相关文档

- 主文档: `../TTFT_QUICK_START.md`
- 详细文档: `../opencompass/models/sparity_model/utils/TTFT_README.md`
- 测试脚本: `../opencompass/models/sparity_model/utils/test_ttft.py`
