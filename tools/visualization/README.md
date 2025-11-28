# KV Cache 可视化工具使用说明

本文档详细说明了 KV Cache 相关数据的保存和可视化功能。

## 📁 目录结构

所有数据统一保存在 `kv_cache_logs` 目录下：

```
kv_cache_logs/
├── topk_indices_logs/              # TopK 索引数据
│   └── {timestamp}/                # 时间戳（如: 20251127_043533）
│       └── {dataset_name}/         # 数据集名称
│           └── cap_{capacity}/     # 容量配置
│               └── *.json          # 数据文件
│
├── key_states_logs/                # Key States 数据
│   └── {timestamp}/
│       └── {dataset_name}/
│           └── cap_{capacity}/
│               └── *_keystates.json
│
├── query_states_logs/              # Query States 数据（Window部分）
│   └── {timestamp}/
│       └── {dataset_name}/
│           └── cap_{capacity}/
│               └── *_querystates.json
│
└── visualizations/                 # 可视化输出
    ├── topk_indices/               # TopK 索引可视化
    ├── key_states/                 # Key States 可视化
    └── query_states/               # Query States 可视化
```

## ⚙️ 配置参数

在配置文件中（如 `run_Longbench.py`）启用数据保存：

```python
cache_kwargs=dict(
    # ... 其他参数 ...

    # Visualization parameters
    save_indices=False,           # 启用索引保存
    save_key_states=False,        # 启用 key states 保存
    save_query_states=True,       # 启用 query states 保存（window部分）
    visualize_layer=None,         # 保存所有层（0=只保存第0层）
    max_samples_to_save=10,       # 保存前10个样本
)
```

### 参数说明：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `save_indices` | bool | False | 是否保存 TopK 索引 |
| `save_key_states` | bool | False | 是否保存 Key States（压缩前部分） |
| `save_query_states` | bool | False | 是否保存 Query States（window部分） |
| `visualize_layer` | int/None | None | 指定保存的层，None=所有层 |
| `max_samples_to_save` | int | 10 | 最多保存多少个样本 |

---

## 🎨 可视化工具

### 1. TopK 索引可视化

**脚本**: `visualize_topk_indices.py`

#### 功能说明
- 可视化 KV cache 压缩算法选择的 TopK 索引
- 支持多种图表类型：heatmap、distribution、attention、per_head、group_overlap

#### 使用方法

```bash
# 基础用法 - 单个文件
python tools/visualization/visualize_topk_indices.py \
    --input kv_cache_logs/topk_indices_logs/.../sample000_layer0_indices.json \
    --plot_types heatmap

# 批量处理 - 指定目录
python tools/visualization/visualize_topk_indices.py \
    --input_dir kv_cache_logs/topk_indices_logs/20251127_043533 \
    --dataset LongBench_narrativeqa \
    --layer 0 \
    --plot_types all

# 自定义输出目录
python tools/visualization/visualize_topk_indices.py \
    --input_dir kv_cache_logs/topk_indices_logs/20251127_043533/InfiniteBench_ensum/cap_512 \
    --plot_types heatmap distribution \
    --output_dir custom_output_dir
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 单个 JSON 文件路径 | - |
| `--input_dir` | 包含 JSON 文件的目录 | - |
| `--layer` | 指定可视化的层 | - |
| `--sample` | 指定可视化的样本 (0-9) | - |
| `--dataset` | 指定数据集名称 | - |
| `--output_dir` | 输出目录 | `kv_cache_logs/visualizations/topk_indices` |
| `--plot_types` | 图表类型 | `all` |

**可用的图表类型**:
- `heatmap` - 热力图
- `distribution` - 分布图
- `attention` - 注意力分数图
- `per_head` - 每个 head 的分析
- `group_overlap` - Group 重叠分析
- `all` - 生成所有类型

---

### 2. Key States 3D 可视化

**脚本**: `visualize_key_states_3d.py`

#### 功能说明
- 3D 可视化 Key States（压缩前的部分）
- 数据形状: `[num_kv_heads, seq_len, head_dim]`
- 坐标轴:
  - X轴: Token Position (0 到 seq_len-1)
  - Y轴: Head Dimension (0 到 head_dim-1)
  - Z轴: Key Value (实际值，不降维)

#### 使用方法

```bash
# 单个 head 的 matplotlib 3D 可视化
python tools/visualization/visualize_key_states_3d.py \
    --input kv_cache_logs/key_states_logs/.../sample000_layer0_keystates.json \
    --mode matplotlib \
    --head 0 \
    --plot_style wireframe

# 使用 plotly 交互式 3D 图
python tools/visualization/visualize_key_states_3d.py \
    --input kv_cache_logs/key_states_logs/.../sample000_layer0_keystates.json \
    --mode plotly \
    --head 0

# 批量处理所有 heads
python tools/visualization/visualize_key_states_3d.py \
    --input_dir kv_cache_logs/key_states_logs/20251127_043533/LongBench_narrativeqa/cap_512 \
    --mode matplotlib \
    --all_heads \
    --plot_style wireframe \
    --layer 0

# Heatmap 模式
python tools/visualization/visualize_key_states_3d.py \
    --input_dir kv_cache_logs/key_states_logs/20251127_043533/LongBench_narrativeqa/cap_512 \
    --mode heatmap \
    --head 0

# Comparison 模式（所有 heads 对比）
python tools/visualization/visualize_key_states_3d.py \
    --input_dir kv_cache_logs/key_states_logs/20251127_043533/LongBench_narrativeqa/cap_512 \
    --mode comparison

# Subplots 模式（所有 heads 在一张图中）
python tools/visualization/visualize_key_states_3d.py \
    --input_dir kv_cache_logs/key_states_logs/20251127_043533/LongBench_narrativeqa/cap_512 \
    --mode subplots \
    --plot_style wireframe
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--input` | 单个 JSON 文件路径 | - |
| `--input_dir` | 包含 JSON 文件的目录 | - |
| `--output_dir` | 输出目录 | `kv_cache_logs/visualizations/key_states` |
| `--layer` | 过滤指定层 | - |
| `--sample` | 过滤指定样本 | - |
| `--mode` | 可视化模式 | `matplotlib` |
| `--head` | 要可视化的 head 索引 | 0 |
| `--all_heads` | 可视化所有 heads | False |
| `--sample_tokens` | Token 采样间隔 | 自动 |
| `--sample_dims` | Dimension 采样间隔 | 自动 |
| `--plot_style` | 绘图风格 | `wireframe` |

**可用的模式**:
- `matplotlib` - 使用 matplotlib 3D 图
- `plotly` - 使用 plotly 交互式 3D 图
- `both` - 同时生成 matplotlib 和 plotly
- `heatmap` - 2D 热力图
- `comparison` - 所有 heads 对比（降维）
- `subplots` - 所有 heads 在一张图中（3D subplots）

**绘图风格**:
- `wireframe` - 细线网格（推荐，清晰）
- `surface` - 实体曲面

#### 采样设置

当数据量较大时，自动采样可以提高性能：

```bash
# 自定义采样间隔
python tools/visualization/visualize_key_states_3d.py \
    --input_dir kv_cache_logs/key_states_logs/... \
    --mode matplotlib \
    --head 0 \
    --sample_tokens 10 \    # 每隔10个token采样一次
    --sample_dims 8         # 每隔8个dimension采样一次
```

**默认采样**:
- Token 维度: `max(1, seq_len // 100)` - 最多显示100个tokens
- Dimension 维度: `max(1, head_dim // 64)` - 最多显示64个维度

#### 输出说明

对于包含负数的数据，会自动生成两个版本：

```
kv_cache_logs/visualizations/key_states/
├── matplotlib/
│   ├── original/           # 原始数据（保留负数）
│   │   └── *.png
│   └── shifted/            # 平移数据（最小值=0，便于观察差异）
│       └── *.png
└── plotly/
    └── *.html
```

---

### 3. Query States 3D 可视化

**脚本**: `visualize_query_states_3d.py`

#### 功能说明
- 3D 可视化 Query States（Window 部分）
- 数据形状: `[num_query_heads, window_size, head_dim]`
- 注意: 保存的是参与注意力重要性计算的 query（window 部分）
- 坐标轴:
  - X轴: Token Position (window 中的位置, 0 到 window_size-1)
  - Y轴: Head Dimension (0 到 head_dim-1)
  - Z轴: Query Value (实际值，不降维)

#### 使用方法

用法与 `visualize_key_states_3d.py` 基本相同：

```bash
# 单个 head 的 matplotlib 3D 可视化
python tools/visualization/visualize_query_states_3d.py \
    --input kv_cache_logs/query_states_logs/.../sample000_layer0_querystates.json \
    --mode matplotlib \
    --head 0

# 批量处理所有 heads
python tools/visualization/visualize_query_states_3d.py \
    --input_dir kv_cache_logs/query_states_logs/20251127_043533/LongBench_narrativeqa/cap_512 \
    --mode matplotlib \
    --all_heads \
    --plot_style wireframe

# 🎯 GQA Group 对比模式：查看同一 group 内不同 heads 的差异
# 模式1：overlay - 所有 heads 在一个 3D 图中，用不同颜色区分
python tools/visualization/visualize_query_states_3d.py \
    --input kv_cache_logs/query_states_logs/.../sample000_layer0_querystates.json \
    --mode group_comparison \
    --group_idx 0 \
    --comparison_type overlay

# 模式2：subplots - 每个 head 单独显示在子图中
python tools/visualization/visualize_query_states_3d.py \
    --input kv_cache_logs/query_states_logs/.../sample000_layer0_querystates.json \
    --mode group_comparison \
    --group_idx 0 \
    --comparison_type subplots

# 模式3：difference - 显示与第一个 head 的差异（使用 diverging colormap）
python tools/visualization/visualize_query_states_3d.py \
    --input kv_cache_logs/query_states_logs/.../sample000_layer0_querystates.json \
    --mode group_comparison \
    --group_idx 0 \
    --comparison_type difference
```

#### 参数说明

参数与 `visualize_key_states_3d.py` 基本相同，新增以下参数：

- `--mode group_comparison`: 启用 GQA group 对比模式
- `--group_idx`: 指定要可视化的 GQA group 索引（默认: 0）
- `--comparison_type`: 对比类型
  - `overlay`: 所有 heads 叠加在一个 3D 图中，用不同颜色和图例区分
  - `subplots`: 每个 head 单独显示在子图网格中，便于逐个对比
  - `difference`: 显示每个 head 与第一个 head（参考 head）的差异，使用 diverging colormap（红蓝色）突出差异

**应用场景**：
- **overlay**: 快速查看整体趋势，适合发现明显异常的 head
- **subplots**: 详细对比每个 head 的特征，适合精确分析
- **difference**: 量化 head 之间的差异，适合寻找最相似/不同的 head

**输出说明**：
- 如果数据包含负数，会自动生成两个版本：
  - `original/`: 保留原始值（包括负数）
  - `shifted/`: 将所有值向上平移，使最小值为 0（便于观察相对差异）
- 两个版本的图在标题中会有相应标注

**默认输出目录**: `kv_cache_logs/visualizations/query_states`

---

## 🔍 数据文件格式

### TopK 索引文件 (`*_indices.json`)

```json
{
  "timestamp": "2025-11-27 04:35:33",
  "sample_id": 0,
  "layer_idx": 0,
  "method": "snapkv",
  "capacity": 512,
  "indices_shape": [1, 4, 448],
  "indices": [...],
  "attn_scores_shape": [1, 4, 448],
  "attn_scores": [...],
  "group_info": {
    "num_kv_heads": 4,
    "num_query_heads": 32,
    "num_key_value_groups": 8,
    "is_gqa": true
  }
}
```

### Key States 文件 (`*_keystates.json`)

```json
{
  "timestamp": "2025-11-27 04:35:33",
  "sample_id": 0,
  "layer_idx": 0,
  "method": "snapkv",
  "capacity": 512,
  "shape": [1, 4, 448, 128],
  "key_states": [...],
  "group_info": {
    "num_kv_heads": 4,
    "num_query_heads": 32,
    "num_key_value_groups": 8,
    "is_gqa": true
  }
}
```

### Query States 文件 (`*_querystates.json`)

```json
{
  "timestamp": "2025-11-27 04:35:33",
  "sample_id": 0,
  "layer_idx": 0,
  "method": "snapkv",
  "capacity": 512,
  "shape": [1, 32, 64, 128],
  "query_states": [...],
  "group_info": {
    "num_kv_heads": 4,
    "num_query_heads": 32,
    "num_key_value_groups": 8,
    "is_gqa": true
  }
}
```

---

## 💡 使用技巧

### 1. 快速查看特定层的所有可视化

```bash
# 创建一个脚本批量可视化
LAYER=0
TIMESTAMP="20251127_043533"
DATASET="LongBench_narrativeqa"
CAP="cap_512"

# TopK 索引
python tools/visualization/visualize_topk_indices.py \
    --input_dir kv_cache_logs/topk_indices_logs/$TIMESTAMP/$DATASET/$CAP \
    --layer $LAYER \
    --plot_types all

# Key States
python tools/visualization/visualize_key_states_3d.py \
    --input_dir kv_cache_logs/key_states_logs/$TIMESTAMP/$DATASET/$CAP \
    --mode subplots \
    --layer $LAYER

# Query States
python tools/visualization/visualize_query_states_3d.py \
    --input_dir kv_cache_logs/query_states_logs/$TIMESTAMP/$DATASET/$CAP \
    --mode subplots \
    --layer $LAYER
```

### 2. 交互式探索（使用 plotly）

```bash
# 生成交互式 HTML 文件
python tools/visualization/visualize_key_states_3d.py \
    --input_dir kv_cache_logs/key_states_logs/.../cap_512 \
    --mode plotly \
    --head 0

# 在浏览器中打开生成的 HTML 文件进行交互式探索
```

### 3. 对比不同容量配置

```bash
# 可视化不同容量配置的结果
for cap in 256 512 1024; do
    python tools/visualization/visualize_topk_indices.py \
        --input_dir kv_cache_logs/topk_indices_logs/$TIMESTAMP/$DATASET/cap_$cap \
        --layer 0 \
        --plot_types heatmap
done
```

### 4. 数据过大时的优化

如果数据量很大导致可视化缓慢：

```bash
# 增加采样间隔
python tools/visualization/visualize_key_states_3d.py \
    --input_dir kv_cache_logs/key_states_logs/.../cap_512 \
    --mode matplotlib \
    --head 0 \
    --sample_tokens 50 \    # 更稀疏的采样
    --sample_dims 16        # 更稀疏的采样
```

### 5. 分析 GQA 中的 Query Head 差异

在 Grouped Query Attention 中，多个 query heads 共享同一个 KV head。使用 group_comparison 模式可以分析同一 group 内不同 query heads 的行为差异：

```bash
# 步骤1：先查看数据的 group 信息
python tools/visualization/visualize_query_states_3d.py \
    --input kv_cache_logs/query_states_logs/.../sample000_layer0_querystates.json \
    --mode matplotlib \
    --head 0
# 输出会显示: num_query_heads=32, num_kv_heads=4, 每个 group 有 8 个 query heads

# 步骤2：对比第 0 个 group 内的所有 query heads (heads 0-7)
# 使用 subplots 模式查看每个 head 的细节
python tools/visualization/visualize_query_states_3d.py \
    --input kv_cache_logs/query_states_logs/.../sample000_layer0_querystates.json \
    --mode group_comparison \
    --group_idx 0 \
    --comparison_type subplots

# 步骤3：使用 difference 模式量化差异
python tools/visualization/visualize_query_states_3d.py \
    --input kv_cache_logs/query_states_logs/.../sample000_layer0_querystates.json \
    --mode group_comparison \
    --group_idx 0 \
    --comparison_type difference

# 步骤4：对比不同 groups (例如 group 0 vs group 1)
python tools/visualization/visualize_query_states_3d.py \
    --input kv_cache_logs/query_states_logs/.../sample000_layer0_querystates.json \
    --mode group_comparison \
    --group_idx 0 \
    --comparison_type overlay

python tools/visualization/visualize_query_states_3d.py \
    --input kv_cache_logs/query_states_logs/.../sample000_layer0_querystates.json \
    --mode group_comparison \
    --group_idx 1 \
    --comparison_type overlay
```

**分析要点**：
- 如果同一 group 内的 heads 差异很大（difference 模式显示较大值），说明这些 heads 关注不同的特征
- 如果 heads 非常相似（difference 模式接近 0），可能存在冗余，compression 对这些 heads 影响较小
- 通过对比不同 groups，可以理解模型如何分配注意力资源

---

## 🐛 常见问题

### Q1: 为什么生成了 `original` 和 `shifted` 两个版本？

**A**: 当数据中包含负数时，会自动生成两个版本：
- `original/` - 保留原始数据的绝对值（包括负数）
- `shifted/` - 将所有值向上平移，使最小值为0，便于观察相对差异

### Q2: 如何只保存特定层的数据？

**A**: 在配置文件中设置 `visualize_layer`：
```python
visualize_layer=0,  # 只保存第0层
```

### Q3: 数据保存在哪里？

**A**: 所有数据统一保存在项目根目录下的 `kv_cache_logs/` 文件夹中。

### Q4: 如何清理旧数据？

**A**: 直接删除对应时间戳的文件夹：
```bash
rm -rf kv_cache_logs/topk_indices_logs/20251127_043533
rm -rf kv_cache_logs/key_states_logs/20251127_043533
rm -rf kv_cache_logs/query_states_logs/20251127_043533
```

### Q5: 可视化脚本支持哪些文件格式？

**A**: 输入只支持 JSON 格式，输出支持：
- matplotlib: PNG (默认 300 DPI)
- plotly: HTML (交互式)

### Q6: 什么时候使用 group_comparison 模式？

**A**: `group_comparison` 模式专门用于分析 GQA (Grouped Query Attention) 中同一 group 内不同 query heads 的行为差异。适用场景：

1. **理解 Query Head 冗余性**：如果同一 group 内的多个 heads 非常相似，说明可能存在冗余
2. **分析注意力多样性**：通过对比 heads 差异，了解模型如何分配注意力资源
3. **调试 KV 压缩算法**：观察压缩后不同 heads 的表现差异

**三种对比类型的选择**：
- `overlay`: 快速概览，适合发现异常 head
- `subplots`: 详细对比，适合逐个分析每个 head
- `difference`: 量化差异，适合寻找最相似/不同的 head pairs

**示例**：如果模型有 32 个 query heads 和 4 个 KV heads，每个 group 包含 8 个 query heads。使用 `--group_idx 0` 会对比 heads 0-7 的差异。

---

## 📊 输出示例

### TopK 索引 Heatmap
显示每个 head 选择的 token 位置热力图。

### Key States 3D Wireframe
使用细线网格显示 key states 在 token position 和 dimension 维度的分布。

### Query States Subplots
在一张图中显示所有 query heads 的 3D 分布，便于对比。

---

## 📝 更新日志

### 2025-11-27
- ✨ 统一目录结构，所有数据保存到 `kv_cache_logs/`
- ✨ 添加 Query States 保存和可视化功能
- ✨ 可视化输出统一到 `kv_cache_logs/visualizations/`
- ✨ 新增 `group_comparison` 模式：对比同一 GQA group 内不同 query heads 的差异
  - 支持三种对比类型：overlay（叠加）、subplots（子图）、difference（差异）
  - 帮助分析 query head 冗余性和注意力多样性
- 🐛 修复了 `save_key_states` 和 `save_query_states` 缺少默认值检查的问题
- 📝 完善文档和使用说明

---

## 🔗 相关链接

- [OpenCompass 文档](https://opencompass.readthedocs.io/)
- [SnapKV 论文](https://arxiv.org/abs/2404.14469)

---

## 📧 反馈与支持

如有问题或建议，请提交 Issue 或 Pull Request。
