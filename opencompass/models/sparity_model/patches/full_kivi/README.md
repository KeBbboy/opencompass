# KIVI (Key-Value cache quantization with Importance-based eviction)

KIVI 是一种 KV cache 量化方法，通过将旧的 KV cache tokens 量化到低比特精度来减少内存使用，同时保持最近的 tokens 为全精度以确保准确性。

## 特性

- **低比特量化**：支持 1-8 bit 的 Key 和 Value 量化
- **分组量化**：使用可配置的 group_size 进行分组量化
- **混合精度**：最近的 tokens 保持全精度，旧 tokens 使用量化
- **CUDA 加速**：使用 Triton 和 CUDA 内核实现高效计算

## 完整部署流程

### 1. 克隆仓库

```bash
git clone git@github.com:KeBbboy/opencompass.git
cd opencompass
```

### 2. 创建并激活 Conda 环境

```bash
conda create -n opencompass python=3.10 -y
conda activate opencompass
```

### 3. 安装 PyTorch

推荐使用稳定版本：

```bash
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

### 4. 安装项目依赖

```bash
pip install -r requirements.txt
```

### 5. 编译 KIVI CUDA 扩展

KIVI 需要编译 CUDA 扩展才能使用：

```bash
# 方式 1：使用安装脚本（推荐）
cd opencompass/models/sparity_model/patches/full_kivi
./install_kivi.sh

# 方式 2：手动编译
cd opencompass/models/sparity_model/patches/full_kivi/quant
python setup.py install
```

### 6. 依赖验证

检查依赖是否正确安装：

```bash
nvcc --version  # 检查 CUDA
python -c "import torch; print(torch.__version__, torch.version.cuda)"  # 检查 PyTorch
python -c "import triton; print(triton.__version__)"  # 检查 Triton
```

**系统要求**：
- PyTorch with CUDA support (推荐 2.3.1+cu121)
- CUDA Toolkit 12.1+
- Triton
- C++ 编译器（支持 C++17）

## 使用方法

### 配置文件设置

在 `run_Longbench.py` 或其他配置文件中：

```python
models = [
    dict(
        type=Qwen_AttentionConvert,
        method='full_KIVI',  # 设置为 KIVI 方法

        cache_kwargs=dict(
            window_size=64,
            max_capacity_prompt=512,
        ),

        kivi_kwargs=dict(
            k_bits=2,              # Key 量化位数 (1-8)
            v_bits=2,              # Value 量化位数 (1-8)
            group_size=32,         # 量化分组大小
            residual_length=128,   # 保持全精度的最近 token 数
        )
    )
]
```

### 参数说明

| 参数 | 默认值 | 范围 | 说明 |
|------|--------|------|------|
| `k_bits` | 2 | 1-8 | Key 量化位数。越低压缩越大，精度越低 |
| `v_bits` | 2 | 1-8 | Value 量化位数。越低压缩越大，精度越低 |
| `group_size` | 32 | 16/32/64/128 | 量化分组大小。越大压缩越多，粒度越粗 |
| `residual_length` | 128 | 64-512 | 保持全精度的最近 token 数。越大精度越高，内存越多 |

### 推荐配置

```python
# 高压缩（内存优先，适合长上下文）
kivi_kwargs=dict(k_bits=2, v_bits=2, group_size=64, residual_length=64)

# 平衡（推荐默认配置）
kivi_kwargs=dict(k_bits=2, v_bits=2, group_size=32, residual_length=128)

# 高精度（质量优先，适合短上下文）
kivi_kwargs=dict(k_bits=4, v_bits=4, group_size=32, residual_length=256)
```

## 运行

### 基本运行命令

```bash
# 返回项目根目录
cd /path/to/opencompass

# 使用 KIVI 方法运行评测
CUDA_VISIBLE_DEVICES=0 SPARITY_METHOD=full_KIVI \
python run.py opencompass/configs/sparity_config/run_Longbench.py --debug
```

### 多 GPU 运行

```bash
# 使用多个 GPU
CUDA_VISIBLE_DEVICES=0,1,2,3 SPARITY_METHOD=full_KIVI \
python run.py opencompass/configs/sparity_config/run_Longbench.py
```

### 生产环境运行

```bash
# 不使用 debug 模式
CUDA_VISIBLE_DEVICES=0 SPARITY_METHOD=full_KIVI \
python run.py opencompass/configs/sparity_config/run_Longbench.py
```

## 目录结构

```
full_kivi/
├── __init__.py           # 模块初始化
├── patch.py              # Patch 应用逻辑
├── forward.py            # KIVI 前向传播实现
├── install_kivi.sh       # 安装脚本
├── README.md             # 本文档
└── quant/                # 量化工具
    ├── __init__.py       # 量化模块初始化
    ├── new_pack.py       # Triton 量化和打包
    ├── matmul.py         # CUDA 矩阵乘法
    ├── setup.py          # CUDA 扩展编译脚本
    ├── README.md         # 量化工具说明
    └── csrc/             # CUDA 源代码
        ├── gemv_cuda.cu  # CUDA 内核
        ├── gemv_cuda.h   # 头文件
        └── pybind.cpp    # Python 绑定
```

## 工作原理

KIVI 的核心思想：

1. **Prefill 阶段**：
   - 对输入序列进行量化
   - 将序列分为两部分：
     - 前面的 tokens → 量化为低比特（k_bits/v_bits）
     - 最后 `residual_length` 个 tokens → 保持全精度

2. **Decode 阶段**：
   - 新 token 加入全精度 cache
   - 当全精度 cache 达到 `residual_length` 时：
     - 最老的全精度 token 被量化
     - 新 token 加入全精度 cache

3. **注意力计算**：
   - 对量化部分：使用 CUDA 加速的量化矩阵乘法
   - 对全精度部分：使用标准矩阵乘法
   - 合并两部分结果

## 性能优势

- **内存节省**：2-bit 量化可节省约 75% 的 KV cache 内存
- **质量保持**：通过保留最近 tokens 的全精度，保持生成质量
- **计算加速**：CUDA 内核优化的量化操作

## 故障排除

### 编译失败

1. **CUDA 未找到**：
   ```bash
   export CUDA_HOME=/usr/local/cuda
   export PATH=$CUDA_HOME/bin:$PATH
   ```

2. **版本不匹配**：
   确保 CUDA toolkit 版本与 PyTorch CUDA 版本兼容

3. **C++ 编译器问题**：
   ```bash
   # Ubuntu/Debian
   sudo apt-get install build-essential
   ```

### 运行时错误

1. **KIVI_AVAILABLE = False**：
   - CUDA 扩展未编译，运行 `./install_kivi.sh`

2. **Memory Error**：
   - 减小 `residual_length`
   - 增加 `k_bits` 和 `v_bits`（降低压缩率）

## 参考

基于 KIVI 原始实现：
- 源代码：`/home/yichen/Sparity_v1/KIVI/`
- 论文：KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache

## 许可

遵循原 KIVI 项目许可。
