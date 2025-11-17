# KIVI 快速开始指南

## 环境准备

### 第一步：克隆仓库

```bash
git clone git@github.com:KeBbboy/opencompass.git
cd opencompass
```

### 第二步：创建并激活 Conda 环境

```bash
conda create -n opencompass python=3.10 -y
conda activate opencompass
```

### 第三步：安装 PyTorch

推荐使用稳定版本：

```bash
pip install torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1 \
  --index-url https://download.pytorch.org/whl/cu121
```

### 第四步：安装项目依赖

```bash
pip install -r requirements.txt
```

## KIVI 特定配置

### 第五步：编译 KIVI CUDA 扩展

```bash
cd opencompass/models/sparity_model/patches/full_kivi
./install_kivi.sh
```

**预期输出**：
```
=========================================
Installing KIVI CUDA Extension
=========================================
CUDA version:
...
Building CUDA extension...
Installation complete!
=========================================
```

### 第六步：配置参数

在配置文件中已经添加了 KIVI 参数（`run_Longbench.py` 第 111-117 行）：

```python
kivi_kwargs=dict(
    k_bits=2,              # Key 量化位数
    v_bits=2,              # Value 量化位数
    group_size=32,         # 量化分组大小
    residual_length=128,   # 保持全精度的最近 token 数
)
```

## 运行 KIVI 评测

### 第七步：运行测试

```bash
# 返回项目根目录
cd /path/to/opencompass

# 设置方法为 KIVI 并运行
CUDA_VISIBLE_DEVICES=0 SPARITY_METHOD=full_KIVI \
python run.py opencompass/configs/sparity_config/run_Longbench.py --debug
```

**命令说明**：
- `CUDA_VISIBLE_DEVICES=0` - 指定使用 GPU 0
- `SPARITY_METHOD=full_KIVI` - 设置使用 KIVI 方法
- `--debug` - 启用调试模式，输出详细信息

## 验证安装

运行以下命令验证 KIVI 是否正确安装：

```python
python -c "
from opencompass.models.sparity_model.patches.full_kivi.quant import (
    triton_quantize_and_pack_along_last_dim,
    cuda_bmm_fA_qB_outer
)
print('KIVI quantization utilities imported successfully!')
"
```

**成功输出**：
```
KIVI quantization utilities imported successfully!
```

## 参数调优建议

### 场景 1：长上下文（>16K tokens）
优先内存节省：
```python
kivi_kwargs=dict(k_bits=2, v_bits=2, group_size=64, residual_length=64)
```

### 场景 2：中等上下文（8K-16K tokens）
平衡配置（默认）：
```python
kivi_kwargs=dict(k_bits=2, v_bits=2, group_size=32, residual_length=128)
```

### 场景 3：短上下文（<8K tokens）
优先质量：
```python
kivi_kwargs=dict(k_bits=4, v_bits=4, group_size=32, residual_length=256)
```

## 常见问题

### Q: 编译失败 "nvcc: command not found"
**A:** CUDA 未安装或未加入 PATH
```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
```

### Q: RuntimeError: KIVI quantization utilities not available
**A:** CUDA 扩展未编译，运行：
```bash
cd patches/full_kivi
./install_kivi.sh
```

### Q: 内存溢出 (OOM)
**A:** 降低压缩率或减小 residual_length：
```python
kivi_kwargs=dict(k_bits=2, v_bits=2, residual_length=64)  # 减小到 64
```

### Q: 生成质量下降
**A:** 增加量化位数或 residual_length：
```python
kivi_kwargs=dict(k_bits=4, v_bits=4, residual_length=256)  # 提高质量
```

## 性能对比

| 配置 | 内存使用 | 质量 | 适用场景 |
|------|---------|------|---------|
| 2-bit, res=64 | ~20% | 中等 | 超长上下文 (>32K) |
| 2-bit, res=128 | ~25% | 良好 | 长上下文 (16K-32K) |
| 4-bit, res=256 | ~40% | 优秀 | 中等上下文 (<16K) |

*相对于 full precision KV cache 的百分比

## 下一步

- 查看 `README.md` 了解详细技术说明
- 查看 `quant/README.md` 了解量化实现细节
- 调整参数进行性能测试
