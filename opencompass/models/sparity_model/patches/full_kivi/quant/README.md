# KIVI Quantization CUDA Extension

This directory contains the KIVI quantization utilities, including CUDA kernels for efficient KV cache quantization.

## Installation

### 方式 1：使用安装脚本（推荐）

从 `full_kivi` 目录运行：

```bash
cd /path/to/opencompass/models/sparity_model/patches/full_kivi
./install_kivi.sh
```

### 方式 2：手动编译

```bash
cd /path/to/opencompass/models/sparity_model/patches/full_kivi/quant
python setup.py install
```

## Requirements

- PyTorch with CUDA support
- CUDA Toolkit (compatible with your PyTorch version)
- Triton
- C++ compiler with C++17 support

## Files

- `new_pack.py`: Triton-based quantization and packing kernels
- `matmul.py`: CUDA-based quantized matrix multiplication
- `setup.py`: Build script for CUDA extension
- `csrc/`: CUDA source code
  - `gemv_cuda.cu`: CUDA kernels for GEMV operations
  - `pybind.cpp`: Python bindings
  - `gemv_cuda.h`: Header file

## Usage

After compilation, the module can be imported:

```python
from patches.full_kivi.quant import (
    triton_quantize_and_pack_along_last_dim,
    cuda_bmm_fA_qB_outer
)
```

## Troubleshooting

If compilation fails:

1. Check CUDA installation: `nvcc --version`
2. Check PyTorch CUDA version: `python -c "import torch; print(torch.version.cuda)"`
3. Ensure CUDA toolkit version matches PyTorch
4. Check compiler support for C++17
