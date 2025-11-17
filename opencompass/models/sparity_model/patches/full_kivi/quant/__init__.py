"""
KIVI Quantization utilities.

This module provides KIVI's KV cache quantization functions:
- triton_quantize_and_pack_along_last_dim: Quantize and pack tensors using Triton
- cuda_bmm_fA_qB_outer: CUDA-accelerated batch matrix multiplication with quantized tensors

Note: Requires compilation of CUDA extension. Run:
    cd quant && python setup.py install
"""

from .new_pack import triton_quantize_and_pack_along_last_dim
from .matmul import cuda_bmm_fA_qB_outer

__all__ = [
    'triton_quantize_and_pack_along_last_dim',
    'cuda_bmm_fA_qB_outer',
]
