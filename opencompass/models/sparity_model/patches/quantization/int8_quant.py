"""
INT8 Quantization utilities for KV cache compression.

This module provides functions for quantizing and dequantizing KV cache tensors
to INT8 format, reducing memory usage while maintaining reasonable accuracy.
"""

from typing import Tuple
import torch


def quantize_kv_int8_per_token(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    将KV缓存张量按per-token方式量化为int8格式

    Args:
        tensor: 输入张量，形状为 [batch_size, num_heads, seq_len, head_dim]

    Returns:
        quantized_tensor: int8量化后的张量
        scale: 量化比例因子，形状为 [batch_size, num_heads, seq_len, 1]，用于反量化
    """
    # 计算每个token在head_dim维度上的绝对值最大值作为量化比例
    # 这样每个token都有自己的量化比例
    abs_max = torch.abs(tensor).amax(dim=-1, keepdim=True)  # [batch_size, num_heads, seq_len, 1]

    # 避免除零错误
    abs_max = torch.clamp(abs_max, min=1e-8)

    # 计算量化比例，int8范围是[-127, 127]
    scale = abs_max / 127.0

    # 执行量化
    quantized = torch.round(tensor / scale).clamp(-127, 127)
    quantized_int8 = quantized.to(torch.int8)

    return quantized_int8, scale


def dequantize_kv_int8(quantized_tensor: torch.Tensor, scale: torch.Tensor, target_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """
    将int8量化的KV缓存张量反量化为原始精度

    Args:
        quantized_tensor: int8量化的张量
        scale: 量化时使用的比例因子
        target_dtype: 目标数据类型

    Returns:
        dequantized_tensor: 反量化后的张量
    """
    # 将int8转换为float并应用比例因子
    dequantized = quantized_tensor.to(target_dtype) * scale

    return dequantized
