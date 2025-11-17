"""
INT8 Quantization utilities for KV cache compression.

This module provides functions for quantizing and dequantizing KV cache tensors
to INT8 format, reducing memory usage while maintaining reasonable accuracy.
"""

from typing import Tuple
import torch


def quantize_kv_int8_per_token(tensor: torch.Tensor) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    """
    将KV缓存张量按per-token方式量化为int8格式（非对称量化）

    Args:
        tensor: 输入张量，形状为 [batch_size, num_heads, seq_len, head_dim]

    Returns:
        quantized_tensor: int8量化后的张量
        (scale, zero_point): 量化参数元组
            - scale: 量化比例因子，形状为 [batch_size, num_heads, seq_len, 1]
            - zero_point: 零点偏移，形状为 [batch_size, num_heads, seq_len, 1]
    """
    # 计算每个token在head_dim维度上的最小值和最大值
    min_val = tensor.amin(dim=-1, keepdim=True)  # [batch_size, num_heads, seq_len, 1]
    max_val = tensor.amax(dim=-1, keepdim=True)  # [batch_size, num_heads, seq_len, 1]

    # 避免除零错误
    range_val = max_val - min_val
    range_val = torch.clamp(range_val, min=1e-8)

    # int8范围是[-128, 127]，共256个值
    qmin = -128
    qmax = 127

    # 计算量化比例
    scale = range_val / (qmax - qmin)

    # 计算零点（zero point）
    # zero_point = qmin - round(min_val / scale)
    zero_point = qmin - torch.round(min_val / scale)
    zero_point = torch.clamp(zero_point, qmin, qmax)

    # 执行量化
    quantized = torch.round(tensor / scale + zero_point).clamp(qmin, qmax)
    quantized_int8 = quantized.to(torch.int8)

    return quantized_int8, (scale, zero_point)


def dequantize_kv_int8(quantized_tensor: torch.Tensor, quant_params: Tuple[torch.Tensor, torch.Tensor], target_dtype: torch.dtype = torch.float16) -> torch.Tensor:
    """
    将int8量化的KV缓存张量反量化为原始精度（非对称量化）

    Args:
        quantized_tensor: int8量化的张量
        quant_params: 量化参数元组 (scale, zero_point)
            - scale: 量化时使用的比例因子
            - zero_point: 量化时使用的零点偏移
        target_dtype: 目标数据类型

    Returns:
        dequantized_tensor: 反量化后的张量
    """
    scale, zero_point = quant_params

    # 将int8转换为float，减去零点偏移，然后应用比例因子
    # dequantized = (quantized - zero_point) * scale
    dequantized = (quantized_tensor.to(target_dtype) - zero_point) * scale

    return dequantized
