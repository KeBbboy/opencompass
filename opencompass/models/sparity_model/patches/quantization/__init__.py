"""
Quantization utilities for KV cache compression.
"""

from .int8_quant import quantize_kv_int8_per_token, dequantize_kv_int8

__all__ = [
    'quantize_kv_int8_per_token',
    'dequantize_kv_int8',
]
