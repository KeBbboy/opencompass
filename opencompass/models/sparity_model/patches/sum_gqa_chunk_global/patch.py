"""Patch function for SnapKV-GQA-Chunk-Global method."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_sum_gqa_chunk_global


def apply_sum_gqa_chunk_global(self, path, model_kwargs, is_qwen=False):
    """
    Apply SnapKV-GQA-Chunk-Global patch.

    Difference from snapkv_gqa2:
    - Global chunk selection: all KV head groups share the same chunk indices
    - Simplifies implementation and ensures all groups have identical sequence lengths

    Args:
        self: Model instance
        path: Model path
        model_kwargs: Model kwargs
        is_qwen: Whether it's Qwen model
    """
    print('Using SnapKV-GQA-Chunk-Global (global chunk selection)!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_sum_gqa_chunk_global, model_class)
