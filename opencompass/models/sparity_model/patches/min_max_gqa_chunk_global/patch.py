"""Patch function for Min-Max-GQA-Chunk-Global method."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_min_max_gqa_chunk_global


def apply_min_max_gqa_chunk_global(self, path, model_kwargs, is_qwen=False):
    """
    Apply Min-Max-GQA-Chunk-Global patch.

    Difference from min_max_gqa_chunk:
    - Global chunk selection: all KV head groups share the same chunk indices
    - Uses Max+min max+min scoring for chunks
    - Ensures all groups have identical sequence lengths

    Args:
        self: Model instance
        path: Model path
        model_kwargs: Model kwargs
        is_qwen: Whether it's Qwen model
    """
    print('Using Min-Max-GQA-Chunk-Global (Quest max+min with global chunk selection)!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_min_max_gqa_chunk_global, model_class)
