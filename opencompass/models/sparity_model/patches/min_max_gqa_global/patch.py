"""Patch function for Min-Max-GQA-Global method."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_min_max_gqa_global


def apply_min_max_gqa_global(self, path, model_kwargs, is_qwen=False):
    """
    Apply Min-Max-GQA-Global patch.

    Difference from min_max_gqa:
    - Global token selection: all KV heads share the same token indices
    - Uses Max+min max+min scoring aggregated across all heads
    - Simplifies implementation and ensures all heads have identical sequence lengths

    Args:
        self: Model instance
        path: Model path
        model_kwargs: Model kwargs
        is_qwen: Whether it's Qwen model
    """
    print('Using Min-Max-GQA-Global (Quest max+min with global token selection)!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_min_max_gqa_global, model_class)
