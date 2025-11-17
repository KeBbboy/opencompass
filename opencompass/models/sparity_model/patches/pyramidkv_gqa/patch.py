"""PyramidKV_GQA method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_PyramidKV_gqa


def apply_pyramidkv_gqa(self, path, model_kwargs, is_qwen=False):
    """Apply PyramidKV_GQA method patch."""
    print('Using PyramidKV_GQA!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_PyramidKV_gqa, model_class)
