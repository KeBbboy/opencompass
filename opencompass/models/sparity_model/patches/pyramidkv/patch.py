"""PyramidKV method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_PyramidKV


def apply_pyramidkv(self, path, model_kwargs, is_qwen=False):
    """Apply PyramidKV method patch."""
    print('Using PyramidKV!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_PyramidKV, model_class)
