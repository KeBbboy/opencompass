"""FULL_INT8 method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_FULL_INT8_KV


def apply_full_int8(self, path, model_kwargs, is_qwen=False):
    """Apply FULL_INT8 method patch."""
    print('Using FULL_INT8!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_FULL_INT8_KV, model_class)
