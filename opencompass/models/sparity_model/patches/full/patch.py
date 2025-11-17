"""FULL method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_FULL_KV


def apply_full(self, path, model_kwargs, is_qwen=False):
    """Apply FULL method patch."""
    print('Using FULL!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_FULL_KV, model_class)
