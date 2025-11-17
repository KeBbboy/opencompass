"""FULL_KIVI method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_FULL_KIVI


def apply_full_KIVI(self, path, model_kwargs, is_qwen=False):
    """Apply FULL_KIVI method patch."""
    print('Using FULL_KIVI!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_FULL_KIVI, model_class)
