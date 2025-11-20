"""WindowKV method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_windowkv


def apply_windowkv(self, path, model_kwargs, is_qwen=False):
    """Apply WindowKV method patch."""
    print('Using WindowKV!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_windowkv, model_class)
