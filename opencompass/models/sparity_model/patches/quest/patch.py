"""Quest method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_Quest, qwen2_sdpa_attn_forward_Quest


def apply_quest(self, path, model_kwargs, is_qwen=False):
    """Apply Quest method patch."""
    print('Using Quest!')
    model_class = "qwen" if is_qwen else "llama"
    forward_func = qwen2_sdpa_attn_forward_Quest if is_qwen else llama_sdpa_attn_forward_Quest
    apply_simple_patch(self, path, model_kwargs, forward_func, model_class)
