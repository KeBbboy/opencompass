"""FULL method patch."""

from ..common import apply_simple_patch
from .forward import llama_attention_forward_FULL_KV, qwen2_attention_forward_FULL_KV


def apply_full(self, path, model_kwargs, is_qwen=False):
    """Apply FULL method patch - uses original attention with KV memory estimation."""
    print('Using FULL!')
    model_class = "qwen" if is_qwen else "llama"
    forward_func = qwen2_attention_forward_FULL_KV if is_qwen else llama_attention_forward_FULL_KV
    apply_simple_patch(self, path, model_kwargs, forward_func, model_class)
