"""RaaS method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_RaaS, qwen2_sdpa_attn_forward_RaaS


def apply_raas(self, path, model_kwargs, is_qwen=False):
    """Apply RaaS method patch."""
    print('Using RaaS!')
    model_class = "qwen" if is_qwen else "llama"
    forward_func = qwen2_sdpa_attn_forward_RaaS if is_qwen else llama_sdpa_attn_forward_RaaS
    apply_simple_patch(self, path, model_kwargs, forward_func, model_class)
