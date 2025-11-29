"""RQA_mean method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_RQA_mean


def apply_RQA_mean(self, path, model_kwargs, is_qwen=False):
    """Apply RQA_mean method patch."""
    print('Using RQA_mean!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_RQA_mean, model_class)
