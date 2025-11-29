"""RQA_sum method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_RQA_sum


def apply_RQA_sum(self, path, model_kwargs, is_qwen=False):
    """Apply RQA_sum method patch."""
    print('Using RQA_sum!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_RQA_sum, model_class)
