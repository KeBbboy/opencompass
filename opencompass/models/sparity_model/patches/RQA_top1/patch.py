"""RQA_top1 method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_RQA_top1


def apply_RQA_top1(self, path, model_kwargs, is_qwen=False):
    """Apply RQA_top1 method patch."""
    print('Using RQA_top1!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_RQA_top1, model_class)
