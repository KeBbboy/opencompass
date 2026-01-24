"""RQA_per_head_topk method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_RQA_per_head_topk


def apply_RQA_per_head_topk(self, path, model_kwargs, is_qwen=False):
    """Apply RQA_per_head_topk method patch."""
    print('Using RQA_per_head_topk!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_RQA_per_head_topk, model_class)
