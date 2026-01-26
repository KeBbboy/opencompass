"""RQA_L2_hydrid method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_RQA_L2_hydrid


def apply_RQA_L2_hydrid(self, path, model_kwargs, is_qwen=False):
    """Apply RQA_L2_hydrid method patch."""
    print('Using RQA_L2_hydrid!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_RQA_L2_hydrid, model_class)
