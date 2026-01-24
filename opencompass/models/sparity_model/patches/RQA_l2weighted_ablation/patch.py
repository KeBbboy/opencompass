"""RQA_l2weighted_ablation method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_RQA_l2weighted_ablation


def apply_RQA_l2weighted_ablation(self, path, model_kwargs, is_qwen=False):
    """Apply RQA_l2weighted_ablation method patch."""
    print('Using RQA_l2weighted_ablation!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_RQA_l2weighted_ablation, model_class)
