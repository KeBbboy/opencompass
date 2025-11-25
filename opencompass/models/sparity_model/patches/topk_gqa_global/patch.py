"""TopK-GQA-Global method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_TopK_gqa_global


def apply_topk_gqa_global(self, path, model_kwargs, is_qwen=False):
    """Apply TopK-GQA-Global method patch."""
    print('Using TopK-GQA-Global!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_TopK_gqa_global, model_class)
