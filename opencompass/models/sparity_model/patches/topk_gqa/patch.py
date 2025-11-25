"""TopK-GQA method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_TopK_gqa


def apply_topk_gqa(self, path, model_kwargs, is_qwen=False):
    """Apply TopK-GQA method patch."""
    print('Using TopK-GQA!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_TopK_gqa, model_class)
