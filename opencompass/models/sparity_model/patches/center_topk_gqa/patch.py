"""Center-TopK-GQA method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_center_topk_gqa


def apply_center_topk_gqa(self, path, model_kwargs, is_qwen=False):
    """Apply Center-TopK-GQA method patch."""
    print('Using Center-TopK-GQA!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_center_topk_gqa, model_class)
