"""SnapKV_GQA2 method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_SnapKV_gqa2


def apply_snapkv_gqa2(self, path, model_kwargs, is_qwen=False):
    """Apply SnapKV_GQA2 method patch."""
    print('Using SnapKV_GQA2!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_SnapKV_gqa2, model_class)
