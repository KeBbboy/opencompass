"""SnapKV_GQA3 method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_SnapKV_gqa3


def apply_snapkv_gqa3(self, path, model_kwargs, is_qwen=False):
    """Apply SnapKV_GQA3 method patch."""
    print('Using SnapKV_GQA3!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_SnapKV_gqa3, model_class)
