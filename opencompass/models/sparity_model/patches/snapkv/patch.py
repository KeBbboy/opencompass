"""SnapKV method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_SnapKV


def apply_snapkv(self, path, model_kwargs, is_qwen=False):
    """Apply SnapKV method patch."""
    print('Using SnapKV!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_SnapKV, model_class)
