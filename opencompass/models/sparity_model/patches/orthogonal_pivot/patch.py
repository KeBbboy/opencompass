"""Orthogonal Pivot Selection (Dual-Probe Retrieval) method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_orthogonal_pivot


def apply_orthogonal_pivot(self, path, model_kwargs, is_qwen=False):
    """Apply Orthogonal Pivot Selection (Dual-Probe Retrieval) method patch."""
    print('Using Orthogonal Pivot Selection (Dual-Probe Retrieval)!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_orthogonal_pivot, model_class)
