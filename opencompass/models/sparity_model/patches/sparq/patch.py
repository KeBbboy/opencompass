"""SparQ method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_SparQ


def apply_sparq(self, path, model_kwargs, is_qwen=False):
    """Apply SparQ method patch."""
    print('Using SparQ!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_SparQ, model_class)
