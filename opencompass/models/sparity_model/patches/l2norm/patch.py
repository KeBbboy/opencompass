"""L2Norm method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_L2Norm


def apply_l2norm(self, path, model_kwargs, is_qwen=False):
    """Apply L2Norm method patch."""
    print('Using L2Norm!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_L2Norm, model_class)
