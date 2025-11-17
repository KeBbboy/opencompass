"""H2O method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_H2O


def apply_h2o(self, path, model_kwargs, is_qwen=False):
    """Apply H2O method patch."""
    print('Using H2O!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_H2O, model_class)
