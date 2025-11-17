"""CAM method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_CAM


def apply_cam(self, path, model_kwargs, is_qwen=False):
    """Apply CAM method patch."""
    print('Using CAM!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_CAM, model_class)
