"""First Group GQA method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_first_group_gqa


def apply_first_group_gqa(self, path, model_kwargs, is_qwen=False):
    """Apply First Group GQA method patch."""
    print('Using First Group GQA!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_first_group_gqa, model_class)
