"""Min-Max-GQA method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_min_max_gqa


def apply_min_max_gqa(self, path, model_kwargs, is_qwen=False):
    """Apply Min-Max-GQA method patch.

    Max+min token scoring using group max+min for GQA models.
    """
    print('Using Min-Max-GQA (Max+min group max+min scoring)!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_min_max_gqa, model_class)
