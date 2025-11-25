"""Min-Max-GQA-Chunk method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_min_max_gqa_chunk


def apply_min_max_gqa_chunk(self, path, model_kwargs, is_qwen=False):
    """Apply Min-Max-GQA-Chunk method patch.

    Chunk-based compression using Max+min group max+min scoring.
    """
    print('Using Min-Max-GQA-Chunk (Max+min chunk-based group max+min scoring)!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_min_max_gqa_chunk, model_class)
