"""RQA_learned_weights method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_RQA_learned_weights


def apply_RQA_learned_weights(self, path, model_kwargs, is_qwen=False):
    """Apply RQA_learned_weights method patch."""
    print('Using RQA_learned_weights!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_RQA_learned_weights, model_class)
