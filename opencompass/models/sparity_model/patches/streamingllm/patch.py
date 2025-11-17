"""StreamingLLM method patch."""

from ..common import apply_simple_patch
from .forward import llama_sdpa_attn_forward_StreamingLLM


def apply_streamingllm(self, path, model_kwargs, is_qwen=False):
    """Apply StreamingLLM method patch."""
    print('Using StreamingLLM!')
    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, llama_sdpa_attn_forward_StreamingLLM, model_class)
