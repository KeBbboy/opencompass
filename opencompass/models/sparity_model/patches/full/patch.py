"""FULL method patch."""

from ..common import apply_simple_patch
from .forward import (
    llama_attention_forward_FULL_KV,
    qwen2_attention_forward_FULL_KV,
    qwen3moe_attention_forward_FULL_KV,
)


def apply_full(self, path, model_kwargs, is_qwen=False):
    """Apply FULL method patch - uses original attention with KV memory estimation."""
    print('Using FULL!')
    model_class = "qwen" if is_qwen else "llama"

    # Choose the correct forward function based on model type
    if is_qwen:
        # Check if it's Qwen3MoE or Qwen2 from path or self.model_type
        # Use self.model_type (set in __init__) instead of self.model.config
        model_type = getattr(self, 'model_type', path).lower()
        if 'qwen3' in model_type or 'moe' in model_type:
            print(f'[FULL] Detected Qwen3MoE model (model_type: {model_type})')
            forward_func = qwen3moe_attention_forward_FULL_KV
        else:
            print(f'[FULL] Detected Qwen2 model (model_type: {model_type})')
            forward_func = qwen2_attention_forward_FULL_KV
    else:
        forward_func = llama_attention_forward_FULL_KV

    apply_simple_patch(self, path, model_kwargs, forward_func, model_class)
