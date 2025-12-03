"""sum_gqa method patch."""

from ..common import apply_simple_patch
from .forward import (
    llama_attention_forward_sum_gqa,
    qwen2_attention_forward_sum_gqa,
    qwen3moe_attention_forward_sum_gqa,
)


def apply_sum_gqa(self, path, model_kwargs, is_qwen=False):
    """Apply sum_gqa method patch - KV compression with GQA support."""
    print("\n" + "="*60)
    print('[PATCH] Using sum_gqa method!')
    print(f'[PATCH] Model type: {"Qwen" if is_qwen else "Llama"}')
    print("="*60 + "\n")

    model_class = "qwen" if is_qwen else "llama"

    # Choose the correct forward function based on model type
    if is_qwen:
        # Check if it's Qwen3MoE or Qwen2
        # Load config from path since model hasn't been loaded yet
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        model_type = getattr(config, 'model_type', '').lower()
        if 'qwen3' in model_type or 'moe' in model_type:
            print(f'[SUM_GQA] Detected Qwen3MoE model (model_type: {model_type})')
            print(f'[PATCH] Forward function: {qwen3moe_attention_forward_sum_gqa.__name__}')
            forward_func = qwen3moe_attention_forward_sum_gqa
        else:
            print(f'[SUM_GQA] Detected Qwen2 model (model_type: {model_type})')
            print(f'[PATCH] Forward function: {qwen2_attention_forward_sum_gqa.__name__}')
            forward_func = qwen2_attention_forward_sum_gqa
    else:
        print(f'[PATCH] Forward function: {llama_attention_forward_sum_gqa.__name__}')
        forward_func = llama_attention_forward_sum_gqa

    apply_simple_patch(self, path, model_kwargs, forward_func, model_class)
