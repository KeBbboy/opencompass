"""RQA_mean method patch."""

from ..common import apply_simple_patch
from .forward import (
    llama_sdpa_attn_forward_RQA_mean,
    qwen2_attention_forward_RQA_mean,
    qwen3moe_attention_forward_RQA_mean,
)


def apply_RQA_mean(self, path, model_kwargs, is_qwen=False):
    """Apply RQA_mean method patch."""
    print('Using RQA_mean!')

    if is_qwen:
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(path, trust_remote_code=True)
        model_type = getattr(config, 'model_type', '').lower()
        if 'qwen3' in model_type or 'moe' in model_type:
            print(f'[RQA_mean] Detected Qwen3MoE model (model_type: {model_type})')
            forward_func = qwen3moe_attention_forward_RQA_mean
        else:
            print(f'[RQA_mean] Detected Qwen2 model (model_type: {model_type})')
            forward_func = qwen2_attention_forward_RQA_mean
    else:
        forward_func = llama_sdpa_attn_forward_RQA_mean

    model_class = "qwen" if is_qwen else "llama"
    apply_simple_patch(self, path, model_kwargs, forward_func, model_class)
