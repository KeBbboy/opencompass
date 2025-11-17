"""
FULL_KIVI method forward implementation.

This implementation is based on KIVI (Key-Value cache quantization with Importance-based Eviction).
KIVI uses a hybrid approach:
- Quantizes older KV cache tokens to low-bit precision (k_bits, v_bits)
- Keeps recent tokens (residual_length) in full precision for better accuracy
"""

import math
import warnings
from typing import List, Optional, Tuple, Union
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache, DynamicCache
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    logger,
    repeat_kv,
    StaticCache
)
from transformers.utils import logging

logger = logging.get_logger(__name__)

# Import KIVI quantization utilities
try:
    from .quant import triton_quantize_and_pack_along_last_dim, cuda_bmm_fA_qB_outer
    KIVI_AVAILABLE = True
except ImportError as e:
    logger.warning(
        f"KIVI quantization utilities not available: {e}\n"
        "Please compile the CUDA extension:\n"
        "  cd opencompass/models/sparity_model/patches/full_kivi/quant\n"
        "  python setup.py install"
    )
    KIVI_AVAILABLE = False


def llama_sdpa_attn_forward_FULL_KIVI(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    KIVI-based attention forward pass with KV cache quantization.

    KIVI maintains:
    - Quantized KV cache for older tokens (using k_bits and v_bits)
    - Full precision KV cache for recent tokens (residual_length)
    """

    if not KIVI_AVAILABLE:
        raise RuntimeError(
            "KIVI quantization utilities are not available. "
            "Please ensure KIVI/quant modules are properly installed."
        )

    if output_attentions:
        logger.warning_once(
            'LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` '
            'does not support `output_attentions=True`. Falling back to the manual attention implementation, '
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. '
            'This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
        )
        return super().forward(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_value=past_key_value,
            output_attentions=output_attentions,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
        )

    # Get KIVI configuration parameters
    k_bits = getattr(self.config, 'k_bits', 2)  # Default 2-bit for keys
    v_bits = getattr(self.config, 'v_bits', 2)  # Default 2-bit for values
    group_size = getattr(self.config, 'group_size', 32)  # Default group size 32
    residual_length = getattr(self.config, 'residual_length', 128)  # Default keep last 128 tokens

    # Initialize KIVI cache storage
    if not hasattr(self, 'kivi_cache'):
        self.kivi_cache = {}

    bsz, q_len, _ = hidden_states.size()

    # Project to Q, K, V
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Reshape to multi-head format
    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # Calculate KV sequence length
    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                'with a layer index.'
            )

        if hasattr(self, 'kv_seq_len'):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)

    # Apply RoPE
    if position_embeddings is None:
        logger.warning_once(
            'The attention layers in this model are transitioning from computing the RoPE embeddings internally '
            'through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed '
            '`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be '
            'removed and `position_embeddings` will be mandatory.'
        )
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    # KIVI KV cache management
    if past_key_value is not None:
        cache_kwargs = {'sin': sin, 'cos': cos, 'cache_position': cache_position}

        # Retrieve KIVI cache components for this layer
        layer_cache = self.kivi_cache.get(self.layer_idx, {})
        key_states_quant_trans = layer_cache.get('key_quant', None)
        key_states_full = layer_cache.get('key_full', None)
        key_scale_trans = layer_cache.get('key_scale', None)
        key_mn_trans = layer_cache.get('key_mn', None)
        value_states_quant = layer_cache.get('value_quant', None)
        value_states_full = layer_cache.get('value_full', None)
        value_scale = layer_cache.get('value_scale', None)
        value_mn = layer_cache.get('value_mn', None)

        if key_states.shape[-2] == kv_seq_len:
            # Prefill phase: quantize and store KV cache
            self.kv_seq_len = kv_seq_len

            print(f"\n[KIVI - Layer {self.layer_idx}] Prefill phase - Quantizing KV cache")
            print(f"  k_bits={k_bits}, v_bits={v_bits}, group_size={group_size}, residual_length={residual_length}")
            print(f"  Key states shape: {key_states.shape}")

            # Quantize keys
            if key_states.shape[-2] % residual_length != 0:
                if key_states.shape[-2] < residual_length:
                    key_states_quant = None
                    key_states_full = key_states
                else:
                    key_states_quant = key_states[:, :, :-(key_states.shape[-2] % residual_length), :].contiguous()
                    key_states_full = key_states[:, :, -(key_states.shape[-2] % residual_length):, :].contiguous()
            else:
                key_states_quant = key_states
                key_states_full = None

            if key_states_quant is not None:
                key_states_quant_trans, key_scale_trans, key_mn_trans = triton_quantize_and_pack_along_last_dim(
                    key_states_quant.transpose(2, 3).contiguous(), group_size, k_bits
                )
            else:
                key_states_quant_trans = None
                key_scale_trans = None
                key_mn_trans = None

            # Quantize values
            if value_states.shape[-2] <= residual_length:
                value_states_quant = None
                value_states_full = value_states
                value_scale = None
                value_mn = None
            else:
                value_states_quant = value_states[:, :, :-residual_length, :].contiguous()
                value_states_full = value_states[:, :, -residual_length:, :].contiguous()
                value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(
                    value_states_quant, group_size, v_bits
                )

            # Store quantized cache
            self.kivi_cache[self.layer_idx] = {
                'key_quant': key_states_quant_trans,
                'key_full': key_states_full,
                'key_scale': key_scale_trans,
                'key_mn': key_mn_trans,
                'value_quant': value_states_quant,
                'value_full': value_states_full,
                'value_scale': value_scale,
                'value_mn': value_mn,
            }

        else:
            # Decode phase: update cache with new tokens
            self.kv_seq_len += q_len

            print(f"\n[KIVI - Layer {self.layer_idx}] Decode phase - Updating cache")

            # Compute attention with quantized keys
            if key_states_quant_trans is not None:
                att_qkquant = cuda_bmm_fA_qB_outer(
                    group_size, query_states, key_states_quant_trans,
                    key_scale_trans, key_mn_trans, k_bits
                )
            else:
                att_qkquant = None

            # Update full precision key cache
            if key_states_full is not None:
                key_states_full = torch.cat([key_states_full, key_states], dim=2)
            else:
                key_states_full = key_states

            # Compute attention with full precision keys
            att_qkfull = torch.matmul(query_states, repeat_kv(key_states_full, self.num_key_value_groups).transpose(2, 3))

            # Combine quantized and full attention scores
            if att_qkquant is not None:
                attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(self.head_dim)
            else:
                attn_weights = att_qkfull / math.sqrt(self.head_dim)

            # Quantize residual keys if threshold reached
            if key_states_full.shape[-2] == residual_length:
                assert residual_length % group_size == 0
                key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = triton_quantize_and_pack_along_last_dim(
                    key_states_full.transpose(2, 3).contiguous(), group_size, k_bits
                )
                key_states_full = None

                if key_states_quant_trans is not None:
                    key_states_quant_trans = torch.cat([key_states_quant_trans, key_states_quant_trans_new], dim=3)
                    key_scale_trans = torch.cat([key_scale_trans, key_scale_trans_new], dim=3)
                    key_mn_trans = torch.cat([key_mn_trans, key_mn_trans_new], dim=3)
                else:
                    key_states_quant_trans = key_states_quant_trans_new
                    key_scale_trans = key_scale_trans_new
                    key_mn_trans = key_mn_trans_new

            # Apply attention mask
            causal_mask = attention_mask
            if attention_mask is not None:
                causal_mask = causal_mask[:, :, :, :kv_seq_len]
                attn_weights = attn_weights + causal_mask
                attn_weights = torch.max(attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min))

            # Softmax
            attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

            # Update value cache
            value_states_full = torch.cat([value_states_full, value_states], dim=2)
            value_full_length = value_states_full.shape[-2]

            # Compute attention output with quantized values
            if value_states_quant is None:
                attn_output = torch.matmul(attn_weights, repeat_kv(value_states_full, self.num_key_value_groups))
            else:
                attn_output = cuda_bmm_fA_qB_outer(
                    group_size, attn_weights[:, :, :, :-value_full_length],
                    value_states_quant, value_scale, value_mn, v_bits
                )
                attn_output += torch.matmul(
                    attn_weights[:, :, :, -value_full_length:],
                    repeat_kv(value_states_full, self.num_key_value_groups)
                )

            # Quantize residual values if threshold reached
            if value_full_length > residual_length:
                assert value_full_length == residual_length + 1
                value_states_quant_new, scale, mn = triton_quantize_and_pack_along_last_dim(
                    value_states_full[:, :, :1, :].contiguous(), group_size, v_bits
                )
                value_states_full = value_states_full[:, :, 1:, :].contiguous()

                if value_states_quant is not None:
                    value_states_quant = torch.cat([value_states_quant, value_states_quant_new], dim=2)
                    value_scale = torch.cat([value_scale, scale], dim=2)
                    value_mn = torch.cat([value_mn, mn], dim=2)
                else:
                    value_states_quant = value_states_quant_new
                    value_scale = scale
                    value_mn = mn

            # Update cache
            self.kivi_cache[self.layer_idx] = {
                'key_quant': key_states_quant_trans,
                'key_full': key_states_full,
                'key_scale': key_scale_trans,
                'key_mn': key_mn_trans,
                'value_quant': value_states_quant,
                'value_full': value_states_full,
                'value_scale': value_scale,
                'value_mn': value_mn,
            }

        past_key_value._seen_tokens = self.kv_seq_len

    else:
        # No cache: standard attention
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, :key_states.shape[-2]]

        # Ensure contiguous for CUDA
        if query_states.device.type == 'cuda' and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.view(bsz, q_len, self.hidden_size)
    attn_output = self.o_proj(attn_output)

    return attn_output, None, past_key_value
