"""
FULL_KIVI method forward implementation.

This implementation is based on KIVI (Key-Value cache quantization with Importance-based Eviction).
KIVI uses a hybrid approach:
- Quantizes older KV cache tokens to low-bit precision (k_bits, v_bits)
- Keeps recent tokens (residual_length) in full precision for better accuracy
"""

import math
import warnings
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import (
    apply_rotary_pos_emb,
    repeat_kv,
)
from transformers.utils import logging

logger = logging.get_logger(__name__)


# KIVI Cache class to make it compatible with transformers' Cache API
class KIVICache(Cache):
    """
    Custom Cache class for KIVI that stores quantized KV cache.

    This class extends transformers.cache_utils.Cache to be properly recognized
    by Qwen2Model and other transformer models.
    """
    def __init__(self):
        super().__init__()
        # Store KIVI cache tuples for each layer
        # Each tuple: (k_quant, k_full, k_scale, k_mn, v_quant, v_full, v_scale, v_mn, kv_seq_len)
        self.key_cache = []  # List of KIVI cache tuples per layer
        self._seen_tokens = 0  # For compatibility

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):  # noqa: ARG002
        """
        Update cache for a specific layer with KIVI cache tuple.

        For KIVI, key_states is the full KIVI cache tuple, value_states is unused.
        """
        # For KIVI, we expect key_states to be the full cache tuple
        if isinstance(key_states, tuple) and len(key_states) == 9:
            # Ensure we have enough layers
            while len(self.key_cache) <= layer_idx:
                self.key_cache.append(None)

            self.key_cache[layer_idx] = key_states
            # Update sequence length from cache tuple
            self._seen_tokens = key_states[-1]  # kv_seq_len is the last element

            # Return empty tensors (not used in KIVI forward)
            return key_states[1], key_states[5]  # key_states_full, value_states_full
        else:
            raise ValueError(f"KIVI Cache expects 9-tuple, got {type(key_states)}")

    def get_seq_length(self, layer_idx=0):
        """Get the sequence length for a specific layer."""
        if layer_idx < len(self.key_cache) and self.key_cache[layer_idx] is not None:
            return self.key_cache[layer_idx][-1]  # kv_seq_len
        return 0

    def get_max_length(self):
        """Get maximum cache length (for compatibility)."""
        return None  # Dynamic cache has no max length

    def to_legacy_cache(self):
        """Convert to legacy tuple format (tuple of layer caches)."""
        return tuple(self.key_cache)

    def __len__(self):
        return len(self.key_cache)

    def __getitem__(self, layer_idx):
        """Get cache for a specific layer."""
        if layer_idx < len(self.key_cache):
            return self.key_cache[layer_idx]
        return None

    def __repr__(self):
        return f"KIVICache(num_layers={len(self.key_cache)}, seq_length={self._seen_tokens})"

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

# Import unified memory estimation utility
from ..utils.kv_utils import estimate_kivi_memory


def llama_sdpa_attn_forward_FULL_KIVI(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,  # noqa: ARG001
    use_cache: bool = False,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[Tuple[torch.Tensor]]]:
    """
    KIVI-based attention forward pass - LlamaFlashAttention_KIVI style.

    Cache structure (tuple):
        (key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans,
         value_states_quant, value_states_full, value_scale, value_mn, kv_seq_len)
    """

    if not KIVI_AVAILABLE:
        raise RuntimeError(
            "KIVI quantization utilities are not available. "
            "Please ensure KIVI/quant modules are properly installed."
        )

    if "padding_mask" in kwargs:
        warnings.warn(
            "Passing `padding_mask` is deprecated and will be removed in v4.37. "
            "Please make sure use `attention_mask` instead."
        )

    # Get KIVI configuration parameters
    k_bits = getattr(self.config, 'k_bits', 2)
    v_bits = getattr(self.config, 'v_bits', 2)
    group_size = getattr(self.config, 'group_size', 32)
    residual_length = getattr(self.config, 'residual_length', 32)

    # Debug: Print KIVI parameters once per model (using layer 0)
    layer_idx = self.layer_idx if hasattr(self, 'layer_idx') else 0
    if layer_idx == 0 and not hasattr(self, '_kivi_params_printed'):
        print("\n" + "="*60)
        print("[KIVI Forward] Reading parameters from self.config:")
        print(f"  k_bits: {k_bits}")
        print(f"  v_bits: {v_bits}")
        print(f"  group_size: {group_size}")
        print(f"  residual_length: {residual_length}")
        print(f"  hasattr(self.config, 'k_bits'): {hasattr(self.config, 'k_bits')}")
        if hasattr(self.config, 'k_bits'):
            print(f"  self.config.k_bits: {self.config.k_bits}")
        print("="*60 + "\n")
        self._kivi_params_printed = True

    bsz, q_len, _ = hidden_states.size()


    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    # Handle different cache types
    original_cache = past_key_value  # Keep reference to original cache object
    layer_past_kv = None
    layer_idx = self.layer_idx if hasattr(self, 'layer_idx') else 0

    if past_key_value is not None:
        if isinstance(past_key_value, KIVICache):
            # Extract cache for this layer
            layer_past_kv = past_key_value[layer_idx]

        elif isinstance(past_key_value, tuple):
            # Legacy format: could be tuple of tuples (all layers) or single layer tuple
            if len(past_key_value) > 0 and isinstance(past_key_value[0], tuple):
                # Tuple of tuples - extract this layer's cache
                if layer_idx < len(past_key_value):
                    layer_past_kv = past_key_value[layer_idx]
            elif len(past_key_value) == 9:
                # Single layer KIVI tuple
                layer_past_kv = past_key_value
            else:
                # Unknown tuple format
                layer_past_kv = None

        else:
            # Try other Cache types (DynamicCache, etc.)
            if isinstance(past_key_value, Cache):
                # Convert DynamicCache to KIVICache ONCE (shared across all layers)
                past_key_value_tuple = past_key_value.to_legacy_cache()
                if len(past_key_value_tuple) == 0:
                    # Empty cache - create new KIVICache for this forward pass
                    original_cache = KIVICache()
                    layer_past_kv = None
                else:
                    # Non-empty cache - convert entire DynamicCache to KIVICache
                    original_cache = KIVICache()
                    for idx, layer_cache in enumerate(past_key_value_tuple):
                        if layer_cache is not None:
                            original_cache.update(layer_cache, None, idx)
                    # Extract this layer's cache
                    layer_past_kv = original_cache[layer_idx]
            else:
                raise TypeError(
                    f"KIVI forward expects KIVICache, tuple, or Cache object, but got {type(past_key_value)}"
                )

    # Rename for clarity: layer_past_kv is the KIVI tuple for this specific layer
    past_key_value = layer_past_kv

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
        # KIVI uses custom tuple format: (k_quant, k_full, k_scale, k_mn, v_quant, v_full, v_scale, v_mn, seq_len)
        kv_seq_len += past_key_value[-1]

    cos, sin = self.rotary_emb(value_states, position_ids)
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin, position_ids)

    # KIVI cache management - Decode phase
    if past_key_value is not None:
        key_states_quant_trans = past_key_value[0]
        key_states_full = past_key_value[1]
        key_scale_trans = past_key_value[2]
        key_mn_trans = past_key_value[3]
        value_states_quant = past_key_value[4]
        value_states_full = past_key_value[5]
        value_scale = past_key_value[6]
        value_mn = past_key_value[7]

        if key_states_quant_trans is not None:
            att_qkquant = cuda_bmm_fA_qB_outer(
                group_size, query_states, key_states_quant_trans,
                key_scale_trans, key_mn_trans, k_bits
            )
        else:
            att_qkquant = None

        if key_states_full is not None:
            key_states_full = torch.cat([key_states_full, key_states], dim=2)
        else:
            key_states_full = key_states

        att_qkfull = torch.matmul(query_states, repeat_kv(key_states_full, self.num_key_value_groups).transpose(2, 3))

        if att_qkquant is not None:
            attn_weights = torch.cat([att_qkquant, att_qkfull], dim=-1) / math.sqrt(self.head_dim)
        else:
            attn_weights = att_qkfull / math.sqrt(self.head_dim)

        if key_states_full.shape[-2] == residual_length:
            assert residual_length % group_size == 0
            key_states_quant_trans_new, key_scale_trans_new, key_mn_trans_new = triton_quantize_and_pack_along_last_dim(
                key_states_full.transpose(2, 3).contiguous(),
                group_size,
                k_bits
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

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights + attention_mask
            attn_weights = torch.max(
                attn_weights, torch.tensor(torch.finfo(attn_weights.dtype).min)
            )

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        value_states_full = torch.cat([value_states_full, value_states], dim=2)
        value_full_length = value_states_full.shape[-2]

        if value_states_quant is None:
            attn_output = torch.matmul(attn_weights, value_states_full)
        else:
            attn_output = cuda_bmm_fA_qB_outer(
                group_size, attn_weights[:, :, :, :-value_full_length], value_states_quant,
                value_scale, value_mn, v_bits
            )
            attn_output += torch.matmul(attn_weights[:, :, :, -value_full_length:], repeat_kv(value_states_full, self.num_key_value_groups))

        attn_output = attn_output.transpose(1, 2).contiguous()

        if value_full_length > residual_length:
            assert value_full_length == residual_length + 1
            value_states_quant_new, scale, mn = triton_quantize_and_pack_along_last_dim(
                value_states_full[:, :, :1, :].contiguous(),
                group_size,
                v_bits
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

    else:
        # Prefill phase: use scaled_dot_product_attention
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Use SDPA
        key_states_repeat = repeat_kv(key_states, self.num_key_value_groups)
        value_states_repeat = repeat_kv(value_states, self.num_key_value_groups)

        causal_mask = attention_mask
        if attention_mask is not None:
            causal_mask = causal_mask[:, :, :, :key_states_repeat.shape[-2]]

        if query_states.device.type == 'cuda' and causal_mask is not None:
            query_states = query_states.contiguous()
            key_states_repeat = key_states_repeat.contiguous()
            value_states_repeat = value_states_repeat.contiguous()

        is_causal = True if causal_mask is None and q_len > 1 else False

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states_repeat,
            value_states_repeat,
            attn_mask=causal_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=is_causal,
        )
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Quantize for cache
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

        if value_states.shape[-2] <= residual_length:
            value_states_quant = None
            value_states_full = value_states
            value_scale = None
            value_mn = None
        else:
            value_states_quant = value_states[:, :, :-residual_length, :].contiguous()
            value_states_full = value_states[:, :, -residual_length:, :].contiguous()
            value_states_quant, value_scale, value_mn = triton_quantize_and_pack_along_last_dim(
                value_states_quant,
                group_size,
                v_bits
            )

    # Prepare cache for return
    if use_cache:
        # Create KIVI cache tuple for this layer
        kivi_cache_tuple = (
            key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans,
            value_states_quant, value_states_full, value_scale, value_mn, kv_seq_len
        )

        # Estimate memory usage at layer 27 (only during prefill)
        if layer_idx == 27 and q_len > 1:
            method = getattr(self.config, 'method', 'full_kivi')
            print(f"\n[KIVI Memory] Prefill phase completed (q_len={q_len})")
            estimate_kivi_memory(kivi_cache_tuple, k_bits, v_bits, group_size, residual_length, method)

        # Update cache using the Cache API
        if isinstance(original_cache, KIVICache):
            # Update KIVICache and return it (cache is shared across layers)
            original_cache.update(kivi_cache_tuple, None, layer_idx)
            return_cache = original_cache
        elif original_cache is None:
            # First layer in first forward pass - create new KIVICache
            return_cache = KIVICache()
            return_cache.update(kivi_cache_tuple, None, layer_idx)
        else:
            # Shouldn't reach here if logic above is correct
            raise RuntimeError(
                f"Unexpected cache type after processing: {type(original_cache)}. "
                "Expected KIVICache or None."
            )
    else:
        return_cache = None

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)


    attn_output = self.o_proj(attn_output)

    attn_weights = None
    return attn_output, attn_weights, return_cache
