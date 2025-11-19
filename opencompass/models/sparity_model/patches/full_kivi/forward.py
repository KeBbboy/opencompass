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

# Import unified memory estimation utility
from ..utils.kv_utils import estimate_kivi_memory


def llama_sdpa_attn_forward_FULL_KIVI(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Tuple[torch.Tensor]] = None,
    output_attentions: bool = False,
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

    bsz, q_len, _ = hidden_states.size()

    # Project to Q, K, V
    if self.config.pretraining_tp > 1:
        key_value_slicing = (self.num_key_value_heads * self.head_dim) // self.config.pretraining_tp
        query_slices = self.q_proj.weight.split(
            (self.num_heads * self.head_dim) // self.config.pretraining_tp, dim=0
        )
        key_slices = self.k_proj.weight.split(key_value_slicing, dim=0)
        value_slices = self.v_proj.weight.split(key_value_slicing, dim=0)

        query_states = [F.linear(hidden_states, query_slices[i]) for i in range(self.config.pretraining_tp)]
        query_states = torch.cat(query_states, dim=-1)

        key_states = [F.linear(hidden_states, key_slices[i]) for i in range(self.config.pretraining_tp)]
        key_states = torch.cat(key_states, dim=-1)

        value_states = [F.linear(hidden_states, value_slices[i]) for i in range(self.config.pretraining_tp)]
        value_states = torch.cat(value_states, dim=-1)
    else:
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    if past_key_value is not None:
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

    past_key_value = (
        key_states_quant_trans, key_states_full, key_scale_trans, key_mn_trans,
        value_states_quant, value_states_full, value_scale, value_mn, kv_seq_len
    ) if use_cache else None

    attn_output = attn_output.reshape(bsz, q_len, self.hidden_size)

    if self.config.pretraining_tp > 1:
        attn_output = attn_output.split(self.hidden_size // self.config.pretraining_tp, dim=2)
        o_proj_slices = self.o_proj.weight.split(self.hidden_size // self.config.pretraining_tp, dim=1)
        attn_output = sum([F.linear(attn_output[i], o_proj_slices[i]) for i in range(self.config.pretraining_tp)])
    else:
        attn_output = self.o_proj(attn_output)

    # Estimate memory usage at layer 27 (only during prefill)
    if self.layer_idx == 27 and past_key_value is not None and q_len > 1:
        method = getattr(self.config, 'method', 'full_kivi')
        print(f"\n[KIVI Memory] Prefill phase completed (q_len={q_len})")
        estimate_kivi_memory(past_key_value, k_bits, v_bits, group_size, residual_length, method)

    attn_weights = None
    return attn_output, attn_weights, past_key_value
