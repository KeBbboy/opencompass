"""Forward pass implementation for Min-Max-GQA."""

import math
import warnings
from typing import List, Optional, Tuple, Union
import os
import time
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

# Import local init function
from .init_utils import init_min_max_gqa
from ..utils.kv_utils import estimate_kv_memory


def llama_sdpa_attn_forward_min_max_gqa(
    self,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_value: Optional[Cache] = None,
    output_attentions: bool = False,
    use_cache: bool = False,
    cache_position: Optional[torch.LongTensor] = None,
    position_embeddings: Optional[Tuple[
        torch.Tensor, torch.Tensor]] = None,  # will become mandatory in v4.45
) -> Tuple[torch.Tensor, Optional[torch.Tensor],
           Optional[Tuple[torch.Tensor]]]:
    """
    Min-Max-GQA forward pass.

    Uses Max+min group max+min scoring to compress KV cache.

    Key idea:
    1. For each KV head group (containing multiple Q heads in GQA)
    2. Compute attention scores for all Q heads in the group
    3. Use max(scores) + min(scores) as representative score for each token
    4. Select top-k tokens based on these Max+min scores
    """
    if output_attentions:
        logger.warning_once(
            'LlamaModel is using LlamaSdpaAttention, but `torch.nn.functional.scaled_dot_product_attention` does not support `output_attentions=True`. Falling back to the manual attention implementation, '
            'but specifying the manual implementation will be required from Transformers version v5.0.0 onwards. This warning can be removed using the argument `attn_implementation="eager"` when loading the model.'
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

    # Initialize Min-Max-GQA components
    init_min_max_gqa(self)
    bsz, q_len, _ = hidden_states.size()

    # Project to Q, K, V
    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    # Reshape to [bsz, num_heads, seq_len, head_dim]
    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]

    # Apply RoPE
    if position_embeddings is None:
        logger.warning_once(
            'The attention layers in this model are transitioning from computing the RoPE embeddings internally '
            'through `position_ids` (2D tensor with the indexes of the tokens), to using externally computed '
            '`position_embeddings` (Tuple of tensors, containing cos and sin). In v4.45 `position_ids` will be '
            'removed and `position_embeddings` will be mandatory.')
        cos, sin = self.rotary_emb(value_states, position_ids)
    else:
        cos, sin = position_embeddings

    query_states, key_states = apply_rotary_pos_emb(query_states, key_states,
                                                    cos, sin)

    # Update KV cache with Max+min compression
    if past_key_value is not None:
        cache_kwargs = {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        }

        # Prefill phase: compress using Max+min scoring
        if key_states.shape[-2] != 1:
            # Apply Max+min compression
            # Parameters (window_size, max_capacity_prompt, etc.) are already set in kv_cluster
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states,
                query_states,
                value_states,
                attention_mask,
                self.num_key_value_groups,
            )

            # Update cache with compressed KV
            past_key_value.update(key_states_compress, value_states_compress,
                                  self.layer_idx, cache_kwargs)

            # Estimate memory usage (debugging, only at last layer)
            if self.layer_idx == 27:
                method = getattr(self.config, 'method', 'min_max_gqa')
                max_capacity_prompt_cfg = None
                if hasattr(self.config, 'max_capacity_prompt'):
                    max_capacity_prompt_cfg = self.config.max_capacity_prompt
                elif hasattr(self.config, 'cache_kwargs') and 'max_capacity_prompt' in self.config.cache_kwargs:
                    max_capacity_prompt_cfg = self.config.cache_kwargs['max_capacity_prompt']

                if isinstance(method, str) and method.lower() == "full":
                    max_capacity_prompt_cfg = None

                estimate_kv_memory(past_key_value, method=method, max_capacity_prompt=max_capacity_prompt_cfg)

        # Decode phase: just append new token
        else:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx, cache_kwargs)


    # Repeat KV for GQA
    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    # Prepare causal mask
    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, :key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs
    # Reference: https://github.com/pytorch/pytorch/issues/112577
    if query_states.device.type == 'cuda' and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # Dispatch to SDPA
    is_causal = True if causal_mask is None and q_len > 1 else False

    attn_output = torch.nn.functional.scaled_dot_product_attention(
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
