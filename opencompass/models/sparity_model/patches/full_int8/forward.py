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


def llama_sdpa_attn_forward_FULL_INT8_KV(
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
    if output_attentions:
        # TODO: Improve this warning with e.g. `model.config.attn_implementation = "manual"` once this is implemented.
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

    # Initialize scale storage for INT8 quantization if not exists
    if not hasattr(self, 'key_scale_cache'):
        self.key_scale_cache = {}
    if not hasattr(self, 'value_scale_cache'):
        self.value_scale_cache = {}

    bsz, q_len, _ = hidden_states.size()

    query_states = self.q_proj(hidden_states)
    key_states = self.k_proj(hidden_states)
    value_states = self.v_proj(hidden_states)

    query_states = query_states.view(bsz, q_len, self.num_heads,
                                     self.head_dim).transpose(1, 2)
    key_states = key_states.view(bsz, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)
    value_states = value_states.view(bsz, q_len, self.num_key_value_heads,
                                     self.head_dim).transpose(1, 2)

    kv_seq_len = key_states.shape[-2]
    # if past_key_value is not None:
    #     kv_seq_len += past_key_value.get_usable_length(kv_seq_len, self.layer_idx)
    if past_key_value is not None:
        if self.layer_idx is None:
            raise ValueError(
                f'The cache structure has changed since version v4.36. If you are using {self.__class__.__name__} '
                'for auto-regressive decoding with k/v caching, please make sure to initialize the attention class '
                'with a layer index.')

        if hasattr(self, 'kv_seq_len'):
            if self.kv_seq_len != 0:
                kv_seq_len += self.kv_seq_len
            else:
                kv_seq_len += past_key_value.get_usable_length(
                    kv_seq_len, self.layer_idx)
        else:
            kv_seq_len += past_key_value.get_usable_length(
                kv_seq_len, self.layer_idx)

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

    if past_key_value is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {
            'sin': sin,
            'cos': cos,
            'cache_position': cache_position
        }

        # 获取原始数据类型用于反量化
        original_dtype = key_states.dtype

        if key_states.shape[-2] == kv_seq_len:
            # Prefill阶段：量化并存储整个KV cache
            self.kv_seq_len = kv_seq_len

            # INT8量化 key_states 和 value_states
            print(f"\n[INT8 Quantization - Layer {self.layer_idx}] Prefill阶段 - 量化KV cache")
            print(f"  Key states shape: {key_states.shape}, dtype: {key_states.dtype}")

            key_states_int8, key_scale = quantize_kv_int8_per_token(key_states)
            value_states_int8, value_scale = quantize_kv_int8_per_token(value_states)

            # 存储量化的 scale
            self.key_scale_cache[self.layer_idx] = key_scale
            self.value_scale_cache[self.layer_idx] = value_scale

            # 将量化后的 int8 tensor 存储到 cache
            past_key_value.update(key_states_int8, value_states_int8,
                                  self.layer_idx, cache_kwargs)


            if self.layer_idx == 27:
                # 获取 method 和 max_capacity_prompt 参数
                method = getattr(self.config, 'method', 'unknown')
                max_capacity_prompt = None
                if hasattr(self.config, 'max_capacity_prompt'):
                    max_capacity_prompt = self.config.max_capacity_prompt
                elif hasattr(self.config, 'cache_kwargs') and 'max_capacity_prompt' in self.config.cache_kwargs:
                    max_capacity_prompt = self.config.cache_kwargs['max_capacity_prompt']

                # 如果 method 是 "full"，则不在文件名中添加 max_capacity_prompt
                if isinstance(method, str) and method.lower() == "full":
                    max_capacity_prompt = None

                estimate_kv_memory(past_key_value, method=method, max_capacity_prompt=max_capacity_prompt)
        else:
            # Decode阶段：量化新token，拼接到已有cache，然后反量化
            self.kv_seq_len += q_len

            # INT8量化新的 key_states 和 value_states
            print(f"\n[INT8 Quantization - Layer {self.layer_idx}] Decode阶段 - 量化新token")
            print(f"  New key states shape: {key_states.shape}, dtype: {key_states.dtype}")

            key_states_int8, key_scale_new = quantize_kv_int8_per_token(key_states)
            value_states_int8, value_scale_new = quantize_kv_int8_per_token(value_states)

            # 将量化后的新token存储到cache，返回完整的量化cache
            key_states_int8_full, value_states_int8_full = past_key_value.update(
                key_states_int8, value_states_int8, self.layer_idx, cache_kwargs)

            # 更新scale：拼接新token的scale到已有scale
            if self.layer_idx in self.key_scale_cache:
                self.key_scale_cache[self.layer_idx] = torch.cat([
                    self.key_scale_cache[self.layer_idx], key_scale_new
                ], dim=2)  # 在seq_len维度拼接
                self.value_scale_cache[self.layer_idx] = torch.cat([
                    self.value_scale_cache[self.layer_idx], value_scale_new
                ], dim=2)
            else:
                # 如果是第一次decode（之前没有prefill），直接存储
                self.key_scale_cache[self.layer_idx] = key_scale_new
                self.value_scale_cache[self.layer_idx] = value_scale_new

            # 反量化完整的cache以供计算使用
            key_states = dequantize_kv_int8(
                key_states_int8_full,
                self.key_scale_cache[self.layer_idx],
                original_dtype
            )
            value_states = dequantize_kv_int8(
                value_states_int8_full,
                self.value_scale_cache[self.layer_idx],
                original_dtype
            )

        past_key_value._seen_tokens = self.kv_seq_len

    key_states = repeat_kv(key_states, self.num_key_value_groups)
    value_states = repeat_kv(value_states, self.num_key_value_groups)

    causal_mask = attention_mask
    if attention_mask is not None:
        causal_mask = causal_mask[:, :, :, :key_states.shape[-2]]

    # SDPA with memory-efficient backend is currently (torch==2.1.2) bugged with non-contiguous inputs with custom attn_mask,
    # Reference: https://github.com/pytorch/pytorch/issues/112577.
    if query_states.device.type == 'cuda' and causal_mask is not None:
        query_states = query_states.contiguous()
        key_states = key_states.contiguous()
        value_states = value_states.contiguous()

    # We dispatch to SDPA's Flash Attention or Efficient kernels via this if statement instead of an
    # inline conditional assignment to support both torch.compile's `dynamic=True` and `fullgraph=True`
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





