import math
import warnings
from tkinter import NO
from typing import List, Optional, Tuple, Union
import os
import time
import csv
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
from flash_attn import flash_attn_func, flash_attn_varlen_func
from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input
from transformers.cache_utils import Cache, DynamicCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import (apply_rotary_pos_emb,
                                                      logger, repeat_kv)
from transformers.utils import logging
import os
import pandas as pd
from torch import topk,softmax
try:
    from transformers.models.llama.modeling_llama import StaticCache
except ImportError:
    from transformers import StaticCache

logger = logging.get_logger(__name__)

def build_generation_inputs(
    model,
    input_ids,
    attention_mask,
    past_key_values=None,
    cache_position=None,
    use_cache=True,
    inputs_embeds=None
):
    position_ids = None
    if not isinstance(past_key_values, tuple):
        if len(past_key_values.key_cache) == 0:
            for layer in model.model.layers:
                layer.self_attn.kv_seq_len = 0

    if past_key_values is not None:
        if inputs_embeds is not None:
            input_ids = input_ids[:, -cache_position.shape[0]:]
        elif input_ids.shape[1] != cache_position.shape[0]:
            input_ids = input_ids[:, cache_position]

    if attention_mask is not None and position_ids is None:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]
            position_ids = position_ids.clone(memory_format=torch.contiguous_format)

    if inputs_embeds is not None and cache_position[0] == 0:
        model_inputs = {'inputs_embeds': inputs_embeds, 'input_ids': None}
    else:
        model_inputs = {
            'input_ids': input_ids.clone(memory_format=torch.contiguous_format),
            'inputs_embeds': None
        }

    if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
        if model_inputs['inputs_embeds'] is not None:
            batch_size, sequence_length, _ = model_inputs['inputs_embeds'].shape
            device = model_inputs['inputs_embeds'].device
        else:
            batch_size, sequence_length = model_inputs['input_ids'].shape
            device = model_inputs['input_ids'].device

        dtype = model.lm_head.weight.dtype
        min_dtype = torch.finfo(dtype).min

        attention_mask = _prepare_4d_causal_attention_mask_with_cache_position(
            attention_mask,
            sequence_length=sequence_length,
            target_length=past_key_values.get_max_length(),
            dtype=dtype,
            device=device,
            min_dtype=min_dtype,
            cache_position=cache_position,
            batch_size=batch_size,
        )

    model_inputs.update({
        'position_ids': position_ids,
        'cache_position': cache_position,
        'past_key_values': past_key_values,
        'use_cache': use_cache,
        'attention_mask': attention_mask,
    })

    return model_inputs
def _prepare_4d_causal_attention_mask_with_cache_position(
    attention_mask: torch.Tensor,
    sequence_length: int,
    target_length: int,
    dtype: torch.dtype,
    device: torch.device,
    min_dtype: float,
    cache_position: torch.Tensor,
    batch_size: int,
):
    """Creates a causal 4D mask of shape `(batch_size, 1, query_length,
    key_value_length)` from a 2D mask of shape `(batch_size,
    key_value_length)`, or if the input `attention_mask` is already 4D, do
    nothing.

    Args:
        attention_mask (`torch.Tensor`):
            A 2D attention mask of shape `(batch_size, key_value_length)` or a 4D attention mask of shape `(batch_size, 1, query_length, key_value_length)`.
        sequence_length (`int`):
            The sequence length being processed.
        target_length (`int`):
            The target length: when generating with static cache, the mask should be as long as the static cache, to account for the 0 padding, the part of the cache that is not filled yet.
        dtype (`torch.dtype`):
            The dtype to use for the 4D attention mask.
        device (`torch.device`):
            The device to plcae the 4D attention mask on.
        min_dtype (`float`):
            The minimum value representable with the dtype `dtype`.
        cache_position (`torch.Tensor`):
            Indices depicting the position of the input sequence tokens in the sequence.
        batch_size (`torch.Tensor`):
            Batch size.
    """
    if attention_mask is not None and attention_mask.dim() == 4:
        # In this case we assume that the mask comes already in inverted form and requires no inversion or slicing.
        causal_mask = attention_mask
    else:
        causal_mask = torch.full((sequence_length, target_length),
                                 fill_value=min_dtype,
                                 dtype=dtype,
                                 device=device)
        if sequence_length != 1:
            causal_mask = torch.triu(causal_mask, diagonal=1)
        causal_mask *= torch.arange(
            target_length, device=device) > cache_position.reshape(-1, 1)
        causal_mask = causal_mask[None,
                                  None, :, :].expand(batch_size, 1, -1, -1)
        if attention_mask is not None:
            causal_mask = causal_mask.clone(
            )  # copy to contiguous memory for in-place edit
            mask_length = attention_mask.shape[-1]
            padding_mask = causal_mask[:, :, :, :
                                       mask_length] + attention_mask[:, None,
                                                                     None, :]
            padding_mask = padding_mask == 0
            causal_mask[:, :, :, :
                        mask_length] = causal_mask[:, :, :, :
                                                   mask_length].masked_fill(
                                                       padding_mask, min_dtype)

    return causal_mask