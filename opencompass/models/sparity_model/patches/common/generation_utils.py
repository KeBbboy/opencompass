"""Generation utilities for model patching."""

import torch
from transformers.models.llama.modeling_llama import StaticCache


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


def prepare_inputs_for_generation_llama_new(
    self,
    input_ids,
    past_key_values=None,
    attention_mask=None,
    inputs_embeds=None,
    cache_position=None,
    position_ids=None,
    use_cache=True,
    **kwargs,
):
    """Prepare inputs for generation with custom KV cache handling."""
    from transformers.cache_utils import DynamicCache, Cache


    # If we have cache: let's slice `input_ids` through `cache_position`, to keep only the unprocessed tokens
    # Exception 1: when passing input_embeds, input_ids may be missing entries
    # Exception 2: some generation methods do special slicing of input_ids, so we don't need to do it here
    if past_key_values is not None:
        if inputs_embeds is not None:  # Exception 1
            input_ids = input_ids[:, -cache_position.shape[0]:]
        elif input_ids.shape[1] != cache_position.shape[
                0]:  # Default case (the "else", a no op, is Exception 2)
            input_ids = input_ids[:, cache_position]

    if attention_mask is not None and position_ids is None:
        # create position_ids on the fly for batch generation
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 1)
        if past_key_values:
            position_ids = position_ids[:, -input_ids.shape[1]:]

            # This `clone` call is needed to avoid recapturing cuda graphs with `torch.compile`'s  `mode="reduce-overhead`, as otherwise the input `position_ids` would have various stride during the decoding. Here, simply using `.contiguous()` is not sufficient as in the batch size = 1 case, `position_ids` is already contiguous but with varying stride which retriggers a capture.
            position_ids = position_ids.clone(
                memory_format=torch.contiguous_format)

    # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
    if inputs_embeds is not None and cache_position[0] == 0:
        model_inputs = {'inputs_embeds': inputs_embeds, 'input_ids': None}
    else:
        # The clone here is for the same reason as for `position_ids`.
        model_inputs = {
            'input_ids':
            input_ids.clone(memory_format=torch.contiguous_format),
            'inputs_embeds': None
        }

    if isinstance(past_key_values, StaticCache) and attention_mask.ndim == 2:
        if model_inputs['inputs_embeds'] is not None:
            batch_size, sequence_length, _ = model_inputs[
                'inputs_embeds'].shape
            device = model_inputs['inputs_embeds'].device
        else:
            batch_size, sequence_length = model_inputs['input_ids'].shape
            device = model_inputs['input_ids'].device

        dtype = self.lm_head.weight.dtype
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
