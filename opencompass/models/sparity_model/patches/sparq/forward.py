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
    repeat_kv)
try:
    from transformers.models.llama.modeling_llama import StaticCache
except ImportError:
    from transformers import StaticCache
from transformers.utils import logging

logger = logging.get_logger(__name__)

# Import local init function
from .init_utils import init_sparq



def llama_sdpa_attn_forward_SparQ(
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

    init_snapkv(self)

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
        if key_states.shape[-2] == kv_seq_len:
            self.kv_seq_len = kv_seq_len
            past_key_value.update(key_states, value_states,self.layer_idx, cache_kwargs)
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
        else:
            past_key_value._seen_tokens = self.kv_seq_len
            self.kv_seq_len += q_len
            key_states, value_states = past_key_value.update(key_states, value_states, self.layer_idx, cache_kwargs)
            # key_states = repeat_kv(key_states, self.num_key_value_groups)
            # value_states = repeat_kv(value_states, self.num_key_value_groups)
            batch_size = query_states.shape[0]
            n_heads =  query_states.shape[1]
            n_kv_heads = key_states.shape[1]
            head_size =  key_states.shape[3]
            seuqence = key_states.shape[2]

            query_states = query_states.reshape(batch_size,n_kv_heads, n_heads //  n_kv_heads, 1, head_size)
            key_states = key_states.reshape(batch_size, n_kv_heads, 1, seuqence,head_size)
            value_states = value_states.reshape(batch_size, n_kv_heads, 1, seuqence, head_size)
            # =========================== 进行sparq =====================
            Q = query_states
            K = key_states
            V = value_states


            r = self.config.window_size
            k = self.config.max_capacity_prompt

            i1 = topk(abs(Q).sum(dim=2, keepdim=True), r, -1).indices

            
            # 内联gather: Q_hat = gather(Q, -1, i1)
            dim_Q = -1 + (-1 < 0) * Q.ndim
            Q_hat = Q.gather(dim_Q, i1.expand(*Q.shape[:dim_Q], i1.shape[dim_Q], *Q.shape[dim_Q + 1:]))
            
            # 内联gather: K_hat = gather(K, -1, i1)
            dim_K = -1 + (-1 < 0) * K.ndim
            K_hat = K.gather(dim_K, i1.expand(*K.shape[:dim_K], i1.shape[dim_K], *K.shape[dim_K + 1:]))
            
            scale = torch.sqrt(
                Q.shape[-1]
                * abs(Q_hat).sum(dim=-1, keepdim=True)
                / abs(Q).sum(dim=-1, keepdim=True)
            )

            # 创建基本的因果掩码 (下三角矩阵)
            causal_mask = torch.tril(torch.ones(seuqence, seuqence))
            causal_mask = causal_mask.masked_fill(causal_mask == 0, float('-inf'))
            last_token_mask = causal_mask[-1:, :]  # 形状: [1, seq_len]

            expanded_mask = last_token_mask.unsqueeze(0).unsqueeze(0).unsqueeze(0)
            M = expanded_mask.expand(batch_size, n_kv_heads,  n_heads // n_kv_heads, 1, seuqence)
            M = M.to(Q_hat.device)

            s_hat = softmax(Q_hat @ K_hat.transpose(-1, -2) / scale + M, dim=-1)
            # (batch_size, n_kv_heads, n_heads // n_kv_heads, 1, sequence)

            
            
            # # 2. Gather top k positions based on approximate attention scores
            actual_k = min(k, s_hat.shape[-1])  # 确保不会越界
            i2 = topk(s_hat.sum(dim=2, keepdim=True), actual_k, -1).indices
            iKV = i2[..., 0, :, None]

            
            # # 内联gather: K_top = gather(K, -2, iKV)
            dim_K_top = -2 + (-2 < 0) * K.ndim
            K_top = K.gather(dim_K_top, iKV.expand(*K.shape[:dim_K_top], iKV.shape[dim_K_top], *K.shape[dim_K_top + 1:]))
            
            # # 内联gather: V_top = gather(V, -2, iKV)
            dim_V_top = -2 + (-2 < 0) * V.ndim
            V_top = V.gather(dim_V_top, iKV.expand(*V.shape[:dim_V_top], iKV.shape[dim_V_top], *V.shape[dim_V_top + 1:]))
            
            # # 内联gather: M_top = gather(M, -1, i2)
            dim_M = -1 + (-1 < 0) * M.ndim
            M_top = M.gather(dim_M, i2.expand(*M.shape[:dim_M], i2.shape[dim_M], *M.shape[dim_M + 1:]))
            
            # # 内联attn函数
            s = (Q @ K_top.transpose(-1, -2)) / torch.sqrt(torch.tensor(Q.shape[-1], device=Q.device)) + M_top
            y_ = torch.softmax(s.to(V_top.dtype), dim=-1) @ V_top

            
            # # 3. Estimate the total score of the top k, and interpolate with V_mean
            # # 内联gather: alpha = gather(s_hat, -1, i2)
            dim_s_hat = -1 + (-1 < 0) * s_hat.ndim
            alpha = s_hat.gather(dim_s_hat, i2.expand(*s_hat.shape[:dim_s_hat], i2.shape[dim_s_hat], *s_hat.shape[dim_s_hat + 1:])).sum(-1, keepdim=True)
            
            V_mean = V.mean(dim=-2, keepdim=True)
            attn_output = alpha * y_ + (1 - alpha) * V_mean

            batch_size, n_heads, n_kv_heads, seq_len, head_size = attn_output.shape
            attn_output = attn_output.reshape(batch_size, n_kv_heads * n_heads, seq_len, head_size)  # -> (batch_size, n_heads, head_size)

            
            

            attn_output = attn_output.transpose(1, 2).contiguous()
            attn_output = attn_output.view(bsz, seq_len, self.hidden_size)

            attn_output = attn_output.to(self.o_proj.weight.dtype)
            attn_output = self.o_proj(attn_output)

            return attn_output, None, past_key_value

        past_key_value._seen_tokens = self.kv_seq_len

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
    


