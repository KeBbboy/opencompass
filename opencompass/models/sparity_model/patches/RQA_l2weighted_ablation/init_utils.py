"""Initialization utilities for RQA_l2weighted_ablation."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from transformers.cache_utils import Cache


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """This is the equivalent of torch.repeat_interleave(x, dim=1,
    repeats=n_rep).

    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :,
                                  None, :, :].expand(batch,
                                                     num_key_value_heads,
                                                     n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                                 head_dim)


class SnapKVCluster_RQA_l2weighted_ablation():

    def __init__(self,
                 window_size=64,
                 max_capacity_prompt=256 + 64,
                 kernel_size=5,
                 pooling='avgpool',
                 merge=None,
                 recent_size=32,
                 ratio=0.4,
                 weight_temperature=1.0):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.ratio = ratio
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.recent_size = recent_size
        self.ratio = ratio
        self.weight_temperature = weight_temperature  

    def reset(self,
              window_size=64,
              max_capacity_prompt=256 + 64,
              kernel_size=5,
              pooling='avgpool',
              ratio = 0,
              merge=None,
              weight_temperature=1.0):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.ratio = ratio
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.weight_temperature = weight_temperature


    def update_kv(self, key_states, query_states, value_states, attention_mask,
                  num_key_value_groups):
            # check if prefix phase
            # 保存原始的 GQA 格式的 key_states 和 value_states
            key_states_gqa = key_states  # [bsz, num_key_value_heads, seq_len, head_dim]
            value_states_gqa = value_states

            assert key_states.shape[-2] == query_states.shape[-2]
            bsz, num_heads, q_len, head_dim = query_states.shape
            bsz_k, num_key_value_heads, q_len_k, head_dim_k = key_states_gqa.shape

            if q_len < self.max_capacity_prompt:
                return key_states_gqa, value_states_gqa
            else:
                # RQA_l2weighted_ablation: 使用 L2 范数加权聚合，然后每个 head 独立选择相同数量的 tokens
                # query_states: [bsz, num_heads, seq_len, head_dim]
                # 目标: [bsz, num_key_value_heads, window_size, head_dim]

                # 先选择最近的 window_size 个 token
                query_states_window = query_states[..., -self.window_size:, :]  # [bsz, num_heads, window_size, head_dim]

                # 将 query_states_window 重塑为 [bsz, num_key_value_heads, num_key_value_groups, window_size, head_dim]
                query_states_grouped = query_states_window.view(bsz, num_key_value_heads, num_key_value_groups, self.window_size, head_dim)

                # 计算每个 query 的 L2 范数
                l2_norms = torch.norm(query_states_grouped, p=2, dim=-1)  # [bsz, num_key_value_heads, num_key_value_groups, window_size]

                # 使用温度缩放的 softmax 计算权重
                # weights: [bsz, num_key_value_heads, num_key_value_groups, window_size, 1]
                weights = F.softmax(l2_norms / self.weight_temperature, dim=2).unsqueeze(-1)

                # 加权求和
                weighted_queries = query_states_grouped * weights
                query_states_aggregated = weighted_queries.sum(dim=2)  # [bsz, num_key_value_heads, window_size, head_dim]

                # 直接和 key_states_gqa 计算 attention weights
                attn_weights = torch.matmul(
                    query_states_aggregated,
                    key_states_gqa.transpose(2, 3)) / math.sqrt(head_dim)

                mask = torch.full((self.window_size, self.window_size),
                                torch.finfo(attn_weights.dtype).min,
                                device=attn_weights.device)
                mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
                mask.masked_fill_(
                    mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
                mask = mask.to(attn_weights.device)
                attention_mask = mask[None, None, :, :]

                attn_weights[:, :, -self.window_size:,
                            -self.window_size:] += attention_mask

                attn_weights = nn.functional.softmax(attn_weights,
                                                    dim=-1,
                                                    dtype=torch.float32).to(
                                                        query_states.dtype)
                attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.
                                                window_size].sum(dim=-2)

                # Pooling 平滑（已经在 GQA 格式上）
                if self.pooling == 'avgpool':
                    attn_cache = F.avg_pool1d(attn_weights_sum,
                                            kernel_size=self.kernel_size,
                                            padding=self.kernel_size // 2,
                                            stride=1)
                elif self.pooling == 'maxpool':
                    attn_cache = F.max_pool1d(attn_weights_sum,
                                            kernel_size=self.kernel_size,
                                            padding=self.kernel_size // 2,
                                            stride=1)
                else:
                    raise ValueError('Pooling method not supported')

                # 此时 attn_cache 已经是 GQA 格式: [bsz, num_key_value_heads, seq_len]
                # 不需要再做聚合，直接选择 top-k indices
                indices = attn_cache.topk(self.max_capacity_prompt -
                                        self.window_size,
                                        dim=-1).indices
                indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

                if self.merge is not None:
                    key_states_gqa, value_states_gqa = merge_kv(key_states_gqa, value_states_gqa,
                                                        indices, self.window_size,
                                                        self.merge)
                    return key_states_gqa, value_states_gqa

                # 对原始 GQA 格式的 key_states 和 value_states 应用 indices
                k_past_compress = key_states_gqa[:, :, :-self.window_size, :].gather(
                    dim=2, index=indices)
                v_past_compress = value_states_gqa[:, :, :-self.window_size, :].gather(
                    dim=2, index=indices)
                k_cur = key_states_gqa[:, :, -self.window_size:, :]
                v_cur = value_states_gqa[:, :, -self.window_size:, :]
                key_states_gqa = torch.cat([k_past_compress, k_cur], dim=2)
                value_states_gqa = torch.cat([v_past_compress, v_cur], dim=2)

                # 返回 GQA 格式: [bsz, num_key_value_heads, compressed_seq_len, head_dim]
                return key_states_gqa, value_states_gqa




def init_RQA_l2weighted_ablation(self):
    if not hasattr(self, 'kv_cluster'):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 16
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 64
        if not hasattr(self.config, 'ratio'):
            self.config.ratio = 0.4
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 7
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'maxpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None
        if not hasattr(self.config, 'weight_temperature'):
            self.config.weight_temperature = 1.0  # 默认温度参数

        self.kv_cluster = SnapKVCluster_RQA_l2weighted_ablation(
            window_size=self.config.window_size,
            max_capacity_prompt=self.config.max_capacity_prompt,
            ratio = 0.4,
            kernel_size=7,
            pooling=self.config.pooling,
            merge=self.config.merge,
            weight_temperature=self.config.weight_temperature,
        )
