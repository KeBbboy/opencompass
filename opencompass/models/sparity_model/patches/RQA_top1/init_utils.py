"""Initialization utilities for RQA_top1.

RQA_top1: Query head-level Top-1 based KV compression.
Each query head independently selects its top-1 tokens, then aggregates to KV head level.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from transformers.cache_utils import Cache


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


class RQATop1Cluster():
    """
    RQA_top1: Query head-level KV compression.

    Key difference from TopK-GQA:
    - TopK-GQA: operates at KV head level, selects top-k query heads within each group
    - RQA_top1: operates at query head level, each query head independently selects top-1 tokens

    Algorithm:
    1. Each query head computes attention weights independently
    2. Each query head selects its top-1 most important token
    3. Aggregate all query heads' selections at KV head level
    4. Select final top-k tokens for each KV head
    """

    def __init__(self,
                 window_size=64,
                 max_capacity_prompt=256 + 64,
                 kernel_size=5,
                 pooling='avgpool',
                 merge=None,
                 recent_size=32,
                 ratio=0.4):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.ratio = ratio
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.recent_size = recent_size
        self.ratio = ratio

    def reset(self,
              window_size=64,
              max_capacity_prompt=256 + 64,
              kernel_size=5,
              pooling='avgpool',
              ratio=0,
              merge=None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.ratio = ratio
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def update_kv(self, key_states, query_states, value_states, attention_mask,
                  num_key_value_groups):
        """
        Update KV cache using query head-level Top-1 selection.

        Args:
            key_states: [bsz, num_key_value_heads, seq_len, head_dim]
            query_states: [bsz, num_heads, seq_len, head_dim]
            value_states: [bsz, num_key_value_heads, seq_len, head_dim]
            attention_mask: attention mask
            num_key_value_groups: number of query heads per KV head (GQA ratio)

        Returns:
            Compressed key_states and value_states in GQA format
        """
        # 保存原始的 GQA 格式的 key_states 和 value_states
        key_states_gqa = key_states  # [bsz, num_key_value_heads, seq_len, head_dim]
        value_states_gqa = value_states

        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        bsz_k, num_key_value_heads, q_len_k, head_dim_k = key_states_gqa.shape

        if q_len < self.max_capacity_prompt:
            return key_states_gqa, value_states_gqa
        else:
            # === RQA_top1 策略：直接选择数值最大的query head进行聚合 ===

            # 1. 先选择最近的 window_size 个 query tokens
            query_states_window = query_states[..., -self.window_size:, :]  # [bsz, num_heads, window_size, head_dim]

            # 2. 将 query_states 重塑为 GQA 分组格式
            # [bsz, num_heads, window_size, head_dim] -> [bsz, num_key_value_heads, num_key_value_groups, window_size, head_dim]
            query_states_grouped = query_states_window.view(bsz, num_key_value_heads, num_key_value_groups, self.window_size, head_dim)

            # 3. 对每个 query 在 head_dim 维度上取最大值，作为该query的代表值
            query_max_values, _ = torch.max(query_states_grouped, dim=-1)  # [bsz, num_key_value_heads, num_key_value_groups, window_size]

            # 4. 对每个 KV head，在每个 window position 上选择数值最大的那个 query head (top-1)
            # 在 num_key_value_groups 维度上选择 top-1（最大值）
            top1_values, top1_indices = torch.topk(query_max_values, k=1, dim=2)  # [bsz, num_key_value_heads, 1, window_size]

            # 5. 使用 top-1 indices 选择对应的 query states
            # top1_indices: [bsz, num_key_value_heads, 1, window_size]
            # 需要扩展到 [bsz, num_key_value_heads, 1, window_size, head_dim]
            top1_indices_expanded = top1_indices.unsqueeze(-1).expand(-1, -1, -1, -1, head_dim)

            # 从 query_states_grouped 中选择数值最大的 query
            query_states_aggregated = torch.gather(query_states_grouped, dim=2, index=top1_indices_expanded)
            # query_states_aggregated: [bsz, num_key_value_heads, 1, window_size, head_dim]

            # 去掉多余的维度
            query_states_aggregated = query_states_aggregated.squeeze(2)  # [bsz, num_key_value_heads, window_size, head_dim]

            # 6. 直接和 key_states_gqa 计算 attention weights（不需要扩展 key_states）
            attn_weights = torch.matmul(
                query_states_aggregated,
                key_states_gqa.transpose(2, 3)) / math.sqrt(head_dim)
            # attn_weights: [bsz, num_key_value_heads, window_size, seq_len]

            # 7. 创建 causal mask
            mask = torch.full((self.window_size, self.window_size),
                            torch.finfo(attn_weights.dtype).min,
                            device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(
                mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask_causal = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:,
                        -self.window_size:] += attention_mask_causal

            attn_weights = nn.functional.softmax(attn_weights,
                                                dim=-1,
                                                dtype=torch.float32).to(
                                                    query_states.dtype)

            # 8. 计算历史 tokens 的注意力分数
            attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim=-2)
            # attn_weights_sum: [bsz, num_key_value_heads, history_len]

            # 9. Pooling：在 token 维度上进行平滑
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
            # attn_cache: [bsz, num_key_value_heads, history_len]

            # 10. 基于 GQA 格式选择 top-k indices
            indices = attn_cache.topk(self.max_capacity_prompt -
                                    self.window_size,
                                    dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            # 11. 对原始 GQA 格式的 key_states 和 value_states 应用 indices
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


def init_RQA_top1(self):
    """Initialize RQA_top1 cluster."""
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

    self.kv_cluster = RQATop1Cluster(
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        ratio=0.4,
        kernel_size=7,
        pooling=self.config.pooling,
        merge=self.config.merge,
    )
