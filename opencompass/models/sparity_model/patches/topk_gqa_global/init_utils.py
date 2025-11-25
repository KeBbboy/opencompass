"""Initialization utilities for TopK-GQA-Global.

TopK-based KV compression: select top-k Q heads GLOBALLY across all KV head groups,
then sum their scores. All KV heads share the same token indices.
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


class TopKKVCluster_gqa_global():
    """
    TopK-based KV compression for GQA with global token selection.

    Key difference from TopK-GQA:
    - TopK-GQA: Each KV head group independently selects tokens
    - TopK-GQA-Global: All KV heads share the same token indices (global selection)

    Algorithm:
    1. Pool attention scores across token dimension
    2. Reshape to GQA format
    3. Select top-k Q heads for each token (per KV head group)
    4. Sum across top-k heads for each KV head group
    5. Average scores across all KV head groups
    6. Select tokens based on global average score (same indices for all groups)
    """

    def __init__(self,
                 window_size=64,
                 max_capacity_prompt=256 + 64,
                 kernel_size=5,
                 pooling='avgpool',
                 merge=None,
                 recent_size=32,
                 ratio=0.4,
                 topk_heads=2):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.ratio = ratio
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.recent_size = recent_size
        self.ratio = ratio
        self.topk_heads = topk_heads

    def reset(self,
              window_size=64,
              max_capacity_prompt=256 + 64,
              kernel_size=5,
              pooling='avgpool',
              ratio=0,
              merge=None,
              topk_heads=2):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.ratio = ratio
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.topk_heads = topk_heads

    def update_kv(self, key_states, query_states, value_states, attention_mask,
                  num_key_value_groups):
        """
        Update KV cache using TopK-based group-wise selection with GLOBAL token indices.

        Args:
            key_states: [bsz, num_key_value_heads, seq_len, head_dim]
            query_states: [bsz, num_heads, seq_len, head_dim]
            value_states: [bsz, num_key_value_heads, seq_len, head_dim]
            attention_mask: attention mask
            num_key_value_groups: number of query heads per KV head (GQA ratio)

        Returns:
            Compressed key_states and value_states in GQA format
            All KV head groups use the SAME token indices
        """
        key_states_gqa = key_states
        value_states_gqa = value_states

        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        bsz_k, num_key_value_heads, q_len_k, head_dim_k = key_states_gqa.shape

        if q_len < self.max_capacity_prompt:
            return key_states_gqa, value_states_gqa
        else:
            # 临时扩展 key_states 用于计算 attention weights
            key_states_expanded = repeat_kv(key_states_gqa, num_key_value_groups)

            attn_weights = torch.matmul(
                query_states[..., -self.window_size:, :],
                key_states_expanded.transpose(2, 3)) / math.sqrt(head_dim)

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

            # === 全局策略：先 Pool，再组内 TopK，最后全局平均得到统一的 token 索引 ===

            # 1. 计算历史 tokens 的注意力分数
            attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim=-2)
            # attn_weights_sum: [bsz, num_heads, history_len]

            # 2. 先 Pool：在每个 head 的 token 维度上进行平滑
            if self.pooling == 'avgpool':
                attn_weights_pooled = F.avg_pool1d(attn_weights_sum,
                                                  kernel_size=self.kernel_size,
                                                  padding=self.kernel_size // 2,
                                                  stride=1)
            elif self.pooling == 'maxpool':
                attn_weights_pooled = F.max_pool1d(attn_weights_sum,
                                                  kernel_size=self.kernel_size,
                                                  padding=self.kernel_size // 2,
                                                  stride=1)
            else:
                raise ValueError('Pooling method not supported')
            # attn_weights_pooled: [bsz, num_heads, history_len]

            # 3. 将 attention weights 从 MHA 格式转换为 GQA 格式
            history_len = attn_weights_pooled.shape[-1]
            attn_weights_gqa = attn_weights_pooled.view(bsz, num_key_value_heads, num_key_value_groups, history_len)

            # 4. 对每个 token，在组内选择 top-k 个最大的 Q head 分数
            k = min(self.topk_heads, num_key_value_groups)
            topk_scores, _ = torch.topk(attn_weights_gqa, k=k, dim=2, largest=True)
            # topk_scores: [bsz, num_key_value_heads, k, history_len]

            # 5. 对 top-k 个 max 值求和，得到每个 KV head 组对每个 token 的分数
            attn_scores_per_group = topk_scores.sum(dim=2)
            # attn_scores_per_group: [bsz, num_key_value_heads, history_len]

            # === 关键：全局平均，所有 KV heads 共享相同的 token 索引 ===
            # 6. 对所有 KV head 组的分数求平均，得到全局分数
            attn_cache_global = attn_scores_per_group.mean(dim=1, keepdim=True)
            # attn_cache_global: [bsz, 1, history_len]

            # 7. 基于全局分数选择 top-k token indices（所有组共享）
            indices_global = attn_cache_global.topk(
                self.max_capacity_prompt - self.window_size,
                dim=-1
            ).indices
            # indices_global: [bsz, 1, num_selected_tokens]

            # 扩展 indices 到所有 KV heads（广播）
            indices = indices_global.expand(bsz, num_key_value_heads, -1)
            # indices: [bsz, num_key_value_heads, num_selected_tokens]
            # 注意：所有 KV heads 的 indices 是相同的！

            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            # indices: [bsz, num_key_value_heads, num_selected_tokens, head_dim]

            # 对原始 GQA 格式的 key_states 和 value_states 应用相同的 indices
            k_past_compress = key_states_gqa[:, :, :-self.window_size, :].gather(
                dim=2, index=indices)
            v_past_compress = value_states_gqa[:, :, :-self.window_size, :].gather(
                dim=2, index=indices)
            k_cur = key_states_gqa[:, :, -self.window_size:, :]
            v_cur = value_states_gqa[:, :, -self.window_size:, :]
            key_states_gqa = torch.cat([k_past_compress, k_cur], dim=2)
            value_states_gqa = torch.cat([v_past_compress, v_cur], dim=2)

            # 返回 GQA 格式: [bsz, num_key_value_heads, compressed_seq_len, head_dim]
            # 所有 KV heads 保留相同的 tokens
            return key_states_gqa, value_states_gqa


def init_topk_gqa_global(self):
    """Initialize TopK-GQA-Global cluster."""
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
        if not hasattr(self.config, 'topk_heads'):
            self.config.topk_heads = 2

    self.kv_cluster = TopKKVCluster_gqa_global(
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        ratio=0.4,
        kernel_size=7,
        pooling=self.config.pooling,
        merge=self.config.merge,
        topk_heads=self.config.topk_heads,
    )
