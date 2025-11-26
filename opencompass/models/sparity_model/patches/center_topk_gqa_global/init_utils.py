"""Initialization utilities for Center-TopK-GQA-Global.

Center-based KV compression with global centroid computation:
1. Computes centroid (center) using ALL heads (global)
2. Calculates distance from each token to the global centroid
3. Selects top-k tokens that are farthest from the centroid (most informative)
4. All KV head groups share the same selected tokens
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


class CenterTopKKVCluster_gqa_global():
    """
    Center-based TopK KV compression for GQA with global centroid.

    Algorithm:
    1. Compute GLOBAL centroid across ALL heads: center = mean of all head attention scores
    2. Calculate distance from each token to the GLOBAL centroid
    3. Select top-k tokens with largest distance (farthest from center)
    4. ALL KV head groups share the same selected tokens (global selection)

    Key difference from center_topk_gqa:
    - center_topk_gqa: Each KV head group independently computes its own centroid
    - center_topk_gqa_global: Single global centroid computed from all heads

    Key insight: Tokens far from the global centroid are important across all heads,
    providing a more consistent and stable token selection strategy.
    """

    def __init__(self,
                 window_size=64,
                 max_capacity_prompt=256 + 64,
                 kernel_size=5,
                 pooling='avgpool',
                 merge=None,
                 recent_size=32,
                 ratio=0.4,
                 distance_metric='l2'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.ratio = ratio
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.recent_size = recent_size
        self.ratio = ratio
        self.distance_metric = distance_metric  # 'l2' or 'l1'

    def reset(self,
              window_size=64,
              max_capacity_prompt=256 + 64,
              kernel_size=5,
              pooling='avgpool',
              ratio=0,
              merge=None,
              distance_metric='l2'):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.ratio = ratio
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.distance_metric = distance_metric

    def update_kv(self, key_states, query_states, value_states, attention_mask,
                  num_key_value_groups):
        """
        Update KV cache using global center-based token selection.

        Key difference from center_topk_gqa:
        - center_topk_gqa: Computes centroid per KV head group (local)
        - center_topk_gqa_global: Computes single centroid across all heads (global)
        """
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            # === 计算注意力权重（与 center_topk_gqa 相同）===
            attn_weights = torch.matmul(
                query_states[..., -self.window_size:, :],
                key_states.transpose(2, 3)) / math.sqrt(head_dim)

            # Causal mask
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

            # 聚合注意力分数（与 center_topk_gqa 相同）
            attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim=-2)
            # attn_weights_sum: [bsz, num_heads, history_len]

            # Pool 平滑（与 center_topk_gqa 相同）
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

            # === GLOBAL Center-based 策略：基于所有 heads 的全局质心 ===

            # 1. 计算全局质心（所有 heads 的平均注意力分数）
            # 不再按 KV head 分组，而是直接对所有 heads 求平均
            global_center = attn_weights_pooled.mean(dim=1, keepdim=True)
            # global_center: [bsz, 1, history_len]
            # 质心 = 所有 heads 对每个 token 的平均注意力

            # 2. 计算每个 head 的注意力分数与全局质心的距离
            # 这里我们需要计算全局距离：所有 heads 对每个 token 的偏差总和
            global_token_dists = (attn_weights_pooled - global_center).abs().sum(dim=1)
            # global_token_dists: [bsz, history_len]
            # 距离越大 = 该 token 在所有 heads 中引起的注意力"分歧"越大 = 越重要

            # 3. 基于全局距离选择 tokens（所有 KV heads 共享相同的选择）
            num_key_value_heads = key_states.shape[1]
            history_len = global_token_dists.shape[-1]

            # 选择距离全局质心最远的 tokens
            indices_1d = global_token_dists.topk(self.max_capacity_prompt - self.window_size,
                                                  dim=-1, largest=True).indices
            # indices_1d: [bsz, num_selected]

            # 扩展到所有 KV heads（所有 heads 使用相同的 token 选择）
            indices = indices_1d.unsqueeze(1).unsqueeze(-1).expand(
                bsz, num_key_value_heads, -1, head_dim)
            # indices: [bsz, num_kv_heads, num_selected, head_dim]

            # 4. Gather 压缩的 KV
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(
                dim=2, index=indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(
                dim=2, index=indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            return key_states, value_states


def init_center_topk_gqa_global(self):
    """Initialize Center-TopK-GQA-Global cluster."""
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
        if not hasattr(self.config, 'distance_metric'):
            self.config.distance_metric = 'l2'  # 默认使用 L2 距离

    self.kv_cluster = CenterTopKKVCluster_gqa_global(
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        ratio=0.4,
        kernel_size=7,
        pooling=self.config.pooling,
        merge=self.config.merge,
        distance_metric=self.config.distance_metric,
    )
