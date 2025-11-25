"""Initialization utilities for Center-TopK-GQA.

Center-based KV compression: Each KV head group independently:
1. Computes centroid (center) using max and min
2. Calculates distance from each token to the centroid
3. Selects top-k tokens that are farthest from the centroid (most informative)
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


class CenterTopKKVCluster_gqa():
    """
    Center-based TopK KV compression for GQA.

    Algorithm:
    1. For each KV head group, compute centroid: center = (max + min) / 2
    2. Calculate distance from each token to the centroid
    3. Select top-k tokens with largest distance (farthest from center)
    4. Each KV head group independently selects different tokens

    Key insight: Tokens far from centroid contain more unique information,
    while tokens near centroid are redundant and can be compressed.
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
        Update KV cache using center-based token selection on attention scores.

        Similar to SnapKV but:
        - SnapKV: directly uses pooled attention scores
        - Center-TopK-GQA: computes centroid of attention scores per group,
          then selects tokens based on distance from centroid
        """
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            # === 计算注意力权重（与 SnapKV 相同）===
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

            # 聚合注意力分数（与 SnapKV 相同）
            attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim=-2)
            # attn_weights_sum: [bsz, num_heads, history_len]

            # Pool 平滑（与 SnapKV 相同）
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

            # === Center-based 策略：基于 pooled 注意力分数的质心 ===

            # 1. 计算每个 head 组的质心（注意力分数的中心）
            # 将 MHA 转换为 GQA 格式
            history_len = attn_weights_pooled.shape[-1]
            num_key_value_heads = key_states.shape[1]
            attn_weights_gqa = attn_weights_pooled.view(bsz, num_key_value_heads, num_key_value_groups, history_len)
            # [bsz, num_kv_heads, num_kv_groups, history_len]

            # 计算每个 KV head 组的质心（每个 token 在组内的平均注意力）
            centers = attn_weights_gqa.mean(dim=2, keepdim=True)
            # centers: [bsz, num_kv_heads, 1, history_len]
            # 质心 = 组内所有 Q heads 对每个 token 的平均注意力

            # 2. 计算每个 token 到质心的距离
            # 距离 = 组内各 Q heads 的注意力分数与质心的偏差
            token_dists = (attn_weights_gqa - centers).abs().sum(dim=2)
            # token_dists: [bsz, num_kv_heads, history_len]
            # 距离越大 = 该 token 在组内引起的注意力"分歧"越大 = 越重要

            # 3. 选择距离质心最远的 tokens（注意力分歧最大）
            indices = token_dists.topk(self.max_capacity_prompt - self.window_size,
                                      dim=-1, largest=True).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

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


def init_center_topk_gqa(self):
    """Initialize Center-TopK-GQA cluster."""
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

    self.kv_cluster = CenterTopKKVCluster_gqa(
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        ratio=0.4,
        kernel_size=7,
        pooling=self.config.pooling,
        merge=self.config.merge,
        distance_metric=self.config.distance_metric,
    )
