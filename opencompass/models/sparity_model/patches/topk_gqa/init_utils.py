"""Initialization utilities for TopK-GQA.

TopK-based KV compression: select top-k Q heads within each group and sum their scores.
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


class TopKKVCluster_gqa():
    """
    TopK-based KV compression for GQA.

    Key difference from SnapKV-GQA:
    - SnapKV-GQA: mean over all Q heads in group
    - TopK-GQA: select top-k Q heads in group, then sum their scores

    This allows focusing on the most important Q heads within each group.
    """

    def __init__(self,
                 window_size=64,
                 max_capacity_prompt=256 + 64,
                 kernel_size=5,
                 pooling='avgpool',
                 merge=None,
                 recent_size=32,
                 ratio=0.4,
                 topk_heads=1):  # 新增参数：每组选取 top-k 个 Q heads
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
        Update KV cache using TopK-based group-wise selection.

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

            # === 新策略：先 Pool，再组内 TopK Max，最后用 Max 和排序 ===

            # 1. 计算历史 tokens 的注意力分数
            attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim=-2)
            # attn_weights_sum: [bsz, num_heads, history_len]

            # 2. Pool：在 token 维度上进行平滑
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
            # [bsz, num_heads, history_len] -> [bsz, num_key_value_heads, num_key_value_groups, history_len]
            attn_weights_gqa = attn_weights_pooled.view(bsz, num_key_value_heads, num_key_value_groups, history_len)

            # 4. 对每个 token，在组内选择 top-k 个最大的 Q head 分数
            # 确保 topk_heads 不超过 num_key_value_groups
            k = min(self.topk_heads, num_key_value_groups)

            # attn_weights_gqa: [bsz, num_key_value_heads, num_key_value_groups, history_len]
            # topk 在 dim=2 (num_key_value_groups) 上操作，对每个 token 独立选择 top-k heads
            topk_scores, _ = torch.topk(attn_weights_gqa, k=k, dim=2, largest=True)
            # topk_scores: [bsz, num_key_value_heads, k, history_len]
            # 含义：对每个 token，保留组内 top-k 个 Q heads 的最大注意力分数

            # 5. 对 top-k 个 max 值求和，得到每个 token 的最终代表分数
            # 这个和决定了 token 在组内的重要性排序
            attn_cache = topk_scores.sum(dim=2)
            # attn_cache: [bsz, num_key_value_heads, history_len]
            # 含义：每个 token 的分数 = 组内 top-k 个最关注它的 Q heads 的分数之和

            # 6. 基于 GQA 格式选择 top-k indices
            indices = attn_cache.topk(self.max_capacity_prompt -
                                    self.window_size,
                                    dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

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


def init_topk_gqa(self):
    """Initialize TopK-GQA cluster."""
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
            self.config.topk_heads = 2  # 默认选取 top-2 个 Q heads

    self.kv_cluster = TopKKVCluster_gqa(
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        ratio=0.4,
        kernel_size=7,
        pooling=self.config.pooling,
        merge=self.config.merge,
        topk_heads=self.config.topk_heads,
    )
