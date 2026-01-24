"""Initialization utilities for RQA_per_head_topk."""

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


class SnapKVCluster_RQA_per_head_topk():

    def __init__(self,
                 window_size=64,
                 max_capacity_prompt=256 + 64,
                 kernel_size=5,
                 pooling='avgpool',
                 merge=None,
                 recent_size=32,
                 ratio=0.4,
                 weight_temperature=1.0,
                 target_layers=None):
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
        # 新增：指定哪些层需要应用该算法，如果为None则所有层都应用
        self.target_layers = target_layers if target_layers is not None else []

    def reset(self,
              window_size=64,
              max_capacity_prompt=256 + 64,
              kernel_size=5,
              pooling='avgpool',
              ratio=0,
              merge=None,
              weight_temperature=1.0,
              target_layers=None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.ratio = ratio
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.weight_temperature = weight_temperature
        self.target_layers = target_layers if target_layers is not None else []

    def update_kv(self, key_states, query_states, value_states, attention_mask,
                  num_key_value_groups, layer_idx):
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
                # 如果 target_layers 不为空，且当前层不在目标层列表中
                # 使用 snapkv 的方法，但对每个 query head 独立选择 topk
                if len(self.target_layers) > 0 and layer_idx not in self.target_layers:
                    # SnapKV-style compression for each query head independently
                    # 扩展 key_states_gqa 到所有 query heads
                    key_states_expanded = repeat_kv(key_states_gqa, num_key_value_groups)

                    # 每个 query head 独立计算 attention weights
                    attn_weights = torch.matmul(
                        query_states[..., -self.window_size:, :],
                        key_states_expanded.transpose(2, 3)) / math.sqrt(head_dim)

                    # 创建 causal mask
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

                    # 计算每个 head 的 attention weights sum
                    attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim=-2)
                    # attn_weights_sum: [bsz, num_heads, seq_len - window_size]

                    # Pooling 平滑（对每个 head 独立进行）
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

                    # 将 attn_cache 重塑为 GQA 格式并聚合
                    # [bsz, num_heads, seq_len - window_size] -> [bsz, num_key_value_heads, num_key_value_groups, seq_len - window_size]
                    attn_cache_grouped = attn_cache.view(bsz, num_key_value_heads, num_key_value_groups, -1)

                    # 对每个 kv head，聚合所有对应 query heads 的 attention cache
                    attn_cache_aggregated = attn_cache_grouped.mean(dim=2)  # [bsz, num_key_value_heads, seq_len - window_size]

                    # 选择 topk
                    indices = attn_cache_aggregated.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
                    indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                    # indices: [bsz, num_key_value_heads, topk, head_dim]

                    # 应用压缩
                    k_past_compress = key_states_gqa[:, :, :-self.window_size, :].gather(
                        dim=2, index=indices)
                    v_past_compress = value_states_gqa[:, :, :-self.window_size, :].gather(
                        dim=2, index=indices)
                    k_cur = key_states_gqa[:, :, -self.window_size:, :]
                    v_cur = value_states_gqa[:, :, -self.window_size:, :]
                    key_states_gqa = torch.cat([k_past_compress, k_cur], dim=2)
                    value_states_gqa = torch.cat([v_past_compress, v_cur], dim=2)

                    return key_states_gqa, value_states_gqa

                # RQA_per_head_topk: 每个 query head 独立选择 topk 的 kv tokens
                # query_states: [bsz, num_heads, seq_len, head_dim]
                # 目标: 每个 query head 独立选择，最终聚合成 [bsz, num_key_value_heads, window_size, head_dim]

                # 先选择最近的 window_size 个 token
                query_states_window = query_states[..., -self.window_size:, :]  # [bsz, num_heads, window_size, head_dim]

                # 将 query_states 重塑为 [bsz, num_key_value_heads, num_key_value_groups, window_size, head_dim]
                query_states_grouped = query_states_window.view(bsz, num_key_value_heads, num_key_value_groups, self.window_size, head_dim)

                # 计算每个 query 的 L2 范数
                l2_norms = torch.norm(query_states_grouped, p=2, dim=-1)  # [bsz, num_key_value_heads, num_key_value_groups, window_size]

                # 使用温度缩放的 softmax 计算权重
                weights = F.softmax(l2_norms / self.weight_temperature, dim=2).unsqueeze(-1)

                # 加权求和
                weighted_queries = query_states_grouped * weights
                query_states_aggregated = weighted_queries.sum(dim=2)  # [bsz, num_key_value_heads, window_size, head_dim]

                # 扩展 key_states_gqa 到所有 query heads 以便每个 query head 独立计算
                # [bsz, num_key_value_heads, seq_len, head_dim] -> [bsz, num_heads, seq_len, head_dim]
                key_states_expanded = repeat_kv(key_states_gqa, num_key_value_groups)

                # 每个 query head 独立计算 attention weights
                # query_states_window: [bsz, num_heads, window_size, head_dim]
                # key_states_expanded: [bsz, num_heads, seq_len, head_dim]
                attn_weights_per_head = torch.matmul(
                    query_states_window,
                    key_states_expanded.transpose(2, 3)) / math.sqrt(head_dim)
                # attn_weights_per_head: [bsz, num_heads, window_size, seq_len]

                # 创建 causal mask
                mask = torch.full((self.window_size, self.window_size),
                                torch.finfo(attn_weights_per_head.dtype).min,
                                device=attn_weights_per_head.device)
                mask_cond = torch.arange(mask.size(-1), device=attn_weights_per_head.device)
                mask.masked_fill_(
                    mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
                mask = mask.to(attn_weights_per_head.device)
                attention_mask = mask[None, None, :, :]

                attn_weights_per_head[:, :, -self.window_size:,
                            -self.window_size:] += attention_mask

                attn_weights_per_head = nn.functional.softmax(attn_weights_per_head,
                                                    dim=-1,
                                                    dtype=torch.float32).to(
                                                        query_states.dtype)

                # 计算每个 head 的 attention weights sum
                attn_weights_sum_per_head = attn_weights_per_head[:, :, -self.window_size:, :-self.window_size].sum(dim=-2)
                # attn_weights_sum_per_head: [bsz, num_heads, seq_len - window_size]

                # Pooling 平滑（对每个 head 独立进行）
                if self.pooling == 'avgpool':
                    attn_cache_per_head = F.avg_pool1d(attn_weights_sum_per_head,
                                            kernel_size=self.kernel_size,
                                            padding=self.kernel_size // 2,
                                            stride=1)
                elif self.pooling == 'maxpool':
                    attn_cache_per_head = F.max_pool1d(attn_weights_sum_per_head,
                                            kernel_size=self.kernel_size,
                                            padding=self.kernel_size // 2,
                                            stride=1)
                else:
                    raise ValueError('Pooling method not supported')

                # 每个 query head 独立选择 top-k indices
                # attn_cache_per_head: [bsz, num_heads, seq_len - window_size]
                indices_per_head = attn_cache_per_head.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
                # indices_per_head: [bsz, num_heads, topk]

                # 将 indices_per_head 重塑回 GQA 格式
                # [bsz, num_heads, topk] -> [bsz, num_key_value_heads, num_key_value_groups, topk]
                indices_grouped = indices_per_head.view(bsz, num_key_value_heads, num_key_value_groups, -1)

                # 对每个 kv head，聚合来自多个 query heads 的 indices
                # 方法：取所有 query heads 选择的 indices 的并集
                # 为了简化，我们可以先将所有 indices 展平，然后去重
                indices_flat = indices_grouped.reshape(bsz, num_key_value_heads, -1)  # [bsz, num_key_value_heads, num_key_value_groups * topk]

                # 对每个 kv head，我们需要选择最多 topk 个唯一的 indices
                # 使用 unique 操作可能会导致不同 kv head 有不同数量的 indices，这会导致形状不一致
                # 因此，我们采用另一种方法：对所有候选 indices 的频率进行排序，选择出现频率最高的 topk 个

                # 为了保持简单，我们直接取所有 query heads 选择的 indices 的平均位置
                # 或者简单地选择第一个 query head 的 indices（这不太合理）
                # 更好的方法：对每个位置的得分求和，然后选择 topk

                # 重新计算：对每个 kv head，求所有对应 query heads 的 attn_cache 的平均
                attn_cache_grouped = attn_cache_per_head.view(bsz, num_key_value_heads, num_key_value_groups, -1)
                attn_cache_aggregated = attn_cache_grouped.mean(dim=2)  # [bsz, num_key_value_heads, seq_len - window_size]

                # 现在对聚合后的 attention cache 选择 topk
                indices = attn_cache_aggregated.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
                indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                # indices: [bsz, num_key_value_heads, topk, head_dim]

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




def init_RQA_per_head_topk(self):
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
            self.config.weight_temperature = 1.0
        if not hasattr(self.config, 'target_layers'):
            # 默认为空列表，表示所有层都应用
            self.config.target_layers = []

        self.kv_cluster = SnapKVCluster_RQA_per_head_topk(
            window_size=self.config.window_size,
            max_capacity_prompt=self.config.max_capacity_prompt,
            ratio=0.4,
            kernel_size=7,
            pooling=self.config.pooling,
            merge=self.config.merge,
            weight_temperature=self.config.weight_temperature,
            target_layers=self.config.target_layers,
        )
