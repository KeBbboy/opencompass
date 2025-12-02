"""Initialization utilities for Orthogonal Pivot Selection (Dual-Probe Retrieval)."""

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


class SnapKVCluster_OrthogonalPivot():

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
        Dual-Probe Retrieval Algorithm:
        1. Compute centroid: Q_mean = mean(Q_i)
        2. Compute residuals: R_i = Q_i - Q_mean
        3. Find max outlier: R_max = argmax(||R_i||_2)
        4. Dual-probe retrieval:
           - Use Q_mean to select top-K/2 keys
           - Use R_max to select top-K/2 keys
           - Take union of both sets
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
            # Step 1: 先选择最近的 window_size 个 token 作为查询窗口
            query_states_window = query_states[..., -self.window_size:, :]  # [bsz, num_heads, window_size, head_dim]

            # 将 query_states_window 重塑为 [bsz, num_key_value_heads, num_key_value_groups, window_size, head_dim]
            query_states_grouped = query_states_window.view(bsz, num_key_value_heads, num_key_value_groups, self.window_size, head_dim)

            # Step 2: 计算质心 Q_mean - 对组内的 head 求 mean
            # [bsz, num_key_value_heads, window_size, head_dim]
            query_centroid = query_states_grouped.mean(dim=2)

            # Step 3: 计算残差 R_i = Q_i - Q_mean
            # 扩展 query_centroid 以便进行广播
            # [bsz, num_key_value_heads, 1, window_size, head_dim]
            query_centroid_expanded = query_centroid.unsqueeze(2)

            # 计算残差: [bsz, num_key_value_heads, num_key_value_groups, window_size, head_dim]
            residuals = query_states_grouped - query_centroid_expanded

            # Step 4: 找到最大残差向量（按 L2 范数）
            # 计算每个 head 的 L2 范数: [bsz, num_key_value_heads, num_key_value_groups, window_size]
            residuals_norm = torch.norm(residuals, p=2, dim=-1)

            # 对 window_size 维度求和得到每个 head 的总 L2 范数
            # [bsz, num_key_value_heads, num_key_value_groups]
            residuals_norm_sum = residuals_norm.sum(dim=-1)

            # 找到每个 GQA 组中 L2 范数最大的 head
            # [bsz, num_key_value_heads]
            max_outlier_idx = residuals_norm_sum.argmax(dim=2)

            # 提取最大残差向量 R_max
            # [bsz, num_key_value_heads, window_size, head_dim]
            batch_indices = torch.arange(bsz, device=query_states.device)[:, None]
            kv_head_indices = torch.arange(num_key_value_heads, device=query_states.device)[None, :]
            query_residual_max = residuals[batch_indices, kv_head_indices, max_outlier_idx]

            # Step 5: 双探针检索
            # 5.1 使用 Q_mean (质心) 计算 attention weights
            attn_weights_centroid = torch.matmul(
                query_centroid,
                key_states_gqa.transpose(2, 3)) / math.sqrt(head_dim)

            # 添加 causal mask
            mask = torch.full((self.window_size, self.window_size),
                            torch.finfo(attn_weights_centroid.dtype).min,
                            device=attn_weights_centroid.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights_centroid.device)
            mask.masked_fill_(
                mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights_centroid.device)
            attention_mask_causal = mask[None, None, :, :]

            attn_weights_centroid[:, :, -self.window_size:,
                                -self.window_size:] += attention_mask_causal

            attn_weights_centroid = nn.functional.softmax(attn_weights_centroid,
                                                dim=-1,
                                                dtype=torch.float32).to(
                                                    query_states.dtype)
            attn_weights_centroid_sum = attn_weights_centroid[:, :, -self.window_size:, :-self.window_size].sum(dim=-2)

            # 5.2 使用 R_max (最大残差) 计算 attention weights
            attn_weights_residual = torch.matmul(
                query_residual_max,
                key_states_gqa.transpose(2, 3)) / math.sqrt(head_dim)

            attn_weights_residual[:, :, -self.window_size:,
                                -self.window_size:] += attention_mask_causal

            attn_weights_residual = nn.functional.softmax(attn_weights_residual,
                                                dim=-1,
                                                dtype=torch.float32).to(
                                                    query_states.dtype)
            attn_weights_residual_sum = attn_weights_residual[:, :, -self.window_size:, :-self.window_size].sum(dim=-2)

            # Step 6: 对两个 attention weights 进行 pooling 平滑
            if self.pooling == 'avgpool':
                attn_cache_centroid = F.avg_pool1d(attn_weights_centroid_sum,
                                        kernel_size=self.kernel_size,
                                        padding=self.kernel_size // 2,
                                        stride=1)
                attn_cache_residual = F.avg_pool1d(attn_weights_residual_sum,
                                        kernel_size=self.kernel_size,
                                        padding=self.kernel_size // 2,
                                        stride=1)
            elif self.pooling == 'maxpool':
                attn_cache_centroid = F.max_pool1d(attn_weights_centroid_sum,
                                        kernel_size=self.kernel_size,
                                        padding=self.kernel_size // 2,
                                        stride=1)
                attn_cache_residual = F.max_pool1d(attn_weights_residual_sum,
                                        kernel_size=self.kernel_size,
                                        padding=self.kernel_size // 2,
                                        stride=1)
            else:
                raise ValueError('Pooling method not supported')

            # Step 7: 分别选择 top-K/2 的 indices，然后取并集
            k_half = (self.max_capacity_prompt - self.window_size) // 2

            # 从质心探针选择 top-K/2
            indices_centroid = attn_cache_centroid.topk(k_half, dim=-1).indices

            # 从残差探针选择 top-K/2
            indices_residual = attn_cache_residual.topk(k_half, dim=-1).indices

            # 取并集（去重）
            # 拼接两组 indices: [bsz, num_key_value_heads, k_half * 2]
            indices_combined = torch.cat([indices_centroid, indices_residual], dim=-1)

            # 对每个 batch 和 head 进行去重和排序
            indices_list = []
            for b in range(bsz):
                batch_indices = []
                for h in range(num_key_value_heads):
                    # 获取当前 batch 和 head 的 indices
                    current_indices = indices_combined[b, h]
                    # 去重并排序
                    unique_indices = torch.unique(current_indices, sorted=True)
                    # 如果去重后的数量不足，补充更多的 indices
                    if unique_indices.shape[0] < (self.max_capacity_prompt - self.window_size):
                        # 从 attn_cache_centroid + attn_cache_residual 中选择
                        combined_scores = attn_cache_centroid[b, h] + attn_cache_residual[b, h]
                        # 排除已选择的 indices
                        mask = torch.ones_like(combined_scores, dtype=torch.bool)
                        mask[unique_indices] = False
                        remaining_indices = torch.arange(combined_scores.shape[0], device=combined_scores.device)[mask]
                        remaining_scores = combined_scores[mask]
                        # 选择剩余的 top indices
                        num_remaining = (self.max_capacity_prompt - self.window_size) - unique_indices.shape[0]
                        if num_remaining > 0 and remaining_scores.shape[0] > 0:
                            top_remaining = remaining_scores.topk(min(num_remaining, remaining_scores.shape[0]), dim=-1).indices
                            additional_indices = remaining_indices[top_remaining]
                            unique_indices = torch.cat([unique_indices, additional_indices])
                            unique_indices = torch.sort(unique_indices)[0]

                    batch_indices.append(unique_indices)
                indices_list.append(torch.stack(batch_indices))

            indices = torch.stack(indices_list)  # [bsz, num_key_value_heads, selected_size]

            # 扩展 indices 以匹配 head_dim
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            # Step 8: 使用 indices 压缩 key 和 value states
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


def init_orthogonal_pivot(self):
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

    self.kv_cluster = SnapKVCluster_OrthogonalPivot(
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        ratio=0.4,
        kernel_size=7,
        pooling=self.config.pooling,
        merge=self.config.merge,
    )
