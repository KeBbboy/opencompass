"""Initialization utilities for snapkv_gqa_chunk_global."""

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


class SnapKVCluster_chunk_global():

    def __init__(self,
                 window_size=64,
                 max_capacity_prompt=256 + 64,
                 chunk_size=64,
                 kernel_size=5,
                 pooling='avgpool',
                 merge=None,
                 recent_size=32,
                 ratio=0.4):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.chunk_size = chunk_size
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
              chunk_size=64,
              kernel_size=5,
              pooling='avgpool',
              ratio=0,
              merge=None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.chunk_size = chunk_size
        self.ratio = ratio
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def update_kv(self, key_states, query_states, value_states, attention_mask,
                  num_key_value_groups):
        """
        Update KV cache using global chunk-based selection.

        Difference from snapkv_gqa2:
        - All KV head groups share the same chunk indices (global selection)
        - Ensures all groups have identical compressed sequence lengths

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

        # 如果序列长度小于最大容量，不压缩
        if q_len < self.max_capacity_prompt:
            return key_states_gqa, value_states_gqa

        # 扩展 key_states 用于计算 attention weights
        key_states_expanded = repeat_kv(key_states_gqa, num_key_value_groups)

        # 计算注意力权重：使用最近的 window_size 个 query
        attn_weights = torch.matmul(
            query_states[..., -self.window_size:, :],
            key_states_expanded.transpose(2, 3)) / math.sqrt(head_dim)

        # 创建因果掩码
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

        # Softmax 归一化
        attn_weights = nn.functional.softmax(attn_weights,
                                            dim=-1,
                                            dtype=torch.float32).to(
                                                query_states.dtype)

        # 计算历史 tokens 的累积注意力分数
        # attn_weights: [bsz, num_heads, window_size, seq_len]
        # 我们只关心历史部分（不包括最后的 window）
        history_len = q_len - self.window_size
        attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim=-2)
        # attn_weights_sum: [bsz, num_heads, history_len]

        # === GROUP-WISE 处理：按 GQA groups 分组聚合 ===
        # 将 num_heads 维度 reshape 成 [num_key_value_heads, num_key_value_groups]
        # 然后在每个 group 内求和
        attn_weights_sum = attn_weights_sum.view(bsz, num_key_value_heads, num_key_value_groups, history_len)
        # 在 group 内求和（对于 top-k 选择，求和和求平均效果相同）
        attn_weights_sum = attn_weights_sum.sum(dim=2)  # [bsz, num_key_value_heads, history_len]

        # === CHUNK-BASED 选择 ===
        # 将历史序列划分为 chunks
        num_chunks = (history_len + self.chunk_size - 1) // self.chunk_size  # 向上取整

        # Pad 到完整的 chunks（用于方便处理）
        padded_len = num_chunks * self.chunk_size
        pad_len = padded_len - history_len

        if pad_len > 0:
            # Pad with very small values (won't be selected)
            # Use torch.finfo to get the minimum value for the dtype (FP16-safe)
            min_value = torch.finfo(attn_weights_sum.dtype).min
            padding = torch.full((bsz, num_key_value_heads, pad_len),
                               min_value,
                               dtype=attn_weights_sum.dtype,
                               device=attn_weights_sum.device)
            attn_weights_padded = torch.cat([attn_weights_sum, padding], dim=-1)
        else:
            attn_weights_padded = attn_weights_sum

        # Reshape 成 chunks: [bsz, num_key_value_heads, num_chunks, chunk_size]
        attn_weights_chunks = attn_weights_padded.view(bsz, num_key_value_heads, num_chunks, self.chunk_size)

        # 计算每个 chunk 的重要性分数（chunk 内所有 tokens 的注意力分数之和）
        chunk_importance = attn_weights_chunks.sum(dim=-1)  # [bsz, num_key_value_heads, num_chunks]

        # === GLOBAL CHUNK SELECTION: 聚合所有 groups 的 chunk importance ===
        # 对所有 KV head groups 求和，得到全局 chunk importance
        chunk_importance_global = chunk_importance.sum(dim=1)  # [bsz, num_chunks]

        # 应用 pooling 进行平滑（可选）
        if self.pooling == 'avgpool':
            chunk_importance_global = F.avg_pool1d(chunk_importance_global.unsqueeze(1),
                                          kernel_size=min(self.kernel_size, num_chunks),
                                          padding=min(self.kernel_size, num_chunks) // 2,
                                          stride=1).squeeze(1)
        elif self.pooling == 'maxpool':
            chunk_importance_global = F.max_pool1d(chunk_importance_global.unsqueeze(1),
                                          kernel_size=min(self.kernel_size, num_chunks),
                                          padding=min(self.kernel_size, num_chunks) // 2,
                                          stride=1).squeeze(1)

        # 计算需要保留的 chunks 数量
        num_tokens_to_keep = self.max_capacity_prompt - self.window_size
        num_chunks_to_keep = (num_tokens_to_keep + self.chunk_size - 1) // self.chunk_size
        num_chunks_to_keep = min(num_chunks_to_keep, num_chunks)  # 不能超过总 chunk 数

        # 全局选择 top-k chunks（所有 groups 共享）
        # chunk_importance_global: [bsz, num_chunks]
        topk_chunk_indices = chunk_importance_global.topk(num_chunks_to_keep, dim=-1).indices
        # topk_chunk_indices: [bsz, num_chunks_to_keep]

        # === 根据选中的 chunks 提取对应的 tokens ===
        # 所有 groups 使用相同的 chunk indices
        # 策略：对于每个选中的 chunk，提取该 chunk 的所有 tokens（最后一个 chunk 可能不满）

        # 为每个 batch 生成 token indices
        batch_token_indices = []
        for b in range(bsz):
            token_indices = []
            for chunk_idx in topk_chunk_indices[b]:
                chunk_idx = chunk_idx.item()
                start_idx = chunk_idx * self.chunk_size
                end_idx = min(start_idx + self.chunk_size, history_len)
                token_indices.extend(range(start_idx, end_idx))
            batch_token_indices.append(token_indices)

        # 转换为 tensor，注意每个 batch 可能长度不同（因为最后一个 chunk）
        # 为了统一处理，我们 pad 到相同长度
        max_len = max(len(indices) for indices in batch_token_indices)
        token_indices_tensor = torch.zeros((bsz, max_len), dtype=torch.long, device=key_states_gqa.device)

        for b in range(bsz):
            indices = batch_token_indices[b]
            token_indices_tensor[b, :len(indices)] = torch.tensor(indices, dtype=torch.long, device=key_states_gqa.device)

        # 扩展到 num_key_value_heads 和 head_dim
        # [bsz, max_len] -> [bsz, num_key_value_heads, max_len, head_dim]
        token_indices_tensor = token_indices_tensor.unsqueeze(1).unsqueeze(-1).expand(bsz, num_key_value_heads, -1, head_dim)

        # 从 key_states_gqa 和 value_states_gqa 中 gather
        # 所有 groups 使用相同的 indices
        k_past_compress = key_states_gqa[:, :, :-self.window_size, :].gather(dim=2, index=token_indices_tensor)
        v_past_compress = value_states_gqa[:, :, :-self.window_size, :].gather(dim=2, index=token_indices_tensor)

        # 添加最近的 window tokens
        k_cur = key_states_gqa[:, :, -self.window_size:, :]
        v_cur = value_states_gqa[:, :, -self.window_size:, :]

        key_states_gqa = torch.cat([k_past_compress, k_cur], dim=2)
        value_states_gqa = torch.cat([v_past_compress, v_cur], dim=2)

        # 返回 GQA 格式: [bsz, num_key_value_heads, compressed_seq_len, head_dim]
        # 注意：所有 groups 使用相同的 token indices
        return key_states_gqa, value_states_gqa


def init_snapkv_gqa_chunk_global(self):
    """Initialize SnapKV GQA Chunk Global cluster."""
    if not hasattr(self, 'kv_cluster'):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 16
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 64
        if not hasattr(self.config, 'chunk_size'):
            self.config.chunk_size = 64
        if not hasattr(self.config, 'ratio'):
            self.config.ratio = 0.4
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 7
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'maxpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None

    self.kv_cluster = SnapKVCluster_chunk_global(
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        chunk_size=self.config.chunk_size,
        ratio=0.4,
        kernel_size=7,
        pooling=self.config.pooling,
        merge=self.config.merge,
    )
