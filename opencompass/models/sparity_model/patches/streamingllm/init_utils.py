"""Initialization utilities for streamingllm."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from transformers.cache_utils import Cache


class StreamingLLMKVCluster():

    def __init__(self,
                 window_size=64,
                 max_capacity_prompt=256 + 64,
                 kernel_size=5,
                 pooling='avgpool',
                 merge=None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def reset(self,
              window_size=64,
              max_capacity_prompt=256 + 64,
              kernel_size=5,
              pooling='avgpool',
              merge=None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge

    def update_kv(self, key_states, query_states, value_states, attention_mask,
                  num_key_value_groups):

        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        print(f'StreamingLLM max_capacity_prompt {self.max_capacity_prompt}')

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:

            indices = torch.tensor(range(self.max_capacity_prompt -
                                         self.window_size),
                                   dtype=torch.int64).to(key_states.device)
            indices = indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(
                bsz, num_heads, 1, head_dim)

            if self.merge is not None:
                key_states, value_states = merge_kv(key_states, value_states,
                                                    indices, self.window_size,
                                                    self.merge)
                return key_states, value_states

            k_past_compress = key_states[:, :, :-self.window_size, :].gather(
                dim=2, index=indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(
                dim=2, index=indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)
            return key_states, value_states


class AdaKVCluster():
    """adapt from https://github.com/FFY0/AdaKV."""

    def __init__(self,
                 window_size=32,
                 kernel_size=7,
                 pooling='maxpool',
                 max_capacity_prompt=None,
                 floor=None,
                 normalize=None,
                 layer_idx=None,
                 num_hidden_layers=None):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.base_capacity = max_capacity_prompt - window_size
        self.floor_ratio = floor
        self.floor_capacity = int(self.base_capacity * self.floor_ratio)
        self.adaptive_capacity = self.base_capacity - self.floor_capacity
        self.num_hidden_layers = num_hidden_layers

        self.normalize = normalize
        self.layer_idx = layer_idx

        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None

    def calcul_attn_sore(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :],
                                    key_states.transpose(
                                        2, 3)) / math.sqrt(head_dim)
        mask = torch.full((self.window_size, self.window_size),
                          torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1),
                          0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:,
                     -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights,
                                             dim=-1,
                                             dtype=torch.float32).to(
                                                 query_states.dtype)
        attn_weights_mean = attn_weights[:, :, -self.window_size:, :-self.
                                         window_size].mean(dim=-2)
        if self.pooling == 'avgpool':
            attn_weights_mean_pooling = F.avg_pool1d(
                attn_weights_mean,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1)
        elif self.pooling == 'maxpool':
            attn_weights_mean_pooling = F.max_pool1d(
                attn_weights_mean,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1)
        else:
            raise ValueError('Pooling method not supported')
        return attn_weights_mean_pooling

    def update_kv(self, key_states, query_states, value_states):
        # check if prefix phase        assert key_states.shape[-2] == query_states.shape[-2]
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_score = self.calcul_attn_sore(key_states, query_states)
        origin_heads_key_states = torch.split(key_states, 1, dim=1)
        origin_heads_value_states = torch.split(value_states, 1, dim=1)

        def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
            # init metadata
            self.head_lens = torch.tensor(k_lens,
                                          dtype=torch.int32,
                                          device=_device)
            self.klen_sum = klen_sum
            self.max_seqlen_k = max_seqlen_k
            self.cu_headlens = torch.cumsum(self.head_lens,
                                            dim=0,
                                            dtype=torch.int32)
            # init varlen flash attention metadata
            self.cu_klen = self.cu_headlens - self.head_lens
            self.cu_klen = torch.cat([
                self.cu_klen,
                torch.tensor(
                    [self.klen_sum], dtype=torch.int32, device=_device)
            ],
                                     dim=0)
            self.layer_qlens = torch.ones(num_heads,
                                          dtype=torch.int32,
                                          device=_device)
            self.qlen_sum = num_heads
            self.cu_qlen = torch.cumsum(
                self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat([
                self.cu_qlen,
                torch.tensor(
                    [self.qlen_sum], dtype=torch.int32, device=_device)
            ],
                                     dim=0)
            self.cu_offset = torch.arange(0,
                                          num_heads + 1,
                                          dtype=torch.int32,
                                          device=_device)
            self.cu_head_offset = torch.arange(1,
                                               num_heads + 1,
                                               dtype=torch.int32,
                                               device=_device)

        if self.base_capacity > attn_score.size(-1):
            init_metadata(num_heads, [q_len] * num_heads, q_len * num_heads,
                          q_len)
            # not compress
            return key_states.reshape(-1, head_dim), value_states.reshape(
                -1, head_dim)

        # if you need to weight the attn_score
        sorted_attn_score, sorted_attn_score_indices = attn_score.sort(
            dim=-1, descending=True)
        adaptive_attn_score = sorted_attn_score
        length = adaptive_attn_score.size(dim=-1)
        if self.normalize:
            ratio_weight = sorted_attn_score[..., :self.base_capacity].sum(
                dim=-1, keepdim=True) / sorted_attn_score.sum(dim=-1,
                                                              keepdim=True)
            adaptive_attn_score = adaptive_attn_score * ratio_weight
        adaptive_attn_score = adaptive_attn_score.reshape(
            bsz, length * num_heads)
        sorted_indices = torch.topk(adaptive_attn_score,
                                    k=num_heads * self.base_capacity,
                                    dim=-1).indices
        sorted_indices = sorted_indices // length
        # floor capacity set
        head_adaptive_capacity = torch.zeros((bsz, num_heads),
                                             device=_device,
                                             dtype=sorted_indices.dtype)
        head_adaptive_capacity.scatter_add_(
            -1,
            sorted_indices,
            torch.ones_like(sorted_indices,
                            dtype=head_adaptive_capacity.dtype),
        )
        assert head_adaptive_capacity.sum().item(
        ) == num_heads * self.base_capacity
        head_adaptive_capacity = torch.round(head_adaptive_capacity *
                                             (1 - self.floor_ratio) +
                                             self.floor_capacity).int()
        sorted_attn_score_indices = sorted_attn_score_indices.split(1, dim=1)

        heads_key_states = []
        heads_value_states = []
        assert bsz == 1

        # per head
        # reinit varlen metadata
        k_lens = []
        klen_sum = 0
        max_seqlen_k = 0
        self.cu_klen = 0

        for head_idx in range(num_heads):
            cache_index = sorted_attn_score_indices[head_idx][
                ..., :head_adaptive_capacity[0][head_idx]]

            l = cache_index.shape[-1] + self.window_size
            k_lens.append(l)
            max_seqlen_k = max(max_seqlen_k, l)
            klen_sum += l

            cache_index = cache_index.view(1, 1, -1,
                                           1).expand(-1, -1, -1, head_dim)
            top_Kcache = origin_heads_key_states[head_idx].gather(
                dim=2, index=cache_index)
            top_Vcache = origin_heads_value_states[head_idx].gather(
                dim=2, index=cache_index)
            selected_k = torch.cat([
                top_Kcache,
                origin_heads_key_states[head_idx][:, :, -self.window_size:, :]
            ],
                                   dim=2)
            selected_v = torch.cat([
                top_Vcache,
                origin_heads_value_states[head_idx][:, :,
                                                    -self.window_size:, :]
            ],
                                   dim=2)

            # NOTE: flatten view
            heads_key_states.append(selected_k.view(-1, head_dim))
            heads_value_states.append(selected_v.view(-1, head_dim))

        init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)

        # NOTE: compose as flatten view
        heads_key_states = torch.cat(heads_key_states, dim=0)
        heads_value_states = torch.cat(heads_value_states, dim=0)

        return heads_key_states, heads_value_states


class HeadKVCluster():
    """adapt from https://github.com/FFY0/AdaKV."""

    def __init__(self,
                 window_size=32,
                 kernel_size=7,
                 pooling='maxpool',
                 max_capacity_prompt=None,
                 layer_idx=None,
                 num_hidden_layers=None,
                 head_capacity=None):
        self.window_size = window_size
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.base_capacity = max_capacity_prompt - window_size
        self.head_adaptive_capacity = head_capacity
        self.num_hidden_layers = num_hidden_layers

        self.layer_idx = layer_idx

        self.head_lens = None
        self.max_seqlen_k = 0
        self.klen_sum = 0
        self.cu_klen = 0
        self.cu_offset = None
        self.cu_headlens = None

    def calcul_attn_sore(self, key_states, query_states):
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_weights = torch.matmul(query_states[..., -self.window_size:, :],
                                    key_states.transpose(
                                        2, 3)) / math.sqrt(head_dim)
        mask = torch.full((self.window_size, self.window_size),
                          torch.finfo(attn_weights.dtype).min,
                          device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1),
                          0)
        mask = mask.to(attn_weights.device)
        attention_mask = mask[None, None, :, :]

        attn_weights[:, :, -self.window_size:,
                     -self.window_size:] += attention_mask

        attn_weights = nn.functional.softmax(attn_weights,
                                             dim=-1,
                                             dtype=torch.float32).to(
                                                 query_states.dtype)
        attn_weights_mean = attn_weights[:, :, -self.window_size:, :-self.
                                         window_size].mean(dim=-2)
        if self.pooling == 'avgpool':
            attn_weights_mean_pooling = F.avg_pool1d(
                attn_weights_mean,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1)
        elif self.pooling == 'maxpool':
            attn_weights_mean_pooling = F.max_pool1d(
                attn_weights_mean,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1)
        else:
            raise ValueError('Pooling method not supported')
        return attn_weights_mean_pooling

    def update_kv(self, key_states, query_states, value_states):
        # check if prefix phase        assert key_states.shape[-2] == query_states.shape[-2]
        _device = key_states.device
        bsz, num_heads, q_len, head_dim = query_states.shape
        attn_score = self.calcul_attn_sore(key_states, query_states)
        origin_heads_key_states = torch.split(key_states, 1, dim=1)
        origin_heads_value_states = torch.split(value_states, 1, dim=1)

        def init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k):
            # init metadata
            self.head_lens = torch.tensor(k_lens,
                                          dtype=torch.int32,
                                          device=_device)
            self.klen_sum = klen_sum
            self.max_seqlen_k = max_seqlen_k
            self.cu_headlens = torch.cumsum(self.head_lens,
                                            dim=0,
                                            dtype=torch.int32)
            # init varlen flash attention metadata
            self.cu_klen = self.cu_headlens - self.head_lens
            self.cu_klen = torch.cat([
                self.cu_klen,
                torch.tensor(
                    [self.klen_sum], dtype=torch.int32, device=_device)
            ],
                                     dim=0)
            self.layer_qlens = torch.ones(num_heads,
                                          dtype=torch.int32,
                                          device=_device)
            self.qlen_sum = num_heads
            self.cu_qlen = torch.cumsum(
                self.layer_qlens, dim=0, dtype=torch.int32) - self.layer_qlens
            self.cu_qlen = torch.cat([
                self.cu_qlen,
                torch.tensor(
                    [self.qlen_sum], dtype=torch.int32, device=_device)
            ],
                                     dim=0)
            self.cu_offset = torch.arange(0,
                                          num_heads + 1,
                                          dtype=torch.int32,
                                          device=_device)
            self.cu_head_offset = torch.arange(1,
                                               num_heads + 1,
                                               dtype=torch.int32,
                                               device=_device)

        if self.base_capacity > attn_score.size(-1):
            init_metadata(num_heads, [q_len] * num_heads, q_len * num_heads,
                          q_len)
            # not compress
            return key_states.reshape(-1, head_dim), value_states.reshape(
                -1, head_dim)

        # if you need to weight the attn_score
        _, sorted_attn_score_indices = attn_score.sort(dim=-1, descending=True)
        sorted_attn_score_indices = sorted_attn_score_indices.split(1, dim=1)

        heads_key_states = []
        heads_value_states = []
        assert bsz == 1

        # per head
        # reinit varlen metadata
        k_lens = []
        klen_sum = 0
        max_seqlen_k = 0
        self.cu_klen = 0

        for head_idx in range(num_heads):
            cache_index = sorted_attn_score_indices[head_idx][
                ..., :self.head_adaptive_capacity[self.layer_idx][head_idx]]

            l = cache_index.shape[-1] + self.window_size
            k_lens.append(l)
            max_seqlen_k = max(max_seqlen_k, l)
            klen_sum += l

            cache_index = cache_index.view(1, 1, -1,
                                           1).expand(-1, -1, -1, head_dim)
            top_Kcache = origin_heads_key_states[head_idx].gather(
                dim=2, index=cache_index)
            top_Vcache = origin_heads_value_states[head_idx].gather(
                dim=2, index=cache_index)
            selected_k = torch.cat([
                top_Kcache,
                origin_heads_key_states[head_idx][:, :, -self.window_size:, :]
            ],
                                   dim=2)
            selected_v = torch.cat([
                top_Vcache,
                origin_heads_value_states[head_idx][:, :,
                                                    -self.window_size:, :]
            ],
                                   dim=2)

            # NOTE: flatten view
            heads_key_states.append(selected_k.view(-1, head_dim))
            heads_value_states.append(selected_v.view(-1, head_dim))

        init_metadata(num_heads, k_lens, klen_sum, max_seqlen_k)

        # NOTE: compose as flatten view
        heads_key_states = torch.cat(heads_key_states, dim=0)
        heads_value_states = torch.cat(heads_value_states, dim=0)

        return heads_key_states, heads_value_states




def init_StreamingLLM(self):
    if not hasattr(self, 'kv_cluster'):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 128
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 256
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None

    self.kv_cluster = StreamingLLMKVCluster(
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        kernel_size=self.config.kernel_size,
        pooling=self.config.pooling,
        merge=self.config.merge,
    )


