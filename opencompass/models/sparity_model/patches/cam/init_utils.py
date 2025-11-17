"""Initialization utilities for cam."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from transformers.cache_utils import Cache


class CAMKVCluster:

    def __init__(self,
                 start_budget_ratio=0.1,
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
        self.start_budget_ratio = start_budget_ratio
        self.merge = merge

    def reset(self,
              start_budget_ratio=0.1,
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
        self.start_budget_ratio = start_budget_ratio
        self.merge = merge

    def update_kv(self, key_states, query_states, value_states, attention_mask,
                  num_key_value_groups):

        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        print(f'CAM max_capacity_prompt {self.max_capacity_prompt}')

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(
                query_states[..., -self.window_size:, :],
                key_states.transpose(2, 3)) / math.sqrt(head_dim)
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
            attn_weights_sum = attn_weights[:, :, :, :-self.window_size].sum(
                dim=-2)
            # if self.pooling == 'avgpool':
            #     attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            # elif self.pooling == 'maxpool':
            #     attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            # else:
            #     raise ValueError('Pooling method not supported')
            attn_cache = attn_weights_sum

            # merge recent tokens
            start_budget = math.ceil(self.start_budget_ratio * q_len)
            recent_budget = self.window_size
            # start_budget = math.ceil(self.start_budget_ratio * attn_weights.shape[-1])
            # recent_budget = math.ceil(self.recent_budget_ratio * attn_weights.shape[-1])
            # print(f"start_budget {start_budget}")
            # print(f"recent_budget {recent_budget}")

            # CAM merge
            seq_length = attn_weights.shape[-1]
            padding_length = 0
            merge_budget = recent_budget
            for token_index in range(
                    start_budget + padding_length + recent_budget, seq_length):
                if token_index - recent_budget < 0 or token_index - recent_budget >= value_states.shape[
                        2]:
                    continue
                attn_score = torch.mean(
                    attn_weights[:, :, :token_index, :token_index], dim=-2)
                mean_attn = torch.max(torch.cat(
                    (attn_score[0, :, :start_budget],
                     attn_score[0, :,
                                token_index - recent_budget:token_index]),
                    dim=-1),
                                      dim=-1)[0]
                merge_prob = attn_score[0, :, token_index -
                                        recent_budget] / mean_attn
                if torch.isnan(merge_prob).any():
                    merge_prob[torch.isnan(merge_prob)] = 0
                if torch.isinf(merge_prob).any():
                    merge_prob[torch.isinf(merge_prob)] = 1
                merge_mask = torch.bernoulli(merge_prob.clamp(min=0, max=1))
                score1 = value_states[:, :, token_index - recent_budget,
                                      ...].clone() * merge_mask.unsqueeze(
                                          -1) / merge_budget
                value_states[:, :, token_index - recent_budget +
                             1:token_index - recent_budget + merge_budget +
                             1, :] += score1.unsqueeze(2)

            indices = attn_cache.topk(self.max_capacity_prompt -
                                      self.window_size,
                                      dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(
                dim=2, index=indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(
                dim=2, index=indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim=2)
            value_states = torch.cat([v_past_compress, v_cur], dim=2)

            return key_states, value_states




def init_CAM(self):
    if not hasattr(self, 'kv_cluster'):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'

    self.kv_cluster = CAMKVCluster(
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        kernel_size=self.config.kernel_size,
        pooling=self.config.pooling,
        merge=self.config.merge,
    )


