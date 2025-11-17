"""Initialization utilities for sparq."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple
from transformers.cache_utils import Cache


class SparQCluster():

    def __init__(self,
                 r=64,
                 k=256 + 64,
                 kernel_size=5,
                 pooling='avgpool',
                 merge=None,
                 recent_size=32,
                 ratio=0.4):
        self.r = r
        self.k = k
        assert self.k - self.r > 0
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
              merge=None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.ratio = ratio
        self.recent_size = recent_size

    @staticmethod
    def gather(t, dim, i):
        dim += (dim < 0) * t.ndim
        return t.gather(
            dim, i.expand(*t.shape[:dim], i.shape[dim], *t.shape[dim + 1:]))

    @staticmethod
    def attn(Q, K, V, M):
        s = (Q @ K.transpose(-1, -2)) / sqrt(
            tensor(Q.shape[-1], device=Q.device)) + M
        y = F.softmax(s, dim=-1) @ V
        return y

    def update_kv(
        self,
        key_states,
        query_states,
        value_states,
        M,
        num_key_value_groups,
    ):

        # check if prefix phase
        # assert M.shape == (batch, n_kv_heads, n_heads_per_kv, 1, seq)

        batch_size, num_heads, q_len, head_dim = query_states.shape
        batch_size, num_heads_K, k_len, head_dim = key_states.shape

        Q = query_states.view(batch_size, num_heads // num_key_value_groups,
                              num_key_value_groups, q_len, head_dim)
        K = key_states.view(batch_size, num_heads // num_key_value_groups, 1,
                            k_len, head_dim)
        V = value_states.view(batch_size, num_heads // num_key_value_groups, 1,
                              k_len, head_dim)
        device = Q.device
        dtype = Q.dtype
        M = torch.zeros(batch_size,
                        num_heads // num_key_value_groups,
                        num_key_value_groups,
                        1,
                        k_len,
                        device=device,
                        dtype=dtype)
        Q_abs_sum = torch.abs(Q).sum(dim=2, keepdim=True)
        i1 = torch.topk(Q_abs_sum, self.r, dim=-1).indices
        Q_hat = self.gather(Q, -1, i1)
        K_hat = self.gather(K, -1, i1)
        scale = sqrt(Q.shape[-1] * torch.abs(Q_hat).sum(dim=-1, keepdim=True) /
                     torch.abs(Q).sum(dim=-1, keepdim=True))
        s_hat_raw = Q_hat @ K_hat.transpose(-1, -2)
        s_hat = F.softmax(s_hat_raw / scale + M, dim=-1)

        s_hat_sum = s_hat.sum(
            dim=2, keepdim=True)  # [batch, n_kv_heads, 1, 1, seq_len]
        i2 = torch.topk(s_hat_sum, self.k,
                        dim=-1).indices  # [batch, n_kv_heads, 1, 1, k]

        # 收集 K, V, M 中对应的位置
        iKV = i2[..., 0, :, None]  # [batch, n_kv_heads, 1, k, 1]
        K_top = self.gather(K, -2, iKV)  # [batch, n_kv_heads, 1, k, head_size]
        V_top = self.gather(V, -2, iKV)  # [batch, n_kv_heads, 1, k, head_size]
        M_top = self.gather(M, -1, i2)  # [batch, 1, 1, k]
        y_top = self.attn(
            Q, K_top, V_top,
            M_top)  # [batch, n_kv_heads, n_heads//n_kv_heads, 1, head_size]

        s_hat_topk = self.gather(
            s_hat, -1, i2)  # [batch, n_kv_heads, n_heads//n_kv_heads, 1, k]
        alpha = s_hat_topk.sum(
            -1, keepdim=True)  # [batch, n_kv_heads, n_heads//n_kv_heads, 1, 1]

        V_mean = V.mean(dim=-2, keepdim=True)  # 在序列维度上取平均
        y = alpha * y_top + (
            1 - alpha
        ) * V_mean  # [batch, n_kv_heads, n_heads//n_kv_heads, 1, head_size]

        return y




def init_sparq(self):
    if not hasattr(self, 'kv_cluster'):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 4
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 32
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None

    self.kv_cluster = SparQCluster(
        r=self.config.window_size,
        k=self.config.max_capacity_prompt,
        kernel_size=self.config.kernel_size,
        pooling=self.config.pooling,
        merge=self.config.merge,
    )
