"""Initialization utilities for snapkv."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils.kv_utils import save_topk_indices, save_key_states, save_query_states


class SnapKVCluster():

    def __init__(self,
                 window_size=64,
                 max_capacity_prompt=256 + 64,
                 kernel_size=5,
                 pooling='avgpool',
                 merge=None,
                 recent_size=32,
                 ratio=0.4,
                 save_indices=False,
                 save_key_states=False,
                 save_query_states=False,
                 visualize_layer=None,
                 max_samples_to_save=10,
                 run_timestamp=None,
                 global_counter_ref=None,
                 dataset_name=None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.ratio = ratio
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge
        self.recent_size = recent_size
        self.ratio = ratio
        self.save_indices = save_indices  # 是否保存索引
        self.save_key_states = save_key_states  # 是否保存key states
        self.save_query_states = save_query_states  # 是否保存query states
        self.visualize_layer = visualize_layer  # 指定要可视化的层，None表示所有层
        self.max_samples_to_save = max_samples_to_save  # 最多保存多少个样本
        self.run_timestamp = run_timestamp  # 运行时间戳（用于文件夹组织）
        self.global_counter_ref = global_counter_ref  # 全局计数器引用（config对象）
        self.dataset_name = dataset_name  # 数据集名称（用于文件夹组织）

    def reset(self,
              window_size=64,
              max_capacity_prompt=256 + 64,
              kernel_size=5,
              pooling='avgpool',
              ratio = 0,
              merge=None):
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.ratio = ratio
        assert self.max_capacity_prompt - self.window_size > 0
        self.kernel_size = kernel_size
        self.pooling = pooling
        self.merge = merge


    def update_kv(self, key_states, query_states, value_states, attention_mask,
                  num_key_value_groups, layer_idx=None):
            # check if prefix phase
            # Removed debug print for performance
            assert key_states.shape[-2] == query_states.shape[-2]
            bsz, num_heads, q_len, head_dim = query_states.shape

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
                attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.
                                                window_size].sum(dim=-2)
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

                indices = attn_cache.topk(self.max_capacity_prompt -
                                        self.window_size,
                                        dim=-1).indices

                # 可视化保存 topk 索引、key states 和 query states
                # save_indices、save_key_states 和 save_query_states 独立控制
                if (self.save_indices or self.save_key_states or self.save_query_states) and layer_idx is not None and self.global_counter_ref is not None:
                    current_sample_id = self.global_counter_ref._global_sample_counter

                    # 只在前 max_samples_to_save 个样本中保存
                    if current_sample_id < self.max_samples_to_save:
                        # 如果指定了 visualize_layer，只保存该层；否则保存所有层
                        if self.visualize_layer is None or layer_idx == self.visualize_layer:
                            # 保存 topk 索引（仅在 save_indices=True 时）
                            if self.save_indices:
                                save_topk_indices(
                                    indices=indices,
                                    layer_idx=layer_idx,
                                    attn_weights_sum=attn_cache,  # 保存池化后的注意力分数
                                    method="snapkv",
                                    sample_id=current_sample_id,
                                    capacity=self.max_capacity_prompt,
                                    run_timestamp=self.run_timestamp,
                                    num_key_value_groups=num_key_value_groups,  # 传递 GQA group 信息
                                    dataset_name=self.dataset_name  # 传递数据集名称
                                )

                            # 保存 key states（仅在 save_key_states=True 时）
                            if self.save_key_states:
                                save_key_states(
                                    key_states=key_states[:, :, :-self.window_size, :],  # 排除window部分，只保存要压缩的部分
                                    layer_idx=layer_idx,
                                    attn_weights_sum=attn_cache,  # 保存池化后的注意力分数
                                    method="snapkv",
                                    sample_id=current_sample_id,
                                    capacity=self.max_capacity_prompt,
                                    run_timestamp=self.run_timestamp,
                                    num_key_value_groups=num_key_value_groups,  # 传递 GQA group 信息
                                    dataset_name=self.dataset_name  # 传递数据集名称
                                )

                            # 保存 query states（仅在 save_query_states=True 时）
                            if self.save_query_states:
                                save_query_states(
                                    query_states=query_states[..., -self.window_size:, :],  # 只保存window部分的query
                                    layer_idx=layer_idx,
                                    attn_weights_sum=attn_cache,  # 保存池化后的注意力分数
                                    method="snapkv",
                                    sample_id=current_sample_id,
                                    capacity=self.max_capacity_prompt,
                                    run_timestamp=self.run_timestamp,
                                    num_key_value_groups=num_key_value_groups,  # 传递 GQA group 信息
                                    dataset_name=self.dataset_name  # 传递数据集名称
                                )

                            # 只在第一层（layer 0）或指定的 visualize_layer 增加计数器
                            # 避免在每层都重复计数
                            if layer_idx == (self.visualize_layer if self.visualize_layer is not None else 0):
                                self.global_counter_ref._global_sample_counter += 1
                                if self.global_counter_ref._global_sample_counter == self.max_samples_to_save:
                                    save_status = []
                                    if self.save_indices:
                                        save_status.append("indices")
                                    if self.save_key_states:
                                        save_status.append("key_states")
                                    if self.save_query_states:
                                        save_status.append("query_states")
                                    print(f"[Visualization] Reached max samples ({self.max_samples_to_save}), stopping {'/'.join(save_status)} saving.")

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


def init_snapkv(self):
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
        if not hasattr(self.config, 'save_indices'):
            self.config.save_indices = False
        if not hasattr(self.config, 'save_key_states'):
            self.config.save_key_states = False
        if not hasattr(self.config, 'save_query_states'):
            self.config.save_query_states = False
        if not hasattr(self.config, 'visualize_layer'):
            self.config.visualize_layer = None
        if not hasattr(self.config, 'max_samples_to_save'):
            self.config.max_samples_to_save = 10
        if not hasattr(self.config, 'run_timestamp'):
            # Try to read from environment variable, or use None
            import os
            self.config.run_timestamp = os.getenv('RUN_TIMESTAMP', None)

        if not hasattr(self.config, 'dataset_name'):
            # Try to read from environment variable, or use None
            import os
            self.config.dataset_name = os.getenv('DATASET_NAME', None)

        # Initialize global sample counter (shared across all layers)
        if not hasattr(self.config, '_global_sample_counter'):
            self.config._global_sample_counter = 0


    self.kv_cluster = SnapKVCluster(
            window_size=self.config.window_size,
            max_capacity_prompt=self.config.max_capacity_prompt,
            ratio = 0.4,
            kernel_size=7,
            pooling=self.config.pooling,
            merge=self.config.merge,
            save_indices=self.config.save_indices,
            save_key_states=self.config.save_key_states,
            save_query_states=self.config.save_query_states,
            visualize_layer=self.config.visualize_layer,
            max_samples_to_save=self.config.max_samples_to_save,
            run_timestamp=self.config.run_timestamp,
            global_counter_ref=self.config,  # 传递 config 对象作为全局计数器引用
            dataset_name=self.config.dataset_name,  # 传递数据集名称
    )
