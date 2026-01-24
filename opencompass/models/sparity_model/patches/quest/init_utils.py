"""Initialization utilities for Quest."""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class QuestCluster():
    """Quest attention cluster for KV cache compression using chunk-based selection."""

    def __init__(self,
                 token_budget=256,
                 chunk_size=16,
                 window_size=64,
                 max_capacity_prompt=256 + 64,
                 save_indices=False,
                 save_key_states=False,
                 save_query_states=False,
                 visualize_layer=None,
                 max_samples_to_save=10,
                 run_timestamp=None,
                 global_counter_ref=None,
                 dataset_name=None):
        self.token_budget = token_budget
        self.chunk_size = chunk_size
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.save_indices = save_indices
        self.save_key_states = save_key_states
        self.save_query_states = save_query_states
        self.visualize_layer = visualize_layer
        self.max_samples_to_save = max_samples_to_save
        self.run_timestamp = run_timestamp
        self.global_counter_ref = global_counter_ref
        self.dataset_name = dataset_name

    def local_heavy_hitter_mask(self, attn_weights, token_budget, chunk_size):
        """Create mask for heavy hitter tokens based on chunk-based selection."""
        # attn_weights (BS, head, query, keys)
        seq_length = attn_weights.shape[-1]
        padding_length = chunk_size - ((seq_length - 1) % chunk_size + 1)

        attn_weights = torch.cat(
            [
                attn_weights,
                torch.ones(
                    (
                        attn_weights.shape[0],
                        attn_weights.shape[1],
                        attn_weights.shape[2],
                        padding_length,
                    ),
                    device=attn_weights.device,
                )
                * torch.tensor(torch.finfo(attn_weights.dtype).min),
            ],
            dim=-1,
        )

        # chunk attn_weights into chunk_size tokens
        chunk_attn_weights = attn_weights.reshape(
            attn_weights.shape[0],
            attn_weights.shape[1],
            attn_weights.shape[2],
            attn_weights.shape[3] // chunk_size,
            chunk_size,
        ).amax(dim=-1)

        _, topk = chunk_attn_weights.topk(
            k=min(max(3, token_budget // chunk_size), chunk_attn_weights.size(-1)), dim=-1
        )

        # repeat topk chunk_size times and recover the original indexes
        topk = topk.unsqueeze(-1).repeat(
            1, 1, 1, 1, chunk_size
        ) * chunk_size + torch.arange(chunk_size, device=topk.device)
        topk = topk.reshape(topk.shape[0], topk.shape[1], topk.shape[2], -1)
        mask_bottom = torch.zeros_like(attn_weights, dtype=torch.bool)
        mask_bottom.scatter_(-1, topk, True)

        # remove the padding
        mask_bottom = mask_bottom[:, :, :, :seq_length]

        return mask_bottom

    def update_kv(self, key_states, query_states, value_states, attention_mask,
                  num_key_value_groups, layer_idx=None):
        """Update KV cache using Quest's chunk-based selection method."""
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        if q_len < self.max_capacity_prompt:
            return key_states, value_states

        # Quest's quantized weight estimation
        sign = (query_states > 0) + (~(query_states > 0)) * -1
        max_key = key_states * sign
        positive_query = query_states * sign

        # Expand max_key to be divisible by chunk_size
        seq_length = max_key.shape[-2]
        padding_length = self.chunk_size - ((seq_length - 1) % self.chunk_size + 1)
        max_key = torch.cat(
            [
                max_key,
                torch.ones(
                    (max_key.shape[0], max_key.shape[1], padding_length, max_key.shape[3]),
                    device=max_key.device,
                )
                * torch.tensor(torch.finfo(max_key.dtype).min),
            ],
            dim=-2,
        )

        # Chunk max_key into chunk_size tokens
        chunk_max_key = max_key.reshape(
            max_key.shape[0],
            max_key.shape[1],
            max_key.shape[2] // self.chunk_size,
            self.chunk_size,
            max_key.shape[3],
        ).amax(dim=-2)

        # Duplicate chunk_max_key chunk_size times
        chunk_max_key = chunk_max_key.unsqueeze(-2).repeat(1, 1, 1, self.chunk_size, 1)
        chunk_max_key = chunk_max_key.reshape(
            chunk_max_key.shape[0], chunk_max_key.shape[1], -1, chunk_max_key.shape[-1]
        )[:, :, :seq_length, :]

        # Compute quantized attention weights using last window_size queries
        quantized_weight = torch.matmul(
            positive_query[..., -self.window_size:, :].float(),
            chunk_max_key.transpose(2, 3),
        )

        # Apply attention mask if provided
        if attention_mask is not None:
            # Create window mask for causal attention
            window_mask = attention_mask[:, :, -self.window_size:, :seq_length]
            quantized_weight = quantized_weight + window_mask
            quantized_weight = torch.max(
                quantized_weight, torch.tensor(torch.finfo(quantized_weight.dtype).min)
            )

        # Sum attention weights across window queries
        attn_weights_sum = quantized_weight.sum(dim=-2)

        # Select top-k tokens based on quantized weights
        token_budget = min(seq_length, self.max_capacity_prompt - self.window_size)
        indices = attn_weights_sum.topk(token_budget, dim=-1).indices
        indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

        # Compress KV cache
        k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
        v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim=2, index=indices)
        k_cur = key_states[:, :, -self.window_size:, :]
        v_cur = value_states[:, :, -self.window_size:, :]

        key_states = torch.cat([k_past_compress, k_cur], dim=2)
        value_states = torch.cat([v_past_compress, v_cur], dim=2)

        return key_states, value_states


def init_quest(self):
    """Initialize Quest cluster for attention module."""
    if not hasattr(self, 'kv_cluster'):
        if not hasattr(self.config, 'token_budget'):
            self.config.token_budget = 256
        if not hasattr(self.config, 'chunk_size'):
            self.config.chunk_size = 16
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 64
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 256 + 64
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
            import os
            self.config.run_timestamp = os.getenv('RUN_TIMESTAMP', None)
        if not hasattr(self.config, 'dataset_name'):
            import os
            self.config.dataset_name = os.getenv('DATASET_NAME', None)
        if not hasattr(self.config, '_global_sample_counter'):
            self.config._global_sample_counter = 0

    self.kv_cluster = QuestCluster(
        token_budget=self.config.token_budget,
        chunk_size=self.config.chunk_size,
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        save_indices=self.config.save_indices,
        save_key_states=self.config.save_key_states,
        save_query_states=self.config.save_query_states,
        visualize_layer=self.config.visualize_layer,
        max_samples_to_save=self.config.max_samples_to_save,
        run_timestamp=self.config.run_timestamp,
        global_counter_ref=self.config,
        dataset_name=self.config.dataset_name,
    )
