"""Initialization utilities for Min-Max-GQA."""

import torch
import torch.nn as nn


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    Repeat KV heads for GQA.
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep).

    The hidden states go from (batch, num_key_value_heads, seqlen, head_dim) to
    (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


class MinMaxKVCluster(nn.Module):
    """
    Max+min KV compression using group-based max+min token scoring.

    Core idea:
    1. For each KV head group, compute attention scores for all Q heads
    2. Use max(scores) + min(scores) as the representative score for each token
    3. Select top-k tokens based on these scores

    This is inspired by Quest's max/min bound estimation but applied at token level.

    Strategy (same as SnapKV):
    - Keep recent window_size tokens (local context)
    - Select (max_capacity_prompt - window_size) important tokens from history using Quest scoring
    """

    def __init__(
        self,
        num_hidden_layers=32,
        layer_idx=0,
        window_size=64,
        max_capacity_prompt=256 + 64,
        kernel_size=5,
        pooling='avgpool',
    ):
        super().__init__()
        self.num_hidden_layers = num_hidden_layers
        self.layer_idx = layer_idx
        self.window_size = window_size
        self.max_capacity_prompt = max_capacity_prompt
        self.kernel_size = kernel_size
        self.pooling = pooling

        assert self.max_capacity_prompt - self.window_size > 0, \
            f"max_capacity_prompt ({max_capacity_prompt}) must be larger than window_size ({window_size})"

    def update_kv(
        self,
        key_states,        # [bsz, num_kv_heads, seq_len, head_dim]
        query_states,      # [bsz, num_q_heads, q_len, head_dim]
        value_states,      # [bsz, num_kv_heads, seq_len, head_dim]
        attention_mask,
        num_key_value_groups,  # num_q_heads / num_kv_heads
    ):
        """
        Update KV cache with Max+min group scoring.

        Strategy (same as SnapKV):
        1. Keep recent window_size tokens (local context)
        2. From history, select (max_capacity_prompt - window_size) tokens using Quest scoring
        3. Concatenate: [historical_important_tokens, recent_tokens]

        Args:
            key_states: [bsz, num_kv_heads, seq_len, head_dim]
            query_states: [bsz, num_q_heads, q_len, head_dim]
            value_states: [bsz, num_kv_heads, seq_len, head_dim]
            attention_mask: attention mask
            num_key_value_groups: number of Q heads per KV head (GQA group size)

        Returns:
            compressed_key: [bsz, num_kv_heads, max_capacity_prompt, head_dim]
            compressed_value: [bsz, num_kv_heads, max_capacity_prompt, head_dim]
        """
        bsz, num_kv_heads, seq_len, head_dim = key_states.shape
        num_q_heads = query_states.shape[1]

        # If sequence is short enough, no compression needed
        if seq_len < self.max_capacity_prompt:
            return key_states, value_states

        # Step 1: Compute Max+min scores for HISTORICAL tokens only
        # We only score tokens in history, not the recent window
        historical_len = seq_len - self.window_size

        # Expand key_states for computing attention with all Q heads
        # [bsz, num_q_heads, seq_len, head_dim]
        key_states_expanded = repeat_kv(key_states, num_key_value_groups)

        # Compute attention weights using observation window (last window_size queries)
        # [bsz, num_q_heads, window_size, seq_len]
        attn_weights = torch.matmul(
            query_states[..., -self.window_size:, :],
            key_states_expanded.transpose(2, 3)
        ) / (head_dim ** 0.5)

        # Apply causal mask to the observation window
        import math
        mask = torch.full(
            (self.window_size, self.window_size),
            torch.finfo(attn_weights.dtype).min,
            device=attn_weights.device
        )
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        attention_mask_window = mask[None, None, :, :]
        attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask_window

        # Softmax
        attn_weights = torch.nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)

        # Sum over observation window, only for HISTORICAL tokens
        # [bsz, num_q_heads, historical_len]
        attn_weights_sum = attn_weights[:, :, -self.window_size:, :-self.window_size].sum(dim=-2)

        # Apply pooling first (smooth each Q head independently)
        if self.pooling == 'avgpool':
            import torch.nn.functional as F
            attn_weights_sum = F.avg_pool1d(
                attn_weights_sum,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1
            )
        elif self.pooling == 'maxpool':
            import torch.nn.functional as F
            attn_weights_sum = F.max_pool1d(
                attn_weights_sum,
                kernel_size=self.kernel_size,
                padding=self.kernel_size // 2,
                stride=1
            )

        # Aggregate from MHA to GQA format (after pooling)
        # [bsz, num_q_heads, historical_len] -> [bsz, num_kv_heads, num_key_value_groups, historical_len]
        attn_weights_sum = attn_weights_sum.view(
            bsz, num_kv_heads, num_key_value_groups, historical_len
        )

        # Max+min scoring: max + min across group (based on smoothed scores)
        # [bsz, num_kv_heads, historical_len]
        group_max_scores = attn_weights_sum.max(dim=2)[0]  # max over Q heads in group
        group_min_scores = attn_weights_sum.min(dim=2)[0]  # min over Q heads in group
        min_max_scores = group_max_scores + group_min_scores

        # Step 2: Select top-k from historical tokens
        num_select_from_history = self.max_capacity_prompt - self.window_size

        # [bsz, num_kv_heads, num_select_from_history]
        _, topk_indices = torch.topk(
            min_max_scores, k=num_select_from_history, dim=-1, largest=True
        )

        # Expand indices for gathering
        indices_expanded = topk_indices.unsqueeze(-1).expand(
            bsz, num_kv_heads, num_select_from_history, head_dim
        )

        # Step 3: Gather historical important tokens
        k_past_compress = key_states[:, :, :-self.window_size, :].gather(
            dim=2, index=indices_expanded
        )
        v_past_compress = value_states[:, :, :-self.window_size, :].gather(
            dim=2, index=indices_expanded
        )

        # Step 4: Get recent tokens
        k_recent = key_states[:, :, -self.window_size:, :]
        v_recent = value_states[:, :, -self.window_size:, :]

        # Step 5: Concatenate [historical_important, recent]
        compressed_key = torch.cat([k_past_compress, k_recent], dim=2)
        compressed_value = torch.cat([v_past_compress, v_recent], dim=2)

        # Final shape: [bsz, num_kv_heads, max_capacity_prompt, head_dim]
        assert compressed_key.shape[2] == self.max_capacity_prompt

        return compressed_key, compressed_value


def init_min_max_gqa(self):
    """Initialize Min-Max-GQA components for the attention layer."""
    if not hasattr(self, 'kv_cluster'):
        # Get layer index
        layer_idx = 0
        if hasattr(self, 'layer_idx'):
            layer_idx = self.layer_idx

        # Get number of layers
        num_hidden_layers = 32  # default
        if hasattr(self, 'config') and hasattr(self.config, 'num_hidden_layers'):
            num_hidden_layers = self.config.num_hidden_layers

        # Get configuration parameters
        window_size = getattr(self.config, 'window_size', 64)
        max_capacity_prompt = getattr(self.config, 'max_capacity_prompt', 256 + 64)
        kernel_size = getattr(self.config, 'kernel_size', 5)
        pooling = getattr(self.config, 'pooling', 'avgpool')

        # Create KV cluster module
        self.kv_cluster = MinMaxKVCluster(
            num_hidden_layers=num_hidden_layers,
            layer_idx=layer_idx,
            window_size=window_size,
            max_capacity_prompt=max_capacity_prompt,
            kernel_size=kernel_size,
            pooling=pooling,
        )

        # Move to same device as the layer
        if hasattr(self, 'q_proj') and hasattr(self.q_proj, 'weight'):
            device = self.q_proj.weight.device
            self.kv_cluster = self.kv_cluster.to(device)
