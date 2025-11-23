import math
import torch
import torch.nn.functional as F
from transformers.models.llama.modeling_llama import repeat_kv


class ChunkKVCluster():
    """
    ChunkKV: Semantic-preserving compression with chunk-wise token selection.

    Based on the algorithm described in ChunkKV paper (https://arxiv.org/abs/2502.00299).
    Instead of selecting individual tokens globally, ChunkKV:
    1. Computes global importance scores for all tokens
    2. Aggregates scores at chunk level
    3. Selects top-k chunks based on aggregated scores
    4. Preserves all tokens within selected chunks
    """

    def __init__(self, chunk_length=20, max_budget_tokens=2048, observation_window=64):
        """
        Initialize ChunkKV cluster.

        Args:
            chunk_length (int): Length of each chunk for token selection. Default: 20
            max_budget_tokens (int): Maximum number of tokens to keep. Default: 2048
            observation_window (int): Number of recent query tokens to use for computing importance scores. Default: 64
        """
        self.chunk_length = chunk_length
        self.max_budget_tokens = max_budget_tokens
        self.observation_window = observation_window

        # Validate that budget can be evenly divided by chunk_length
        budget_for_past = max_budget_tokens - observation_window
        assert budget_for_past > 0, \
            f"max_budget_tokens ({max_budget_tokens}) must be greater than observation_window ({observation_window})"
        assert budget_for_past % chunk_length == 0, \
            f"Budget for past tokens ({budget_for_past}) must be divisible by chunk_length ({chunk_length}). " \
            f"Adjust max_budget_tokens to {(budget_for_past // chunk_length) * chunk_length + observation_window} " \
            f"or {((budget_for_past // chunk_length) + 1) * chunk_length + observation_window}"

    def compute_token_scores(self, query_states, key_states, head_dim, num_key_value_groups):
        """
        Compute importance scores for each token using attention mechanism.
        Similar to SnapKV, uses only the last observation_window queries for efficiency.

        Args:
            query_states: Query tensor [bsz, num_heads, q_len, head_dim]
            key_states: Key tensor [bsz, num_heads, kv_len, head_dim] (expanded)
            head_dim: Dimension of each attention head
            num_key_value_groups: Number of query heads per KV head

        Returns:
            Tensor of shape [bsz, num_kv_heads, kv_len] with importance scores (aggregated to GQA format)
        """
        bsz, num_heads, q_len, head_dim = query_states.shape
        kv_len = key_states.shape[2]
        num_kv_heads = num_heads // num_key_value_groups

        # Use only the last observation_window query tokens (like SnapKV)
        window_size = min(self.observation_window, q_len)
        query_window = query_states[..., -window_size:, :]

        # Compute attention weights: Q_window @ K^T / sqrt(d)
        attn_weights = torch.matmul(query_window, key_states.transpose(2, 3)) / math.sqrt(head_dim)

        # Apply causal mask to the observation window (prevent future tokens from attending)
        mask = torch.full((window_size, window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
        mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
        mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
        attention_mask = mask[None, None, :, :]
        attn_weights[:, :, -window_size:, -window_size:] += attention_mask

        # Apply softmax to get attention probabilities
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Sum attention weights over the observation window (exclude the window itself from past tokens)
        # Shape: [bsz, num_heads, kv_len - window_size]
        attn_weights_sum = attn_weights[:, :, :, :-window_size].sum(dim=2)

        # Aggregate from MHA format to GQA format: [bsz, num_heads, seq_len] -> [bsz, num_kv_heads, seq_len]
        attn_weights_sum = attn_weights_sum.view(bsz, num_kv_heads, num_key_value_groups, -1)
        attn_weights_sum = attn_weights_sum.mean(dim=2)  # Average over query heads in each group

        return attn_weights_sum

    def select_chunks(self, token_scores, kv_len, window_size):
        """
        Select top-k chunks based on aggregated token scores.

        Args:
            token_scores: Tensor [bsz, num_kv_heads, past_len] with token importance scores
            kv_len: Total number of tokens (including window)
            window_size: Size of the observation window (to be preserved at the end)

        Returns:
            List of selected token indices
        """
        bsz = token_scores.shape[0]
        past_len = kv_len - window_size  # Length of past tokens (excluding window)

        # Calculate number of complete chunks and remaining tokens in past
        num_complete_chunks = past_len // self.chunk_length
        remaining_tokens = past_len % self.chunk_length

        # If no complete chunks, return all indices (no compression)
        if num_complete_chunks == 0:
            return torch.arange(kv_len, device=token_scores.device)

        # Aggregate scores across KV heads (sum over head dimension)
        # Shape: [bsz, past_len]
        global_scores = token_scores.sum(dim=1)

        # Calculate chunk scores for complete chunks
        if num_complete_chunks > 0:
            main_scores = global_scores[:, :num_complete_chunks * self.chunk_length]
            # Reshape to [bsz, num_chunks, chunk_length]
            main_chunk_scores = main_scores.view(bsz, num_complete_chunks, self.chunk_length)
            # Average score per chunk: [bsz, num_chunks]
            main_chunk_scores = main_chunk_scores.mean(dim=-1)
        else:
            main_chunk_scores = torch.empty((bsz, 0), device=token_scores.device)

        # Handle remaining tokens as a partial chunk
        if remaining_tokens > 0:
            remaining_scores = global_scores[:, -remaining_tokens:]
            # Average score for the partial chunk: [bsz, 1]
            remaining_chunk_score = remaining_scores.mean(dim=-1, keepdim=True)
            # Concatenate: [bsz, num_chunks + 1]
            chunk_scores = torch.cat([main_chunk_scores, remaining_chunk_score], dim=-1)
        else:
            chunk_scores = main_chunk_scores

        # Calculate number of chunks to keep based on max_budget_tokens (excluding window)
        total_chunks = num_complete_chunks + (1 if remaining_tokens > 0 else 0)
        budget_for_past = self.max_budget_tokens - window_size

        # If past_len is already within budget, keep all chunks
        if past_len <= budget_for_past:
            n_chunks_kept = total_chunks
        else:
            # Calculate how many chunks we need to keep to stay within budget
            n_chunks_kept = max(1, budget_for_past // self.chunk_length)
            n_chunks_kept = min(n_chunks_kept, total_chunks)  # Don't exceed total chunks

        # Select top-k chunks
        top_chunks = chunk_scores.topk(n_chunks_kept, dim=-1)

        # Convert chunk indices to token indices
        indices = []
        for chunk_idx in top_chunks.indices[0]:
            if chunk_idx < num_complete_chunks:
                # Complete chunk
                start_idx = chunk_idx * self.chunk_length
                chunk_indices = torch.arange(start_idx, start_idx + self.chunk_length, device=token_scores.device)
            else:
                # Partial chunk (remaining tokens)
                chunk_indices = torch.arange(
                    num_complete_chunks * self.chunk_length,
                    past_len,
                    device=token_scores.device
                )
            indices.append(chunk_indices)

        # Concatenate and sort to maintain sequential order
        indices = torch.cat(indices).sort()[0]

        return indices

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        """
        GQA-optimized KV compression with chunk-wise selection.
        Similar to SnapKV, uses observation window and preserves recent tokens.

        Input shapes:
            key_states:   [bsz, num_key_value_heads, seq_len, head_dim] (e.g., [1, 4, 32768, 128])
            query_states: [bsz, num_heads, seq_len, head_dim]            (e.g., [1, 28, 32768, 128])
            value_states: [bsz, num_key_value_heads, seq_len, head_dim]

        Output shapes:
            key_states_compressed:   [bsz, num_key_value_heads, compressed_len, head_dim]
            value_states_compressed: [bsz, num_key_value_heads, compressed_len, head_dim]
        """
        # Check if prefilling phase
        assert key_states.shape[-2] == query_states.shape[-2], "Prefilling phase expected"

        bsz, num_kv_heads, kv_len, head_dim = key_states.shape

        # Skip compression if sequence is within budget
        if kv_len <= self.max_budget_tokens:
            return key_states, value_states

        # Determine window size (min of observation_window and sequence length)
        window_size = min(self.observation_window, kv_len)

        # Temporarily expand KV heads for attention computation
        key_states_expanded = repeat_kv(key_states, num_key_value_groups)

        # Compute token importance scores using expanded keys (only for past tokens, not the window)
        token_scores = self.compute_token_scores(query_states, key_states_expanded, head_dim, num_key_value_groups)

        # Select chunks based on scores (returns indices for past tokens only)
        selected_indices = self.select_chunks(token_scores, kv_len, window_size)

        # Apply compression to ORIGINAL key_states (num_key_value_heads, not expanded)
        # Split past and current (window) tokens
        k_past = key_states[:, :, :-window_size, :]
        v_past = value_states[:, :, :-window_size, :]
        k_cur = key_states[:, :, -window_size:, :]
        v_cur = value_states[:, :, -window_size:, :]

        # Expand indices for gather operation on past tokens
        # Shape: [bsz, num_kv_heads, num_selected, head_dim]
        indices_expanded = selected_indices.view(1, 1, -1, 1).expand(
            bsz, num_kv_heads, -1, head_dim
        )

        # Gather selected past keys and values from original (non-expanded) tensors
        k_past_compressed = k_past.gather(2, indices_expanded).contiguous()
        v_past_compressed = v_past.gather(2, indices_expanded).contiguous()

        # Concatenate compressed past with current window
        key_states_compressed = torch.cat([k_past_compressed, k_cur], dim=2)
        value_states_compressed = torch.cat([v_past_compressed, v_cur], dim=2)

        # Return compressed KV with original num_key_value_heads shape
        # e.g., [bsz, 4, compressed_len, head_dim] instead of [bsz, 28, compressed_len, head_dim]
        return key_states_compressed, value_states_compressed


def init_ChunkKV(self, chunk_length=20, max_budget_tokens=2048, observation_window=64):
    """
    Initialize ChunkKV cluster for attention layer.

    Args:
        self: Attention layer instance
        chunk_length: Length of each chunk. Default: 20
        max_budget_tokens: Maximum number of tokens to keep. Default: 2048
        observation_window: Number of recent query tokens to use for scoring. Default: 64
    """
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = ChunkKVCluster(
            chunk_length=self.config.chunk_length,
            max_budget_tokens=self.config.max_capacity_prompt,
            observation_window=getattr(self.config, 'observation_window', 64),
        )
