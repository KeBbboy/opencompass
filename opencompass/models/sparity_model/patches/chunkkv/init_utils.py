import math
import torch
import torch.nn.functional as F


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

    def __init__(self, chunk_length=20, max_budget_tokens=2048):
        """
        Initialize ChunkKV cluster.

        Args:
            chunk_length (int): Length of each chunk for token selection. Default: 20
            max_budget_tokens (int): Maximum number of tokens to keep. Default: 2048
        """
        self.chunk_length = chunk_length
        self.max_budget_tokens = max_budget_tokens

    def compute_token_scores(self, query_states, key_states, head_dim):
        """
        Compute importance scores for each token using attention mechanism.

        Args:
            query_states: Query tensor [bsz, num_heads, q_len, head_dim]
            key_states: Key tensor [bsz, num_heads, kv_len, head_dim]
            head_dim: Dimension of each attention head

        Returns:
            Tensor of shape [bsz, num_heads, kv_len] with importance scores
        """
        # Compute attention scores: Q @ K^T / sqrt(d)
        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)

        # Apply softmax to get attention probabilities
        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)

        # Sum attention scores across query positions to get token importance
        # Shape: [bsz, num_heads, kv_len]
        token_scores = attn_weights.sum(dim=-2)

        return token_scores

    def select_chunks(self, token_scores, kv_len):
        """
        Select top-k chunks based on aggregated token scores.

        Args:
            token_scores: Tensor [bsz, num_heads, kv_len] with token importance scores
            kv_len: Total number of tokens

        Returns:
            List of selected token indices
        """
        bsz = token_scores.shape[0]

        # Calculate number of complete chunks and remaining tokens
        num_complete_chunks = kv_len // self.chunk_length
        remaining_tokens = kv_len % self.chunk_length

        # If no complete chunks, return all indices (no compression)
        if num_complete_chunks == 0:
            return torch.arange(kv_len, device=token_scores.device)

        # Aggregate scores across heads (sum over head dimension)
        # Shape: [bsz, kv_len]
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

        # Calculate number of chunks to keep based on max_budget_tokens
        total_chunks = num_complete_chunks + (1 if remaining_tokens > 0 else 0)

        # If kv_len is already within budget, keep all chunks
        if kv_len <= self.max_budget_tokens:
            n_chunks_kept = total_chunks
        else:
            # Calculate how many chunks we need to keep to stay within budget
            n_chunks_kept = max(1, self.max_budget_tokens // self.chunk_length)
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
                    kv_len,
                    device=token_scores.device
                )
            indices.append(chunk_indices)

        # Concatenate and sort to maintain sequential order
        indices = torch.cat(indices).sort()[0]

        return indices

    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        """
        Update KV cache with chunk-wise compression.

        Args:
            key_states: Key tensor [bsz, num_heads, kv_len, head_dim]
            query_states: Query tensor [bsz, num_heads, q_len, head_dim]
            value_states: Value tensor [bsz, num_heads, kv_len, head_dim]
            attention_mask: Attention mask
            num_key_value_groups: Number of key-value groups

        Returns:
            Compressed key_states and value_states
        """
        # Check if prefilling phase
        assert key_states.shape[-2] == query_states.shape[-2], "Prefilling phase expected"

        bsz, num_heads, kv_len, head_dim = key_states.shape

        # Skip compression if sequence is within budget or too short
        if kv_len <= self.max_budget_tokens or kv_len <= self.chunk_length:
            return key_states, value_states

        # Compute token importance scores
        token_scores = self.compute_token_scores(query_states, key_states, head_dim)

        # Select chunks based on scores
        selected_indices = self.select_chunks(token_scores, kv_len)

        # Expand indices for gather operation
        # Shape: [bsz, num_heads, num_selected, head_dim]
        indices_expanded = selected_indices.view(1, 1, -1, 1).expand(
            bsz, num_heads, -1, head_dim
        )

        # Gather selected keys and values
        key_states_compressed = key_states.gather(2, indices_expanded).contiguous()
        value_states_compressed = value_states.gather(2, indices_expanded).contiguous()

        return key_states_compressed, value_states_compressed


def init_ChunkKV(self, chunk_length=20, max_budget_tokens=2048):
    """
    Initialize ChunkKV cluster for attention layer.

    Args:
        self: Attention layer instance
        num_hidden_layers: Total number of hidden layers in the model
        chunk_length: Length of each chunk. Default: 20
        max_budget_tokens: Maximum number of tokens to keep. Default: 2048
    """
    if not hasattr(self, "kv_cluster"):
        self.kv_cluster = ChunkKVCluster(
            chunk_length=self.config.chunk_length,
            max_budget_tokens=self.config.max_capacity_prompt,
        )
