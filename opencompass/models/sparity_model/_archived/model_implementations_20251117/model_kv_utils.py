import math
import time
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import sqrt, tensor
from transformers.cache_utils import Cache


def key_pruner_query_driven(kv_states, q_states, recent_size=128, ratio=0.3):
    _, _, seqlen, head_dim = kv_states.shape
    k = int(head_dim * ratio)
    # new efficient implementation
    queries_norm = torch.pow(q_states[..., -32:, :], 2).mean(dim=2)
    keys_norm = torch.pow(kv_states, 2).mean(dim=2)
    key = queries_norm * keys_norm
    _, indices = torch.topk(key, k, dim=-1, largest=False)
    keep_idx = indices.sort().values
    mask = torch.zeros(key.shape, dtype=torch.bool).to(kv_states.device)
    mask = mask.scatter_(-1, keep_idx, 1)
    mask_k = mask.unsqueeze(2).expand(-1, -1, seqlen - recent_size, -1)

    return kv_states[:, :, :seqlen - recent_size, :][~mask_k].reshape(
        1, -1, seqlen - recent_size,
        head_dim - k), kv_states[:, :, seqlen - recent_size:, :], ~mask


class DynamicCacheSplitHeadFlatten(Cache):
    """adapt from https://github.com/FFY0/AdaKV."""

    def __init__(self) -> None:
        # Token wise List[]  Head wise KV List[torch.Tensor]
        super().__init__()
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        self._seen_tokens = 0

    def __len__(self):
        return len(self.key_cache)

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (tuple(self.key_cache[layer_idx]),
                   tuple(self.value_cache[layer_idx]))

    def __getitem__(
            self,
            layer_idx: int) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        if layer_idx < len(self):
            return (tuple(self.key_cache[layer_idx]),
                    tuple(self.value_cache[layer_idx]))
        else:
            raise KeyError(
                f'Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}'
            )

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        if len(self.key_cache) <= layer_idx:
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            assert self.key_cache[layer_idx].dim() == 2
            bs, head, seqlen, dim = key_states.shape
            assert bs == 1 and seqlen == 1
            head_lens = cache_kwargs['head_lens']
            cu_klen = cache_kwargs['cu_klen']

            import nvtx
            copy_old_rng = nvtx.start_range('copy old')
            from tiny_api_cuda import update_flatten_view
            new_key_cache = update_flatten_view(
                self.key_cache[layer_idx].view(-1, dim),
                key_states.view(-1, dim), head_lens, cu_klen)
            new_value_cache = update_flatten_view(
                self.value_cache[layer_idx].view(-1, dim),
                value_states.view(-1, dim), head_lens, cu_klen)

            nvtx.end_range(copy_old_rng)

            self.key_cache[layer_idx] = new_key_cache
            self.value_cache[layer_idx] = new_value_cache

        return self.key_cache[layer_idx], self.value_cache[layer_idx]

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        if len(self.key_cache) <= layer_idx:
            return 0
        # TODO: return 1 to means has content for now
        return 1
        # return max(map(lambda states: states.shape[-2], self.key_cache[layer_idx]))

    def get_max_length(self) -> Optional[int]:
        return None

    def to_legacy_cache(
            self) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
        """Converts the `DynamicCache` instance into the its equivalent in the
        legacy cache format."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            legacy_cache += ((self.key_cache[layer_idx],
                              self.value_cache[layer_idx]), )
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    ) -> 'DynamicCacheEachHead':
        """Converts a cache in the legacy cache format into an equivalent
        `DynamicCache`."""
        cache = cls()
        if past_key_values is not None:
            for layer_idx in range(len(past_key_values)):
                key_states, value_states = past_key_values[layer_idx]
                cache.update(key_states, value_states, layer_idx)
        return cache


class Int8KVCache(Cache):
    """
    A cache that stores key-value tensors in int8 format to reduce memory usage.
    Supports per-token quantization for better accuracy.
    """

    def __init__(self, quantization_method: str = 'per_token') -> None:
        """
        Initialize Int8KVCache.

        Args:
            quantization_method: 'per_token' or 'per_channel'
                - per_token: Each token has its own scale (more accurate, more memory for scales)
                - per_channel: Each head has one scale (less accurate, less memory)
        """
        super().__init__()
        self.quantization_method = quantization_method

        # Storage for quantized key/value tensors (int8)
        self.key_cache_quantized: List[torch.Tensor] = []
        self.value_cache_quantized: List[torch.Tensor] = []

        # Storage for scales (fp16/bf16)
        self.key_scales: List[torch.Tensor] = []
        self.value_scales: List[torch.Tensor] = []

        # Track sequence length
        self._seen_tokens = 0

    def __len__(self):
        return len(self.key_cache_quantized)

    def __iter__(self):
        """Iterate over layers, yielding dequantized key-value pairs."""
        for layer_idx in range(len(self)):
            # Dequantize on-the-fly for iteration
            key_deq = self._dequantize(
                self.key_cache_quantized[layer_idx],
                self.key_scales[layer_idx]
            )
            value_deq = self._dequantize(
                self.value_cache_quantized[layer_idx],
                self.value_scales[layer_idx]
            )
            yield (key_deq, value_deq)

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get dequantized key-value pair for a specific layer."""
        if layer_idx < len(self):
            key_deq = self._dequantize(
                self.key_cache_quantized[layer_idx],
                self.key_scales[layer_idx]
            )
            value_deq = self._dequantize(
                self.value_cache_quantized[layer_idx],
                self.value_scales[layer_idx]
            )
            return (key_deq, value_deq)
        else:
            raise KeyError(
                f'Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}'
            )

    def _quantize(self, tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize tensor to int8."""
        if self.quantization_method == 'per_token':
            # Per-token quantization: scale shape [B, H, L, 1]
            abs_max = torch.abs(tensor).amax(dim=-1, keepdim=True)
            abs_max = torch.clamp(abs_max, min=1e-8)
            scale = abs_max / 127.0
            quantized = torch.round(tensor / scale).clamp(-127, 127).to(torch.int8)
        elif self.quantization_method == 'per_channel':
            # Per-channel quantization: scale shape [B, H, 1, 1]
            abs_max = torch.abs(tensor).amax(dim=(-2, -1), keepdim=True)
            abs_max = torch.clamp(abs_max, min=1e-8)
            scale = abs_max / 127.0
            quantized = torch.round(tensor / scale).clamp(-127, 127).to(torch.int8)
        else:
            raise ValueError(f"Unknown quantization method: {self.quantization_method}")

        return quantized, scale

    def _dequantize(self, quantized: torch.Tensor, scale: torch.Tensor,
                    target_dtype: torch.dtype = None) -> torch.Tensor:
        """Dequantize int8 tensor back to float."""
        if target_dtype is None:
            target_dtype = scale.dtype
        return quantized.to(target_dtype) * scale

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: dict = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with new key-value states.

        Args:
            key_states: New key states [B, H, L, D]
            value_states: New value states [B, H, L, D]
            layer_idx: Layer index
            cache_kwargs: Optional cache arguments (unused but kept for compatibility)

        Returns:
            Dequantized key and value states for immediate use in attention
        """
        if len(self.key_cache_quantized) <= layer_idx:
            # First update for this layer - quantize and store
            key_q, key_s = self._quantize(key_states)
            value_q, value_s = self._quantize(value_states)

            self.key_cache_quantized.append(key_q)
            self.value_cache_quantized.append(value_q)
            self.key_scales.append(key_s)
            self.value_scales.append(value_s)

            # Update seen tokens (assuming first update is prefill)
            if layer_idx == 0:
                self._seen_tokens = key_states.shape[2]

            # Return dequantized for immediate attention computation
            return key_states, value_states
        else:
            # Incremental update (decode phase)
            # Quantize new tokens
            key_q_new, key_s_new = self._quantize(key_states)
            value_q_new, value_s_new = self._quantize(value_states)

            # Concatenate with existing cache
            key_q_full = torch.cat([self.key_cache_quantized[layer_idx], key_q_new], dim=2)
            value_q_full = torch.cat([self.value_cache_quantized[layer_idx], value_q_new], dim=2)
            key_s_full = torch.cat([self.key_scales[layer_idx], key_s_new], dim=2)
            value_s_full = torch.cat([self.value_scales[layer_idx], value_s_new], dim=2)

            # Update cache
            self.key_cache_quantized[layer_idx] = key_q_full
            self.value_cache_quantized[layer_idx] = value_q_full
            self.key_scales[layer_idx] = key_s_full
            self.value_scales[layer_idx] = value_s_full

            # Update seen tokens
            if layer_idx == 0:
                self._seen_tokens += key_states.shape[2]

            # Dequantize full sequence for attention
            target_dtype = key_states.dtype
            key_full = self._dequantize(key_q_full, key_s_full, target_dtype)
            value_full = self._dequantize(value_q_full, value_s_full, target_dtype)

            return key_full, value_full

    def get_seq_length(self, layer_idx: int = 0) -> int:
        """Return the sequence length of the cached states."""
        if len(self.key_cache_quantized) <= layer_idx:
            return 0
        return self.key_cache_quantized[layer_idx].shape[2]

    def get_max_length(self) -> None:
        """This cache type has no maximum length."""
        return None

    def get_usable_length(self, new_seq_length: int, layer_idx: int = 0) -> int:
        """Return the usable length of the cache."""
        return self.get_seq_length(layer_idx)

    def to_legacy_cache(self) -> Tuple[Tuple[torch.Tensor, torch.Tensor], ...]:
        """Convert to legacy cache format (dequantized)."""
        legacy_cache = ()
        for layer_idx in range(len(self)):
            key_deq = self._dequantize(
                self.key_cache_quantized[layer_idx],
                self.key_scales[layer_idx]
            )
            value_deq = self._dequantize(
                self.value_cache_quantized[layer_idx],
                self.value_scales[layer_idx]
            )
            legacy_cache += ((key_deq, value_deq),)
        return legacy_cache

    @classmethod
    def from_legacy_cache(
        cls,
        past_key_values: Tuple[Tuple[torch.FloatTensor, torch.FloatTensor], ...],
        quantization_method: str = 'per_token'
    ) -> 'Int8KVCache':
        """Create Int8KVCache from legacy cache format."""
        cache = cls(quantization_method=quantization_method)
        if past_key_values is not None:
            for layer_idx, (key_states, value_states) in enumerate(past_key_values):
                cache.update(key_states, value_states, layer_idx)
        return cache

    def get_memory_usage(self) -> dict:
        """Calculate and return memory usage statistics."""
        total_quantized = 0
        total_scales = 0

        for layer_idx in range(len(self)):
            # Int8 data
            total_quantized += self.key_cache_quantized[layer_idx].numel() * 1  # 1 byte per int8
            total_quantized += self.value_cache_quantized[layer_idx].numel() * 1

            # Scales (fp16 or bf16)
            scale_bytes = self.key_scales[layer_idx].element_size()
            total_scales += self.key_scales[layer_idx].numel() * scale_bytes
            total_scales += self.value_scales[layer_idx].numel() * scale_bytes

        return {
            'quantized_data_bytes': total_quantized,
            'scales_bytes': total_scales,
            'total_bytes': total_quantized + total_scales,
            'num_layers': len(self),
            'quantization_method': self.quantization_method
        }


# perform qk calculation and get indices
# this version will not update in inference mode


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


def merge_kv(key_states, value_states, indices, window_size, merge):
    # merge methods in LOOK-M

    bsz, num_heads, k_len, head_dim = key_states.shape

    # kv-selected
    selected_keys = key_states.gather(
        dim=2, index=indices)  # [bsz, num_heads, topk_len, head_dim]
    selected_values = value_states.gather(
        dim=2, index=indices)  # [bsz, num_heads, topk_len, head_dim]

    # kv-drop
    all_indices = torch.arange(
        k_len, device=key_states.device).unsqueeze(0).unsqueeze(0).expand(
            bsz, num_heads, k_len)
    all_indices_flattened = all_indices.flatten(
    )  # [bsz * num_heads * (k_len-window_size)]
    selected_indices_flattened = indices.flatten(
    )  # [bsz * num_heads * topk_len]
    is_selected = torch.isin(all_indices_flattened, selected_indices_flattened)
    drop_indices_flattened = all_indices_flattened[~is_selected]
    drop_len = drop_indices_flattened.shape[0] // (all_indices.shape[0] *
                                                   all_indices.shape[1])
    drop_indices = drop_indices_flattened.reshape(
        all_indices.shape[0], all_indices.shape[1],
        drop_len)  # [bsz * num_heads * (k_len-window_size-topk_len)]
    drop_indices = drop_indices.unsqueeze(-1).expand(
        -1, -1, -1,
        head_dim)  # [bsz, num_heads, (k_len-window_size-topk_len), head_dim]
    drop_keys = key_states.gather(dim=2, index=drop_indices)
    drop_values = value_states.gather(dim=2, index=drop_indices)

    # kv-recent
    recent_keys = key_states[:, :, -window_size:, :]

    ##### apply merge #####
    # prepare for merge
    k_hh_pruned = drop_keys  # [bsz, num_heads, k_len-topk_len-window_size, head_dim]
    k_hh_recent = torch.cat(
        [recent_keys, selected_keys],
        dim=2)  # [bsz, num_heads, topk_len+window_size, head_dim]
    v_hh_pruned = drop_values  # [bsz, num_heads, k_len-topk_len-window_size, head_dim]
    v_hh_recent = torch.cat(
        [selected_values, value_states[:, :, -window_size:, :]],
        dim=2)  # [bsz, num_heads, topk_len+window_size, head_dim]
    # similarity matrix
    similarity = (k_hh_pruned / torch.norm(
        k_hh_pruned, dim=-1).unsqueeze(-1).repeat(1, 1, 1, 128)) @ (
            (k_hh_recent /
             (torch.norm(k_hh_recent, dim=-1).unsqueeze(-1).repeat(
                 1, 1, 1, 128))).transpose(-1, -2))  # cosin
    max_values, max_indices = similarity.max(dim=-1)

    # pivot merge
    if merge == 'pivot':
        print('Pivot merge')
        merged_indices = max_indices.unsqueeze(-1).repeat(1, 1, 1, 128)
        k_hh_selected = torch.gather(input=k_hh_recent,
                                     dim=2,
                                     index=merged_indices)
        k_hh_merged = (k_hh_pruned + k_hh_selected) / 2
        k_hh_recent = torch.scatter_reduce(
            input=k_hh_recent,
            dim=2,
            index=merged_indices,
            src=k_hh_merged,
            reduce='mean',
            include_self=True
        )  # include_self=True seems decrease the performance
        v_hh_selected = torch.gather(input=v_hh_recent,
                                     dim=2,
                                     index=merged_indices)
        v_hh_merged = (v_hh_pruned + v_hh_selected) / 2
        v_hh_recent = torch.scatter_reduce(input=v_hh_recent,
                                           dim=2,
                                           index=merged_indices,
                                           src=v_hh_merged,
                                           reduce='mean',
                                           include_self=True)
    else:
        raise ValueError('Merge method not supported')

    # TODO: other merge strategies
    # average merge
    # weight merge

    return k_hh_recent, v_hh_recent


class PyramidKVCluster():

    def __init__(self,
                 num_hidden_layers=32,
                 window_size=64,
                 max_capacity_prompt=256 + 64,
                 kernel_size=5,
                 pooling='avgpool',
                 beta=20,
                 num_layers=80,
                 layer_idx=None,
                 merge=None):

        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        self.steps = -1
        self.beta = beta

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

        # TODO
        # window_sizes = 32
        min_num = (self.max_capacity_prompt - self.window_size) // self.beta
        max_num = (self.max_capacity_prompt - self.window_size) * 2 - min_num

        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (self.max_capacity_prompt -
                       self.window_size) * 2 - max_num

        steps = (max_num - min_num) // (self.num_hidden_layers - 1)
        max_capacity_prompt = max_num - self.layer_idx * steps
        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        elif q_len < (self.max_capacity_prompt - self.window_size) * 2:
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
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

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
            indices = attn_cache.topk(max_capacity_prompt, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

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


class PyramidKVCluster_gqa():
    """PyramidKV with GQA support - properly handles grouped query attention."""

    def __init__(self,
                 num_hidden_layers=32,
                 window_size=64,
                 max_capacity_prompt=256 + 64,
                 kernel_size=5,
                 pooling='avgpool',
                 beta=20,
                 num_layers=80,
                 layer_idx=None,
                 merge=None):

        self.layer_idx = layer_idx
        self.num_hidden_layers = num_hidden_layers

        self.steps = -1
        self.beta = beta

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

        # Save original GQA format
        key_states_gqa = key_states  # [bsz, num_key_value_heads, seq_len, head_dim]
        value_states_gqa = value_states
        bsz_k, num_key_value_heads, q_len_k, head_dim_k = key_states_gqa.shape

        # Calculate dynamic max_capacity_prompt based on layer_idx
        min_num = (self.max_capacity_prompt - self.window_size) // self.beta
        max_num = (self.max_capacity_prompt - self.window_size) * 2 - min_num

        if max_num >= q_len - self.window_size:
            max_num = q_len - self.window_size
            min_num = (self.max_capacity_prompt -
                       self.window_size) * 2 - max_num

        steps = (max_num - min_num) // (self.num_hidden_layers - 1)
        max_capacity_prompt = max_num - self.layer_idx * steps

        if q_len < self.max_capacity_prompt:
            return key_states_gqa, value_states_gqa
        elif q_len < (self.max_capacity_prompt - self.window_size) * 2:
            # Expand key_states for attention computation
            key_states_expanded = repeat_kv(key_states_gqa, num_key_value_groups)

            attn_weights = torch.matmul(
                query_states[..., -self.window_size:, :],
                key_states_expanded.transpose(2, 3)) / math.sqrt(head_dim)
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

            # Aggregate attention weights back to GQA format
            # [bsz, num_heads, seq_len] -> [bsz, num_key_value_heads, seq_len]
            attn_weights_sum = attn_weights_sum.view(bsz, num_key_value_heads, num_key_value_groups, -1)
            attn_weights_sum = attn_weights_sum.mean(dim=2)  # Average over query head groups

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

            # Select top-k based on GQA format
            indices = attn_cache.topk(self.max_capacity_prompt -
                                      self.window_size,
                                      dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            if self.merge is not None:
                key_states_gqa, value_states_gqa = merge_kv(key_states_gqa, value_states_gqa,
                                                    indices, self.window_size,
                                                    self.merge)
                return key_states_gqa, value_states_gqa

            # Apply compression on original GQA format
            k_past_compress = key_states_gqa[:, :, :-self.window_size, :].gather(
                dim=2, index=indices)
            v_past_compress = value_states_gqa[:, :, :-self.window_size, :].gather(
                dim=2, index=indices)
            k_cur = key_states_gqa[:, :, -self.window_size:, :]
            v_cur = value_states_gqa[:, :, -self.window_size:, :]
            key_states_gqa = torch.cat([k_past_compress, k_cur], dim=2)
            value_states_gqa = torch.cat([v_past_compress, v_cur], dim=2)

            # Return GQA format: [bsz, num_key_value_heads, compressed_seq_len, head_dim]
            return key_states_gqa, value_states_gqa
        else:
            # Expand key_states for attention computation
            key_states_expanded = repeat_kv(key_states_gqa, num_key_value_groups)

            attn_weights = torch.matmul(
                query_states[..., -self.window_size:, :],
                key_states_expanded.transpose(2, 3)) / math.sqrt(head_dim)
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

            # Aggregate attention weights back to GQA format
            # [bsz, num_heads, seq_len] -> [bsz, num_key_value_heads, seq_len]
            attn_weights_sum = attn_weights_sum.view(bsz, num_key_value_heads, num_key_value_groups, -1)
            attn_weights_sum = attn_weights_sum.mean(dim=2)  # Average over query head groups

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

            # Select top-k based on GQA format
            indices = attn_cache.topk(max_capacity_prompt, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

            if self.merge is not None:
                key_states_gqa, value_states_gqa = merge_kv(key_states_gqa, value_states_gqa,
                                                    indices, self.window_size,
                                                    self.merge)
                return key_states_gqa, value_states_gqa

            # Apply compression on original GQA format
            k_past_compress = key_states_gqa[:, :, :-self.window_size, :].gather(
                dim=2, index=indices)
            v_past_compress = value_states_gqa[:, :, :-self.window_size, :].gather(
                dim=2, index=indices)
            k_cur = key_states_gqa[:, :, -self.window_size:, :]
            v_cur = value_states_gqa[:, :, -self.window_size:, :]
            key_states_gqa = torch.cat([k_past_compress, k_cur], dim=2)
            value_states_gqa = torch.cat([v_past_compress, v_cur], dim=2)

            # Return GQA format: [bsz, num_key_value_heads, compressed_seq_len, head_dim]
            return key_states_gqa, value_states_gqa


class SnapKVCluster():

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
        # print(f"🔍 recent_size = {recent_size}, ratio = {ratio}")
        # print(f"🔍 window_size = {window_size}, max_capacity_prompt = {max_capacity_prompt}, kernel_size = {kernel_size}, pooling = {pooling}, merge = {merge}")

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
                  num_key_value_groups):
            # check if prefix phase
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
                indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

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
        

    def update_think(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        
        if q_len < self.max_capacity_prompt:
            kv_pruned, kv_recent, mask = key_pruner_query_driven(key_states, query_states, self.recent_size, self.ratio)
            return kv_pruned, kv_recent, mask, value_states
        else:
            attn_weights = torch.matmul(query_states[..., -self.window_size:, :], key_states.transpose(2, 3)) / math.sqrt(head_dim)
            mask = torch.full((self.window_size, self.window_size), torch.finfo(attn_weights.dtype).min, device=attn_weights.device)
            mask_cond = torch.arange(mask.size(-1), device=attn_weights.device)
            mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
            mask = mask.to(attn_weights.device)
            attention_mask = mask[None, None, :, :]

            attn_weights[:, :, -self.window_size:, -self.window_size:] += attention_mask

            attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
            attn_weights_sum = attn_weights[:, :, -self.window_size:, : -self.window_size].sum(dim = -2)
            if self.pooling == 'avgpool':
                attn_cache = F.avg_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            elif self.pooling == 'maxpool':
                attn_cache = F.max_pool1d(attn_weights_sum, kernel_size = self.kernel_size, padding=self.kernel_size//2, stride=1)
            else:
                raise ValueError('Pooling method not supported')
            indices = attn_cache.topk(self.max_capacity_prompt - self.window_size, dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
            k_past_compress = key_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            v_past_compress = value_states[:, :, :-self.window_size, :].gather(dim = 2, index = indices)
            k_cur = key_states[:, :, -self.window_size:, :]
            v_cur = value_states[:, :, -self.window_size:, :]
            key_states = torch.cat([k_past_compress, k_cur], dim = 2)
            value_states = torch.cat([v_past_compress, v_cur], dim = 2)
            kv_pruned, kv_recent, mask = key_pruner_query_driven(key_states, query_states, self.recent_size, self.ratio)
            return kv_pruned, kv_recent, mask, value_states



class SnapKVCluster_gqa():

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
        # print(f"🔍 recent_size = {recent_size}, ratio = {ratio}")
        # print(f"🔍 window_size = {window_size}, max_capacity_prompt = {max_capacity_prompt}, kernel_size = {kernel_size}, pooling = {pooling}, merge = {merge}")

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
                  num_key_value_groups):
            # check if prefix phase
            print("===========================update static sparity (max_capacity_prompt)===========================")
            # 保存原始的 GQA 格式的 key_states 和 value_states
            key_states_gqa = key_states  # [bsz, num_key_value_heads, seq_len, head_dim]
            value_states_gqa = value_states

            assert key_states.shape[-2] == query_states.shape[-2]
            bsz, num_heads, q_len, head_dim = query_states.shape
            bsz_k, num_key_value_heads, q_len_k, head_dim_k = key_states_gqa.shape

            if q_len < self.max_capacity_prompt:
                return key_states_gqa, value_states_gqa
            else:
                # 临时扩展 key_states 用于计算 attention weights
                key_states_expanded = repeat_kv(key_states_gqa, num_key_value_groups)

                attn_weights = torch.matmul(
                    query_states[..., -self.window_size:, :],
                    key_states_expanded.transpose(2, 3)) / math.sqrt(head_dim)
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

                # 将 attention weights 从 MHA 格式聚合回 GQA 格式
                # [bsz, num_heads, seq_len] -> [bsz, num_key_value_heads, seq_len]
                attn_weights_sum = attn_weights_sum.view(bsz, num_key_value_heads, num_key_value_groups, -1)
                attn_weights_sum = attn_weights_sum.mean(dim=2)  # 对每组的多个 query head 求平均

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

                # 基于 GQA 格式选择 top-k indices
                indices = attn_cache.topk(self.max_capacity_prompt -
                                        self.window_size,
                                        dim=-1).indices
                indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

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
        


class ALLKVCluster():
    """
    Prefill: ALLKV
    decoding: None, slide, adaptive, discontinuous
    """

    allkv_max_capacity_prompt = 0
    current_decoding_step = 0
    jump_step = 0
    jump_layer = 0

    def __init__(self, decoding_metric = 'None', delta=15, num_hidden_layers = 32, decoding_window_size = 1024, decoding_recent_size = 256):
        ##### Add decoding window #####
        self.decoding_metric = decoding_metric
        self.decoding_window_size = decoding_window_size
        self.decoding_recent_size = decoding_recent_size
        print(f"🔍 decoding_window_size = {self.decoding_window_size}, decoding_recent_size = {self.decoding_recent_size}")
        assert self.decoding_window_size - self.decoding_recent_size > 0
        ##### Add delta #####
        self.delta = delta
        ##### Add num_hidden_layers #####
        self.num_hidden_layers = num_hidden_layers            

    def reset(self, decoding_metric = 'None', delta=15, num_hidden_layers = 32, decoding_window_size = 1024, decoding_recent_size = 256):
        ##### Add decoding window #####
        self.decoding_metric = decoding_metric
        self.decoding_window_size = decoding_window_size
        self.decoding_recent_size = decoding_recent_size
        assert self.decoding_window_size - self.decoding_recent_size > 0
        ##### Add delta #####
        self.delta = delta
        ##### Add num_hidden_layers #####
        self.num_hidden_layers = num_hidden_layers   
    
    def update_kv(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape
        # print(f"ALLKV: prefill not compressed")

        ##### Record max_capacity_prompt #####
        ALLKVCluster.max_capacity_prompt = key_states.shape[-2]

        # reset decoding step
        ALLKVCluster.current_decoding_step = 0
        ALLKVCluster.jump_step = 0

        return key_states, value_states
    
    def update_kv_in_decoding(self, key_states, query_states, value_states, attention_mask, num_key_value_groups):
        
        # check if decoding phase
        assert query_states.shape[-2]==1
        bsz, num_heads, k_len, head_dim = key_states.shape
        q_len = query_states.shape[-2]

        ##### Set decoding compress strategy #####
        if self.decoding_metric == 'None':
            return key_states, value_states
        elif self.decoding_metric == 'slide':
            decoding_window_size = self.decoding_window_size
            window_size = self.decoding_recent_size
            
            if k_len < ALLKVCluster.max_capacity_prompt + decoding_window_size:
                print(f"k_len: {k_len}, ALLKVCluster.max_capacity_prompt: {ALLKVCluster.max_capacity_prompt}, decoding_window_size: {decoding_window_size}")
                return key_states, value_states
            else:
                print("===========================update ALLKV in decoding (slide)==========================")
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, :, : -window_size].sum(dim = -2)
                attn_cache = attn_weights_sum
                
                # prefill cache
                prefill_indices = torch.tensor(range(ALLKVCluster.max_capacity_prompt), dtype=torch.int64).to(key_states.device)
                prefill_indices = prefill_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

                # decoding cache
                decoding_indices = attn_cache[:, :, ALLKVCluster.max_capacity_prompt:].topk(decoding_window_size - window_size, dim=-1).indices
                decoding_indices += ALLKVCluster.max_capacity_prompt # add offset
                decoding_indices = decoding_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                
                indices = torch.cat((prefill_indices, decoding_indices), dim=2)

                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
        elif self.decoding_metric == 'adaptive':
            window_size = self.decoding_recent_size
            decoding_window_size = window_size + ALLKVCluster.current_decoding_step//(self.delta*self.num_hidden_layers) # TODO: change the step size
            ALLKVCluster.current_decoding_step += 1
            
            if k_len < ALLKVCluster.max_capacity_prompt + decoding_window_size:
                return key_states, value_states
            else:
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, :, : -window_size].sum(dim = -2)
                attn_cache = attn_weights_sum
                
                # prefill cache
                prefill_indices = torch.tensor(range(ALLKVCluster.max_capacity_prompt), dtype=torch.int64).to(key_states.device)
                prefill_indices = prefill_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

                # decoding cache
                decoding_indices = attn_cache[:, :, ALLKVCluster.max_capacity_prompt:].topk(decoding_window_size - window_size, dim=-1).indices
                decoding_indices += ALLKVCluster.max_capacity_prompt # add offset
                decoding_indices = decoding_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                
                # all cache
                indices = torch.cat((prefill_indices, decoding_indices), dim=2)

                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
        elif self.decoding_metric == 'discontinuous':
            window_size = self.decoding_recent_size
            decoding_window_size = window_size + ALLKVCluster.current_decoding_step//(self.delta*self.num_hidden_layers) # TODO: change the step size
            ALLKVCluster.current_decoding_step += 1
            
            if k_len < ALLKVCluster.max_capacity_prompt + decoding_window_size:
                return key_states, value_states
            elif ALLKVCluster.jump_step < self.delta*self.num_hidden_layers:
                ALLKVCluster.jump_step += 1
                return key_states, value_states
            else:
                ALLKVCluster.jump_layer += 1
                if ALLKVCluster.jump_layer == self.num_hidden_layers:
                    ALLKVCluster.jump_step = 0
                    ALLKVCluster.jump_layer = 0
                
                attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(head_dim)
                attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
                attn_weights_sum = attn_weights[:, :, :, : -window_size].sum(dim = -2)
                attn_cache = attn_weights_sum
                
                # prefill cache
                prefill_indices = torch.tensor(range(ALLKVCluster.max_capacity_prompt), dtype=torch.int64).to(key_states.device)
                prefill_indices = prefill_indices.unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(bsz, num_heads, 1, head_dim)

                # decoding cache
                decoding_indices = attn_cache[:, :, ALLKVCluster.max_capacity_prompt:].topk(decoding_window_size - window_size, dim=-1).indices
                decoding_indices += ALLKVCluster.max_capacity_prompt # add offset
                decoding_indices = decoding_indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)
                
                # all cache
                indices = torch.cat((prefill_indices, decoding_indices), dim=2)

                k_past_compress = key_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                v_past_compress = value_states[:, :, :-window_size, :].gather(dim = 2, index = indices)
                k_cur = key_states[:, :, -window_size:, :]
                v_cur = value_states[:, :, -window_size:, :]
                key_states = torch.cat([k_past_compress, k_cur], dim = 2)
                value_states = torch.cat([v_past_compress, v_cur], dim = 2)
                return key_states, value_states
        else:
            raise ValueError('Decoding metric not supported')




class L2NormCluster():

    def __init__(self,
                 max_capacity_prompt: int = 256 + 64,
                 layer_idx: int = 0,
                 skip_layers: List[int] = []):
        self.max_capacity_prompt = max_capacity_prompt
        self.layer_idx = layer_idx
        self.skip_layers = skip_layers

    def reset(self,
              max_capacity_prompt: int = 256 + 64,
              layer_idx: int = 0,
              skip_layers: List[int] = []):
        self.max_capacity_prompt = max_capacity_prompt
        self.layer_idx = layer_idx
        self.skip_layers = skip_layers

    def update_kv(self, key_states, query_states, value_states, attention_mask,
                  num_key_value_groups):

        # check if prefix phase
        assert key_states.shape[-2] == query_states.shape[-2]
        bsz, num_heads, q_len, head_dim = query_states.shape

        print(f'L2Norm max_capacity_prompt {self.max_capacity_prompt}')

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        elif self.layer_idx in self.skip_layers:
            return key_states, value_states
        else:
            head_dim = key_states.size(-1)
            token_norms = torch.norm(key_states, p=2, dim=-1)
            sorted_indices = token_norms.squeeze(-1).argsort(dim=-1)
            sorted_indices_expanded = sorted_indices.unsqueeze(-1).expand(
                -1, -1, -1, head_dim)

            sorted_key_states = key_states.gather(
                dim=2, index=sorted_indices_expanded)
            sorted_value_states = value_states.gather(
                dim=2, index=sorted_indices_expanded)

            key_states = sorted_key_states[:, :, :self.max_capacity_prompt, :]
            value_states = sorted_value_states[:, :, :self.
                                               max_capacity_prompt, :]

            return key_states, value_states


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


class H2OKVCluster():

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

        # print(f"q_len :  {q_len}")
        # print(f"H2O max_capacity_prompt {self.max_capacity_prompt}")

        if q_len < self.max_capacity_prompt:
            return key_states, value_states
        else:
            attn_weights = torch.matmul(query_states, key_states.transpose(
                2, 3)) / math.sqrt(head_dim)
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
            indices = attn_cache.topk(self.max_capacity_prompt -
                                      self.window_size,
                                      dim=-1).indices
            indices = indices.unsqueeze(-1).expand(-1, -1, -1, head_dim)

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
            # print("key.shape",key_states.shape)
            # print("value_states.shape",value_states.shape)
            return key_states, value_states


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


def init_pyramidkv(self, num_hidden_layers):
    if not hasattr(self, 'kv_cluster'):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 16
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 64
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None

    self.kv_cluster = PyramidKVCluster(
        num_hidden_layers=num_hidden_layers,
        layer_idx=self.layer_idx,
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        kernel_size=self.config.kernel_size,
        pooling=self.config.pooling,
        merge=self.config.merge,
    )


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
    
    print(f"SnapKVCluster self.config.window_size,: {self.config.window_size} self.config.max_capacity_prompt {self.config.max_capacity_prompt}")
    self.kv_cluster = SnapKVCluster(
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        ratio = 0.4,
        kernel_size=7,
        pooling=self.config.pooling,
        merge=self.config.merge,
    )

def init_snapkv_gqa(self):
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
    
    self.kv_cluster = SnapKVCluster_gqa(
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        ratio = 0.4,
        kernel_size=7,
        pooling=self.config.pooling,
        merge=self.config.merge,
    )


def init_ALLKV(self, num_hidden_layers):
    if not hasattr(self, "kv_cluster"):
        if not hasattr(self.config, 'decoding_metric'):
            self.config.decoding_metric = 'None'
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 16
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 64
    
    
    self.kv_cluster = ALLKVCluster(
        num_hidden_layers = num_hidden_layers,
        delta=self.config.delta,
        decoding_metric=self.config.decoding_metric,
        decoding_window_size=self.config.decoding_window_size,
        decoding_recent_size=self.config.decoding_recent_size,
        )


def init_think(self):
    if not hasattr(self, 'kv_cluster'):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None
        if not hasattr(self.config, 'recent_size'):
            self.config.recent_size = 32
        if not hasattr(self.config, 'ratio'):
            self.config.ratio = 0.4

    self.kv_cluster = SnapKVCluster(
        window_size=self.config.window_size,
        max_capacity_prompt=self.config.max_capacity_prompt,
        kernel_size=self.config.kernel_size,
        pooling=self.config.pooling,
        merge=self.config.merge,
        recent_size=self.config.recent_size,
        ratio=self.config.ratio)


def init_l2norm(self):

    if not hasattr(self, 'kv_cluster'):
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 4096
        if not hasattr(self.config, 'layer_idx'):
            self.config.layer_idx = 0
        if not hasattr(self.config, 'skip_layers'):
            self.config.skip_layers = [0, 1]

    self.kv_cluster = L2NormCluster(
        max_capacity_prompt=self.config.max_capacity_prompt,
        layer_idx=self.layer_idx,
        skip_layers=self.config.skip_layers)


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


def init_H2O(self):
    if not hasattr(self, 'kv_cluster'):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'avgpool'
        if not hasattr(self.config, 'merge'):
            self.config.merge = None

        self.kv_cluster = H2OKVCluster(
            window_size=self.config.window_size,
            max_capacity_prompt=self.config.max_capacity_prompt,
            kernel_size=self.config.kernel_size,
            pooling=self.config.pooling,
            merge=self.config.merge,
        )


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


def init_adakv(self):
    if not hasattr(self, 'kv_cluster'):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 16
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 64
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'maxpool'
        if not hasattr(self.config, 'floor_ratio'):
            self.config.floor_ratio = 0.2
        if not hasattr(self.config, 'normalize'):
            self.config.normalize = True
    # max_capacity_prompt --> base_capacity
    # init only once
    if not hasattr(self, 'kv_cluster'):
        self.kv_cluster = AdaKVCluster(
            num_hidden_layers=self.config.num_hidden_layers,
            layer_idx=self.layer_idx,
            window_size=self.config.window_size,
            max_capacity_prompt=self.config.max_capacity_prompt,
            kernel_size=self.config.kernel_size,
            pooling=self.config.pooling,
            floor=0.2,
            normalize=self.config.normalize)


def init_headkv(self):
    if not hasattr(self, 'kv_cluster'):
        if not hasattr(self.config, 'window_size'):
            self.config.window_size = 32
        if not hasattr(self.config, 'max_capacity_prompt'):
            self.config.max_capacity_prompt = 2048
        if not hasattr(self.config, 'kernel_size'):
            self.config.kernel_size = 5
        if not hasattr(self.config, 'pooling'):
            self.config.pooling = 'maxpool'
        if not hasattr(self.config, 'head_capacity'):
            raise ValueError('Must have head_capacity')
    # max_capacity_prompt --> base_capacity
    # init only once
    if not hasattr(self, 'kv_cluster'):
        self.kv_cluster = HeadKVCluster(
            num_hidden_layers=self.config.num_hidden_layers,
            layer_idx=self.layer_idx,
            window_size=self.config.window_size,
            max_capacity_prompt=self.config.max_capacity_prompt,
            kernel_size=self.config.kernel_size,
            pooling=self.config.pooling,
            head_capacity=self.config.head_capacity)


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
