import os
import csv
import json
from datetime import datetime
from typing import List, Optional, Tuple

import torch
from transformers.cache_utils import Cache


def estimate_kv_memory(past_key_value, method="unknown", max_capacity_prompt=None, csv_file=None) -> float:
    """Estimate and log KV cache memory usage."""
    total_kv_memory = 0

    key_cache = past_key_value.key_cache
    value_cache = past_key_value.value_cache

    print(f"KV dtype: {key_cache[0].dtype}")
    print(f"key_cache层数: {len(key_cache)}")

    for i, (k, v) in enumerate(zip(key_cache, value_cache)):
        if k is not None and v is not None:
            key_shape = k.shape
            value_shape = v.shape
            key_numel = k.numel()
            val_numel = v.numel()
            key_mem = key_numel * k.element_size()
            val_mem = val_numel * v.element_size()
            layer_mem = (key_mem + val_mem) / (1024 ** 2)
            if i == 20:
                print(f"[Layer {i}] key_cache shape: {key_shape}, value_cache shape: {value_shape}")
                print(f"  ↳ key.numel(): {key_numel}, value.numel(): {val_numel}")
                print(f"  ↳ key mem: {key_mem / (1024 ** 2):.2f} MB, value mem: {val_mem / (1024 ** 2):.2f} MB")

            total_kv_memory += key_mem + val_mem

    kv_mem_MB = total_kv_memory / (1024 ** 2)
    print(f"[KV Cache] 当前past_key_value占用内存: {kv_mem_MB:.2f} MB")

    # ===== ✅ 写入 CSV（自动创建，文件名包含 method 和 max_capacity_prompt） =====
    if csv_file is None:
        # 创建保存文件夹
        save_dir = "kv_memory_logs"
        os.makedirs(save_dir, exist_ok=True)

        # 构建文件名
        if max_capacity_prompt is not None:
            filename = f"{method}_{max_capacity_prompt}_kv_mem_log.csv"
        else:
            filename = f"{method}_kv_mem_log.csv"

        csv_file = os.path.join(save_dir, filename)

    print(f"[KV Memory] 保存到文件: {csv_file}")

    header = ["timestamp", "kv_memory_MB"]
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        f"{kv_mem_MB:.2f}"
    ]
    need_header = not os.path.exists(csv_file)

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(header)
        writer.writerow(row)
    # ==================================

    return kv_mem_MB


def estimate_kivi_memory(past_key_value, k_bits, v_bits, group_size, residual_length, method="full_kivi") -> float:
    """
    Estimate and log KIVI cache memory usage from past_key_value tuple.

    Args:
        past_key_value: Tuple (key_quant_trans, key_full, key_scale_trans, key_mn_trans,
                                value_quant, value_full, value_scale, value_mn, kv_seq_len)
        k_bits: Key quantization bits
        v_bits: Value quantization bits
        group_size: Quantization group size
        residual_length: Residual length
        method: Method name for logging

    Returns:
        Total memory in MB
    """
    total_memory = 0

    # Extract from tuple
    key_states_quant_trans = past_key_value[0]
    key_states_full = past_key_value[1]
    key_scale_trans = past_key_value[2]
    key_mn_trans = past_key_value[3]
    value_states_quant = past_key_value[4]
    value_states_full = past_key_value[5]
    value_scale = past_key_value[6]
    value_mn = past_key_value[7]

    print(f"\n[KIVI Memory] Estimating cache memory")
    print(f"  Params: k_bits={k_bits}, v_bits={v_bits}, group_size={group_size}, residual={residual_length}")

    # Calculate memory for each component
    if key_states_quant_trans is not None:
        key_quant_mem = key_states_quant_trans.numel() * key_states_quant_trans.element_size()
        total_memory += key_quant_mem
        print(f"  key_quant: {key_states_quant_trans.shape}, {key_quant_mem / (1024**2):.2f} MB")

    if key_scale_trans is not None:
        key_scale_mem = key_scale_trans.numel() * key_scale_trans.element_size()
        total_memory += key_scale_mem

    if key_mn_trans is not None:
        key_mn_mem = key_mn_trans.numel() * key_mn_trans.element_size()
        total_memory += key_mn_mem

    if key_states_full is not None:
        key_full_mem = key_states_full.numel() * key_states_full.element_size()
        total_memory += key_full_mem
        print(f"  key_full: {key_states_full.shape}, {key_full_mem / (1024**2):.2f} MB")

    if value_states_quant is not None:
        value_quant_mem = value_states_quant.numel() * value_states_quant.element_size()
        total_memory += value_quant_mem
        print(f"  value_quant: {value_states_quant.shape}, {value_quant_mem / (1024**2):.2f} MB")

    if value_scale is not None:
        value_scale_mem = value_scale.numel() * value_scale.element_size()
        total_memory += value_scale_mem

    if value_mn is not None:
        value_mn_mem = value_mn.numel() * value_mn.element_size()
        total_memory += value_mn_mem

    if value_states_full is not None:
        value_full_mem = value_states_full.numel() * value_states_full.element_size()
        total_memory += value_full_mem
        print(f"  value_full: {value_states_full.shape}, {value_full_mem / (1024**2):.2f} MB")

    total_memory_mb = total_memory / (1024 ** 2)
    print(f"\n[KIVI Cache] Total memory usage: {total_memory_mb:.2f} MB")

    # Log to CSV (same directory as estimate_kv_memory)
    save_dir = "kv_memory_logs"
    os.makedirs(save_dir, exist_ok=True)
    filename = f"{method}_kbits{k_bits}_vbits{v_bits}_group{group_size}_residual{residual_length}_kv_mem_log.csv"
    csv_file = os.path.join(save_dir, filename)

    print(f"[KIVI Memory] Saving to file: {csv_file}")

    header = ["timestamp", "kv_memory_MB", "k_bits", "v_bits", "group_size", "residual_length"]
    row = [
        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        f"{total_memory_mb:.2f}",
        str(k_bits),
        str(v_bits),
        str(group_size),
        str(residual_length)
    ]
    need_header = not os.path.exists(csv_file)

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.writer(f)
        if need_header:
            writer.writerow(header)
        writer.writerow(row)

    return total_memory_mb


def save_topk_indices(indices, layer_idx, attn_weights_sum=None, method="snapkv", save_dir="kv_cache_logs", sample_id=0, capacity=None, run_timestamp=None, num_key_value_groups=None, dataset_name=None):
    """
    Save topk indices selected by KV compression algorithms.

    Args:
        indices: torch.Tensor, shape [bsz, num_heads, topk, head_dim] or [bsz, num_heads, topk]
        layer_idx: int, current layer index
        attn_weights_sum: Optional torch.Tensor, attention weight scores before topk
        method: str, compression method name
        save_dir: str, base directory to save logs
        sample_id: int, sample sequence number (for multiple samples)
        capacity: int, max_capacity_prompt value (for folder organization)
        run_timestamp: str, timestamp for this run (for folder organization)
        num_key_value_groups: int, number of key-value groups (for GQA visualization)
        dataset_name: str, dataset name (for folder organization)
    """
    # Build folder structure: save_dir/topk_indices_logs/timestamp/dataset_name/cap_xxx/
    save_dir = os.path.join(save_dir, "topk_indices_logs")

    if run_timestamp and capacity is not None and dataset_name:
        # Full organization: timestamp/dataset/capacity
        final_dir = os.path.join(save_dir, run_timestamp, dataset_name, f"cap_{capacity}")
    elif run_timestamp and capacity is not None:
        # Use provided timestamp and capacity (backward compatible)
        final_dir = os.path.join(save_dir, run_timestamp, f"cap_{capacity}")
    elif capacity is not None:
        # Only capacity provided, use current timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_dir = os.path.join(save_dir, timestamp, f"cap_{capacity}")
    else:
        # No organization, use base dir directly (backward compatible)
        final_dir = save_dir

    os.makedirs(final_dir, exist_ok=True)

    # Extract indices (remove head_dim dimension if exists)
    if indices.dim() == 4:
        indices_2d = indices[:, :, :, 0]  # [bsz, num_heads, topk]
    else:
        indices_2d = indices  # Already [bsz, num_heads, topk]

    # Convert to CPU and numpy for saving
    indices_np = indices_2d.cpu().numpy()

    # Save to JSON file with sample_id in filename
    json_file = os.path.join(final_dir, f"{method}_sample{sample_id:03d}_layer{layer_idx}_indices.json")

    # Calculate group assignments for GQA
    num_heads = indices_np.shape[1]
    group_info = None
    if num_key_value_groups is not None and num_key_value_groups > 0:
        # For GQA: each group has (num_heads // num_kv_heads) query heads
        # num_kv_heads = num_heads // num_key_value_groups
        heads_per_group = num_key_value_groups
        group_assignments = [head_idx // heads_per_group for head_idx in range(num_heads)]
        num_groups = num_heads // heads_per_group

        group_info = {
            "num_key_value_groups": num_key_value_groups,
            "num_query_heads": num_heads,
            "heads_per_group": heads_per_group,
            "num_groups": num_groups,
            "group_assignments": group_assignments  # [0,0,0,0, 1,1,1,1, 2,2,2,2, ...]
        }

    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sample_id": sample_id,
        "layer_idx": layer_idx,
        "method": method,
        "capacity": capacity,
        "run_timestamp": run_timestamp,
        "dataset_name": dataset_name,  # Dataset name for organization
        "shape": list(indices_np.shape),  # [bsz, num_heads, topk]
        "indices": indices_np.tolist(),
        "group_info": group_info  # GQA group information
    }

    # Optionally save attention scores
    if attn_weights_sum is not None:
        scores_np = attn_weights_sum.cpu().numpy()
        data["attn_scores_shape"] = list(scores_np.shape)
        data["attn_scores"] = scores_np.tolist()

    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"[TopK Indices] Saved sample {sample_id} layer {layer_idx} to {json_file}")

    return json_file


def save_key_states(key_states, layer_idx, attn_weights_sum=None, method="snapkv", save_dir="kv_cache_logs", sample_id=0, capacity=None, run_timestamp=None, num_key_value_groups=None, dataset_name=None):
    """
    Save complete key states (before repeat_kv) for 3D visualization.

    Args:
        key_states: torch.Tensor, shape [bsz, num_heads, seq_len, head_dim], the key states AFTER repeat_kv
        layer_idx: int, current layer index
        attn_weights_sum: Optional torch.Tensor, attention weight scores
        method: str, compression method name
        save_dir: str, base directory to save logs
        sample_id: int, sample sequence number (for multiple samples)
        capacity: int, max_capacity_prompt value (for folder organization)
        run_timestamp: str, timestamp for this run (for folder organization)
        num_key_value_groups: int, number of query heads per KV group (for extracting original KV heads)
        dataset_name: str, dataset name (for folder organization)
    """
    # Build folder structure: save_dir/key_states_logs/timestamp/dataset_name/cap_xxx/
    save_dir = os.path.join(save_dir, "key_states_logs")

    if run_timestamp and capacity is not None and dataset_name:
        final_dir = os.path.join(save_dir, run_timestamp, dataset_name, f"cap_{capacity}")
    elif run_timestamp and capacity is not None:
        final_dir = os.path.join(save_dir, run_timestamp, f"cap_{capacity}")
    elif capacity is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_dir = os.path.join(save_dir, timestamp, f"cap_{capacity}")
    else:
        final_dir = save_dir

    os.makedirs(final_dir, exist_ok=True)

    # Extract original key states before repeat_kv
    # If GQA is used, key_states has been repeated, we need to extract the original KV heads
    # Original: [bsz, num_kv_heads, seq_len, head_dim]
    # After repeat_kv: [bsz, num_heads, seq_len, head_dim] where num_heads = num_kv_heads * num_key_value_groups
    bsz, num_heads, seq_len, head_dim = key_states.shape

    if num_key_value_groups is not None and num_key_value_groups > 1:
        # Extract original KV heads (every num_key_value_groups-th head is the same)
        # Take the first head from each group: heads [0, num_key_value_groups, 2*num_key_value_groups, ...]
        num_kv_heads = num_heads // num_key_value_groups
        original_key_states = key_states[:, ::num_key_value_groups, :, :]  # [bsz, num_kv_heads, seq_len, head_dim]
    else:
        # No GQA, key_states is already original
        original_key_states = key_states
        num_kv_heads = num_heads

    # Convert to CPU and numpy for saving - save ALL key states, not just topk selected ones
    key_states_np = original_key_states.cpu().numpy()  # [bsz, num_kv_heads, seq_len, head_dim]

    # Save to JSON file with sample_id in filename
    json_file = os.path.join(final_dir, f"{method}_sample{sample_id:03d}_layer{layer_idx}_keystates.json")

    # Group information
    group_info = {
        "num_kv_heads": num_kv_heads,
        "num_query_heads": num_heads,
        "num_key_value_groups": num_key_value_groups if num_key_value_groups else 1,
        "is_gqa": num_key_value_groups is not None and num_key_value_groups > 1
    }

    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sample_id": sample_id,
        "layer_idx": layer_idx,
        "method": method,
        "capacity": capacity,
        "run_timestamp": run_timestamp,
        "dataset_name": dataset_name,
        "shape": list(key_states_np.shape),  # [bsz, num_kv_heads, seq_len, head_dim]
        "key_states": key_states_np.tolist(),  # Complete key states for 3D visualization (before repeat_kv)
        "group_info": group_info
    }

    # Optionally save attention scores
    if attn_weights_sum is not None:
        # Also extract original attention scores for KV heads
        if num_key_value_groups is not None and num_key_value_groups > 1:
            original_scores = attn_weights_sum[:, ::num_key_value_groups, :]
            scores_np = original_scores.cpu().numpy()
        else:
            scores_np = attn_weights_sum.cpu().numpy()
        data["attn_scores_shape"] = list(scores_np.shape)
        data["attn_scores"] = scores_np.tolist()

    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"[Key States] Saved sample {sample_id} layer {layer_idx} to {json_file}")

    return json_file


def save_query_states(query_states, layer_idx, attn_weights_sum=None, method="snapkv", save_dir="kv_cache_logs", sample_id=0, capacity=None, run_timestamp=None, num_key_value_groups=None, dataset_name=None):
    """
    Save query states (window part) for 3D visualization.

    Args:
        query_states: torch.Tensor, shape [bsz, num_heads, seq_len, head_dim], the query states (window part)
        layer_idx: int, current layer index
        attn_weights_sum: Optional torch.Tensor, attention weight scores
        method: str, compression method name
        save_dir: str, base directory to save logs
        sample_id: int, sample sequence number (for multiple samples)
        capacity: int, max_capacity_prompt value (for folder organization)
        run_timestamp: str, timestamp for this run (for folder organization)
        num_key_value_groups: int, number of query heads per KV group
        dataset_name: str, dataset name (for folder organization)
    """
    # Build folder structure: save_dir/query_states_logs/timestamp/dataset_name/cap_xxx/
    save_dir = os.path.join(save_dir, "query_states_logs")

    if run_timestamp and capacity is not None and dataset_name:
        final_dir = os.path.join(save_dir, run_timestamp, dataset_name, f"cap_{capacity}")
    elif run_timestamp and capacity is not None:
        final_dir = os.path.join(save_dir, run_timestamp, f"cap_{capacity}")
    elif capacity is not None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_dir = os.path.join(save_dir, timestamp, f"cap_{capacity}")
    else:
        final_dir = save_dir

    os.makedirs(final_dir, exist_ok=True)

    # Query states shape: [bsz, num_heads, seq_len, head_dim]
    # Note: query_states are NOT repeated (each head has unique query)
    bsz, num_heads, seq_len, head_dim = query_states.shape

    # For GQA, we have num_heads query heads, but only num_kv_heads key/value heads
    if num_key_value_groups is not None and num_key_value_groups > 1:
        num_kv_heads = num_heads // num_key_value_groups
    else:
        num_kv_heads = num_heads

    # Convert to CPU and numpy for saving
    query_states_np = query_states.cpu().numpy()  # [bsz, num_heads, seq_len, head_dim]

    # Save to JSON file with sample_id in filename
    json_file = os.path.join(final_dir, f"{method}_sample{sample_id:03d}_layer{layer_idx}_querystates.json")

    # Group information
    group_info = {
        "num_kv_heads": num_kv_heads,
        "num_query_heads": num_heads,
        "num_key_value_groups": num_key_value_groups if num_key_value_groups else 1,
        "is_gqa": num_key_value_groups is not None and num_key_value_groups > 1
    }

    data = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sample_id": sample_id,
        "layer_idx": layer_idx,
        "method": method,
        "capacity": capacity,
        "run_timestamp": run_timestamp,
        "dataset_name": dataset_name,
        "shape": list(query_states_np.shape),  # [bsz, num_heads, seq_len, head_dim]
        "query_states": query_states_np.tolist(),  # Complete query states for window part
        "group_info": group_info
    }

    # Optionally save attention scores
    if attn_weights_sum is not None:
        # Query heads are not repeated, so use scores as-is
        scores_np = attn_weights_sum.cpu().numpy()
        data["attn_scores_shape"] = list(scores_np.shape)
        data["attn_scores"] = scores_np.tolist()

    with open(json_file, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"[Query States] Saved sample {sample_id} layer {layer_idx} to {json_file}")

    return json_file


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
