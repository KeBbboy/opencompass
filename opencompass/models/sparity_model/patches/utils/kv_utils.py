import os
import csv
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
