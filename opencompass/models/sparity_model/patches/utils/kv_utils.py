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
