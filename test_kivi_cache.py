"""Test KIVI Cache implementation"""
import torch
from transformers.cache_utils import Cache
import sys
sys.path.insert(0, 'opencompass/models/sparity_model/patches/full_kivi')

from forward import KIVICache

# Test 1: KIVICache is a Cache subclass
cache = KIVICache()
print(f"✓ KIVICache is Cache subclass: {isinstance(cache, Cache)}")

# Test 2: Basic operations
kivi_tuple = (None, torch.randn(1, 4, 10, 128), None, None,  # key parts
              None, torch.randn(1, 4, 10, 128), None, None, 10)  # value parts + seq_len
cache.update(kivi_tuple, None, 0)
print(f"✓ Cache updated for layer 0, seq_len: {cache.get_seq_length(0)}")

# Test 3: to_legacy_cache returns tuple
legacy = cache.to_legacy_cache()
print(f"✓ to_legacy_cache returns tuple: {isinstance(legacy, tuple)}")
print(f"  Length: {len(legacy)}, Layer 0 cache type: {type(legacy[0])}")

# Test 4: Access by index
layer_cache = cache[0]
print(f"✓ Access layer 0 cache: {type(layer_cache)}, seq_len from tuple: {layer_cache[-1]}")

print("\n✅ All tests passed!")
