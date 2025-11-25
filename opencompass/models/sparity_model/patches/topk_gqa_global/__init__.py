"""TopK-GQA-Global method patch module.

TopK-based KV compression using global top-k Q heads scoring across all groups.
All KV heads share the same token indices.
"""

from .patch import apply_topk_gqa_global

__all__ = ['apply_topk_gqa_global']
