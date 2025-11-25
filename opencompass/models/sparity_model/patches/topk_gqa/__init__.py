"""TopK-GQA method patch module.

TopK-based KV compression using top-k Q heads scoring within each group.
"""

from .patch import apply_topk_gqa

__all__ = ['apply_topk_gqa']
