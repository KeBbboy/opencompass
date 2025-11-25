"""Center-TopK-GQA method patch module.

Center-based KV compression: each KV head group independently selects tokens
based on their distance from the centroid (center point).
"""

from .patch import apply_center_topk_gqa

__all__ = ['apply_center_topk_gqa']
