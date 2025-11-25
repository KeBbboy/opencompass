"""Min-Max-GQA-Chunk method patch module.

Max+min chunk-based KV compression using group max+min scoring.
Similar to SnapKV-GQA2 but uses Quest's max+min aggregation instead of sum.
"""

from .patch import apply_min_max_gqa_chunk

__all__ = ['apply_min_max_gqa_chunk']
