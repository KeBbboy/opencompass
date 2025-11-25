"""Min-Max-GQA method patch module.

Max+min KV compression for GQA using group-based max+min token scoring.
"""

from .patch import apply_min_max_gqa

__all__ = ['apply_min_max_gqa']
