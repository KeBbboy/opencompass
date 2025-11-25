"""Patches module for various sparsity methods."""

from .pyramidkv import apply_pyramidkv
from .snapkv import apply_snapkv
from .streamingllm import apply_streamingllm
from .h2o import apply_h2o
from .cam import apply_cam
from .l2norm import apply_l2norm
from .sparq import apply_sparq
from .full import apply_full
from .full_int8 import apply_full_int8
from .full_kivi import apply_full_KIVI
from .pyramidkv_gqa import apply_pyramidkv_gqa
from .snapkv_gqa import apply_snapkv_gqa
from .snapkv_gqa2 import apply_snapkv_gqa2
from .snapkv_gqa3 import apply_snapkv_gqa3
from .windowkv import apply_windowkv
from .windowkv_gqa import apply_windowkv_gqa
from .chunkkv import apply_chunkkv
from .min_max_gqa import apply_min_max_gqa
from .min_max_gqa_global import apply_min_max_gqa_global
from .min_max_gqa_chunk import apply_min_max_gqa_chunk
from .min_max_gqa_chunk_global import apply_min_max_gqa_chunk_global
from .snapkv_gqa_chunk_global import apply_snapkv_gqa_chunk_global

__all__ = [
    'apply_pyramidkv',
    'apply_snapkv',
    'apply_streamingllm',
    'apply_h2o',
    'apply_cam',
    'apply_l2norm',
    'apply_sparq',
    'apply_full',
    'apply_full_int8',
    'apply_full_KIVI',
    'apply_pyramidkv_gqa',
    'apply_snapkv_gqa',
    'apply_snapkv_gqa2',
    'apply_snapkv_gqa3',
    'apply_windowkv',
    'apply_windowkv_gqa',
    'apply_chunkkv',
    'apply_min_max_gqa',
    'apply_min_max_gqa_global',
    'apply_min_max_gqa_chunk',
    'apply_min_max_gqa_chunk_global',
    'apply_snapkv_gqa_chunk_global',
]
