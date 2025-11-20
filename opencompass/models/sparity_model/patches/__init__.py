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
from .windowkv import apply_windowkv
from .chunkkv import apply_chunkkv

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
    'apply_windowkv',
    'apply_chunkkv',
]
