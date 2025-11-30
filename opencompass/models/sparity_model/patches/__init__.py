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
from .sum_gqa import apply_sum_gqa
from .sum_gqa_chunk import apply_sum_gqa_chunk
from .sum_gqa_global import apply_sum_gqa_global
from .windowkv import apply_windowkv
from .windowkv_gqa import apply_windowkv_gqa
from .chunkkv import apply_chunkkv
from .min_max_gqa import apply_min_max_gqa
from .min_max_gqa_global import apply_min_max_gqa_global
from .min_max_gqa_chunk import apply_min_max_gqa_chunk
from .min_max_gqa_chunk_global import apply_min_max_gqa_chunk_global
from .sum_gqa_chunk_global import apply_sum_gqa_chunk_global
from .topk_gqa import apply_topk_gqa
from .topk_gqa_global import apply_topk_gqa_global
from .center_topk_gqa import apply_center_topk_gqa
from .first_group_gqa import apply_first_group_gqa
from .RQA_sum import apply_RQA_sum
from .RQA_mean import apply_RQA_mean
from .RQA_mean_softmax import apply_RQA_mean_softmax
from .RQA_mean_improved import apply_RQA_mean_improved

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
    'apply_sum_gqa',
    'apply_sum_gqa_chunk',
    'apply_sum_gqa_global',
    'apply_windowkv',
    'apply_windowkv_gqa',
    'apply_chunkkv',
    'apply_min_max_gqa',
    'apply_min_max_gqa_global',
    'apply_min_max_gqa_chunk',
    'apply_min_max_gqa_chunk_global',
    'apply_sum_gqa_chunk_global',
    'apply_topk_gqa',
    'apply_topk_gqa_global',
    'apply_center_topk_gqa',
    'apply_first_group_gqa',
    'apply_RQA_sum',
    'apply_RQA_mean',
    'apply_RQA_mean_softmax',
    'apply_RQA_mean_improved',
]
