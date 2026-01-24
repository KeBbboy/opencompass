"""Patches module for various sparsity methods."""

from .pyramidkv import apply_pyramidkv
from .snapkv import apply_snapkv
from .streamingllm import apply_streamingllm
from .h2o import apply_h2o
from .cam import apply_cam
from .l2norm import apply_l2norm
from .sparq import apply_sparq
from .full import apply_full
from .sum_gqa import apply_sum_gqa
from .windowkv import apply_windowkv
from .windowkv_gqa import apply_windowkv_gqa
from .chunkkv import apply_chunkkv
from .min_max_gqa import apply_min_max_gqa
from .topk_gqa import apply_topk_gqa
from .center_topk_gqa import apply_center_topk_gqa
from .RQA_sum import apply_RQA_sum
from .RQA_mean import apply_RQA_mean
from .RQA_mean_softmax import apply_RQA_mean_softmax
from .RQA_mean_improved import apply_RQA_mean_improved
from .RQA_l2weighted_ablation import apply_RQA_l2weighted_ablation
from .RQA_per_head_topk import apply_RQA_per_head_topk
from .RQA_top1 import apply_RQA_top1
from .quest import apply_quest
from .raas import apply_raas

__all__ = [
    'apply_pyramidkv',
    'apply_snapkv',
    'apply_streamingllm',
    'apply_h2o',
    'apply_cam',
    'apply_l2norm',
    'apply_sparq',
    'apply_full',
    'apply_sum_gqa',
    'apply_windowkv',
    'apply_windowkv_gqa',
    'apply_chunkkv',
    'apply_min_max_gqa',
    'apply_topk_gqa',
    'apply_center_topk_gqa',
    'apply_RQA_sum',
    'apply_RQA_mean',
    'apply_RQA_mean_softmax',
    'apply_RQA_mean_improved',
    'apply_RQA_l2weighted_ablation',
    'apply_RQA_per_head_topk',
    'apply_RQA_top1',
    'apply_quest',
    'apply_raas',
]
