"""Common utilities for model patching."""

from .utils import (
    load_model_with_fallback,
    configure_basic_cache,
)
from .simple_patch import apply_simple_patch
from .generation_utils import prepare_inputs_for_generation_llama_new

__all__ = [
    'load_model_with_fallback',
    'configure_basic_cache',
    'apply_simple_patch',
    'prepare_inputs_for_generation_llama_new',
]
