"""Common utility functions for model patching."""

import os
from transformers import AutoModel, AutoModelForCausalLM


def load_model_with_fallback(path, model_kwargs):
    """Load model with AutoModelForCausalLM, fallback to AutoModel if needed."""
    try:
        return AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
    except ValueError:
        return AutoModel.from_pretrained(path, **model_kwargs)


def configure_basic_cache(model, cache_kwargs, method=None):
    """Configure basic cache settings (window_size, max_capacity_prompt, and method)."""
    model.config.window_size = cache_kwargs.window_size
    model.config.max_capacity_prompt = cache_kwargs.max_capacity_prompt
    if method is not None:
        model.config.method = method

