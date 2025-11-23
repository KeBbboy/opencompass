"""Common utility functions for model patching."""

import os
from transformers import AutoModel, AutoModelForCausalLM


def load_model_with_fallback(path, model_kwargs):
    """Load model with AutoModelForCausalLM, fallback to AutoModel if needed."""
    import torch

    try:
        model = AutoModelForCausalLM.from_pretrained(path, **model_kwargs)
    except ValueError:
        model = AutoModel.from_pretrained(path, **model_kwargs)

    # Ensure model is on CUDA if available and device_map didn't work
    # Check if model is on CPU when CUDA is available
    if torch.cuda.is_available():
        try:
            model_device = next(model.parameters()).device
            if model_device.type == 'cpu':
                print(f"[WARNING] Model loaded on CPU despite device_map={model_kwargs.get('device_map')}")
                print(f"[WARNING] Manually moving model to CUDA...")
                model = model.cuda()
        except StopIteration:
            # No parameters (empty model?)
            pass

    return model


def configure_basic_cache(model, cache_kwargs, method=None):
    """Configure basic cache settings (window_size, max_capacity_prompt, and method)."""
    model.config.window_size = cache_kwargs.window_size
    model.config.max_capacity_prompt = cache_kwargs.max_capacity_prompt
    if method is not None:
        model.config.method = method

