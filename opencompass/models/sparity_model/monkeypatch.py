"""
Refactored monkeypatch module with modular method patches.

This module provides model patching functionality for various sparsity methods.
Each method is now organized in its own subdirectory under patches/.
"""

import torch
import transformers
from transformers import AutoConfig

from .patches import (
    apply_pyramidkv,
    apply_snapkv,
    apply_streamingllm,
    apply_h2o,
    apply_cam,
    apply_l2norm,
    apply_sparq,
    apply_full,
    apply_full_int8,
    apply_pyramidkv_gqa,
    apply_snapkv_gqa,
)
from .patches.common import (
    load_model_with_fallback,
    prepare_inputs_for_generation_llama_new,
)


def _apply_method_patches(self, path, model_kwargs, model_name, is_qwen=False):
    """
    Apply method-specific patches based on self.method.
    Unified handler for both Llama and Qwen models.
    """
    method = self.method
    model_class = "qwen" if is_qwen else "llama"

    # Method dispatch mapping
    method_handlers = {
        'pyramidkv': apply_pyramidkv,
        'snapkv': apply_snapkv,
        'streamingllm': apply_streamingllm,
        'h2o': apply_h2o,
        'cam': apply_cam,
        'l2norm': apply_l2norm,
        'sparq': apply_sparq,
        
        'full': apply_full,
        'full_INT8': apply_full_int8,

        'pyramidkv_gqa': apply_pyramidkv_gqa,
        'snapkv_gqa': apply_snapkv_gqa,
    }
    
    if method in method_handlers:
        # Call the appropriate patch handler
        method_handlers[method](self, path, model_kwargs, is_qwen)
    else:
        raise ValueError(f"Unknown method: {method}. Supported methods: {list(method_handlers.keys())}")

    # Apply generation patch for non-fullkv methods
    if method not in ['fullkv']:
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = \
            prepare_inputs_for_generation_llama_new


def replace_model(self, path=None, model_kwargs=None,
                  model_name='meta-llama/Meta-Llama-3.1-8B-Instruct'):
    """Main function to replace and configure model based on method."""

    self.logger.debug(f'using model_kwargs: {path}')

    model_kwargs['torch_dtype'] = torch.bfloat16

    self.model = load_model_with_fallback(path, model_kwargs)
    # Configure model settings
    model_type = self.model_type
    print(f"================{model_type}===================")
    if "qwen" in model_type.lower():
        _apply_method_patches(self, path, model_kwargs, model_name, is_qwen=True)
    elif "llama" in model_type.lower():
        _apply_method_patches(self, path, model_kwargs, model_name, is_qwen=False)

    # Configure model settings (after patches to avoid being overwritten)
    self.model.config.window_size = self.cache_kwargs.get('window_size', 64)
    self.model.config.max_capacity_prompt = self.cache_kwargs.get('max_capacity_prompt', 512)
    self.model.config.file_name = f"{path}_{self.model.config.window_size}_{self.model.config.max_capacity_prompt}"
    self.model.config.block_size = self.cache_kwargs.get('block_size', 32)
    self.model.config.ratio = self.cache_kwargs.get('ratio', 0.4)

    # Log configuration
    self.logger.debug(
        "[Model Config Parameters]\\n"
        f"  file_name: {self.model.config.file_name}\\n"
        f"  window_size: {self.model.config.window_size}\\n"
        f"  max_capacity_prompt: {self.model.config.max_capacity_prompt}\\n"
        f"  block_size: {self.model.config.block_size}"
    )

