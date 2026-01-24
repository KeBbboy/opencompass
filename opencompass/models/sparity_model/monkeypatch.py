"""
Refactored monkeypatch module with modular method patches.

This module provides model patching functionality for various sparsity methods.
Each method is now organized in its own subdirectory under patches/.
"""

import torch
import transformers

from .patches import (
    apply_pyramidkv,
    apply_snapkv,
    apply_streamingllm,
    apply_h2o,
    apply_cam,
    apply_l2norm,
    apply_sparq,
    apply_full,
    apply_sum_gqa,
    apply_windowkv,
    apply_windowkv_gqa,
    apply_chunkkv,
    apply_min_max_gqa,
    apply_topk_gqa,
    apply_center_topk_gqa,
    apply_RQA_sum,
    apply_RQA_mean,
    apply_RQA_mean_softmax,
    apply_RQA_mean_improved,
    apply_RQA_l2weighted_ablation,
    apply_RQA_per_head_topk,
    apply_RQA_top1,
)
from .patches.common import (
    load_model_with_fallback,
    prepare_inputs_for_generation_llama_new,
)


def _apply_method_patches(self, path, model_kwargs, model_name, is_qwen=False):  # noqa: ARG001
    """
    Apply method-specific patches based on self.method.
    Unified handler for both Llama and Qwen models.
    """
    method = self.method

    # Method dispatch mapping
    method_handlers = {
        'pyramidkv': apply_pyramidkv,
        'snapkv': apply_snapkv,
        'streamingllm': apply_streamingllm,
        'h2o': apply_h2o,
        'cam': apply_cam,
        'l2norm': apply_l2norm,
        'sparq': apply_sparq,
        'windowkv': apply_windowkv,
        'chunkkv': apply_chunkkv,

        'full': apply_full,

        'sum_gqa': apply_sum_gqa,
        'windowkv_gqa': apply_windowkv_gqa,
        'min_max_gqa': apply_min_max_gqa,
        'topk_gqa': apply_topk_gqa,
        'center_topk_gqa': apply_center_topk_gqa,
        'RQA_sum': apply_RQA_sum,
        'RQA_mean': apply_RQA_mean,
        'RQA_mean_softmax': apply_RQA_mean_softmax,
        'RQA_mean_improved': apply_RQA_mean_improved,
        'RQA_l2weighted_ablation': apply_RQA_l2weighted_ablation,
        'RQA_per_head_topk': apply_RQA_per_head_topk,
        'RQA_top1': apply_RQA_top1,
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

    # Ensure device_map is set if not specified
    if 'device_map' not in model_kwargs:
        model_kwargs['device_map'] = 'auto'

    # Use FP16 for KIVI CUDA kernel compatibility
    # KIVI's CUDA kernels only support FP16, not BFloat16
    model_kwargs['torch_dtype'] = torch.float16

    self.model = load_model_with_fallback(path, model_kwargs)
    # Configure model settings
    model_type = self.model_type
    print(f"================{model_type}===================")

    # Debug: Print model device
    print(f"[DEBUG] Model device after loading: {next(self.model.parameters()).device}")
    print(f"[DEBUG] model_kwargs: {model_kwargs}")

    if "qwen" in model_type.lower():
        _apply_method_patches(self, path, model_kwargs, model_name, is_qwen=True)
    elif "llama" in model_type.lower():
        _apply_method_patches(self, path, model_kwargs, model_name, is_qwen=False)

    # Debug: Print model device after patching
    print(f"[DEBUG] Model device after patching: {next(self.model.parameters()).device}")

    # Configure model settings (after patches to avoid being overwritten)
    self.model.config.window_size = self.cache_kwargs.get('window_size', 64)
    self.model.config.max_capacity_prompt = self.cache_kwargs.get('max_capacity_prompt', 512)
    self.model.config.file_name = f"{path}_{self.model.config.window_size}_{self.model.config.max_capacity_prompt}"
    self.model.config.block_size = self.cache_kwargs.get('block_size', 32)
    self.model.config.ratio = self.cache_kwargs.get('ratio', 0.4)

    # Configure KIVI-specific parameters (if using KIVI method)
    if hasattr(self, 'kivi_kwargs') and self.kivi_kwargs:
        self.model.config.k_bits = self.kivi_kwargs.get('k_bits', 2)
        self.model.config.v_bits = self.kivi_kwargs.get('v_bits', 2)
        self.model.config.group_size = self.kivi_kwargs.get('group_size', 32)
        self.model.config.residual_length = self.kivi_kwargs.get('residual_length', 128)

        # Force print to ensure visibility
        print("\n" + "="*60)
        print("[KIVI Config Parameters]")
        print(f"  k_bits: {self.model.config.k_bits}")
        print(f"  v_bits: {self.model.config.v_bits}")
        print(f"  group_size: {self.model.config.group_size}")
        print(f"  residual_length: {self.model.config.residual_length}")
        print("="*60 + "\n")

        self.logger.debug(
            "[KIVI Config Parameters]\\n"
            f"  k_bits: {self.model.config.k_bits}\\n"
            f"  v_bits: {self.model.config.v_bits}\\n"
            f"  group_size: {self.model.config.group_size}\\n"
            f"  residual_length: {self.model.config.residual_length}"
        )

    # Configure WindowKV-specific parameters (if using WindowKV method)
    if self.method in ['windowkv', 'windowkv_gqa']:
        self.model.config.chunk_length = self.cache_kwargs.get('chunk_length', 8)
        category = "qa"
        if category == "qa":
            self.model.config.window_select_strategy = "max"
        else:
            self.model.config.window_select_strategy = "average"


        self.logger.debug(
            "[WindowKV Config Parameters]\\n"
            f"  window_select_strategy: {self.model.config.window_select_strategy}\\n"
        )

    # Configure ChunkKV-specific parameters (if using ChunkKV method)
    if self.method == 'chunkkv':
        self.model.config.chunk_length = self.cache_kwargs.get('chunk_length', 32)

        self.logger.debug(
            "[ChunkKV Config Parameters]\\n"
            f"  chunk_length: {self.model.config.chunk_length}\\n"
        )

    # Log configuration
    self.logger.debug(
        "[Model Config Parameters]\\n"
        f"  file_name: {self.model.config.file_name}\\n"
        f"  window_size: {self.model.config.window_size}\\n"
        f"  max_capacity_prompt: {self.model.config.max_capacity_prompt}\\n"
        f"  block_size: {self.model.config.block_size}"
    )

