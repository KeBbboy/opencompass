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
    apply_full_int8,
    apply_full_KIVI,
    apply_pyramidkv_gqa,
    apply_sum_gqa,
    apply_sum_gqa_chunk,
    apply_sum_gqa_global,
    apply_windowkv,
    apply_windowkv_gqa,
    apply_chunkkv,
    apply_min_max_gqa,
    apply_min_max_gqa_global,
    apply_min_max_gqa_chunk,
    apply_min_max_gqa_chunk_global,
    apply_sum_gqa_chunk_global,
    apply_topk_gqa,
    apply_topk_gqa_global,
    apply_center_topk_gqa,
    apply_first_group_gqa,
)
from .patches.common import (
    load_model_with_fallback,
    prepare_inputs_for_generation_llama_new,
)


def _patch_qwen2_model_for_tuple_cache():
    """Patch Qwen2Model to skip Cache conversion for KIVICache.

    The issue is that Qwen2Model.forward (line 841-922) tries to convert
    non-Cache objects to DynamicCache, but we need to keep KIVICache as-is.
    """
    from transformers.models.qwen2 import modeling_qwen2

    # Import KIVICache class
    try:
        from .patches.full_kivi.forward import KIVICache
    except ImportError:
        # KIVI not being used, skip patch
        return

    # Store original forward
    original_forward = modeling_qwen2.Qwen2Model.forward

    def patched_forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        cache_position=None,
        **kwargs,
    ):
        """Patched forward that uses KIVICache instead of DynamicCache."""

        # If using KIVI method, replace DynamicCache creation with KIVICache
        # Check if this is None (first call) or already a KIVICache
        if past_key_values is None and use_cache:
            # First call - create KIVICache instead of letting Qwen2 create DynamicCache
            past_key_values = KIVICache()
        elif isinstance(past_key_values, KIVICache):
            # Already KIVICache - pass through
            pass

        # Call original forward - KIVICache will pass isinstance(Cache) check
        result = original_forward(
            self,
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
            **kwargs,
        )

        return result

    modeling_qwen2.Qwen2Model.forward = patched_forward


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
        'full_INT8': apply_full_int8,
        'full_KIVI': apply_full_KIVI,

        'pyramidkv_gqa': apply_pyramidkv_gqa,
        'sum_gqa': apply_sum_gqa,
        'sum_gqa_chunk': apply_sum_gqa_chunk,
        'sum_gqa_global': apply_sum_gqa_global,
        'windowkv_gqa': apply_windowkv_gqa,
        'min_max_gqa': apply_min_max_gqa,
        'min_max_gqa_global': apply_min_max_gqa_global,
        'min_max_gqa_chunk': apply_min_max_gqa_chunk,
        'min_max_gqa_chunk_global': apply_min_max_gqa_chunk_global,
        'sum_gqa_chunk_global': apply_sum_gqa_chunk_global,
        'topk_gqa': apply_topk_gqa,
        'topk_gqa_global': apply_topk_gqa_global,
        'center_topk_gqa': apply_center_topk_gqa,
        'first_group_gqa': apply_first_group_gqa,
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


    if method in ['full_KIVI']:
        # Also patch Qwen2 models
        transformers.models.qwen2.modeling_qwen2.Qwen2ForCausalLM.prepare_inputs_for_generation = \
            prepare_inputs_for_generation_llama_new

        # Patch Qwen2Model to handle tuple caches from KIVI
        _patch_qwen2_model_for_tuple_cache()


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

