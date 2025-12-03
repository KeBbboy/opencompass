"""Forward function for sum_gqa method - based on transformers 4.57.3 API."""
import torch
from typing import Optional, Tuple
from transformers.cache_utils import Cache
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
from transformers.integrations.sdpa_attention import use_gqa_in_sdpa
from transformers.utils.deprecation import deprecate_kwarg
from transformers.utils import logging, is_torch_npu_available

_is_torch_npu_available = is_torch_npu_available()
logger = logging.get_logger(__name__)

# Import local init function
from .init_utils import init_sum_gqa
from ..utils.kv_utils import estimate_kv_memory


@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def llama_attention_forward_sum_gqa(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Forward function for sum_gqa method (Llama version).
    Based on transformers 4.57.3 LlamaAttention.forward with KV compression.
    """
    init_sum_gqa(self)

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # Distinguish between prefill and generation
        if key_states.shape[-2] > 1:  # Prefill phase
            # Apply KV compression/sparsity
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask,
                self.num_key_value_groups
            )

            key_states, value_states = past_key_values.update(
                key_states_compress, value_states_compress, self.layer_idx, cache_kwargs
            )

            # Estimate KV memory at layer 27
            if self.layer_idx == 27:
                method = getattr(self.config, 'method', 'unknown')
                max_capacity_prompt = None
                if hasattr(self.config, 'max_capacity_prompt'):
                    max_capacity_prompt = self.config.max_capacity_prompt
                elif hasattr(self.config, 'cache_kwargs') and 'max_capacity_prompt' in self.config.cache_kwargs:
                    max_capacity_prompt = self.config.cache_kwargs['max_capacity_prompt']

                if isinstance(method, str) and method.lower() == "full":
                    max_capacity_prompt = None

                estimate_kv_memory(past_key_values, method=method, max_capacity_prompt=max_capacity_prompt)
        else:  # Generation phase
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

    # SDPA (Scaled Dot Product Attention) implementation
    # Handle GQA (Grouped Query Attention) - repeat key/value if needed
    sdpa_kwargs = {}
    if hasattr(self, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key_states):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}

    # Adjust attention mask to match key sequence length
    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key_states.shape[-2]]

    # Determine if causal masking should be applied
    is_causal = query_states.shape[2] > 1 and attention_mask is None and getattr(self, "is_causal", True)

    # Convert to bool for JIT tracing
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    # Handle NPU special case (convert attention_mask to bool for FlashAttentionScore)
    if _is_torch_npu_available:
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            attention_mask = torch.logical_not(attention_mask.bool()).to(query_states.device)

    # Apply SDPA
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        scale=self.scaling,
        is_causal=is_causal,
        **sdpa_kwargs,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


@deprecate_kwarg("past_key_value", new_name="past_key_values", version="4.58")
def qwen2_attention_forward_sum_gqa(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Forward function for sum_gqa method (Qwen2 version).
    Based on transformers 4.57.3 Qwen2Attention.forward with KV compression.
    """
    # Debug: Confirm forward is being called (only print once per layer)
    if not hasattr(self, '_sum_gqa_forward_called'):
        print(f"[FORWARD] qwen2_attention_forward_sum_gqa called for layer {self.layer_idx}")
        self._sum_gqa_forward_called = True

    init_sum_gqa(self)

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    query_states = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    key_states = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # Distinguish between prefill and generation
        if key_states.shape[-2] > 1:  # Prefill phase
            # Apply KV compression/sparsity
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask,
                self.num_key_value_groups
            )

            key_states, value_states = past_key_values.update(
                key_states_compress, value_states_compress, self.layer_idx, cache_kwargs
            )


        else:  # Generation phase
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

    # SDPA (Scaled Dot Product Attention) implementation
    # Handle GQA (Grouped Query Attention) - repeat key/value if needed
    sdpa_kwargs = {}
    if hasattr(self, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key_states):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}

    # Adjust attention mask to match key sequence length
    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key_states.shape[-2]]

    # Determine if causal masking should be applied
    # Note: Qwen2 may use sliding_window, but this is handled by attention_mask
    is_causal = query_states.shape[2] > 1 and attention_mask is None and getattr(self, "is_causal", True)

    # Convert to bool for JIT tracing
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    # Handle NPU special case (convert attention_mask to bool for FlashAttentionScore)
    if _is_torch_npu_available:
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            attention_mask = torch.logical_not(attention_mask.bool()).to(query_states.device)

    # Apply SDPA
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        scale=self.scaling,
        is_causal=is_causal,
        **sdpa_kwargs,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


def qwen3moe_attention_forward_sum_gqa(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: Tuple[torch.Tensor, torch.Tensor],
    attention_mask: Optional[torch.Tensor],
    past_key_values: Optional[Cache] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """
    Forward function for sum_gqa method (Qwen3MoE version).
    Based on transformers Qwen3MoeAttention.forward with KV compression.
    Key differences from Qwen2: Q/K normalization.
    """
    # Debug: Confirm forward is being called (only print once per layer)
    if not hasattr(self, '_sum_gqa_forward_called'):
        print(f"[FORWARD] qwen3moe_attention_forward_sum_gqa called for layer {self.layer_idx}")
        self._sum_gqa_forward_called = True

    init_sum_gqa(self)

    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    # Qwen3MoE uses Q/K normalization (unlike Qwen2)
    query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
    value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

    cos, sin = position_embeddings
    query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

    if past_key_values is not None:
        # sin and cos are specific to RoPE models; cache_position needed for the static cache
        cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}

        # Distinguish between prefill and generation
        if key_states.shape[-2] > 1:  # Prefill phase
            # Apply KV compression/sparsity
            key_states_compress, value_states_compress = self.kv_cluster.update_kv(
                key_states, query_states, value_states, attention_mask,
                self.num_key_value_groups
            )

            key_states, value_states = past_key_values.update(
                key_states_compress, value_states_compress, self.layer_idx, cache_kwargs
            )

            # Estimate KV memory at layer 27
            if self.layer_idx == 27:
                method = getattr(self.config, 'method', 'unknown')
                max_capacity_prompt = None
                if hasattr(self.config, 'max_capacity_prompt'):
                    max_capacity_prompt = self.config.max_capacity_prompt
                elif hasattr(self.config, 'cache_kwargs') and 'max_capacity_prompt' in self.config.cache_kwargs:
                    max_capacity_prompt = self.config.cache_kwargs['max_capacity_prompt']

                if isinstance(method, str) and method.lower() == "full":
                    max_capacity_prompt = None

                estimate_kv_memory(past_key_values, method=method, max_capacity_prompt=max_capacity_prompt)
        else:  # Generation phase
            key_states, value_states = past_key_values.update(
                key_states, value_states, self.layer_idx, cache_kwargs
            )

    # SDPA (Scaled Dot Product Attention) implementation
    # Handle GQA (Grouped Query Attention) - repeat key/value if needed
    sdpa_kwargs = {}
    if hasattr(self, "num_key_value_groups"):
        if not use_gqa_in_sdpa(attention_mask, key_states):
            key_states = repeat_kv(key_states, self.num_key_value_groups)
            value_states = repeat_kv(value_states, self.num_key_value_groups)
        else:
            sdpa_kwargs = {"enable_gqa": True}

    # Adjust attention mask to match key sequence length
    if attention_mask is not None and attention_mask.ndim == 4:
        attention_mask = attention_mask[:, :, :, : key_states.shape[-2]]

    # Determine if causal masking should be applied
    # Note: Qwen3MoE may use sliding_window, but this is handled by attention_mask
    is_causal = query_states.shape[2] > 1 and attention_mask is None and getattr(self, "is_causal", True)

    # Convert to bool for JIT tracing
    if torch.jit.is_tracing() and isinstance(is_causal, torch.Tensor):
        is_causal = is_causal.item()

    # Handle NPU special case (convert attention_mask to bool for FlashAttentionScore)
    if _is_torch_npu_available:
        if attention_mask is not None and attention_mask.dtype != torch.bool:
            attention_mask = torch.logical_not(attention_mask.bool()).to(query_states.device)

    # Apply SDPA
    attn_output = torch.nn.functional.scaled_dot_product_attention(
        query_states,
        key_states,
        value_states,
        attn_mask=attention_mask,
        dropout_p=self.attention_dropout if self.training else 0.0,
        scale=self.scaling,
        is_causal=is_causal,
        **sdpa_kwargs,
    )

    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None
