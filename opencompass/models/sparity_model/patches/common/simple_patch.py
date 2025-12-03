"""Generic simple patch implementation for methods that only need forward function replacement."""

import transformers


def apply_simple_patch(self, path, model_kwargs, forward_func, model_class="llama"):
    """Apply a simple forward function patch for methods like PyramidKV, StreamingLLM, etc."""
    # Note: This function is called BEFORE the model is loaded (see monkeypatch.py line 210-218)
    # Model configuration is done in replace_model() after the model is loaded
    # So we only patch the transformers classes here, not configure the model instance

    print("\n" + "="*60)
    print(f"[SIMPLE_PATCH] Patching {model_class} attention forward function")
    print(f"[SIMPLE_PATCH] Method: {self.method}")
    print("="*60 + "\n")

    if model_class == "llama":
        transformers.models.llama.modeling_llama.LlamaAttention.forward = forward_func
        print(f"[SIMPLE_PATCH] Patched LlamaAttention.forward")
    elif model_class == "qwen":
        # Patch both Qwen2 and Qwen3MoE attention classes
        transformers.models.qwen2.modeling_qwen2.Qwen2Attention.forward = forward_func
        print(f"[SIMPLE_PATCH] Patched Qwen2Attention.forward")

        # Also patch Qwen3MoE models
        try:
            import transformers.models.qwen3_moe.modeling_qwen3_moe as qwen3_moe
            qwen3_moe.Qwen3MoeAttention.forward = forward_func
            print(f"[SIMPLE_PATCH] Patched Qwen3MoeAttention.forward")
        except (ImportError, AttributeError) as e:
            print(f"[SIMPLE_PATCH] Note: Qwen3MoeAttention not available (this is OK if not using MoE model): {e}")
