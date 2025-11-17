"""Generic simple patch implementation for methods that only need forward function replacement."""

import transformers
from .utils import load_model_with_fallback, configure_basic_cache


def apply_simple_patch(self, path, model_kwargs, forward_func, model_class="llama"):
    """Apply a simple forward function patch for methods like PyramidKV, StreamingLLM, etc."""
    self.model = load_model_with_fallback(path, model_kwargs)
    configure_basic_cache(self.model, self.cache_kwargs, method=self.method)

    if model_class == "llama":
        transformers.models.llama.modeling_llama.LlamaSdpaAttention.forward = forward_func
    elif model_class == "qwen":
        transformers.models.qwen2.modeling_qwen2.Qwen2SdpaAttention.forward = forward_func
