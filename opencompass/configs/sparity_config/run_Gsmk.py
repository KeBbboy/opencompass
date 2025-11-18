from mmengine.config import read_base
from opencompass.models import QwenAttentionConvert
with read_base():
    from ..datasets.gsm8k.gsm8k_gen import gsm8k_datasets  
  

# Read parameters from environment variables with defaults
import os as _temp_os
SPARITY_METHOD = _temp_os.getenv('SPARITY_METHOD', 'snapkv')
MAX_CAPACITY_PROMPT = int(_temp_os.getenv('MAX_CAPACITY_PROMPT', '512'))
del _temp_os  


# 将需要评测的数据集拼接成 datasets 字段
datasets = [
    *gsm8k_datasets,
]

    

models = [
    dict(
        type=QwenAttentionConvert,
        abbr='qwen2.5-7b-instruct-hf',
        path='Qwen/Qwen2.5-7B-Instruct',
        is_use_sparse=True,
        max_seq_len=32768,
        max_out_len=256,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        method=SPARITY_METHOD,
        cache_kwargs=dict(
                # common parameters
                window_size=64,
                max_capacity_prompt=MAX_CAPACITY_PROMPT,
        ),
        kivi_kwargs=dict(
            # KIVI-specific parameters (only used when method='full_KIVI')
            k_bits=2,              # Key quantization bits (1-8)
            v_bits=2,              # Value quantization bits (1-8)
            group_size=32,         # Quantization group size
            residual_length=32,   # Number of recent tokens kept in full precision
        )
    )
]