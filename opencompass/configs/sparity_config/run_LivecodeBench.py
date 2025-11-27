from mmengine.config import read_base
from opencompass.models import wenLlamaAttentionConvert
from opencompass.models import HuggingFacewithChatTemplate
from opencompass.models import QwenAttentionConvert
with read_base():
    # from ..datasets.gsm8k.gsm8k_gen import gsm8k_datasets  
    from ..datasets.livecodebench.livecodebench_gen_a4f90b import LCB_datasets  # noqa: F401, F403
    from ..datasets.bigcodebench.bigcodebench_hard_instruct_gen import bigcodebench_hard_instruct_datasets
    from ..datasets.bigcodebench.bigcodebench_hard_complete_gen import bigcodebench_hard_complete_datasets
    from ..datasets.aime2025.aime2025_cascade_eval_gen_5e9f4f import aime2025_datasets
    from ..datasets.infinitebench.infinitebenchcodedebug.infinitebench_codedebug_gen import InfiniteBench_codedebug_datasets  # noqa: F401, F403

    from ..datasets.infinitebench.infinitebenchensum.infinitebench_ensum_gen_cfbc08 import InfiniteBench_ensum_datasets
# noqa: F401, F403
needlebench_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])



# Read parameters from environment variables with defaults
import os as _temp_os
SPARITY_METHOD = _temp_os.getenv('SPARITY_METHOD', 'snapkv_global')
MAX_CAPACITY_PROMPT = int(_temp_os.getenv('MAX_CAPACITY_PROMPT', '512'))

# KIVI-specific parameters
KIVI_K_BITS = int(_temp_os.getenv('KIVI_K_BITS', '2'))
KIVI_V_BITS = int(_temp_os.getenv('KIVI_V_BITS', '2'))
KIVI_GROUP_SIZE = int(_temp_os.getenv('KIVI_GROUP_SIZE', '32'))
KIVI_RESIDUAL_LENGTH = int(_temp_os.getenv('KIVI_RESIDUAL_LENGTH', '32'))

del _temp_os  


# 将需要评测的数据集拼接成 datasets 字段
datasets = [
    # *needlebench_datasets
    *InfiniteBench_ensum_datasets,
    # *LongBench_qasper_datasets,
    # *LongBench_multifieldqa_en_datasets,
    # *LongBench_multifieldqa_zh_datasets
]

    

models = [
    dict(
        type=QwenAttentionConvert,
        # abbr='qwen2.5-14b-instruct-hf',
        # path='Qwen/Qwen2.5-14B-Instruct',
        abbr='qwen2.5-7b-instruct-hf',
        path='Qwen/Qwen2.5-7B-Instruct',
        is_use_sparse=True,
        max_seq_len=40000,
        max_out_len=256,
        batch_size=1,
        run_cfg=dict(num_gpus=1),
        method=SPARITY_METHOD,
        cache_kwargs=dict(
                # common parameters
                window_size=64,
                max_capacity_prompt=MAX_CAPACITY_PROMPT,
                chunk_length = 8,
                topk_heads=2,  # 在这里设置 k 值
                # NOTE: For ChunkKV method, ensure (max_capacity_prompt - window_size) % chunk_length == 0
                # Current: (512 - 64) % 8 = 448 % 8 = 0 ✓

                # Visualization parameters
                save_indices=True,           # 启用索引保存
                visualize_layer=None,        # 保存所有层（0=只保存第0层）
                max_samples_to_save=5,      # 保存前10个样本
        ),
        kivi_kwargs=dict(
            # KIVI-specific parameters (only used when method='full_KIVI')
            k_bits=KIVI_K_BITS,              # Key quantization bits (1-8)
            v_bits=KIVI_V_BITS,              # Value quantization bits (1-8)
            group_size=KIVI_GROUP_SIZE,         # Quantization group size
            residual_length=KIVI_RESIDUAL_LENGTH,   # Number of recent tokens kept in full precision
        )
    )
]
