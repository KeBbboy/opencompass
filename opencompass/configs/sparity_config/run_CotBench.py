from mmengine.config import read_base
from opencompass.models import wenLlamaAttentionConvert
from opencompass.models import HuggingFacewithChatTemplate
from opencompass.models import QwenAttentionConvert
with read_base():

   


needlebench_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

# Read parameters from environment variables with defaults
import os as _temp_os

# Debug: Print environment variables when config is loaded

SPARITY_METHOD = _temp_os.getenv('SPARITY_METHOD', 'snapkv_global')
MAX_CAPACITY_PROMPT = int(_temp_os.getenv('MAX_CAPACITY_PROMPT', '512'))

# KIVI-specific parameters
KIVI_K_BITS = int(_temp_os.getenv('KIVI_K_BITS', '2'))
KIVI_V_BITS = int(_temp_os.getenv('KIVI_V_BITS', '2'))
KIVI_GROUP_SIZE = int(_temp_os.getenv('KIVI_GROUP_SIZE', '32'))
KIVI_RESIDUAL_LENGTH = int(_temp_os.getenv('KIVI_RESIDUAL_LENGTH', '32'))

# TTFT measurement parameters
ENABLE_TTFT = _temp_os.getenv('ENABLE_TTFT', 'False').lower() in ('true', '1', 'yes')
TTFT_SAVE_DIR = _temp_os.getenv('TTFT_SAVE_DIR', './ttft_logs')
TTFT_SAVE_TO_FILE = _temp_os.getenv('TTFT_SAVE_TO_FILE', 'True').lower() in ('true', '1', 'yes')

print("\n" + "="*60)
print("[Config File] Final values after reading:")
print(f"  ENABLE_TTFT = {ENABLE_TTFT}")
print(f"  TTFT_SAVE_DIR = {TTFT_SAVE_DIR}")
print(f"  TTFT_SAVE_TO_FILE = {TTFT_SAVE_TO_FILE}")
print("="*60 + "\n")

del _temp_os  


# 将需要评测的数据集拼接成 datasets 字段
datasets = [
    # *needlebench_datasets
    # *LongBench_narrativeqa_datasets,
    # *LongBench_qasper_datasets,
    # *LongBench_multifieldqa_en_datasets,
    # *LongBench_multifieldqa_zh_datasets
    # *ruler_32k_ds
    *SciCode_datasets  # LiveCodeBench v6
]

    

models = [
    dict(
        type=QwenAttentionConvert,
        # abbr='Qwen/Qwen3-30B-A3B-Instruct-2507',
        # path='Qwen/Qwen3-30B-A3B-Instruct-2507',
        abbr='qwen2.5-7b-instruct-hf',
        path='Qwen/Qwen2.5-7B-Instruct',
        is_use_sparse=True,
        max_seq_len=40000,  # Qwen3-30B-A3B supports 32K context
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
                save_indices=False,           # 启用索引保存
                save_key_states=False,       # 启用key states保存
                save_query_states=False,     # 启用query states保存（window部分）
                visualize_layer=None,        # 保存所有层（0=只保存第0层）
                max_samples_to_save=3,      # 保存前10个样本
        ),
        kivi_kwargs=dict(
            # KIVI-specific parameters (only used when method='full_KIVI')
            k_bits=KIVI_K_BITS,              # Key quantization bits (1-8)
            v_bits=KIVI_V_BITS,              # Value quantization bits (1-8)
            group_size=KIVI_GROUP_SIZE,         # Quantization group size
            residual_length=KIVI_RESIDUAL_LENGTH,   # Number of recent tokens kept in full precision
        ),

        # TTFT measurement configuration (完全解耦，不影响原有逻辑)
        enable_ttft=ENABLE_TTFT,            # 启用/禁用 TTFT 测量
        ttft_save_dir=TTFT_SAVE_DIR,        # TTFT 日志保存目录
        ttft_save_to_file=TTFT_SAVE_TO_FILE,  # 是否保存到文件
    )
]
