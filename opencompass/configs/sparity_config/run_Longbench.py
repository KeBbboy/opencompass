from mmengine.config import read_base
from opencompass.models import wenLlamaAttentionConvert
from opencompass.models import HuggingFacewithChatTemplate
from opencompass.models import QwenAttentionConvert
with read_base():
    # from ..datasets.gsm8k.gsm8k_gen import gsm8k_datasets  
    # # ===== Single-document QA =====
    from ..datasets.longbench.longbenchnarrativeqa.longbench_narrativeqa_gen import LongBench_narrativeqa_datasets
    from ..datasets.longbench.longbenchqasper.longbench_qasper_gen import LongBench_qasper_datasets
    from ..datasets.longbench.longbenchmultifieldqa_en.longbench_multifieldqa_en_gen import LongBench_multifieldqa_en_datasets
    from ..datasets.longbench.longbenchmultifieldqa_zh.longbench_multifieldqa_zh_gen import LongBench_multifieldqa_zh_datasets

    # # ===== Multi-document QA =====
    from ..datasets.longbench.longbenchhotpotqa.longbench_hotpotqa_gen import LongBench_hotpotqa_datasets
    from ..datasets.longbench.longbench2wikimqa.longbench_2wikimqa_gen import LongBench_2wikimqa_datasets
    from ..datasets.longbench.longbenchmusique.longbench_musique_gen import LongBench_musique_datasets
    from ..datasets.longbench.longbenchdureader.longbench_dureader_gen import LongBench_dureader_datasets

    # ===== Summarization =====
    from ..datasets.longbench.longbenchgov_report.longbench_gov_report_gen import LongBench_gov_report_datasets
    from ..datasets.longbench.longbenchqmsum.longbench_qmsum_gen import LongBench_qmsum_datasets
    from ..datasets.longbench.longbenchmulti_news.longbench_multi_news_gen import LongBench_multi_news_datasets
    from ..datasets.longbench.longbenchvcsum.longbench_vcsum_gen import LongBench_vcsum_datasets

    # ===== Few-shot Learning =====
    from ..datasets.longbench.longbenchtrec.longbench_trec_gen import LongBench_trec_datasets
    from ..datasets.longbench.longbenchlsht.longbench_lsht_gen import LongBench_lsht_datasets
    from ..datasets.longbench.longbenchsamsum.longbench_samsum_gen import LongBench_samsum_datasets
    from ..datasets.longbench.longbenchtriviaqa.longbench_triviaqa_gen import LongBench_triviaqa_datasets

    # ===== Synthetic Task =====
    from ..datasets.longbench.longbenchpassage_count.longbench_passage_count_gen import LongBench_passage_count_datasets
    from ..datasets.longbench.longbenchpassage_retrieval_en.longbench_passage_retrieval_en_gen import LongBench_passage_retrieval_en_datasets
    from ..datasets.longbench.longbenchpassage_retrieval_zh.longbench_passage_retrieval_zh_gen import LongBench_passage_retrieval_zh_datasets

    # ===== Code Completion =====
    from ..datasets.longbench.longbenchlcc.longbench_lcc_gen import LongBench_lcc_datasets
    from ..datasets.longbench.longbenchrepobench.longbench_repobench_gen import LongBench_repobench_datasets

    # # # 长输入数据
    # from ..datasets.ruler.ruler_1m_gen import ruler_datasets as ruler_1m_ds
    # from ..datasets.ruler.ruler_4k_gen import ruler_datasets as ruler_4k_ds
    # from ..datasets.ruler.ruler_8k_gen import ruler_datasets as ruler_8k_ds
    # from ..datasets.ruler.ruler_16k_gen import ruler_datasets as ruler_16k_ds
    # from ..datasets.ruler.ruler_32k_gen import ruler_datasets as ruler_32k_ds
    # from ..datasets.ruler.ruler_64k_gen import ruler_datasets as ruler_64k_ds
    # from ..datasets.ruler.ruler_128k_gen import ruler_datasets as ruler_128k_ds

    # # ===== InfiniteBench Long Context Code Tasks =====
    # from ..datasets.infinitebench.infinitebenchcodedebug.infinitebench_codedebug_gen import InfiniteBench_codedebug_datasets  # ~114.7K tokens avg
    # from ..datasets.infinitebench.infinitebenchcoderun.infinitebench_coderun_gen import InfiniteBench_coderun_datasets  # ~75.2K tokens avg
    # # from ..datasets.infinitebench.infinitebenchendia.infinitebench_endia_gen import InfiniteBench_endia_datasets
    # # from ..datasets.infinitebench.infinitebenchenmc.infinitebench_enmc_gen import InfiniteBench_enmc_datasets
    # # from ..datasets.infinitebench.infinitebenchenqa.infinitebench_enqa_gen import InfiniteBench_enqa_datasets
    # from ..datasets.infinitebench.infinitebenchensum.infinitebench_ensum_gen import InfiniteBench_ensum_datasets
    # from ..datasets.infinitebench.infinitebenchmathcalc.infinitebench_mathcalc_gen import InfiniteBench_mathcalc_datasets
    # from ..datasets.infinitebench.infinitebenchmathfind.infinitebench_mathfind_gen import InfiniteBench_mathfind_datasets
    # from ..datasets.infinitebench.infinitebenchretrievekv.infinitebench_retrievekv_gen import InfiniteBench_retrievekv_datasets
    # from ..datasets.infinitebench.infinitebenchretrievenumber.infinitebench_retrievenumber_gen import InfiniteBench_retrievenumber_datasets
    # from ..datasets.infinitebench.infinitebenchretrievepasskey.infinitebench_retrievepasskey_gen import InfiniteBench_retrievepasskey_datasets
    # from ..datasets.infinitebench.infinitebenchzhqa.infinitebench_zhqa_gen import InfiniteBench_zhqa_datasets

    # from ..datasets.livecodebench.livecodebench_gen_b2b0fd import LCB_datasets  # noqa: F401, F403
    # from ..datasets.livecodebench.livecodebench_v6_academic import LCB_datasets as LCB_v6_datasets  # noqa: F401, F403

    # # ===== Software Engineering (Repository-level Code Understanding) =====
    # from ..datasets.swebench.swebench_gen import SWEBench_datasets  # noqa: F401, F403

    # from ..datasets.apps.apps_gen_c7893a import APPS_datasets  # noqa: F401, F403

    # # ===== Scientific Computing =====
    # # 长度大于4K
    # from ..datasets.scicode.scicode_gen import SciCode_datasets  # noqa: F401, F403

    # # ===== HumanEval Series =====
    # from ..datasets.humaneval.humaneval_gen_8e312c import humaneval_datasets
    # from ..datasets.humaneval_plus.humaneval_plus_gen_8e312c import humaneval_plus_datasets

    # from ..datasets.bigcodebench.bigcodebench_hard_instruct_gen import bigcodebench_hard_instruct_datasets
    # from ..datasets.bigcodebench.bigcodebench_hard_complete_gen import bigcodebench_hard_complete_datasets

    # # ===== MBPP Series =====
    # from ..datasets.mbpp.mbpp_gen import mbpp_datasets
    # from ..datasets.mbpp_plus.mbpp_plus_gen import mbpp_plus_datasets
    # from ..datasets.mbpp_pro.mbpp_pro_gen import mbpppro_datasets
    # from ..datasets.mbpp_cn.mbpp_cn_gen import mbpp_cn_datasets


needlebench_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

# Read parameters from environment variables with defaults
import os as _temp_os

# Debug: Print environment variables when config is loaded

SPARITY_METHOD = _temp_os.getenv('SPARITY_METHOD', 'snapkv_global')
MAX_CAPACITY_PROMPT = int(_temp_os.getenv('MAX_CAPACITY_PROMPT', '512'))

# TTFT measurement parameters
ENABLE_TTFT = _temp_os.getenv('ENABLE_TTFT', 'False').lower() in ('true', '1', 'yes')
TTFT_SAVE_DIR = _temp_os.getenv('TTFT_SAVE_DIR', './ttft_logs')
TTFT_SAVE_TO_FILE = _temp_os.getenv('TTFT_SAVE_TO_FILE', 'True').lower() in ('true', '1', 'yes')


# Attention Entropy Logging parameters
ENABLE_ENTROPY_LOGGING = _temp_os.getenv('ENABLE_ENTROPY_LOGGING', 'False').lower() in ('true', '1', 'yes')
ENTROPY_SAVE_DIR = _temp_os.getenv('ENTROPY_SAVE_DIR', './attention_entropy_logs')

print("\n" + "="*60)
print("[Config File] Final values after reading:")
print(f"  ENABLE_TTFT = {ENABLE_TTFT}")
print(f"  TTFT_SAVE_DIR = {TTFT_SAVE_DIR}")
print(f"  TTFT_SAVE_TO_FILE = {TTFT_SAVE_TO_FILE}")
print(f"  ENABLE_ENTROPY_LOGGING = {ENABLE_ENTROPY_LOGGING}")
print(f"  ENTROPY_SAVE_DIR = {ENTROPY_SAVE_DIR}")
print("="*60 + "\n")

del _temp_os  


# 将需要评测的数据集拼接成 datasets 字段
datasets = [
    # *needlebench_datasets
    *LongBench_narrativeqa_datasets,
    *LongBench_qasper_datasets,
    *LongBench_multifieldqa_en_datasets,
    *LongBench_multifieldqa_zh_datasets
    # *LongBench_multifieldqa_en_datasets,
    # *LongBench_multifieldqa_zh_datasets
    # *ruler_32k_ds

]

    

models = [
    dict(
        type=QwenAttentionConvert,
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
                topk_heads=1,  # 在这里设置 topk_gqa 算法的 k 值


                # RQA_L2_hydrid 参数
                # target_layers=[0, 1] + list(range(5, 28)),  # 除了2、3、4层外都使用L2加权，只有2、3、4层使用per-head SnapKV
                target_layers=list(range(5, 28)),
                weight_temperature=0.3,  # L2范数加权的温度参数


                # Visualization parameters
                save_indices=False,           # 启用索引保存
                save_key_states=False,       # 启用key states保存
                save_query_states=False,     # 启用query states保存（window部分）
                visualize_layer=None,        # 保存所有层（0=只保存第0层）
                max_samples_to_save=3,      # 保存前10个样本
        ),

        # TTFT measurement configuration (完全解耦，不影响原有逻辑)
        enable_ttft=ENABLE_TTFT,            # 启用/禁用 TTFT 测量
        ttft_save_dir=TTFT_SAVE_DIR,        # TTFT 日志保存目录
        ttft_save_to_file=TTFT_SAVE_TO_FILE,  # 是否保存到文件
    )
]
