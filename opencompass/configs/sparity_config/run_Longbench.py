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

    # # 长输入数据
    # from .datasets.ruler.ruler_1m_gen import ruler_datasets as ruler_1m_ds
    # from .datasets.ruler.ruler_4k_gen import ruler_datasets as ruler_4k_ds
    # from .datasets.ruler.ruler_8k_gen import ruler_datasets as ruler_8k_ds
    # from .datasets.ruler.ruler_16k_gen import ruler_datasets as ruler_16k_ds
    # from .datasets.ruler.ruler_32k_gen import ruler_datasets as ruler_32k_ds
    # from .datasets.ruler.ruler_64k_gen import ruler_datasets as ruler_64k_ds
    # from .datasets.ruler.ruler_128k_gen import ruler_datasets as ruler_128k_ds

    # from  .datasets.ruler.ruler_cwe_gen import cwe_datasets as cwe  # CW

    # =====================32k ========================
    # from .datasets.needlebench.needlebench_32k.needlebench_multi_reasoning_32k import needlebench_2needle_en_datasets as needlebench_multi_2needle_en_datasets
    # from .datasets.needlebench.needlebench_32k.needlebench_multi_reasoning_32k import needlebench_3needle_en_datasets as needlebench_multi_3needle_en_datasets
    # from .datasets.needlebench.needlebench_32k.needlebench_multi_reasoning_32k import needlebench_4needle_en_datasets as needlebench_multi_4needle_en_datasets
    # from .datasets.needlebench.needlebench_32k.needlebench_multi_reasoning_32k import needlebench_5needle_en_datasets as needlebench_multi_5needle_en_datasets
    # from .datasets.needlebench.needlebench_32k.needlebench_multi_reasoning_32k import needlebench_2needle_zh_datasets as needlebench_multi_2needle_zh_datasets
    # from .datasets.needlebench.needlebench_32k.needlebench_multi_reasoning_32k import needlebench_3needle_zh_datasets as needlebench_multi_3needle_zh_datasets
    # from .datasets.needlebench.needlebench_32k.needlebench_multi_reasoning_32k import needlebench_4needle_zh_datasets as needlebench_multi_4needle_zh_datasets
    # from ..datasets.needlebench.needlebench_32k.needlebench_multi_reasoning_32k import needlebench_5needle_zh_datasets as needlebench_multi_5needle_zh_datasets
    # from .datasets.needlebench.needlebench_32k.needlebench_single_32k import needlebench_en_datasets as needlebench_origin_en_datasets
    # from .datasets.needlebench.needlebench_32k.needlebench_single_32k import needlebench_zh_datasets as needlebench_origin_zh_datasets
    # from ..datasets.needlebench.needlebench_32k.needlebench_multi_retrieval_32k import needlebench_en_datasets as needlebench_parallel_en_datasets
    # from ..datasets.needlebench.needlebench_32k.needlebench_multi_retrieval_32k import needlebench_zh_datasets as needlebench_parallel_zh_datasets

    # from .datasets.livecodebench.livecodebench_gen_a4f90b import LCB_datasets 
    
    # from .datasets.bbeh.bbeh_gen import bbeh_datasets  

    # from .datasets.livereasonbench.livereasonbench_gen_f990de import livereasonbench_datasets



    # from ..datasets.infinitebench.infinitebench import infinitebench_datasets
    # from ..datasets.infinitebench.infinitebenchensum.infinitebench_ensum_gen import InfiniteBench_ensum_datasets

needlebench_datasets = sum((v for k, v in locals().items() if k.endswith('_datasets')), [])

# Read parameters from environment variables with defaults
import os as _temp_os
SPARITY_METHOD = _temp_os.getenv('SPARITY_METHOD', 'snapkv')
MAX_CAPACITY_PROMPT = int(_temp_os.getenv('MAX_CAPACITY_PROMPT', '512'))
del _temp_os  


# 将需要评测的数据集拼接成 datasets 字段
datasets = [
    *LongBench_narrativeqa_datasets,
    # *LongBench_qasper_datasets,
    # *LongBench_multifieldqa_en_datasets,
    # *LongBench_multifieldqa_zh_datasets
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
        )

    )
]
