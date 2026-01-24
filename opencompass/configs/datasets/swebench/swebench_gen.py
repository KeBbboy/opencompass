from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import SWEBenchDataset, SWEBenchEvaluator

# SWE-bench Dataset Configuration
# For more information: https://www.swebench.com/

SWEBench_reader_cfg = dict(
    input_columns=['problem_statement', 'repo', 'base_commit'],
    output_column='patch',
    train_split='test'
)

SWEBench_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(
                    role='SYSTEM',
                    fallback_role='HUMAN',
                    prompt='You are an expert software engineer. Your task is to analyze the given issue and generate a patch to fix it.'
                ),
            ],
            round=[
                dict(
                    role='HUMAN',
                    prompt='Repository: {repo}\nBase Commit: {base_commit}\n\nIssue Description:\n{problem_statement}\n\nPlease provide a patch to fix this issue. Your response should contain the code changes needed.'
                ),
                dict(role='BOT', prompt=''),
            ],
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=2048)
)

SWEBench_eval_cfg = dict(
    evaluator=dict(type=SWEBenchEvaluator),
    pred_role='BOT'
)

# SWE-bench_Lite: Smaller, more manageable subset (~300 samples)
SWEBench_lite_datasets = [
    dict(
        type=SWEBenchDataset,
        abbr='SWEBench_Lite',
        path='princeton-nlp/SWE-bench_Lite',
        split='test',
        reader_cfg=SWEBench_reader_cfg,
        infer_cfg=SWEBench_infer_cfg,
        eval_cfg=SWEBench_eval_cfg,
    )
]

# SWE-bench_Verified: Human-validated subset (500 samples)
SWEBench_verified_datasets = [
    dict(
        type=SWEBenchDataset,
        abbr='SWEBench_Verified',
        path='princeton-nlp/SWE-bench_Verified',
        split='test',
        reader_cfg=SWEBench_reader_cfg,
        infer_cfg=SWEBench_infer_cfg,
        eval_cfg=SWEBench_eval_cfg,
    )
]

# Default: Use Lite version (recommended for initial testing)
SWEBench_datasets = SWEBench_lite_datasets
