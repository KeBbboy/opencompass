"""SWE-bench dataset configuration."""

from opencompass.openicl.icl_prompt_template import PromptTemplate
from opencompass.openicl.icl_retriever import ZeroRetriever
from opencompass.openicl.icl_inferencer import GenInferencer
from opencompass.datasets import (
    SWEBenchLiteDataset,
    SWEBenchVerifiedDataset,
    SWEBenchDataset,
    SWEBenchLiteEvaluator,
    SWEBenchVerifiedEvaluator,
    SWEBenchEvaluator,
)


# SWE-bench Lite (300 instances)
swebench_lite_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='patch',
)

swebench_lite_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(
                    role='SYSTEM',
                    fallback_role='HUMAN',
                    prompt='You are an expert software engineer. You will be given a GitHub issue and need to provide a patch to fix it. Output the patch in unified diff format.',
                ),
            ],
            round=[
                dict(
                    role='HUMAN',
                    prompt='{prompt}',
                )
            ],
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=2048)
)

swebench_lite_eval_cfg = dict(
    evaluator=dict(type=SWEBenchLiteEvaluator),
    pred_role='BOT',
)

SWEBenchLite_dataset = dict(
    type=SWEBenchLiteDataset,
    abbr='swebench_lite',
    path='princeton-nlp/SWE-bench_Lite',
    reader_cfg=swebench_lite_reader_cfg,
    infer_cfg=swebench_lite_infer_cfg,
    eval_cfg=swebench_lite_eval_cfg,
)


# SWE-bench Verified (500 instances)
swebench_verified_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='patch',
)

swebench_verified_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(
                    role='SYSTEM',
                    fallback_role='HUMAN',
                    prompt='You are an expert software engineer. You will be given a GitHub issue and need to provide a patch to fix it. Output the patch in unified diff format.',
                ),
            ],
            round=[
                dict(
                    role='HUMAN',
                    prompt='{prompt}',
                )
            ],
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=2048)
)

swebench_verified_eval_cfg = dict(
    evaluator=dict(type=SWEBenchVerifiedEvaluator),
    pred_role='BOT',
)

SWEBenchVerified_dataset = dict(
    type=SWEBenchVerifiedDataset,
    abbr='swebench_verified',
    path='princeton-nlp/SWE-bench_Verified',
    reader_cfg=swebench_verified_reader_cfg,
    infer_cfg=swebench_verified_infer_cfg,
    eval_cfg=swebench_verified_eval_cfg,
)


# SWE-bench Full (2294 instances)
swebench_full_reader_cfg = dict(
    input_columns=['prompt'],
    output_column='patch',
)

swebench_full_infer_cfg = dict(
    prompt_template=dict(
        type=PromptTemplate,
        template=dict(
            begin=[
                dict(
                    role='SYSTEM',
                    fallback_role='HUMAN',
                    prompt='You are an expert software engineer. You will be given a GitHub issue and need to provide a patch to fix it. Output the patch in unified diff format.',
                ),
            ],
            round=[
                dict(
                    role='HUMAN',
                    prompt='{prompt}',
                )
            ],
        )
    ),
    retriever=dict(type=ZeroRetriever),
    inferencer=dict(type=GenInferencer, max_out_len=2048)
)

swebench_full_eval_cfg = dict(
    evaluator=dict(type=SWEBenchEvaluator),
    pred_role='BOT',
)

SWEBenchFull_dataset = dict(
    type=SWEBenchDataset,
    abbr='swebench_full',
    path='princeton-nlp/SWE-bench',
    reader_cfg=swebench_full_reader_cfg,
    infer_cfg=swebench_full_infer_cfg,
    eval_cfg=swebench_full_eval_cfg,
)


# Export all datasets
SWEBench_datasets = [
    SWEBenchLite_dataset,
    SWEBenchVerified_dataset,
    # SWEBenchFull_dataset,  # Commented out by default (large dataset)
]
