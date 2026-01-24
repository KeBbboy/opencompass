from datasets import Dataset, load_dataset

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class SWEBenchDataset(BaseDataset):
    """SWE-bench Dataset for evaluating code generation on GitHub issues.

    This dataset contains real-world software engineering tasks from GitHub issues.
    Each task requires understanding a codebase and generating patches to fix issues.
    """

    @staticmethod
    def load(path: str = 'princeton-nlp/SWE-bench_Lite', split: str = 'test', **kwargs):
        """Load SWE-bench dataset from HuggingFace.

        Args:
            path: HuggingFace dataset path. Options:
                - 'princeton-nlp/SWE-bench_Lite' (default, ~300 samples)
                - 'princeton-nlp/SWE-bench_Verified' (500 human-validated samples)
                - 'princeton-nlp/SWE-bench' (full dataset, ~2300 test samples)
            split: Dataset split to use ('test', 'dev', 'train')
        """
        try:
            dataset = load_dataset(path, split=split, trust_remote_code=True)
        except Exception as e:
            print(f"Failed to load dataset from {path}: {e}")
            print("Creating empty dataset as fallback...")
            dataset = Dataset.from_dict({
                'instance_id': [],
                'repo': [],
                'problem_statement': [],
                'base_commit': [],
                'patch': [],
            })

        return dataset


@ICL_EVALUATORS.register_module()
class SWEBenchEvaluator(BaseEvaluator):
    """Evaluator for SWE-bench.

    Note: Full evaluation requires running unit tests in isolated environments.
    This is a simplified evaluator that checks for basic patch format.
    For complete evaluation, use the official SWE-bench evaluation harness.
    """

    def score(self, predictions, references):
        """Score the predictions.

        Args:
            predictions: List of model predictions (patch code)
            references: List of reference data (ground truth patches)

        Returns:
            Dict containing evaluation metrics
        """
        if not predictions or len(predictions) == 0:
            return {'pass@1': 0.0}

        # Count valid patch formats
        valid_patches = 0
        total = len(predictions)

        for pred in predictions:
            if pred and isinstance(pred, str):
                # Check if prediction contains code (basic heuristic)
                if any(keyword in pred.lower() for keyword in ['def ', 'class ', 'import ', '```python']):
                    valid_patches += 1

        result = {
            'valid_patch_rate': 100.0 * valid_patches / total if total > 0 else 0.0,
            'total_samples': total,
        }

        return result
