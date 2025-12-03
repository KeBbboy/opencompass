"""SWE-bench Dataset implementation for OpenCompass."""

from datasets import load_dataset

from opencompass.registry import LOAD_DATASET
from opencompass.utils import get_data_path

from ..base import BaseDataset


@LOAD_DATASET.register_module()
class SWEBenchDataset(BaseDataset):
    """SWE-bench Dataset for software engineering tasks.

    This dataset contains real-world GitHub issues paired with their solutions.
    Models are tasked with generating patches to resolve the issues.
    """

    @staticmethod
    def load(
        path: str = 'princeton-nlp/SWE-bench_Lite',
        split: str = 'test',
        local_mode: bool = False,
    ):
        """Load SWE-bench dataset.

        Args:
            path: Dataset path or HuggingFace dataset name
            split: Dataset split to use ('test', 'dev', 'train')
            local_mode: Whether to load from local path

        Returns:
            Loaded dataset with problem statements and metadata
        """
        if local_mode:
            path = get_data_path(path, local_mode=True)
            dataset = load_dataset('json', data_files=path, split=split)
        else:
            # Load from HuggingFace
            dataset = load_dataset(path, split=split)

        # Transform dataset to OpenCompass format
        def transform(item):
            """Transform dataset item to required format."""
            # Combine instance info for the prompt
            prompt = f"""You are tasked with fixing a software bug in the following repository.

**Repository**: {item['repo']}
**Instance ID**: {item['instance_id']}

**Problem Statement**:
{item['problem_statement']}

"""
            # Add hints if available
            if item.get('hints_text', '').strip():
                prompt += f"""**Hints**:
{item['hints_text']}

"""

            prompt += """Please provide a patch to fix this issue. Output only the patch in unified diff format.

**Patch**:
"""

            return {
                'prompt': prompt,
                'instance_id': item['instance_id'],
                'repo': item['repo'],
                'base_commit': item['base_commit'],
                'problem_statement': item['problem_statement'],
                'patch': item['patch'],  # Gold patch for evaluation
                'test_patch': item.get('test_patch', ''),
                'FAIL_TO_PASS': item.get('FAIL_TO_PASS', '[]'),
                'PASS_TO_PASS': item.get('PASS_TO_PASS', '[]'),
                'hints_text': item.get('hints_text', ''),
            }

        dataset = dataset.map(transform)
        return dataset


@LOAD_DATASET.register_module()
class SWEBenchLiteDataset(SWEBenchDataset):
    """SWE-bench Lite Dataset (300 curated instances)."""

    @staticmethod
    def load(
        path: str = 'princeton-nlp/SWE-bench_Lite',
        split: str = 'test',
        local_mode: bool = False,
    ):
        return SWEBenchDataset.load(path, split, local_mode)


@LOAD_DATASET.register_module()
class SWEBenchVerifiedDataset(SWEBenchDataset):
    """SWE-bench Verified Dataset (500 verified instances)."""

    @staticmethod
    def load(
        path: str = 'princeton-nlp/SWE-bench_Verified',
        split: str = 'test',
        local_mode: bool = False,
    ):
        return SWEBenchDataset.load(path, split, local_mode)
