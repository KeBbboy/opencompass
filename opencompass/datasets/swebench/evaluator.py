"""SWE-bench Evaluator implementation."""

import re
from difflib import SequenceMatcher
from typing import List

from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS


def extract_patch_from_text(text: str) -> str:
    """Extract patch content from model output.

    Args:
        text: Model output text

    Returns:
        Extracted patch string
    """
    # Try to extract content between ```diff or ```patch markers
    patterns = [
        r'```(?:diff|patch)\n(.*?)\n```',
        r'```\n(.*?)\n```',
        r'(?:^|\n)((?:diff --git|---|\+\+\+).*?)(?=\n\n|\Z)',
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
        if match:
            return match.group(1).strip()

    # If no markers found, try to find diff-like content
    lines = text.split('\n')
    patch_lines = []
    in_patch = False

    for line in lines:
        if line.startswith(('diff --git', '---', '+++')):
            in_patch = True
        if in_patch:
            if line.startswith(('+', '-', '@', 'diff', '---', '+++')):
                patch_lines.append(line)
            elif line.strip() == '':
                continue
            elif patch_lines:
                # End of patch section
                break

    if patch_lines:
        return '\n'.join(patch_lines)

    # Return original text if no patch structure found
    return text.strip()


def normalize_patch(patch: str) -> str:
    """Normalize patch for comparison.

    Args:
        patch: Patch string

    Returns:
        Normalized patch
    """
    # Remove leading/trailing whitespace
    patch = patch.strip()

    # Normalize line endings
    patch = patch.replace('\r\n', '\n')

    # Remove diff metadata lines that may vary
    lines = patch.split('\n')
    normalized_lines = []

    for line in lines:
        # Skip lines that are just metadata
        if line.startswith('diff --git'):
            continue
        if line.startswith('index '):
            continue

        normalized_lines.append(line)

    return '\n'.join(normalized_lines)


def compute_patch_similarity(pred_patch: str, gold_patch: str) -> float:
    """Compute similarity between predicted and gold patches.

    Args:
        pred_patch: Predicted patch
        gold_patch: Gold standard patch

    Returns:
        Similarity score between 0 and 1
    """
    pred_normalized = normalize_patch(pred_patch)
    gold_normalized = normalize_patch(gold_patch)

    if not pred_normalized or not gold_normalized:
        return 0.0

    # Use SequenceMatcher for line-by-line comparison
    matcher = SequenceMatcher(None, gold_normalized, pred_normalized)
    return matcher.ratio()


@ICL_EVALUATORS.register_module()
class SWEBenchEvaluator(BaseEvaluator):
    """Evaluator for SWE-bench dataset.

    This is a simplified evaluator that compares generated patches with
    gold patches using text similarity. For full evaluation, use the
    official SWE-bench evaluation harness which runs actual tests.

    Note:
        This evaluator provides an approximate score based on patch similarity.
        For accurate results, use the official SWE-bench evaluation infrastructure
        which executes tests in proper environments.
    """

    def __init__(self):
        super().__init__()

    def score(self, predictions: List, references: List) -> dict:
        """Score predictions against references.

        Args:
            predictions: List of model predictions (patches)
            references: List of reference patches

        Returns:
            Dictionary with evaluation metrics
        """
        if len(predictions) != len(references):
            return {
                'error': 'Predictions and references length mismatch',
                'patch_similarity': 0.0,
                'exact_match': 0.0,
            }

        total_similarity = 0.0
        exact_matches = 0
        valid_predictions = 0

        for pred, ref in zip(predictions, references):
            # Extract patch from prediction
            pred_patch = extract_patch_from_text(str(pred))

            # Get reference patch
            if isinstance(ref, dict):
                ref_patch = ref.get('patch', '')
            else:
                ref_patch = str(ref)

            if pred_patch and ref_patch:
                valid_predictions += 1

                # Compute similarity
                similarity = compute_patch_similarity(pred_patch, ref_patch)
                total_similarity += similarity

                # Check exact match (after normalization)
                if normalize_patch(pred_patch) == normalize_patch(ref_patch):
                    exact_matches += 1

        # Compute metrics
        n = len(predictions)
        avg_similarity = (
            total_similarity / valid_predictions if valid_predictions > 0 else 0.0
        )
        exact_match_rate = exact_matches / n if n > 0 else 0.0

        return {
            'patch_similarity': avg_similarity * 100,  # Convert to percentage
            'exact_match': exact_match_rate * 100,  # Convert to percentage
            'valid_predictions': valid_predictions,
            'total_samples': n,
        }


@ICL_EVALUATORS.register_module()
class SWEBenchLiteEvaluator(SWEBenchEvaluator):
    """Evaluator for SWE-bench Lite dataset."""

    pass


@ICL_EVALUATORS.register_module()
class SWEBenchVerifiedEvaluator(SWEBenchEvaluator):
    """Evaluator for SWE-bench Verified dataset."""

    pass
