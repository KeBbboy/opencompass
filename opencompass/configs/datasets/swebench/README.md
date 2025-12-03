# SWE-bench Dataset Configuration

SWE-bench is a benchmark for evaluating large language models on real-world software engineering tasks, specifically GitHub issue resolution.

## Overview

- **Task**: Generate patches to fix GitHub issues
- **Input**: Repository info, problem statement, and optional hints
- **Output**: Patch in unified diff format
- **Evaluation**: Patch similarity (simplified) or test execution (full)

## Available Datasets

### 1. SWE-bench Lite (Recommended for testing)
- **Instances**: 300 carefully selected problems
- **Path**: `princeton-nlp/SWE-bench_Lite`
- **Description**: Curated subset for faster evaluation

### 2. SWE-bench Verified
- **Instances**: 500 verified solvable problems
- **Path**: `princeton-nlp/SWE-bench_Verified`
- **Description**: Validated by real software engineers

### 3. SWE-bench Full
- **Instances**: 2,294 problems
- **Path**: `princeton-nlp/SWE-bench`
- **Description**: Complete benchmark (disabled by default)

## Configuration Parameters

### Inference Settings

```python
max_out_len=2048  # Maximum output tokens for patch generation
```

### Reader Configuration

```python
input_columns=['prompt']   # Input field containing formatted problem
output_column='patch'      # Reference patch for evaluation
```

## Usage

### Basic Usage

The dataset is already imported in `run_Longbench.py`:

```python
from ..datasets.swebench.swebench_gen import SWEBench_datasets
```

By default, `SWEBench_datasets` includes:
- SWE-bench Lite
- SWE-bench Verified

### Running Evaluation

To include SWE-bench in your evaluation, add it to the datasets list in your config:

```python
datasets = [
    *SWEBench_datasets,
    # ... other datasets
]
```

### Customizing Output Length

If you need to generate longer patches:

```python
# In swebench_gen.py
inferencer=dict(type=GenInferencer, max_out_len=4096)  # Increase for longer patches
```

## Evaluation Metrics

### Current Implementation (Simplified)

- **patch_similarity**: Text similarity between predicted and gold patches (0-100%)
- **exact_match**: Percentage of exact matches after normalization (0-100%)
- **valid_predictions**: Number of predictions containing valid patch structure

### Full Evaluation (Requires SWE-bench Harness)

For accurate evaluation using test execution:
1. Install the official SWE-bench evaluation harness
2. Run tests in Docker containers
3. Measure FAIL_TO_PASS and PASS_TO_PASS test resolution

**Note**: The current implementation provides approximate scores based on patch similarity. For production evaluation, use the official SWE-bench infrastructure.

## Dataset Fields

Each instance contains:
- `instance_id`: Unique identifier
- `repo`: Repository name
- `problem_statement`: Issue description
- `hints_text`: Optional hints from issue comments
- `patch`: Gold standard patch
- `base_commit`: Commit hash for code state
- `test_patch`: Associated test changes
- `FAIL_TO_PASS`: Tests that should pass after fix
- `PASS_TO_PASS`: Tests that should remain passing

## Example Prompt Format

```
You are tasked with fixing a software bug in the following repository.

**Repository**: owner/repo-name
**Instance ID**: repo__issue-123

**Problem Statement**:
[Issue description with code context]

**Hints**:
[Optional comments from issue discussion]

Please provide a patch to fix this issue. Output only the patch in unified diff format.

**Patch**:
```

## Performance Expectations

Based on SWE-bench leaderboards (2025):

| Model | SWE-bench Lite | SWE-bench Verified |
|-------|----------------|-------------------|
| GPT-4 | ~40-50% | ~70%+ |
| Claude Opus | ~30-40% | ~60%+ |
| Open source | ~10-20% | ~20-30% |

## References

- [SWE-bench Website](https://www.swebench.com/)
- [SWE-bench GitHub](https://github.com/SWE-bench/SWE-bench)
- [Dataset on HuggingFace](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Lite)
- [SWE-bench Paper](https://arxiv.org/abs/2310.06770)
