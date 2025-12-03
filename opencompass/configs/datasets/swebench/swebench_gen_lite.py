"""SWE-bench Lite entry point configuration."""

from mmengine.config import read_base

with read_base():
    from .swebench_gen import SWEBench_datasets  # noqa: F401, F403
