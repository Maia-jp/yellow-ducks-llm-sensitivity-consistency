"""Register the DeepSeek model in every Python process of a DeepSeek run.

Ray starts new worker processes that do not run run_deepseek.py. Python
imports this file at start when scripts/deepseek/ is on PYTHONPATH. It acts
only when DEEPSEEK_REGISTER_PROVIDER=1, which the DeepSeek Slurm jobs set for
the runner. Same method as scripts/sitecustomize.py on branch qwen3.5-27b-jobs.
"""

import os

if os.environ.get("DEEPSEEK_REGISTER_PROVIDER") == "1":
    from deepseek_model import register_deepseek

    register_deepseek()
