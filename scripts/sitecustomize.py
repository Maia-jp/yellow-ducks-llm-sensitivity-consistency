"""Register the Qwen model in every Python process of a Qwen run.

Ray starts new worker processes that do not run scripts/run_qwen.py. Python
imports this file at start when scripts/ is on PYTHONPATH. It acts only when
QWEN_REGISTER_PROVIDER=1, which the Qwen Slurm jobs set for the runner.
"""

import os

if os.environ.get("QWEN_REGISTER_PROVIDER") == "1":
    from qwen_model import register_qwen

    register_qwen()
