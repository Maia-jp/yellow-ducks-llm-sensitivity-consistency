"""Load the Qwen provider in Ray worker processes during HPC experiments."""

import os

if os.environ.get("QWEN_REGISTER_PROVIDER") == "1":
    from qwen_model import register_qwen

    register_qwen()
