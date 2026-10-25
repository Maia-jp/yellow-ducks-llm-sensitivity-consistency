#!/usr/bin/env python3
"""Check the installed GPU runtime without downloading a model."""
import faulthandler
import sys
import time
from importlib.metadata import version

faulthandler.enable()
faulthandler.dump_traceback_later(60, repeat=True)
started = time.monotonic()

def report(message):
    print(f"[{time.monotonic() - started:.1f}s] {message}", flush=True)

report(f"Python: {sys.version}")
for package in ("torch", "vllm", "openai", "numpy", "packaging"):
    report(f"{package}: {version(package)}")
report("Importing torch")
import torch
report(f"Torch CUDA build: {torch.version.cuda}")
assert torch.cuda.is_available(), "CUDA is unavailable"
report(f"GPU: {torch.cuda.get_device_name(0)}")
x = torch.ones(4, device="cuda")
assert (x + x).tolist() == [2.0] * 4
report("GPU tensor test passed")
from vllm import LLM
import vllm._C
report("vLLM imports and compiled extension passed")
faulthandler.cancel_dump_traceback_later()
report("Setup checks passed. Model inference was not tested.")
