#!/usr/bin/env python3
"""Run one short generation with Qwen3-4B under vLLM 0.11.0."""
import json
import os

import torch
import vllm
from vllm import LLM, SamplingParams


def main():
    model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen3-4B")
    print(json.dumps({
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "vllm": vllm.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "model": model_name,
    }), flush=True)

    model = LLM(
        model=model_name,
        dtype="bfloat16",
        enforce_eager=True,
        max_model_len=512,
        max_num_seqs=1,
        gpu_memory_utilization=0.7,
        seed=42,
    )
    results = model.generate(
        ["The capital of Denmark is"],
        SamplingParams(temperature=0, max_tokens=32),
    )
    output = results[0].outputs[0]
    assert output.token_ids, "The model returned no tokens"
    print(json.dumps({
        "status": "PASS",
        "model": model_name,
        "text": output.text,
        "token_ids": output.token_ids,
    }), flush=True)


if __name__ == "__main__":
    main()
