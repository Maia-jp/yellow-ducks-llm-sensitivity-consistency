#!/usr/bin/env python3
"""Generate eight tokens with a small model to check the GPU setup."""
import json
import torch
import vllm
from vllm import LLM, SamplingParams


def main():
    print(json.dumps({'torch': torch.__version__, 'cuda': torch.version.cuda,
                      'vllm': vllm.__version__, 'gpu': torch.cuda.get_device_name(0)}), flush=True)
    model = LLM(model='facebook/opt-125m', dtype='float16',
                enforce_eager=True, max_model_len=128, max_num_seqs=1,
                gpu_memory_utilization=0.15, seed=42)
    results = model.generate(['The capital of Denmark is'],
                             SamplingParams(temperature=0, max_tokens=8))
    output = results[0].outputs[0]
    assert output.token_ids, 'The model returned no tokens'
    print(json.dumps({'status': 'PASS', 'model': 'facebook/opt-125m',
                      'text': output.text, 'token_ids': output.token_ids}), flush=True)

if __name__ == '__main__':
    main()
