# DeepSeek-R1 7B results

Results of the paper's original code with `deepseek-r1:7b`, in the same layout as `mixtral8x7b-Instruct-v0.1_RESULTS/`.
The jobs are in `jobs/deepseek-r1-7b/` (see its README).

## How they were made

| Setting | Value |
|---|---|
| Model | `deepseek-r1:7b` from the Ollama library, ID `755ced02ce7b` (Q4_K_M), context 8192 tokens |
| Server | Ollama 0.40.4 on `asus1` (ARM, NVIDIA GB10), private server, 16 parallel requests |
| Sampling | Temperature 0, seed 42 |
| Prompts | Unchanged notebook settings: 30 rephrasings (the original + 29 written by DeepSeek), 1 answer each, 3 prompt types |
| Reasoning | On. DeepSeek-R1 writes about 250 reasoning tokens before each answer. It could not be turned off (tested: `reasoning_effort=none`, `think=false` on the OpenAI API and on Ollama's own API). Only the final answer goes to the paper's label search. |

## Folders

| Folder | Dataset | Input file | Job | Date |
|---|---|---|---|---|
| `TREC_RESULTS/` | TREC, 500 samples | `TREC/test.txt` | 182120 | 2026-11-16/17 |
| `cb_test_labeled/CB_RESULTS/` | CB, 250 samples | `CB/test_labeled.jsonl` (rebuilt test labels) | 182119 | 2026-11-16/17 |
| `CB_RESULTS/` | CB, 250 samples | `CB/train.jsonl`, the file of the Mixtral run | 183238 | 2026-11-19 |

**Compare with Mixtral only on `CB_RESULTS/`.** The Mixtral CB run used `train.jsonl` (all 250 labels match). `test_labeled.jsonl` has other sentences (123 of 250 labels agree), so `cb_test_labeled/` is an extra held-out result, not a like-for-like comparison.

RTE, DBPedia and WoS were not run in full: at about 1.4 requests per second on `asus1`, each would take 36 to 45 hours.

## Known issue: empty rephrasings

The paper code asks the model itself to rephrase the task instruction. A few times DeepSeek used all of its token limit (2048) for reasoning and wrote no rephrasing, so that prompt variant has an empty instruction line. The rest of the prompt (class list, the text) is unchanged.

| Run | Empty variants (of 30) |
|---|---|
| TREC | 1 (position 18) |
| CB `test_labeled` | 2 (positions 7 and 22) |

Without these variants every metric changes by at most 0.01, so the conclusions do not depend on them.

## Metrics

`scripts/deepseek/compare_models.py` computes sensitivity, consistency, micro-F1 and the N/A share with the formulas of `mixtral8x7b-Instruct-v0.1_RESULTS/Results_and_paper_comparison.ipynb` (branch `main`). It reproduces all 15 Mixtral values of that notebook.

```bash
python3 scripts/deepseek/compare_models.py --out deepseek-r1-7b_RESULTS/comparison \
    --model "Mixtral-8x7B=mixtral8x7b-Instruct-v0.1_RESULTS" \
    --model "DeepSeek-R1-7B=deepseek-r1-7b_RESULTS"
```
