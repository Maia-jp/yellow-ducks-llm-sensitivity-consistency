# Reproduction guide

This guide explains how to replicate the experiments of
[What Did I Do Wrong? Quantifying LLMs’ Sensitivity and Consistency to Prompt Engineering](https://aclanthology.org/2025.naacl-long.73/) (NAACL 2025).
This branch holds the foundation: the paper's code, the verified datasets, and the reference values. It holds no model results yet.

## Paper and code

- [Paper PDF](paper.pdf), [BibTeX](references.bib), [DOI](https://doi.org/10.18653/v1/2025.naacl-long.73)
- Original code: https://github.com/nec-research/sensitivity-consistency-LLM
- Our copy of the code: [../sensitivity-consistency-LLM/](../sensitivity-consistency-LLM/). It comes from commit
  [`657355688d4b58a6e3a2ac20484518c5d402c8d8`](https://github.com/nec-research/sensitivity-consistency-LLM/tree/657355688d4b58a6e3a2ac20484518c5d402c8d8).
  NEC Laboratories Europe GmbH developed it. The [NEC license](../sensitivity-consistency-LLM/LICENSE.txt) applies.
- One change to the code: `LLM TREC.ipynb` is a repaired copy, because the upstream file is cut and is not valid JSON.

The notebooks contain the authors' outputs from 2024-08. Those outputs are not results from our runs.

## What the paper runs

| Item | Paper value |
|---|---|
| Datasets | TREC, CB, RTE, DBPedia, WoS-46985 |
| Open models | Llama-3-70B-Instruct and Mixtral-8x7B-Instruct-v0.1 |
| Closed models | GPT-3.5-turbo-0125 and GPT-4o-2024-08-06 (not needed for our work) |
| Prompt strategies | Simple, Detail (notebook name `instruct`), 1-shot (notebook name `fewshot`) |
| Prompt variants | Q = 30 rephrasings of the task description, one answer for each prompt |
| Temperature | 0 for the rephrasing LLM and the answering LLM |
| Seed | 42 for requests at temperature 0 |
| Hardware in the paper | 4 Tesla V100 GPUs (32 GB each) |

## Datasets

The data is in `sensitivity-consistency-LLM/data/`. Each dataset folder has a `README.md`.
Read the [data index](../sensitivity-consistency-LLM/data/README.md) first. The full check against the official sources is in [datasets/README.md](datasets/README.md).

| Dataset | Paper | What the notebook evaluates | Cut |
|---|---|---|---|
| TREC | 500 test | `TREC/test.txt`, 500 rows | none |
| CB | 250 test | `CB/train.jsonl`, 250 rows (see the CB split note) | none |
| RTE | 2490 test | `RTE/train.jsonl`, 2490 rows | none |
| DBPedia | 2000 balanced test | 2000 of 70,000 rows | seed 42 |
| WoS | 2000 balanced | 2000 of 46,985 rows | seed 42 |

Check the data before every run (on ITU HPC, run it with `srun` or `sbatch`, never on the login node):

```bash
python3 scripts/fetch_datasets.py --verify-only
python3 scripts/fetch_datasets.py                 # restore damaged files, extract WOS46985/X.txt
```

**CB split.** The paper says "250 test samples". The released notebook reads `train.jsonl`. The official test file has no labels, so the paper's CB numbers most likely come from `train.jsonl`.
Run CB on `train.jsonl` to compare with Table 1. Run it on `test_labeled.jsonl` (labels rebuilt from the original annotations) for a held-out test.
Do not use `test_labeled.jsonl` to tune prompts or to choose models. See [CB/README.md](../sensitivity-consistency-LLM/data/CB/README.md).

## Metrics

- **Sensitivity** measures how much a model changes its class predictions across prompt rephrasings. Lower values mean less change. The notebooks print it as "Avg Entropy over samples".
- **Consistency** measures how similar the model's prediction distributions are for examples in the same class. Higher values mean more similarity.
- **Micro-F1** measures classification performance across all examples. Higher values mean better performance.

## Reference results (paper, Table 1)

Each cell is Sensitivity / Consistency / Micro-F1. Compare all three metrics.

**Mixtral-8x7B-Instruct-v0.1**

| Dataset | Simple | Detail | 1-shot |
|---|---|---|---|
| TREC | 0.223 / 0.657 / 0.736 | 0.182 / 0.709 / 0.728 | 0.166 / 0.640 / 0.744 |
| CB | 0.203 / 0.625 / 0.672 | 0.013 (see note) / 0.695 / 0.788 | 0.017 (see note) / 0.600 / 0.652 |
| RTE | 0.195 / 0.669 / 0.814 | 0.176 / 0.676 / 0.756 | 0.046 / 0.754 / 0.847 |
| DBPedia | 0.053 / 0.802 / 0.804 | 0.054 / 0.808 / 0.862 | 0.069 / 0.773 / 0.819 |
| WoS | 0.112 / 0.483 / 0.606 | 0.124 / 0.469 / 0.599 | 0.142 / 0.472 / 0.566 |

**Llama-3-70B-Instruct**

| Dataset | Simple | Detail | 1-shot |
|---|---|---|---|
| TREC | 0.127 / 0.693 / 0.848 | 0.141 / 0.694 / 0.824 | 0.095 / 0.686 / 0.846 |
| CB | 0.018 / 0.956 / 0.920 | 0.016 / 0.963 / 0.924 | 0.014 / 0.962 / 0.936 |
| RTE | 0.157 / 0.701 / 0.569 | 0.213 / 0.727 / 0.509 | 0.048 / 0.755 / 0.818 |
| DBPedia | 0.018 / 0.932 / 0.948 | 0.058 / 0.901 / 0.740 | 0.030 / 0.924 / 0.827 |
| WoS | 0.043 / 0.479 / 0.627 | 0.034 / 0.480 / 0.623 | 0.030 / 0.468 / 0.580 |

**Notes on Table 1** (checked on 2026-10-27, see [datasets/README.md](datasets/README.md)):

- The notebooks keep the authors' stored sensitivity values. 45 of 48 values match Table 1 within 0.001. TREC has no stored output because its notebook is cut.
- **CB, Mixtral, Detail and 1-shot:** the stored values are 0.1255 and 0.1740, not 0.013 and 0.017. Appendix D of the paper gives a standard deviation of 0.188 for both cells. A value between 0 and 1 with mean 0.013 cannot have that deviation. Table 1 most likely has a decimal error. **Use 0.125 and 0.174 as the target.**
- **WoS, Llama3, 1-shot:** the stored value is 0.0353 and Table 1 says 0.030. We have no explanation.
- The paper's rows for GPT-3.5 and GPT-4o are in the paper. We do not reproduce them.

## Setup

Run these commands from `sensitivity-consistency-LLM/`. The code needs Python 3.10 and an OpenAI-compatible model server (vLLM or another).

1. Create an environment and install the pinned packages (`requirements.txt`: langchain 0.2.3, ray 2.6.3, numpy 1.25.2, pandas 2.2.2, scikit-learn 1.3.0, and others):

   ```bash
   python3.10 -m venv .venv/llm
   source .venv/llm/bin/activate
   pip install -r requirements.txt
   ```

   On ITU HPC, run this inside a Slurm job (`sbatch` or `srun`). ITU forbids `pip` on the login node.
2. Serve the model. The code uses the model name `mixtral` for Mixtral-8x7B-Instruct-v0.1.
   The Llama entry uses the model name `llama3instruct` and the stop token ID 128009.
3. Set `MIXTRAL_openai_api_base` (and `LLAMA3_openai_api_base` for Llama) in `constants.py`. Both are `"INSERT HERE"` now.
4. Check the data (see Datasets).
5. Open the notebook of one dataset. In the grid-search cell set `llms = ['mixtral']`. The notebooks default to `['llama3', 'mixtral']`.
6. Run the cells in order.

The notebooks set `Qs = [30]`, `As = [1]`, and both temperatures to 0. Requests at temperature 0 send `seed: 42`.
The evaluator runs sample jobs in parallel with Ray. A full run makes many model requests and takes a long time.

## Run plan

1. Start with TREC (500 samples). Run the full dataset before you start the other four. Do not report a partial run as a full reproduction.
2. Run CB on both files (see the CB split above) and label each result with the file name.
3. Run RTE, DBPedia and WoS.
4. Compare Sensitivity, Consistency and Micro-F1 with Table 1 and its notes.

Rules for a clean run:

- Delete any `data/*_RESULTS/` folder from an older run. The notebooks reuse `data/<DATASET>_RESULTS/data.json` (the sampled data). The evaluator reuses `results_<prompt>.json` in that folder to resume a stopped run.
- Keep the result files of a finished run. Remove them only to start that experiment again.
- A smoke test (a small sample or fewer prompt variants) does not reproduce the paper. Label it as a smoke test.
- The TREC notebook has no plot cells after the grid search (they were lost). Compute the metrics from the saved result files, or copy the plot cells from another notebook (we did not test this).

## Results log

Add the measured values after each full run. Include the dataset, the data file, the model name and version, the prompt strategy, the three metrics, the run date and the server settings.

| Dataset | Data file | Model and version | Strategy | Sensitivity | Consistency | Micro-F1 | Notes |
|---|---|---|---|---:|---:|---:|---|
| | | | | | | | |

## Limits

- The paper calls its evaluation a demonstration, not a broad benchmark.
- Mixtral and Llama-3-70B need large GPU memory. The paper used 4 V100 GPUs with 32 GB each.
- Differences from Table 1 can come from the model version, the server and its settings, and the model's response behavior.
- The CB test labels are a reconstruction from public annotations. They are not an official file.
- We did not run any model for this foundation. All checks in this repository concern the data.

## HPC

See [hpc/shared-ssh.md](hpc/shared-ssh.md) and [hpc/agent-setup.md](hpc/agent-setup.md). The rules for ITU HPC are in [../AGENTS.md](../AGENTS.md):
use `srun` or `sbatch` for every install, download and model job. Never use the login node for them.

## Citation

```bibtex
@inproceedings{errica-etal-2025-wrong,
  author    = {Federico Errica and
               Davide Sanvito and
               Giuseppe Siracusano and
               Roberto Bifulco},
  title     = {What Did I Do Wrong? Quantifying LLMs’ Sensitivity and Consistency to Prompt Engineering},
  booktitle = {Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers)},
  year      = {2025},
  pages     = {1543--1558},
  publisher = {Association for Computational Linguistics},
  doi       = {10.18653/v1/2025.naacl-long.73},
  url       = {https://aclanthology.org/2025.naacl-long.73/},
}
```
