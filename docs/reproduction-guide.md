# Yellow Ducks: LLM Sensitivity and Consistency

This project reproduces selected results from the NAACL 2025 paper, [What Did I Do Wrong? Quantifying LLMs’ Sensitivity and Consistency to Prompt Engineering](https://aclanthology.org/2025.naacl-long.73/).

The main goal is to reproduce the Mixtral-8x7B-Instruct-v0.1 results. The second goal is to compare them with a smaller model. We will report measured results after we run the experiments. The values in this file come from the paper and serve as reference values.

## Group

Yellow Ducks (student-organized)

## Paper and code

- [Paper PDF](paper.pdf)
- [Paper page](https://aclanthology.org/2025.naacl-long.73/)
- [Paper DOI](https://doi.org/10.18653/v1/2025.naacl-long.73)
- [Official PDF](https://aclanthology.org/2025.naacl-long.73.pdf)
- [BibTeX reference](references.bib)
- [Original code repository](https://github.com/nec-research/sensitivity-consistency-LLM)

The repository contains five notebooks. Each notebook runs one dataset: TREC, CommitmentBank (CB), RTE, DBPedia, or Web of Science (WoS). The notebook files contain the dataset examples and the analysis code.

## Source and attribution

NEC Laboratories Europe GmbH developed the original software.
The source copy comes from commit
[`657355688d4b58a6e3a2ac20484518c5d402c8d8`](https://github.com/nec-research/sensitivity-consistency-LLM/tree/657355688d4b58a6e3a2ac20484518c5d402c8d8).
This repository also includes our project README, paper copy, and HPC instructions.
The original software retains its [NEC license](../sensitivity-consistency-LLM/LICENSE.txt).

The copied notebooks contain existing outputs. Those outputs are not results from our reproduction runs.
Record our measured results in the results log after each full run.

## HPC access

Connect to the ITU VPN. From the repository root, start the shared SSH connection:

```bash
./scripts/connect-hpc.sh
```

Enter your password in the terminal. Keep the connection open while the agent uses HPC.
See the [SSH instructions](hpc/shared-ssh.md) and [HPC setup guide](hpc/agent-setup.md).
Submit computation through SLURM, the cluster job scheduler.

## Metrics

- **Sensitivity** measures how much a model changes its class predictions across prompt rephrasings. Lower values mean less change.
- **Consistency** measures how similar the model's prediction distributions are for examples in the same class. Higher values mean more similarity.
- **Micro-F1** measures classification performance across all examples. Higher values mean better performance.

Sensitivity and consistency measure different behavior. Compare both with Micro-F1.

## Reference results for Mixtral

The paper reports sensitivity, consistency, and Micro-F1 for each dataset and prompt strategy. Use these values to compare your output. Small differences can occur because of model versions, server settings, and response behavior.

| Dataset | Simple: S / C / F1 | Detail: S / C / F1 | 1-shot: S / C / F1 |
|---|---:|---:|---:|
| TREC | 0.223 / 0.657 / 0.736 | 0.182 / 0.709 / 0.728 | 0.166 / 0.640 / 0.744 |
| CB | 0.203 / 0.625 / 0.672 | 0.013 / 0.695 / 0.788 | 0.017 / 0.600 / 0.652 |
| RTE | 0.195 / 0.669 / 0.814 | 0.176 / 0.676 / 0.756 | 0.046 / 0.754 / 0.847 |
| DBPedia | 0.053 / 0.802 / 0.804 | 0.054 / 0.808 / 0.862 | 0.069 / 0.773 / 0.819 |
| WoS | 0.112 / 0.483 / 0.606 | 0.124 / 0.469 / 0.599 | 0.142 / 0.472 / 0.566 |

Here, `S` means sensitivity, `C` means consistency, and `F1` means Micro-F1. The paper uses 30 prompt variants, temperature 0, and seed 42.

## CommitmentBank test data

The paper reports 250 CB test examples ([paper, Section 4](https://aclanthology.org/2025.naacl-long.73.pdf)).
The released CB notebook reads `train.jsonl`.
The official SuperGLUE test file has no labels, so the notebook cannot calculate Micro-F1 or class-based consistency from that file.

The [original CB data](https://github.com/mcdm/CommitmentBank) includes annotator responses for each item.
The published [recast rule](https://aclanthology.org/D19-1630.pdf) maps at least 80% agreement to entailment, neutral, or contradiction.
The preparation script matches test examples to these annotations and applies this rule.
It writes `test_labeled.jsonl` and leaves the source files unchanged.

On the pinned source files, the script matched all 250 test examples.
It produced 113 entailment, 16 neutral, and 121 contradiction labels.
These counts match the CB paper's test split table.
The same rule reproduces all labels in the official train and validation files.

Run these commands from the repository root:

```bash
mkdir -p sensitivity-consistency-LLM/data/CB
curl -fsSLo sensitivity-consistency-LLM/data/CB/test.jsonl \
  https://raw.githubusercontent.com/nec-research/sensitivity-consistency-LLM/657355688d4b58a6e3a2ac20484518c5d402c8d8/data/CB/test.jsonl
curl -fsSLo sensitivity-consistency-LLM/data/CommitmentBank-items.csv \
  https://raw.githubusercontent.com/mcdm/CommitmentBank/14ea91f56e3575b011d28fca5c37a1f5e92769b7/CommitmentBank-items.csv
python3 scripts/prepare_cb_test_labels.py \
  --test-jsonl sensitivity-consistency-LLM/data/CB/test.jsonl \
  --items-csv sensitivity-consistency-LLM/data/CommitmentBank-items.csv \
  --output-jsonl sensitivity-consistency-LLM/data/CB/test_labeled.jsonl
```

The CB HPC runner uses `test_labeled.jsonl` by default.
Submit the full run with `sbatch --export=ALL,DATASET=cb scripts/qwen35_reproduce.sbatch`.
Use `SMOKE=1` for a two-example setup check.
Do not use recovered test labels to select models or tune prompts.

The labels come from original public annotations, not an author-provided SuperGLUE gold-label file.
The published class counts and every available train and validation label match.
See the dated evidence log in [findings.md](findings.md#cb-split-finding).

## Setup

From the repository root, enter the code directory before you run the setup commands:

```bash
cd sensitivity-consistency-LLM
```

The code expects Python 3.10 and an OpenAI-compatible model server. The server can run with vLLM, Ollama, or another compatible service. The repository sets the API key to `EMPTY`. Set a key in `util.py` if your server requires one.

1. Create and activate a virtual environment:

   ```bash
   python3.10 -m venv .venv/llm
   source .venv/llm/bin/activate
   ```

2. Install the packages:

   ```bash
   pip install -r requirements.txt
   ```

3. Start a server with the Mixtral-8x7B-Instruct-v0.1 model.

4. Set `MIXTRAL_openai_api_base` in `constants.py` to the server's API base URL.

5. Start Jupyter and open the dataset notebook that you want to run:

   ```bash
   jupyter notebook
   ```

6. Run the notebook cells in order. Set `llms = ['mixtral']` in the grid-search cell to run Mixtral only.

The notebooks set the experiment to 30 prompt variants and one answer per prompt. They also set both temperatures to 0. The code uses seed 42 for zero-temperature requests. The evaluator uses Ray to run sample jobs in parallel. A full run makes many model requests and can take a long time.

## Run plan

The current source copy of `LLM TREC.ipynb` is truncated and is not valid notebook JSON.
The same error exists in source commit `657355688d4b58a6e3a2ac20484518c5d402c8d8`.
Restore a complete TREC notebook before you follow this run plan.
The other four notebooks passed JSON parsing. We did not run their experiments during repository setup.

Start with the TREC notebook. It has 500 test examples and fewer examples than RTE, DBPedia, or WoS. Run the full TREC dataset before you start all five datasets. Do not report a partial run as a full reproduction.

To reduce run time during setup, copy a notebook and use a small sample or fewer prompt variants. Label those results as a smoke test. They do not reproduce the paper's full evaluation.

The evaluator saves intermediate outputs in each notebook's results folder. Keep these files to resume a stopped run. Remove them only when you want to start that experiment again.

## Smaller model comparison

Use **Llama 3.2 3B Instruct** as the smaller model. It is much smaller than Mixtral-8x7B and uses the Llama 3 prompt format. This makes it a practical local comparison model. It also shares the Llama family with the paper's Llama 3-70B model.

The current repository supports `mixtral` and `llama3` names in `util.py`. Its `llama3` entry calls the model `llama3instruct` and uses Llama 3 stop token IDs. To run Llama 3.2 3B, serve that model at the Llama API base URL and check that your server accepts the `llama3instruct` model name. Record the actual model name and server settings. Do not label it as the paper's Llama 3-70B model.

Run the same notebook, dataset, prompts, temperatures, seed, and number of prompt variants for both models. Change the notebook's `llms` list to `['llama3']` for the smaller model. Compare all three metrics. Do not expect matching values because the model weights differ.

## Results log

Add the measured values after each full run. Include the dataset, model name and version, prompt strategy, sensitivity, consistency, Micro-F1, run date, and server settings.

| Dataset | Model and version | Strategy | Sensitivity | Consistency | Micro-F1 | Notes |
|---|---|---|---:|---:|---:|---|
| | | | | | | |

## Limits

The paper describes its evaluation as a demonstration, not a broad benchmark. The smaller-model comparison is an extension. Report model and server settings so readers can interpret differences.

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
