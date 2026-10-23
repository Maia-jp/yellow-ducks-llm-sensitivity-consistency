# Yellow Ducks: LLM Sensitivity and Consistency

This Seminars in Data Science project reproduces selected results from
[What Did I Do Wrong? Quantifying LLMs’ Sensitivity and Consistency to Prompt Engineering](https://aclanthology.org/2025.naacl-long.73/).
Our goal is to reproduce the Mixtral results and compare them with a smaller model.

## Project structure

| Folder | Contents |
|---|---|
| [sensitivity-consistency-LLM/](sensitivity-consistency-LLM/) | Original Python code, five notebooks, dependencies, and NEC license |
| [docs/](docs/README.md) | Paper, citation, reproduction guide, and HPC instructions |
| [scripts/](scripts/) | Shared HPC SSH connection script |

## Start

Read the [reproduction guide](docs/reproduction-guide.md) for setup, reference results, and the experiment plan.
Run the notebook setup commands from `sensitivity-consistency-LLM/`.

For HPC access, connect to the ITU VPN. From this repository root, run:

```bash
./scripts/connect-hpc.sh
```

See the [shared SSH instructions](docs/hpc/shared-ssh.md).

## Qwen3.5-4B experiment

The paper's original open models are Llama-3-70B-Instruct and Mixtral-8x7B-Instruct-v0.1.
Qwen3.5-4B is an added model comparison. It does not replace either original model.
The newest Qwen release is Qwen3.8, but its smallest open checkpoint is 27B. Qwen3.5-4B is the newest Qwen checkpoint at 4B.

From this repository root on HPC, submit a one-prompt model check:

```bash
sbatch scripts/qwen35_smoke.sbatch
```

The job keeps the vLLM environment and model cache in your HPC home directory.
It does not save model weights in the repository.
The job writes its output to `qwen35-smoke-<JOB_ID>.log`.

The full runner reuses the paper's code and notebook prompts for the four intact notebooks.
Run it from `sensitivity-consistency-LLM/` after you install the project requirements and place each notebook's input files in its expected `data/` paths:

```bash
python ../scripts/reproduce_qwen.py --dataset cb
python ../scripts/reproduce_qwen.py --dataset dbpedia
python ../scripts/reproduce_qwen.py --dataset rte
python ../scripts/reproduce_qwen.py --dataset wos
```

Set `QWEN_API_BASE` if the model server does not use `http://127.0.0.1:8000/v1`.
Use `--smoke` for two samples and three prompt variants. These results do not reproduce the paper.
The TREC notebook in the source copy is truncated and cannot be read as notebook JSON.
The runner saves Qwen results under `sensitivity-consistency-LLM/results/qwen35/`.

## Current status

We did not run the reproduction experiments during repository setup.
The copied notebooks contain existing outputs from the source copy.
The source `LLM TREC.ipynb` is incomplete and needs restoration before use.

## Sources

- [Paper PDF](docs/paper.pdf)
- [BibTeX citation](docs/references.bib)
- [Original code repository](https://github.com/nec-research/sensitivity-consistency-LLM)
- [Original NEC license](sensitivity-consistency-LLM/LICENSE.txt)

NEC Laboratories Europe GmbH developed the original software.
The [reproduction guide](docs/reproduction-guide.md#source-and-attribution) records the source commit.
