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
From the repository root on HPC, submit one dataset job after you place its input file in the expected `data/` path:

```bash
sbatch --export=ALL,DATASET=cb scripts/qwen35_reproduce.sbatch
```

Use `DATASET=dbpedia`, `DATASET=rte`, or `DATASET=wos` for the other intact notebooks.
Add `SMOKE=1` to run two samples and three prompt variants. These are technical checks, not paper results.
The job starts Qwen with vLLM, runs a one-prompt check, then runs the notebook evaluation.
It saves model weights and the Python environment in your HPC home directory.
The TREC notebook in the source copy is truncated and cannot be read as notebook JSON.
The runner saves Qwen results under `sensitivity-consistency-LLM/results/qwen35/`.

### Install HPC dependencies

Use the installer before the small vLLM setup test:

```bash
bash scripts/install_hpc_dependencies.sh
```

The installer creates separate environments for vLLM and the paper evaluator.
It uses vLLM 0.11.0 with the CUDA 12.8 PyTorch backend by default.
This version checks the cluster runtime with `facebook/opt-125m`.
It does not claim Qwen3.5 support. The Qwen3.5 job needs a newer vLLM build.
Run the setup test with:

```bash
sbatch scripts/test_vllm011.sbatch
```

To test a supported Qwen model with the same vLLM version, submit:

```bash
sbatch scripts/test_qwen3_vllm011.sbatch
```

This job installs vLLM in temporary node storage and runs `Qwen/Qwen3-4B`.
It stores the model cache in your HPC home directory.
This checks Qwen3 support. It does not test Qwen3.5 or reproduce paper results.

## Current status

We did not run the reproduction experiments during repository setup.
The copied notebooks contain existing outputs from the source copy.
The source `LLM TREC.ipynb` is incomplete and needs restoration before use.
The first Qwen3.5 vLLM job used a CUDA 13 wheel on a CUDA 12.9 node and did not start.
The OPT preflight passed with vLLM 0.11.0 and CUDA 12.8.
It does not test Qwen3.5. The Qwen3 test uses a supported Qwen model instead.

## Sources

- [Paper PDF](docs/paper.pdf)
- [BibTeX citation](docs/references.bib)
- [Original code repository](https://github.com/nec-research/sensitivity-consistency-LLM)
- [Original NEC license](sensitivity-consistency-LLM/LICENSE.txt)

NEC Laboratories Europe GmbH developed the original software.
The [reproduction guide](docs/reproduction-guide.md#source-and-attribution) records the source commit.
