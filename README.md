# Yellow Ducks: LLM Sensitivity and Consistency

This Seminars in Data Science project reproduces selected results from
[What Did I Do Wrong? Quantifying LLMs’ Sensitivity and Consistency to Prompt Engineering](https://aclanthology.org/2025.naacl-long.73/).
Our goal is to reproduce the Mixtral results and compare them with a smaller model.

This branch (`dataset-fidelity`) is the foundation. It holds the paper's code, the verified datasets, and the reference values.
It holds no model results. Experiments with other models belong on separate branches.

## Project structure

| Folder | Contents |
|---|---|
| [sensitivity-consistency-LLM/](sensitivity-consistency-LLM/README.md) | The paper's Python code, five notebooks, pinned dependencies, and the NEC license |
| [sensitivity-consistency-LLM/data/](sensitivity-consistency-LLM/data/README.md) | The five datasets, the 2000-sample subsets, and one explanation file for each dataset |
| [docs/](docs/README.md) | Paper, citation, reproduction guide, dataset checks, and HPC instructions |
| [scripts/](scripts/) | Dataset tools |

## Start

1. Read the [reproduction guide](docs/reproduction-guide.md). It has the setup, the reference values and the run plan.
2. Check the data: `python3 scripts/fetch_datasets.py --verify-only` (see Data).
3. Run the notebook setup commands from `sensitivity-consistency-LLM/`.

## Data

All datasets are in `sensitivity-consistency-LLM/data/`. Each dataset folder has a `README.md` that explains the source, the cut and the 1-shot examples.
We compared every file with its official source. The result is in [docs/datasets/README.md](docs/datasets/README.md).

Git ignores one derived file, `WOS46985/X.txt` (65 MB). Extract it and check every file from the repository root:

```bash
python3 scripts/fetch_datasets.py
```

The script downloads a file only when it is missing or damaged. It checks every file against `docs/datasets/data-manifest.json`.
On HPC, run it with `srun` or `sbatch`. Do not run it on the login node.

Two points to know before you run an experiment:

- **CB:** the paper says "test", but the released notebook reads `train.jsonl`. Read the [CB note](sensitivity-consistency-LLM/data/CB/README.md).
- **Paper Table 1:** two Mixtral CB sensitivity values most likely have a decimal error. The [guide](docs/reproduction-guide.md#reference-results-paper-table-1) gives the values to use.

## ITU HPC

Connect to the cluster with `ssh s25441@hpc3.itu.dk`. This account needs no password.
See the [HPC setup guide](docs/hpc/agent-setup.md). We did not test it on `hpc3` yet.

ITU forbids package managers (`pip`, `uv`, `conda`) on the main login node.
ITU also forbids VS Code Remote, Jupyter and other interactive tools there.
ITU restricted an account for this in October 2026.
Use the login node only to edit files and to run `sbatch`, `squeue` and `sinfo`.
Run installs, model downloads and model servers only inside Slurm jobs.
See [AGENTS.md](AGENTS.md) for the rules that agents follow.

## Sources

- [Paper PDF](docs/paper.pdf)
- [BibTeX citation](docs/references.bib)
- [Original code repository](https://github.com/nec-research/sensitivity-consistency-LLM)
- [Original NEC license](sensitivity-consistency-LLM/LICENSE.txt)

NEC Laboratories Europe GmbH developed the original software.
The [reproduction guide](docs/reproduction-guide.md#paper-and-code) records the source commit.
