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

## Current status
 ????
## Sources

- [Paper PDF](docs/paper.pdf)
- [BibTeX citation](docs/references.bib)
- [Original code repository](https://github.com/nec-research/sensitivity-consistency-LLM)
- [Original NEC license](sensitivity-consistency-LLM/LICENSE.txt)

NEC Laboratories Europe GmbH developed the original software.
The [reproduction guide](docs/reproduction-guide.md#source-and-attribution) records the source commit.
