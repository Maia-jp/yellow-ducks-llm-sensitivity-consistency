# Source code and notebooks

This folder contains the copied code from
[nec-research/sensitivity-consistency-LLM](https://github.com/nec-research/sensitivity-consistency-LLM).
NEC Laboratories Europe GmbH developed the original software.
The original [license](LICENSE.txt) applies to the copied software.

Run the notebook setup commands from this folder.
See the [reproduction guide](../docs/reproduction-guide.md) for dependencies and experiment instructions.

`LLM TREC.ipynb` is a repaired copy. The upstream file is cut inside an image output. Our copy keeps the 7 complete cells (loader, prompts, 1-shot examples, grid search). The plot cells after the cut are lost.

The data is in [data/](data/README.md). Each dataset has an explanation file.
