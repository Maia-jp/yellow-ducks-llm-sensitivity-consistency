# Yellow Ducks: LLM Sensitivity and Consistency

This repository contains our reproducibility project for the **Seminars in Data Science (2026)** course at the IT University of Copenhagen.

We reproduce and extend selected experiments from the paper [*What Did I Do Wrong? Quantifying LLMs' Sensitivity and Consistency to Prompt Engineering*](https://aclanthology.org/2025.naacl-long.73/) (NAACL 2025).

Our objective is to investigate how different large language models respond to variations in prompt engineering, focusing on **prompt sensitivity, consistency, and evaluation performance**.


We reproduce selected results using **Mixtral and Llama‐3** and extend the evaluation to two additional models: **Qwen3.5 and DeepSeek**.


## Project Structure

| Folder | Contents |
|---|---|
| `original-paper/` | Original paper implementation (NEC), evaluation notebooks, dependencies |
| `original-paper/data/` | Five evaluation datasets, subsets, and dataset documentation |
| `notebooks/` | Our exploratory analysis (correlation analysis for Claim 1) |
| `results/models/` | Raw results per model (zipped) |
| `results/correlation/` | Correlation plots and statistics output |
| `scripts/` | Dataset preparation, verification, model runners, and analysis scripts |
| `scripts/hpc/jobs/` | Slurm job files for ITU HPC |
| `docs/` | Original paper, reproduction instructions, dataset verification, and HPC documentation |

## Models

Our experiments cover four language models:

| Model | Purpose |
|---|---|
| Mixtral‐8x7B‐Instruct‐v0.1 | Reproduction of selected results from the original paper |
| Llama‐3‐70B‐Instruct | Reproduction of selected results from the original paper |
| Qwen3.5-27B | Additional model evaluation |
| DeepSeek‐R1 7B | Additional model evaluation |

The additional models allow us to investigate whether the findings on prompt sensitivity and consistency generalize beyond the model evaluated in the original reproduction.

## Datasets

We use the five datasets provided with the original implementation, covering different natural language processing tasks.

All datasets are located in:

`original-paper/data/`

Each dataset directory contains documentation describing its source, preprocessing, and evaluation setup.

Dataset integrity and correspondence with the original sources are documented in [docs/datasets/README.md](docs/datasets/README.md).
