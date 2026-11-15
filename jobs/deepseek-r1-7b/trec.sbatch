#!/bin/bash
# DeepSeek-R1 7B (Ollama) on the trec dataset with the paper's original code.
# Submit from the repository root on the login node:
#   sbatch jobs/deepseek-r1-7b/trec.sbatch
# The job builds the Python environment, downloads the dataset, starts a
# private Ollama server and runs the paper code. See jobs/deepseek-r1-7b/README.md.
#SBATCH --job-name=deepseek7b-trec
#SBATCH --output=jobs/deepseek-r1-7b/slurm-logs/%x-%j.out
#SBATCH --error=jobs/deepseek-r1-7b/slurm-logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --constraint=arch_icelake
#SBATCH --partition=scavenge
#SBATCH --time=1-00:00:00

set -euo pipefail
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"

source "$SLURM_SUBMIT_DIR/scripts/hpc/deepseek_common.sh"
run_dataset trec
