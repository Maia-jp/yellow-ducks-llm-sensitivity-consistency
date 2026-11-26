#!/bin/bash
# DeepSeek-R1 7B on the rte dataset, x86 node with AVX-512 and the ITU Ollama module (0.6.0).
# The default jobs are one folder up and run on asus1. Do not mix the two in one results folder.
# Submit from the repository root on the login node:
#   sbatch scripts/hpc/jobs/deepseek-r1-7b/x86/rte.sbatch
# The job builds the Python environment, downloads the dataset, starts a
# private Ollama server and runs the paper code. See scripts/hpc/jobs/deepseek-r1-7b/README.md.
#SBATCH --job-name=deepseek7b-rte
#SBATCH --output=scripts/hpc/jobs/deepseek-r1-7b/slurm-logs/%x-%j.out
#SBATCH --error=scripts/hpc/jobs/deepseek-r1-7b/slurm-logs/%x-%j.err
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
run_dataset rte
