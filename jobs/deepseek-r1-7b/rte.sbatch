#!/bin/bash
# DeepSeek-R1 7B (Ollama) on the rte dataset with the paper's original code, on asus1.
# Submit from the repository root on the login node:
#   sbatch jobs/deepseek-r1-7b/rte.sbatch
# The job builds the Python environment, downloads the dataset, starts a
# private Ollama server and runs the paper code. See jobs/deepseek-r1-7b/README.md.
#
# asus1 is an ARM node with an NVIDIA GB10 GPU and 128 GB of memory shared by
# CPU and GPU. Slurm has no GPU resource for it, so there is no --gres line.
# Keep --mem: without it the job takes all memory of the node and no other
# job can start there.
#SBATCH --job-name=deepseek7b-rte
#SBATCH --output=jobs/deepseek-r1-7b/slurm-logs/%x-%j.out
#SBATCH --error=jobs/deepseek-r1-7b/slurm-logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodelist=asus1
#SBATCH --partition=asus1
#SBATCH --time=1-00:00:00

set -euo pipefail
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"

source "$SLURM_SUBMIT_DIR/scripts/hpc/deepseek_common.sh"
run_dataset rte
