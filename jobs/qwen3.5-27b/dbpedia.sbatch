#!/bin/bash
# Qwen3.5-27B on the dbpedia dataset with the paper's original code.
# Submit from the repository root on the login node:
#   sbatch jobs/qwen3.5-27b/dbpedia.sbatch
# The job installs the environments, downloads the model and the dataset,
# starts vLLM and runs the paper code. See jobs/qwen3.5-27b/README.md.
#SBATCH --job-name=qwen27b-dbpedia
#SBATCH --output=qwen27b-dbpedia-%j.out
#SBATCH --error=qwen27b-dbpedia-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=60G
#SBATCH --gres=gpu:a100_40gb:2
#SBATCH --partition=acltr
#SBATCH --time=3-00:00:00

set -euo pipefail
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"

source "$SLURM_SUBMIT_DIR/scripts/hpc/qwen_common.sh"
run_dataset dbpedia
