#!/bin/bash
# Optional preparation job without a GPU. It checks the network and the disk,
# builds both Python environments, downloads the model (about 56 GB) and the
# datasets. The GPU jobs then start faster. They also do these steps if needed.
# Submit from the repository root on the login node:
#   sbatch jobs/qwen3.5-27b/prepare.sbatch
# It runs on cn13 (the A100 node) to read that node's NVIDIA driver version.
# It uses scavenge because acltr accepts only jobs with a GPU (QOSMinGRES).
#SBATCH --job-name=qwen27b-prepare
#SBATCH --output=qwen27b-prepare-%j.out
#SBATCH --error=qwen27b-prepare-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --nodelist=cn13
#SBATCH --partition=scavenge
#SBATCH --time=04:00:00

set -euo pipefail
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"

source "$SLURM_SUBMIT_DIR/scripts/hpc/qwen_common.sh"
prepare_all
