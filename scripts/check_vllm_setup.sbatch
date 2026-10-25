#!/usr/bin/env bash
#SBATCH --job-name=vllm-setup
#SBATCH --partition=acltr
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:05:00
#SBATCH --output=vllm-setup-%j.log
set -euo pipefail
module load Python/3.11.3-foss-2023a
nvidia-smi --query-gpu=name,driver_version --format=csv
"$HOME/.venvs/qwen35-vllm-0.32.0-py311/bin/python" -B -u "$SLURM_SUBMIT_DIR/scripts/check_vllm_setup.py"
