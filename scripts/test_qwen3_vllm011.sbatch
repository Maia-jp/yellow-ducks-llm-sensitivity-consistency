#!/usr/bin/env bash
#SBATCH --job-name=qwen3-4b-vllm011
#SBATCH --partition=acltr
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=qwen3-4b-vllm011-%j.log
set -euo pipefail

module load Python/3.11.3-foss-2023a
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=4
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

runtime=$(mktemp -d /tmp/qwen3-vllm011-XXXXXXXX)
trap 'rm -rf "$runtime"' EXIT
export TMPDIR="$runtime"
export UV_CACHE_DIR="$runtime/uv-cache"
export TRITON_CACHE_DIR="$runtime/triton"
export VLLM_CACHE_ROOT="$runtime/vllm-cache"

if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --user --upgrade uv
    export PATH="$HOME/.local/bin:$PATH"
fi

uv venv --python "$(command -v python3)" "$runtime/venv"
uv pip install --python "$runtime/venv/bin/python" \
    --torch-backend=cu128 "vllm==0.11.0"

nvidia-smi --query-gpu=name,driver_version --format=csv
"$runtime/venv/bin/python" -B -u \
    "$SLURM_SUBMIT_DIR/scripts/test_qwen3_vllm011.py"
