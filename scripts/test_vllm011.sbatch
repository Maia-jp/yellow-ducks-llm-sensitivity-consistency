#!/usr/bin/env bash
#SBATCH --job-name=vllm011-test
#SBATCH --partition=acltr
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=vllm011-test-%j.log
set -euo pipefail
module load Python/3.11.3-foss-2023a
export PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=2
export HF_HOME="$HOME/.cache/huggingface"
export VLLM_WORKER_MULTIPROC_METHOD=spawn
runtime=$(mktemp -d /tmp/vllm011-XXXXXXXX)
trap 'rm -rf "$runtime"' EXIT
export TMPDIR="$runtime"
export UV_CACHE_DIR="$runtime/uv-cache"
export TRITON_CACHE_DIR="$runtime/triton"
export VLLM_CACHE_ROOT="$runtime/vllm-cache"
VLLM_VENV="${VLLM_VENV:-$HOME/.venvs/vllm-0.11.0-cu128}"
if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
    echo "Missing $VLLM_VENV. Run scripts/install_hpc_dependencies.sh first." >&2
    exit 2
fi
nvidia-smi --query-gpu=name,driver_version --format=csv
"$VLLM_VENV/bin/python" -B -u "$SLURM_SUBMIT_DIR/scripts/test_vllm011.py"
