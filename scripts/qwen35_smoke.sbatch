#!/usr/bin/env bash
# Run from the repository root with: sbatch scripts/qwen35_smoke.sbatch
# The model and Python environment stay in the user's HPC home directory.
# Slurm writes the test log to the repository root. Git ignores *.log files.
#SBATCH --job-name=qwen35-smoke
#SBATCH --partition=acltr
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --output=qwen35-smoke-%j.log

set -euo pipefail

module load Python/3.12.3-GCCcore-13.3.0

QWEN_VENV="$HOME/.venvs/qwen35-vllm-0.32.0"
HF_HOME="$HOME/.cache/huggingface"
export HF_HOME
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DOWNLOAD_TIMEOUT=60

mkdir -p "$(dirname "$QWEN_VENV")" "$HF_HUB_CACHE"
if [[ ! -x "$QWEN_VENV/bin/python" ]]; then
    python3 -m venv "$QWEN_VENV"
    "$QWEN_VENV/bin/python" -m pip install --upgrade pip
    "$QWEN_VENV/bin/python" -m pip install "vllm==0.32.0"
fi

source "$QWEN_VENV/bin/activate"
MODEL="Qwen/Qwen3.5-4B"
SERVER_LOG="$HOME/qwen35-vllm-${SLURM_JOB_ID}.log"

vllm serve "$MODEL" \
    --host 127.0.0.1 \
    --port 8000 \
    --dtype bfloat16 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.9 \
    --max-num-seqs 16 \
    --generation-config vllm >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!
cleanup() {
    kill "$SERVER_PID" 2>/dev/null || true
    wait "$SERVER_PID" 2>/dev/null || true
}
trap cleanup EXIT

python "$SLURM_SUBMIT_DIR/scripts/smoke_qwen.py" \
    --model "$MODEL" \
    --wait-seconds 3600
