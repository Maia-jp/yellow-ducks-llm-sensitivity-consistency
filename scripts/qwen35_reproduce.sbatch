#!/usr/bin/env bash
# Run from the repository root:
# sbatch --export=ALL,DATASET=cb scripts/qwen35_reproduce.sbatch
# Add SMOKE=1 to run two samples and three prompt variants only.
# The model cache and Python environment stay in the user's HPC home directory.
#SBATCH --job-name=qwen35-repro
#SBATCH --partition=acltr
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=qwen35-repro-%j.log

set -euo pipefail

DATASET=${DATASET:-cb}
SMOKE=${SMOKE:-0}
CODE_DIR="$SLURM_SUBMIT_DIR/sensitivity-consistency-LLM"
VLLM_VENV="$HOME/.venvs/qwen35-vllm-0.32.0-py311"
EVAL_VENV="$HOME/.venvs/qwen35-eval-py311"
HF_HOME="$HOME/.cache/huggingface"
export HF_HOME
export HF_HUB_CACHE="$HF_HOME/hub"
export HF_HUB_DOWNLOAD_TIMEOUT=60
export QWEN_MODEL="Qwen/Qwen3.5-4B"
export QWEN_API_BASE="http://127.0.0.1:8000/v1"

case "$DATASET" in
    cb) DATA_FILE="$CODE_DIR/data/CB/test_labeled.jsonl" ;;
    dbpedia) DATA_FILE="$CODE_DIR/data/DBPEDIA/test.csv" ;;
    rte) DATA_FILE="$CODE_DIR/data/RTE/train.jsonl" ;;
    wos) DATA_FILE="$CODE_DIR/data/WOS46985/X.txt" ;;
    *) echo "DATASET must be cb, dbpedia, rte, or wos" >&2; exit 2 ;;
esac
if [[ ! -f "$DATA_FILE" ]]; then
    echo "Missing dataset input: $DATA_FILE" >&2
    echo "Follow the dataset setup notes in docs/reproduction-guide.md." >&2
    exit 2
fi

module load Python/3.11.3-foss-2023a
mkdir -p "$HOME/.venvs" "$HF_HUB_CACHE"
if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
    python3 -m venv "$VLLM_VENV"
    "$VLLM_VENV/bin/python" -m pip install --upgrade pip uv
    "$VLLM_VENV/bin/uv" pip install --python "$VLLM_VENV/bin/python" \
        --torch-backend=cu129 "vllm==0.32.0"
fi
"$VLLM_VENV/bin/python" -m pip install --upgrade "openai>=2.25.0,<4"

if [[ ! -x "$EVAL_VENV/bin/python" ]]; then
    python3 -m venv "$EVAL_VENV"
    "$EVAL_VENV/bin/python" -m pip install --upgrade pip
    "$EVAL_VENV/bin/python" -m pip install \
        -r "$CODE_DIR/requirements.txt" \
        "langchain_openai==0.1.8"
fi

MODEL="$QWEN_MODEL"
SERVER_LOG="$HOME/qwen35-vllm-${SLURM_JOB_ID}.log"
"$VLLM_VENV/bin/vllm" serve "$MODEL" \
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

"$EVAL_VENV/bin/python" "$SLURM_SUBMIT_DIR/scripts/smoke_qwen.py" --model "$MODEL" --wait-seconds 3600

export PYTHONPATH="$SLURM_SUBMIT_DIR/scripts:$CODE_DIR${PYTHONPATH:+:$PYTHONPATH}"
export QWEN_REGISTER_PROVIDER=1
cd "$CODE_DIR"
RUN_ARGS=(--dataset "$DATASET")
if [[ "$DATASET" == cb ]]; then
    RUN_ARGS+=(--input-file "$DATA_FILE")
fi
if [[ "$SMOKE" == 1 ]]; then
    RUN_ARGS+=(--smoke)
fi
"$EVAL_VENV/bin/python" "$SLURM_SUBMIT_DIR/scripts/reproduce_qwen.py" "${RUN_ARGS[@]}"
