#!/usr/bin/env bash
# Install the HPC environments used by the reproducibility scripts.
# Run from the repository root on HPC:
#   bash scripts/install_hpc_dependencies.sh

set -euo pipefail

if [[ ${1:-} == --help ]]; then
    cat <<'HELP'
Usage: bash scripts/install_hpc_dependencies.sh

Environment variables:
  VLLM_VERSION   vLLM version for the GPU preflight environment (default: 0.11.0)
  TORCH_BACKEND  PyTorch CUDA backend (default: cu128)
  VLLM_VENV      vLLM environment path
  EVAL_VENV      evaluation environment path
  HF_HOME        Hugging Face cache path

The script installs packages only. It does not download a model.
HELP
    exit 0
fi

PYTHON_MODULE=${PYTHON_MODULE:-Python/3.11.3-foss-2023a}
VLLM_VERSION=${VLLM_VERSION:-0.11.0}
TORCH_BACKEND=${TORCH_BACKEND:-cu128}
VLLM_VENV=${VLLM_VENV:-$HOME/.venvs/vllm-${VLLM_VERSION}-${TORCH_BACKEND}}
EVAL_VENV=${EVAL_VENV:-$HOME/.venvs/yellow-ducks-eval-py311}
HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

module load "$PYTHON_MODULE"

if ! command -v uv >/dev/null 2>&1; then
    python3 -m pip install --user --upgrade uv
    export PATH="$HOME/.local/bin:$PATH"
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv was not found after installation. Check your user Python path." >&2
    exit 1
fi

mkdir -p "$(dirname "$VLLM_VENV")" "$(dirname "$EVAL_VENV")" "$HF_HOME/hub"

if [[ ! -x "$VLLM_VENV/bin/python" ]]; then
    uv venv --python "$(command -v python3)" "$VLLM_VENV"
fi
uv pip install --python "$VLLM_VENV/bin/python" \
    --torch-backend="$TORCH_BACKEND" "vllm==$VLLM_VERSION" \
    "transformers==4.57.1"

if [[ ! -x "$EVAL_VENV/bin/python" ]]; then
    uv venv --python "$(command -v python3)" "$EVAL_VENV"
fi
uv pip install --python "$EVAL_VENV/bin/python" \
    -r "$PWD/sensitivity-consistency-LLM/requirements.txt" \
    "langchain_openai==0.1.8"

cat <<EOF
Installed HPC dependencies.
vLLM environment: $VLLM_VENV
Evaluation environment: $EVAL_VENV
Hugging Face cache: $HF_HOME
vLLM version: $VLLM_VERSION
PyTorch backend: $TORCH_BACKEND
EOF
