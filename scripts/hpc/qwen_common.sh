#!/usr/bin/env bash
# Shared steps for the Qwen3.5-27B Slurm jobs in jobs/qwen3.5-27b/.
#
# The job files source this file. Do not run it on the login node:
# it installs packages and downloads the model, so it refuses to start
# outside a Slurm job.
#
# Every step is idempotent. A step that is already done is skipped, so the
# first job builds the environments and later jobs reuse them.
# Shared steps take a lock, so parallel jobs do not build the same thing twice.

set -euo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo "Run this only inside a Slurm job (sbatch). SLURM_JOB_ID is not set." >&2
    exit 1
fi

# ---------------------------------------------------------------- settings ---

REPO_DIR=${SLURM_SUBMIT_DIR:?}
CODE_DIR="$REPO_DIR/sensitivity-consistency-LLM"
LOCK_FILES="$REPO_DIR/jobs/qwen3.5-27b/locks"

MODEL_ID="Qwen/Qwen3.5-27B"
MODEL_REVISION="fc05daec18b0a78c049392ed2e771dde82bdf654"
MODEL_TAG="qwen3.5-27b"
MODEL_MIN_FREE_GB=60

VLLM_VERSION="0.32.0"
UV_VERSION="0.13.6"
UV_SHA256="f29c3e94ffabd7e6f3ba521dba169a9b7ec12100bf3257087f0f24f6123f3645"

# Everything lives under one folder in the home directory.
YD_ROOT=${YD_ROOT:-$HOME/yellow-ducks}
UV_DIR="$YD_ROOT/tools/uv-$UV_VERSION"
EVAL_ENV="$YD_ROOT/envs/eval-py310"
RUNS_DIR="$YD_ROOT/runs/$MODEL_TAG"
LOCK_DIR="$YD_ROOT/locks"

# vLLM server settings. Override them with sbatch --export=ALL,NAME=value.
MAX_MODEL_LEN=${MAX_MODEL_LEN:-8192}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-auto}  # auto: 64 on GPUs with 45 GB or more, else 32
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.92}
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-3600}

export TMPDIR=${TMPDIR:-$HOME/tmp}
JOB_TMP="$TMPDIR/yd-$SLURM_JOB_ID"
export HF_HOME="$YD_ROOT/hf"
export HF_HUB_DOWNLOAD_TIMEOUT=120
# Keep the download cache between jobs: on 2026-10-30 the vLLM packages took
# 58 minutes to download, so a job that stops must not lose them.
export UV_CACHE_DIR="$YD_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$YD_ROOT/python"
export UV_PYTHON_PREFERENCE=only-managed
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=300
export UV_HTTP_RETRIES=8
export UV_NO_PROGRESS=1
export PYTHONUNBUFFERED=1
unset PYTHONPATH RAY_ADDRESS VIRTUAL_ENV CONDA_PREFIX
mkdir -p "$TMPDIR" "$JOB_TMP" "$YD_ROOT" "$LOCK_DIR"

HELD_LOCK=""
SERVER_PID=""
RUNNER_PID=""
VLLM_ENV=""
CUDA_VARIANT=""

log() { printf '[%s] %s\n' "$(date '+%F %T')" "$*"; }
fail() { log "ERROR: $*" >&2; exit 1; }

cleanup() {
    local rc=$?
    set +e
    if [[ -n $RUNNER_PID ]] && kill -0 "$RUNNER_PID" 2>/dev/null; then
        kill "$RUNNER_PID" 2>/dev/null
        wait "$RUNNER_PID" 2>/dev/null
    fi
    if [[ -n $SERVER_PID ]] && kill -0 "$SERVER_PID" 2>/dev/null; then
        kill "$SERVER_PID" 2>/dev/null
        for _ in $(seq 1 30); do kill -0 "$SERVER_PID" 2>/dev/null || break; sleep 1; done
        kill -9 "$SERVER_PID" 2>/dev/null
    fi
    if [[ -n $HELD_LOCK ]]; then
        rm -rf "$HELD_LOCK"
    fi
    rm -rf "$JOB_TMP"
    log "Job finished with exit code $rc."
    exit "$rc"
}
trap cleanup EXIT
trap 'exit 143' TERM INT

# ------------------------------------------------------------------- locks ---

# acquire_lock NAME: wait until no other job builds the same thing.
# A lock from a job that is no longer in squeue is stale and is removed.
acquire_lock() {
    local lock="$LOCK_DIR/$1" waited=0 owner
    until mkdir "$lock" 2>/dev/null; do
        owner=$(cat "$lock/job" 2>/dev/null || true)
        if [[ -n $owner ]] && ! squeue -h -j "$owner" 2>/dev/null | grep -q .; then
            log "Removing stale lock $lock from ended job $owner."
            rm -rf "$lock"
            continue
        fi
        if (( waited >= 10800 )); then
            fail "Waited 3 hours for lock $lock (owner job ${owner:-unknown})."
        fi
        if (( waited % 300 == 0 )); then
            log "Waiting for lock $lock (owner job ${owner:-unknown})."
        fi
        sleep 30
        waited=$((waited + 30))
    done
    echo "$SLURM_JOB_ID" >"$lock/job"
    HELD_LOCK=$lock
}

release_lock() {
    rm -rf "$HELD_LOCK"
    HELD_LOCK=""
}

# ------------------------------------------------------------- node facts ---

# Prints the NVIDIA driver version of this node, or nothing if there is none.
# /proc works also in a job without a GPU allocation.
driver_version() {
    if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1
    elif [[ -r /proc/driver/nvidia/version ]]; then
        grep -oE '[0-9]{3}\.[0-9]+(\.[0-9]+)?' /proc/driver/nvidia/version | head -n 1
    fi
}

# vLLM 0.32.0 has a CUDA 13.0 build (driver 580 or newer) and a CUDA 12.9
# build (driver 525 or newer). Choose the build that the driver supports.
choose_cuda_variant() {
    local version major
    version=$(driver_version)
    [[ -n $version ]] || fail "No NVIDIA driver on $(hostname). Use a GPU node."
    major=${version%%.*}
    if (( major >= 580 )); then
        CUDA_VARIANT=cu130
    elif (( major >= 525 )); then
        CUDA_VARIANT=cu129
    else
        fail "NVIDIA driver $version is too old for vLLM $VLLM_VERSION (needs 525 or newer)."
    fi
    VLLM_ENV="$YD_ROOT/envs/vllm-$VLLM_VERSION-$CUDA_VARIANT"
    log "NVIDIA driver $version: using the $CUDA_VARIANT build of vLLM."
}

report_node() {
    log "Host: $(hostname). Job: $SLURM_JOB_ID. CPUs: ${SLURM_CPUS_PER_TASK:-?}."
    log "GPUs given by Slurm: ${CUDA_VISIBLE_DEVICES:-none}."
    log "glibc: $(ldd --version 2>&1 | head -n 1)"
    log "Driver: $(driver_version || true)"
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv 2>/dev/null || true
    fi
    df -h "$YD_ROOT" "$TMPDIR" 2>/dev/null || true
    if command -v quota >/dev/null 2>&1; then
        quota -s 2>/dev/null || true
    fi
}

# Fails early if a download host is not reachable from this node.
# Each host gets 3 tries: single timeouts happen (seen on 2026-10-30).
check_network() {
    local url try
    for url in \
        https://github.com \
        https://pypi.org/simple/pip/ \
        https://download.pytorch.org/whl/ \
        https://huggingface.co/api/models/$MODEL_ID \
        https://raw.githubusercontent.com/nec-research/sensitivity-consistency-LLM/657355688d4b58a6e3a2ac20484518c5d402c8d8/README.md; do
        for try in 1 2 3; do
            if curl -fsS -o /dev/null --max-time 60 "$url"; then
                log "Network OK: $url"
                continue 2
            fi
            log "Try $try: no answer from $url."
            sleep 20
        done
        fail "This node cannot reach $url after 3 tries. Downloads in the job will fail."
    done
}

# Fails early if the home folder has too little space for what is missing.
# The vLLM environment needs about 12 GB, its download cache about 12 GB more
# while it builds, and the model about 56 GB.
check_disk() {
    local need=5 free_gb
    compgen -G "$YD_ROOT/envs/vllm-$VLLM_VERSION-*/.ready" >/dev/null || need=$((need + 24))
    compgen -G "$HF_HOME/.ready-$MODEL_TAG-*" >/dev/null || need=$((need + 60))
    free_gb=$(df -Pk "$YD_ROOT" | awk 'NR == 2 {print int($4 / 1048576)}')
    log "Free space in $YD_ROOT: ${free_gb} GB. Needed for missing parts: about ${need} GB."
    if (( free_gb < need )); then
        fail "Not enough disk space in $YD_ROOT. Free some space or set YD_ROOT to a larger folder."
    fi
}

# --------------------------------------------------------------- installs ---

ensure_uv() {
    [[ -x $UV_DIR/uv ]] && return 0
    acquire_lock uv
    if [[ ! -x $UV_DIR/uv ]]; then
        log "Installing uv $UV_VERSION into $UV_DIR."
        local archive="$JOB_TMP/uv.tar.gz"
        curl -fsSL --retry 3 -o "$archive" \
            "https://github.com/astral-sh/uv/releases/download/$UV_VERSION/uv-x86_64-unknown-linux-gnu.tar.gz"
        echo "$UV_SHA256  $archive" | sha256sum -c -
        rm -rf "$UV_DIR.partial"
        mkdir -p "$UV_DIR.partial"
        tar -xzf "$archive" -C "$UV_DIR.partial" --strip-components=1
        rm -rf "$UV_DIR"
        mv "$UV_DIR.partial" "$UV_DIR"
    fi
    release_lock
}

uv() { "$UV_DIR/uv" "$@"; }

# build_env PATH PYTHON_VERSION LOCK_FILE [TORCH_BACKEND]
# Installs the exact versions in LOCK_FILE. Writes PATH/.ready at the end,
# so an environment from a failed build is never used.
build_env() {
    local env=$1 python=$2 lock=$3 backend=${4:-}
    [[ -f $env/.ready ]] && return 0
    acquire_lock "env-$(basename "$env")"
    if [[ ! -f $env/.ready ]]; then
        log "Building $env from $(basename "$lock")."
        rm -rf "$env"
        uv venv --python "$python" "$env"
        local args=(pip sync --python "$env/bin/python" "$lock")
        [[ -n $backend ]] && args+=(--torch-backend "$backend")
        uv "${args[@]}"
        cp "$lock" "$env/lock.txt"
        touch "$env/.ready"
        log "Built $env."
    fi
    release_lock
}

ensure_eval_env() {
    build_env "$EVAL_ENV" 3.10 "$LOCK_FILES/eval-py310.txt"
    "$EVAL_ENV/bin/python" -c 'import langchain, langchain_openai, ray, sklearn, seaborn, scipy; print("eval env OK: ray", ray.__version__)'
}

ensure_vllm_env() {
    [[ -n $CUDA_VARIANT ]] || choose_cuda_variant
    build_env "$VLLM_ENV" 3.12 "$LOCK_FILES/vllm-$VLLM_VERSION-$CUDA_VARIANT.txt" "$CUDA_VARIANT"
    "$VLLM_ENV/bin/python" -c 'import torch, vllm; print("vllm env OK: vllm", vllm.__version__, "torch", torch.__version__, "cuda", torch.version.cuda)'
}

# Downloads the model once into HF_HOME. Prints nothing; sets MODEL_PATH.
ensure_model() {
    local marker="$HF_HOME/.ready-${MODEL_TAG}-${MODEL_REVISION}"
    if [[ ! -f $marker ]]; then
        acquire_lock model
        if [[ ! -f $marker ]]; then
            mkdir -p "$HF_HOME"
            local free_gb
            free_gb=$(df -Pk "$HF_HOME" | awk 'NR == 2 {print int($4 / 1048576)}')
            log "Free space for the model: ${free_gb} GB."
            if (( free_gb < MODEL_MIN_FREE_GB )) && [[ ! -d $HF_HOME/hub/models--Qwen--Qwen3.5-27B ]]; then
                fail "The model needs about 56 GB. Only ${free_gb} GB are free in $HF_HOME."
            fi
            log "Downloading $MODEL_ID at revision $MODEL_REVISION."
            local try
            for try in 1 2 3 4 5; do
                # snapshot_download continues partial downloads, so a retry is cheap.
                if "$VLLM_ENV/bin/python" - "$MODEL_ID" "$MODEL_REVISION" <<'EOF'
import sys
from huggingface_hub import snapshot_download
path = snapshot_download(sys.argv[1], revision=sys.argv[2], max_workers=8)
print("model files in", path)
EOF
                then
                    touch "$marker"
                    break
                fi
                log "Model download try $try failed. Waiting 60 s."
                sleep 60
            done
            [[ -f $marker ]] || fail "The model download failed 5 times."
        fi
        release_lock
    fi
    MODEL_PATH=$("$VLLM_ENV/bin/python" - "$MODEL_ID" "$MODEL_REVISION" <<'EOF'
import sys
from huggingface_hub import snapshot_download
print(snapshot_download(sys.argv[1], revision=sys.argv[2], local_files_only=True))
EOF
)
    [[ -f $MODEL_PATH/config.json ]] || fail "Model files are missing in $MODEL_PATH."
    log "Model path: $MODEL_PATH"
}

# Downloads and verifies the datasets into the run folder of one dataset.
# scripts/fetch_datasets.py checks every file against docs/datasets/data-manifest.json.
ensure_data() {
    local run_dir=$1
    mkdir -p "$run_dir"
    acquire_lock "data-$(basename "$run_dir")"
    log "Downloading and verifying the datasets into $run_dir/data."
    "$EVAL_ENV/bin/python" "$REPO_DIR/scripts/fetch_datasets.py" --out "$run_dir/data"
    release_lock
}

# ----------------------------------------------------------- vLLM server ---

gpu_count() {
    local devices=${CUDA_VISIBLE_DEVICES:-}
    [[ -n $devices ]] || { echo 0; return; }
    awk -F, '{print NF}' <<<"$devices"
}

free_port() {
    "$EVAL_ENV/bin/python" -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'
}

# start_server LOG_FILE EXTRA_ARGS...: starts vLLM and waits until it answers.
# Returns 1 if the server process stops before it is ready.
start_server() {
    local server_log=$1
    shift
    local tp
    tp=$(gpu_count)
    PORT=$(free_port)
    log "Starting vLLM on port $PORT with $tp GPU(s): max-num-seqs $MAX_NUM_SEQS, max-model-len $MAX_MODEL_LEN $*"
    "$VLLM_ENV/bin/vllm" serve "$MODEL_PATH" \
        --served-model-name "$MODEL_ID" \
        --host 127.0.0.1 \
        --port "$PORT" \
        --tensor-parallel-size "$tp" \
        --dtype bfloat16 \
        --max-model-len "$MAX_MODEL_LEN" \
        --max-num-seqs "$MAX_NUM_SEQS" \
        --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION" \
        --language-model-only \
        --generation-config vllm \
        --seed 42 \
        "$@" >>"$server_log" 2>&1 &
    SERVER_PID=$!

    local waited=0
    until curl -fsS -o /dev/null --max-time 10 "http://127.0.0.1:$PORT/v1/models"; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            wait "$SERVER_PID" 2>/dev/null || true
            SERVER_PID=""
            log "vLLM stopped before it was ready. Last lines of $server_log:"
            tail -n 40 "$server_log" >&2
            return 1
        fi
        if (( waited >= SERVER_START_TIMEOUT )); then
            log "vLLM was not ready after $SERVER_START_TIMEOUT s. Last lines of $server_log:"
            tail -n 40 "$server_log" >&2
            fail "vLLM start timed out."
        fi
        sleep 15
        waited=$((waited + 15))
        if (( waited % 300 == 0 )); then
            log "vLLM is still loading after $waited s. Last log line: $(tail -n 1 "$server_log")"
        fi
    done
    log "vLLM is ready after about $waited s."
}

serve_model() {
    local server_log=$1 tp total_mib
    tp=$(gpu_count)
    case $tp in
        1 | 2 | 4) ;;
        *) fail "Request 1, 2 or 4 GPUs with --gres. Slurm gave: ${CUDA_VISIBLE_DEVICES:-none}." ;;
    esac
    total_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk '{s += $1} END {print s}')
    if (( total_mib < 75000 )); then
        fail "The 27B model needs about 56 GB for weights plus cache. The GPUs have only ${total_mib} MiB in total."
    fi

    if [[ $MAX_NUM_SEQS == auto ]]; then
        local min_mib
        min_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | sort -n | head -n 1)
        # Each parallel request keeps about 150 MB of model state (float32).
        MAX_NUM_SEQS=$(( min_mib >= 45000 ? 64 : 32 ))
    fi

    mkdir -p "$(dirname "$server_log")"
    if start_server "$server_log"; then
        return 0
    fi
    # Second try with less memory: no CUDA graphs and fewer parallel requests.
    MAX_NUM_SEQS=$(( MAX_NUM_SEQS > 16 ? 16 : MAX_NUM_SEQS ))
    log "Retrying vLLM with --enforce-eager and max-num-seqs $MAX_NUM_SEQS."
    if start_server "$server_log" --enforce-eager; then
        return 0
    fi
    # Third try: the longest prompt has about 2,940 tokens (WoS 1-shot), so
    # 6144 tokens still fit every prompt and its answer.
    MAX_MODEL_LEN=$(( MAX_MODEL_LEN > 6144 ? 6144 : MAX_MODEL_LEN ))
    MAX_NUM_SEQS=$(( MAX_NUM_SEQS > 8 ? 8 : MAX_NUM_SEQS ))
    log "Retrying vLLM with max-model-len $MAX_MODEL_LEN and max-num-seqs $MAX_NUM_SEQS."
    start_server "$server_log" --enforce-eager || fail "vLLM did not start. Read $server_log."
}

# take_run_lock NAME: makes sure that only one job runs a dataset at a time,
# so two jobs never write the same results file. Ends this job (exit 0) if
# another live job runs the same dataset.
take_run_lock() {
    local lock="$LOCK_DIR/run-$1" owner
    until mkdir "$lock" 2>/dev/null; do
        owner=$(cat "$lock/job" 2>/dev/null || true)
        if [[ -n $owner ]] && ! squeue -h -j "$owner" 2>/dev/null | grep -q .; then
            log "Removing stale lock $lock from ended job $owner."
            rm -rf "$lock"
            continue
        fi
        log "Job ${owner:-unknown} already runs $1. This job has nothing to do."
        exit 0
    done
    echo "$SLURM_JOB_ID" >"$lock/job"
    HELD_LOCK=$lock
}

# ------------------------------------------------------------- main steps ---

# Preparation without a GPU: checks, environments, model and data.
prepare_all() {
    report_node
    check_network
    check_disk
    ensure_uv
    ensure_eval_env
    ensure_vllm_env
    ensure_model
    local dataset
    for dataset in trec cb-test_labeled rte dbpedia wos; do
        ensure_data "$RUNS_DIR/$dataset"
    done
    log "Preparation finished. The GPU jobs will reuse these files."
}

# run_dataset DATASET: the full job for one dataset.
run_dataset() {
    local dataset=$1
    local smoke=${SMOKE:-0}
    local cb_file=${CB_FILE:-test_labeled}
    local run_name=$dataset
    [[ $dataset == cb ]] && run_name="cb-$cb_file"
    [[ $smoke == 1 ]] && run_name="$run_name-smoke"
    local run_dir="$RUNS_DIR/$run_name"
    local log_dir="$run_dir/logs"
    mkdir -p "$log_dir"

    report_node
    [[ $(gpu_count) -gt 0 ]] || fail "This job has no GPU. Submit it with --gres=gpu:..."
    check_network
    check_disk
    ensure_uv
    ensure_eval_env
    choose_cuda_variant
    ensure_vllm_env
    ensure_model
    ensure_data "$run_dir"

    local args=(--dataset "$dataset" --work-dir "$run_dir")
    [[ $dataset == cb ]] && args+=(--cb-file "$cb_file")
    [[ $smoke == 1 ]] && args+=(--smoke)

    take_run_lock "$run_name"
    # A continuation job ends here if an earlier job finished the dataset.
    if "$EVAL_ENV/bin/python" "$REPO_DIR/scripts/run_qwen.py" "${args[@]}" --status-only; then
        log "All samples of $run_name are finished. Nothing to do."
        return 0
    fi

    serve_model "$log_dir/vllm-$SLURM_JOB_ID.log"

    args+=(--api-base "http://127.0.0.1:$PORT/v1" --model "$MODEL_ID"
        --max-model-len "$MAX_MODEL_LEN" --ray-cpus "${SLURM_CPUS_PER_TASK:-8}")

    log "Running the paper code for $dataset. Results go to $run_dir/results."
    PYTHONPATH="$REPO_DIR/scripts:$CODE_DIR" \
        QWEN_REGISTER_PROVIDER=1 \
        QWEN_MODEL="$MODEL_ID" \
        QWEN_API_BASE="http://127.0.0.1:$PORT/v1" \
        "$EVAL_ENV/bin/python" -u "$REPO_DIR/scripts/run_qwen.py" "${args[@]}" &
    RUNNER_PID=$!

    # The paper code retries failed requests forever. If vLLM stops, end the
    # job instead of waiting until the time limit.
    local rc=0
    while kill -0 "$RUNNER_PID" 2>/dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log "vLLM stopped during the run. Last lines of its log:"
            tail -n 40 "$log_dir/vllm-$SLURM_JOB_ID.log" >&2
            fail "vLLM stopped. Resubmit the job. Finished samples are kept."
        fi
        sleep 30
    done
    wait "$RUNNER_PID" || rc=$?
    RUNNER_PID=""
    (( rc == 0 )) || fail "The runner stopped with exit code $rc."
    log "Finished $dataset. Results: $run_dir/results"
}
