#!/usr/bin/env bash
# Shared steps for the DeepSeek-R1 7B Slurm jobs in scripts/hpc/jobs/deepseek-r1-7b/.
#
# The job files source this file. Do not run it on the login node:
# it installs packages and can download the model, so it refuses to start
# outside a Slurm job.
#
# Adapted from scripts/hpc/qwen_common.sh (branch qwen3.5-27b-jobs). The
# environment, lock and dataset steps are the same. The model server is
# Ollama in place of vLLM.
#
# Two kinds of nodes work:
# - ARM nodes (asus1): the Ollama program of the node, /usr/local/bin/ollama
#   (version 0.40.4 on 2026-11-15). This is the default of the job files.
# - x86 nodes with AVX-512: the ITU module ollama/0.6.0 (jobs in x86/).
# The two Ollama versions are not the same program. A run folder remembers
# its Ollama version, and a job with another version refuses to continue it.
#
# Every step is idempotent. A step that is already done is skipped, so the
# first job builds the environment and later jobs reuse it.
# Shared steps take a lock, so parallel jobs do not build the same thing twice.

set -euo pipefail

if [[ -z ${SLURM_JOB_ID:-} ]]; then
    echo "Run this only inside a Slurm job (sbatch). SLURM_JOB_ID is not set." >&2
    exit 1
fi

# ---------------------------------------------------------------- settings ---

REPO_DIR=${SLURM_SUBMIT_DIR:?}
CODE_DIR="$REPO_DIR/original-paper"
PY_DIR="$REPO_DIR/scripts/deepseek"
LOCK_FILES="$REPO_DIR/jobs/deepseek-r1-7b/locks"

ARCH=${YD_ARCH:-$(uname -m)}  # x86_64 or aarch64

# Override these with sbatch --export=ALL,NAME=value.
# OLLAMA_MODULE: a module name, or "system" for the ollama program of the node.
if [[ $ARCH == aarch64 ]]; then
    OLLAMA_MODULE=${OLLAMA_MODULE:-system}
else
    OLLAMA_MODULE=${OLLAMA_MODULE:-ollama/0.6.0-GCCcore-13.3.0-CUDA-12.6.0}
fi
BASE_MODEL=${BASE_MODEL:-deepseek-r1:7b}
# ID that "ollama list" showed for deepseek-r1:7b on 2026-11-14. The job stops
# if the tag points to other weights, so every run uses the same model.
BASE_MODEL_ID=${BASE_MODEL_ID:-755ced02ce7b}
MODEL_TAG="deepseek-r1-7b"
CONTEXT_LENGTH=${CONTEXT_LENGTH:-8192}  # tokens for one request: prompt + reasoning + answer
NUM_PARALLEL=${NUM_PARALLEL:-8}         # requests that Ollama answers at the same time
MAX_TOKENS=${MAX_TOKENS:-2048}          # limit for reasoning + answer of one request
RAY_CPUS=${RAY_CPUS:-2}                 # samples that the paper code handles at the same time
SERVER_START_TIMEOUT=${SERVER_START_TIMEOUT:-600}
# Ollama model with the context length set. The OpenAI-compatible API cannot
# set the context for each request, so the job makes this model from BASE_MODEL.
SERVED_MODEL="$MODEL_TAG-ctx$CONTEXT_LENGTH"

UV_VERSION="0.13.6"
# SHA-256 of the x86_64 archive. For aarch64 the job reads the .sha256 file
# of the same release and prints the value into the log.
UV_SHA256_X86_64="f29c3e94ffabd7e6f3ba521dba169a9b7ec12100bf3257087f0f24f6123f3645"

# Everything lives under one folder in the home directory.
# The model is in the Ollama store (default ~/.ollama/models, about 4.7 GB).
YD_ROOT=${YD_ROOT:-$HOME/yellow-ducks}
# The home folder is the same on all nodes, so programs for the two CPU
# types get separate folders.
UV_DIR="$YD_ROOT/tools/uv-$UV_VERSION-$ARCH"
EVAL_ENV="$YD_ROOT/envs/eval-py310-$ARCH"
RUNS_DIR="$YD_ROOT/runs/$MODEL_TAG"
LOCK_DIR="$YD_ROOT/locks"

export TMPDIR=${TMPDIR:-$HOME/tmp}
JOB_TMP="$TMPDIR/yd-$SLURM_JOB_ID"
export UV_CACHE_DIR="$YD_ROOT/cache/uv"
export UV_PYTHON_INSTALL_DIR="$YD_ROOT/python"
export UV_PYTHON_PREFERENCE=only-managed
export UV_LINK_MODE=copy
export UV_HTTP_TIMEOUT=300
export UV_HTTP_RETRIES=8
export UV_NO_PROGRESS=1
export PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
unset PYTHONPATH RAY_ADDRESS VIRTUAL_ENV CONDA_PREFIX
mkdir -p "$TMPDIR" "$JOB_TMP" "$YD_ROOT" "$LOCK_DIR"

# The Ollama module changes LD_LIBRARY_PATH. Python runs with the value from
# before the module load, so the module libraries cannot change its behaviour.
BASE_LD_LIBRARY_PATH=${LD_LIBRARY_PATH:-}

HELD_LOCK=""
SERVER_PID=""
RUNNER_PID=""
HEARTBEAT_PID=""
PORT=""
OLLAMA_VERSION=""

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
    if [[ -n $HEARTBEAT_PID ]]; then
        kill "$HEARTBEAT_PID" 2>/dev/null
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

# While this job holds a lock, a background loop renews the time stamp of
# the lock every minute. Nodes without the squeue command (possible on the
# ARM nodes) use the time stamp to see if the owner of a lock is still alive.
start_heartbeat() {
    (
        while sleep 60; do
            lock=$(cat "$JOB_TMP/held-lock" 2>/dev/null || true)
            if [[ -n $lock && -d $lock ]]; then touch "$lock/job" 2>/dev/null || true; fi
        done
    ) &
    HEARTBEAT_PID=$!
}
start_heartbeat

hold_lock() {
    echo "$SLURM_JOB_ID" >"$1/job"
    HELD_LOCK=$1
    echo "$1" >"$JOB_TMP/held-lock"
}

# lock_is_live LOCK: true if the job that made the lock still runs.
lock_is_live() {
    local lock=$1 owner
    owner=$(cat "$lock/job" 2>/dev/null || true)
    if command -v squeue >/dev/null 2>&1; then
        [[ -n $owner ]] || return 0  # the owner is just writing its job ID
        squeue -h -j "$owner" 2>/dev/null | grep -q .
        return
    fi
    # No squeue here: the lock is live if its time stamp is younger than 10 minutes.
    [[ -n $(find "$lock" -maxdepth 1 -mmin -10 2>/dev/null) ]]
}

# acquire_lock NAME: wait until no other job builds the same thing.
# A lock from a job that no longer runs is stale and is removed.
acquire_lock() {
    local lock="$LOCK_DIR/$1" waited=0
    until mkdir "$lock" 2>/dev/null; do
        if ! lock_is_live "$lock"; then
            log "Removing stale lock $lock from ended job $(cat "$lock/job" 2>/dev/null || echo unknown)."
            rm -rf "$lock"
            continue
        fi
        if (( waited >= 10800 )); then
            fail "Waited 3 hours for lock $lock (owner job $(cat "$lock/job" 2>/dev/null || echo unknown))."
        fi
        if (( waited % 300 == 0 )); then
            log "Waiting for lock $lock (owner job $(cat "$lock/job" 2>/dev/null || echo unknown))."
        fi
        sleep 30
        waited=$((waited + 30))
    done
    hold_lock "$lock"
}

release_lock() {
    rm -rf "$HELD_LOCK"
    HELD_LOCK=""
    : >"$JOB_TMP/held-lock"
}

# ------------------------------------------------------------- node facts ---

# Number of GPUs that this job can use. Asks the driver, because the ARM
# nodes have no Slurm GPU resource (no --gres, no CUDA_VISIBLE_DEVICES).
# On the other nodes a job without --gres sees no GPU.
gpu_count() {
    command -v nvidia-smi >/dev/null 2>&1 || { echo 0; return; }
    nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true
}

report_node() {
    log "Host: $(hostname). Job: $SLURM_JOB_ID. CPUs: ${SLURM_CPUS_PER_TASK:-?}. CPU type: $ARCH."
    log "CPU: $(grep -m 1 -E 'model name|Model' /proc/cpuinfo | cut -d: -f2- | sed 's/^ *//' || true)"
    log "GPUs given by Slurm: ${CUDA_VISIBLE_DEVICES:-none}. GPUs that the driver shows: $(gpu_count)."
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=index,name,memory.total,memory.used,driver_version --format=csv 2>/dev/null || true
    fi
    df -h "$YD_ROOT" "$TMPDIR" 2>/dev/null || true
    if command -v quota >/dev/null 2>&1; then
        quota -s 2>/dev/null || true
    fi
}

# The ITU Ollama module uses AVX-512 instructions. On a CPU without them
# (for example cn9, AMD EPYC 7643) the model load stops with
# "SIGILL: illegal instruction" (seen on 2026-11-14). Stop early with a clear text.
check_cpu() {
    if [[ $OLLAMA_MODULE == system ]]; then
        log "CPU check not needed: the job uses the Ollama program of the node."
        return 0
    fi
    if [[ ${SKIP_CPU_CHECK:-0} == 1 ]]; then
        log "CPU check skipped (SKIP_CPU_CHECK=1)."
        return 0
    fi
    if ! grep -q -w avx512f /proc/cpuinfo; then
        fail "The CPU of $(hostname) has no AVX-512. The Ollama module does not work here. Submit with --constraint=arch_icelake. For an Ollama module that works without AVX-512, submit with --export=ALL,SKIP_CPU_CHECK=1."
    fi
    log "CPU check OK: AVX-512 is available."
}

# Fails early if a download host is not reachable from this node.
# Each host gets 3 tries: single timeouts happen.
check_network() {
    local url try
    for url in \
        https://github.com \
        https://pypi.org/simple/pip/ \
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
# The paper environment needs about 2 GB and the datasets of one run about 0.3 GB.
check_disk() {
    local need=1 free_gb
    [[ -f $EVAL_ENV/.ready ]] || need=$((need + 3))
    free_gb=$(df -Pk "$YD_ROOT" | awk 'NR == 2 {print int($4 / 1048576)}')
    log "Free space in $YD_ROOT: ${free_gb} GB. Needed for missing parts: about ${need} GB."
    if (( free_gb < need )); then
        fail "Not enough disk space in $YD_ROOT. Free some space or set YD_ROOT to a larger folder."
    fi
}

# --------------------------------------------------------------- installs ---

ensure_uv() {
    [[ -x $UV_DIR/uv ]] && return 0
    acquire_lock "uv-$ARCH"
    if [[ ! -x $UV_DIR/uv ]]; then
        log "Installing uv $UV_VERSION ($ARCH) into $UV_DIR."
        local name="uv-$ARCH-unknown-linux-gnu.tar.gz" sha
        local base="https://github.com/astral-sh/uv/releases/download/$UV_VERSION"
        local archive="$JOB_TMP/$name"
        curl -fsSL --retry 3 -o "$archive" "$base/$name"
        if [[ $ARCH == x86_64 ]]; then
            sha=$UV_SHA256_X86_64
        else
            sha=$(curl -fsSL --retry 3 "$base/$name.sha256" | awk '{print $1}')
            [[ $sha =~ ^[0-9a-f]{64}$ ]] || fail "Cannot read the SHA-256 of $name from the uv release."
            log "SHA-256 of $name from the uv release: $sha"
        fi
        echo "$sha  $archive" | sha256sum -c -
        rm -rf "$UV_DIR.partial"
        mkdir -p "$UV_DIR.partial"
        tar -xzf "$archive" -C "$UV_DIR.partial" --strip-components=1
        rm -rf "$UV_DIR"
        mv "$UV_DIR.partial" "$UV_DIR"
    fi
    release_lock
}

uv() { "$UV_DIR/uv" "$@"; }

# Runs the Python of the paper environment without the module libraries.
eval_python() { LD_LIBRARY_PATH="$BASE_LD_LIBRARY_PATH" "$EVAL_ENV/bin/python" "$@"; }

# build_env PATH PYTHON_VERSION LOCK_FILE
# Installs the exact versions in LOCK_FILE. Writes PATH/.ready at the end,
# so an environment from a failed build is never used.
build_env() {
    local env=$1 python=$2 lock=$3
    [[ -f $env/.ready ]] && return 0
    acquire_lock "env-$(basename "$env")"
    if [[ ! -f $env/.ready ]]; then
        log "Building $env from $(basename "$lock")."
        rm -rf "$env"
        uv venv --python "$python" "$env"
        uv pip sync --python "$env/bin/python" "$lock"
        cp "$lock" "$env/lock.txt"
        touch "$env/.ready"
        log "Built $env."
    fi
    release_lock
}

ensure_eval_env() {
    build_env "$EVAL_ENV" 3.10 "$LOCK_FILES/eval-py310.txt"
    eval_python -c 'import langchain, langchain_openai, ray, sklearn, seaborn, scipy; print("eval env OK: ray", ray.__version__)'
}

# Downloads and verifies the datasets into the run folder of one dataset.
# scripts/fetch_datasets.py checks every file against docs/datasets/data-manifest.json.
ensure_data() {
    local run_dir=$1
    mkdir -p "$run_dir"
    acquire_lock "data-$(basename "$run_dir")"
    log "Downloading and verifying the datasets into $run_dir/data."
    eval_python "$REPO_DIR/scripts/fetch_datasets.py" --out "$run_dir/data"
    release_lock
}

# ---------------------------------------------------------- Ollama server ---

# Makes the ollama command available: the program of the node
# (OLLAMA_MODULE=system) or an ITU module. Lmod scripts use unset variables,
# so "set -u" is off while they run.
load_ollama() {
    local file ok=0
    if [[ $OLLAMA_MODULE == system ]]; then
        command -v ollama >/dev/null 2>&1 || fail "This node has no ollama program. Use a node with Ollama, or set OLLAMA_MODULE to a module name."
        log "Using the Ollama program of the node: $(command -v ollama)"
        return 0
    fi
    set +u
    if ! command -v module >/dev/null 2>&1; then
        for file in /etc/profile.d/lmod.sh /etc/profile.d/z00_lmod.sh /etc/profile.d/modules.sh; do
            # shellcheck disable=SC1090
            [[ -r $file ]] && source "$file"
        done
    fi
    if command -v module >/dev/null 2>&1 && module load "$OLLAMA_MODULE"; then
        ok=1
    fi
    set -u
    (( ok == 1 )) || fail "Cannot load the module $OLLAMA_MODULE. Check the name with: module avail ollama"
    command -v ollama >/dev/null 2>&1 || fail "The module $OLLAMA_MODULE is loaded, but the ollama command is missing."
    log "Ollama module loaded: $(command -v ollama)"
}

free_port() {
    eval_python -c 'import socket; s = socket.socket(); s.bind(("127.0.0.1", 0)); print(s.getsockname()[1])'
}

# start_server LOG_FILE: starts a private Ollama server on a free port and
# waits until it answers.
#
# The port matters. Some nodes run an Ollama service of the node on the
# default port 11434 (cn19 and asus1). Without OLLAMA_HOST the
# ollama command talks to that service: another Ollama version, other
# settings and another model store.
start_server() {
    local server_log=$1 waited=0
    mkdir -p "$(dirname "$server_log")"
    PORT=$(free_port)
    export OLLAMA_HOST="127.0.0.1:$PORT"
    export OLLAMA_NUM_PARALLEL="$NUM_PARALLEL"
    export OLLAMA_CONTEXT_LENGTH="$CONTEXT_LENGTH"
    export OLLAMA_MAX_LOADED_MODELS=1
    export OLLAMA_MAX_QUEUE=4096
    export OLLAMA_KEEP_ALIVE=-1
    log "Starting Ollama on port $PORT: $NUM_PARALLEL parallel requests, context $CONTEXT_LENGTH tokens."
    ollama serve >>"$server_log" 2>&1 &
    SERVER_PID=$!

    until curl -fsS -o /dev/null --max-time 10 "http://127.0.0.1:$PORT/api/version" 2>/dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            wait "$SERVER_PID" 2>/dev/null || true
            SERVER_PID=""
            log "Ollama stopped before it was ready. Last lines of $server_log:"
            tail -n 40 "$server_log" >&2
            fail "Ollama did not start."
        fi
        if (( waited >= SERVER_START_TIMEOUT )); then
            tail -n 40 "$server_log" >&2
            fail "Ollama was not ready after $SERVER_START_TIMEOUT s."
        fi
        sleep 2
        waited=$((waited + 2))
    done
    OLLAMA_VERSION=$(curl -fsS --max-time 10 "http://127.0.0.1:$PORT/api/version" | sed -E 's/.*"version" *: *"([^"]*)".*/\1/')
    log "Ollama is ready after about $waited s. Server version: $OLLAMA_VERSION (source: $OLLAMA_MODULE)."
}

# check_run_version RUN_DIR: one run folder holds results of one Ollama
# version only. The first job writes the version, later jobs compare it.
check_run_version() {
    local file="$1/ollama-version.txt" now="$OLLAMA_MODULE $OLLAMA_VERSION $ARCH" before
    if [[ -f $file ]]; then
        before=$(cat "$file")
        if [[ $before != "$now" ]]; then
            fail "This run folder has results from Ollama '$before'. This job uses '$now'. Do not mix them: run on the same kind of node, or set YD_ROOT to another folder."
        fi
    else
        echo "$now" >"$file"
    fi
    log "Run folder $1: Ollama '$now'."
}

# Prints the ID that "ollama list" shows for a model name, or nothing.
model_id() {
    local list_out
    list_out=$(ollama list)
    awk -v name="$1" 'NR > 1 && ($1 == name || $1 == name ":latest") {print $2; exit}' <<<"$list_out"
}

# Makes sure that the base model is in the store, checks its ID and makes the
# served model with the context length.
ensure_model() {
    local id
    acquire_lock model
    if [[ -z $(model_id "$BASE_MODEL") ]]; then
        log "Downloading $BASE_MODEL (about 4.7 GB) into the Ollama store."
        ollama pull "$BASE_MODEL" >"$JOB_TMP/pull.log" 2>&1 || { tail -n 20 "$JOB_TMP/pull.log" >&2; fail "ollama pull $BASE_MODEL failed."; }
    fi
    id=$(model_id "$BASE_MODEL")
    [[ -n $id ]] || fail "$BASE_MODEL is not in the Ollama store after the download."
    log "Base model: $BASE_MODEL, ID $id."
    if [[ $id != "$BASE_MODEL_ID" && ${ALLOW_OTHER_MODEL_ID:-0} != 1 ]]; then
        fail "$BASE_MODEL has ID $id, expected $BASE_MODEL_ID. The tag points to other weights. Submit with --export=ALL,ALLOW_OTHER_MODEL_ID=1 to use it, and record the ID."
    fi
    printf 'FROM %s\nPARAMETER num_ctx %s\n' "$BASE_MODEL" "$CONTEXT_LENGTH" >"$JOB_TMP/Modelfile"
    ollama create "$SERVED_MODEL" -f "$JOB_TMP/Modelfile" >"$JOB_TMP/create.log" 2>&1 \
        || { tail -n 20 "$JOB_TMP/create.log" >&2; fail "ollama create $SERVED_MODEL failed."; }
    [[ -n $(model_id "$SERVED_MODEL") ]] || fail "$SERVED_MODEL is not in the Ollama store after ollama create."
    log "Served model: $SERVED_MODEL (from $BASE_MODEL, num_ctx $CONTEXT_LENGTH)."
    release_lock
}

# Loads the model into the GPU and checks that all of it is on the GPU.
load_model() {
    local server_log=$1 ctx
    log "Loading $SERVED_MODEL into the GPU."
    if ! curl -fsS -o "$JOB_TMP/load.json" --max-time 900 -H 'Content-Type: application/json' \
        -d "{\"model\": \"$SERVED_MODEL\", \"prompt\": \"\", \"keep_alive\": -1}" \
        "http://127.0.0.1:$PORT/api/generate"; then
        log "The model did not load. Last lines of $server_log without the stack trace:"
        grep -v 'swbuilder' "$server_log" | grep -E 'level=(ERROR|WARN)|SIG|fatal|panic' | tail -n 20 >&2 || true
        fail "Ollama could not load $SERVED_MODEL. 'SIGILL' in the log means a CPU without AVX-512."
    fi
    # Read "ollama ps" once into a variable. A pipe such as "ollama ps | head"
    # can end ollama early, and with "set -o pipefail" the test then fails
    # (seen on 2026-11-15: the context check below was skipped).
    local ps_out
    ps_out=$(ollama ps)
    printf '%s\n' "$ps_out"
    if ! grep -F "$SERVED_MODEL" <<<"$ps_out" | grep -q '100% GPU'; then
        fail "$SERVED_MODEL is not fully on the GPU (see 'ollama ps' above). A run on the CPU is too slow."
    fi
    # Newer Ollama versions show the context of one request in "ollama ps" (column CONTEXT).
    if grep -q CONTEXT <<<"$ps_out"; then
        ctx=$(grep -F "$SERVED_MODEL" <<<"$ps_out" | awk '{for (i = 1; i < NF; i++) if ($i == "GPU") {print $(i + 1); exit}}')
        if [[ $ctx == "$CONTEXT_LENGTH" ]]; then
            log "Context check OK: 'ollama ps' shows a context of $ctx tokens."
        else
            fail "'ollama ps' shows a context of '$ctx' tokens, expected $CONTEXT_LENGTH."
        fi
        nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv 2>/dev/null || true
        return 0
    fi
    # Ollama 0.6.0: the server log shows the total context: parallel requests x context of one request.
    ctx=$(grep -oE -- '--ctx-size [0-9]+' "$server_log" | tail -n 1 | awk '{print $2}' || true)
    if [[ -n $ctx ]]; then
        if (( ctx == NUM_PARALLEL * CONTEXT_LENGTH )); then
            log "Context check OK: total context $ctx = $NUM_PARALLEL x $CONTEXT_LENGTH."
        else
            log "WARNING: the server log shows a total context of $ctx tokens, expected $((NUM_PARALLEL * CONTEXT_LENGTH)). Ollama may have lowered the number of parallel requests. The prompt length check of the runner decides."
        fi
    else
        log "WARNING: no --ctx-size in the server log. The prompt length check of the runner decides."
    fi
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv 2>/dev/null || true
}

# take_run_lock NAME: makes sure that only one job runs a dataset at a time,
# so two jobs never write the same results file. Ends this job (exit 0) if
# another live job runs the same dataset.
take_run_lock() {
    local lock="$LOCK_DIR/run-$MODEL_TAG-$1"
    until mkdir "$lock" 2>/dev/null; do
        if ! lock_is_live "$lock"; then
            log "Removing stale lock $lock from ended job $(cat "$lock/job" 2>/dev/null || echo unknown)."
            rm -rf "$lock"
            continue
        fi
        log "Job $(cat "$lock/job" 2>/dev/null || echo unknown) already runs $1. This job has nothing to do."
        exit 0
    done
    hold_lock "$lock"
}

# ------------------------------------------------------------- main steps ---

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
    local server_log="$log_dir/ollama-$SLURM_JOB_ID.log"
    mkdir -p "$log_dir"

    report_node
    [[ $(gpu_count) -gt 0 ]] || fail "This job sees no GPU. On the x86 nodes submit it with --gres=gpu:1."
    check_cpu
    check_network
    check_disk
    ensure_uv
    ensure_eval_env
    ensure_data "$run_dir"

    local args=(--dataset "$dataset" --work-dir "$run_dir")
    [[ $dataset == cb ]] && args+=(--cb-file "$cb_file")
    [[ $smoke == 1 ]] && args+=(--smoke)

    take_run_lock "$run_name"
    # A continuation job ends here if an earlier job finished the dataset.
    if PYTHONPATH="$PY_DIR:$CODE_DIR" eval_python "$PY_DIR/run_deepseek.py" "${args[@]}" --status-only; then
        log "All samples of $run_name are finished. Nothing to do."
        return 0
    fi
    release_lock

    load_ollama
    start_server "$server_log"
    check_run_version "$run_dir"
    ensure_model
    take_run_lock "$run_name"
    load_model "$server_log"

    args+=(--api-base "http://127.0.0.1:$PORT/v1" --model "$SERVED_MODEL"
        --context-length "$CONTEXT_LENGTH" --max-tokens "$MAX_TOKENS" --ray-cpus "$RAY_CPUS")

    log "Running the paper code for $dataset. Results go to $run_dir/results."
    PYTHONPATH="$PY_DIR:$CODE_DIR" \
        LD_LIBRARY_PATH="$BASE_LD_LIBRARY_PATH" \
        DEEPSEEK_REGISTER_PROVIDER=1 \
        DEEPSEEK_MODEL="$SERVED_MODEL" \
        DEEPSEEK_API_BASE="http://127.0.0.1:$PORT/v1" \
        DEEPSEEK_MAX_TOKENS="$MAX_TOKENS" \
        "$EVAL_ENV/bin/python" -u "$PY_DIR/run_deepseek.py" "${args[@]}" &
    RUNNER_PID=$!

    # The paper code retries failed requests forever. If Ollama stops, end the
    # job instead of waiting until the time limit.
    local rc=0
    while kill -0 "$RUNNER_PID" 2>/dev/null; do
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log "Ollama stopped during the run. Last lines of its log:"
            tail -n 40 "$server_log" >&2
            fail "Ollama stopped. Resubmit the job. Finished samples are kept."
        fi
        sleep 30
    done
    wait "$RUNNER_PID" || rc=$?
    RUNNER_PID=""
    (( rc == 0 )) || fail "The runner stopped with exit code $rc."
    log "Finished $dataset. Results: $run_dir/results"
}
