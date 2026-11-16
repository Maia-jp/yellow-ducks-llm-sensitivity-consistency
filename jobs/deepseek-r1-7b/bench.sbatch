#!/bin/bash
# Speed test for DeepSeek-R1 7B on asus1: how many requests per second does
# the private Ollama server answer with 1, 4, 8 and 16 requests at the same time?
# It also tries to turn the reasoning off through Ollama's own API.
# Submit from the repository root on the login node:
#   sbatch jobs/deepseek-r1-7b/bench.sbatch
# It writes no results of the paper. Read the numbers in the .out file.
#SBATCH --job-name=deepseek7b-bench
#SBATCH --output=jobs/deepseek-r1-7b/slurm-logs/%x-%j.out
#SBATCH --error=jobs/deepseek-r1-7b/slurm-logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --nodelist=asus1
#SBATCH --partition=asus1
#SBATCH --time=00:30:00

set -euo pipefail
export TMPDIR=$HOME/tmp
mkdir -p "$TMPDIR"
export NUM_PARALLEL=${NUM_PARALLEL:-16}

source "$SLURM_SUBMIT_DIR/scripts/hpc/deepseek_common.sh"

SERVER_LOG="$RUNS_DIR/bench/logs/ollama-$SLURM_JOB_ID.log"
MESSAGES='"messages":[{"role":"system","content":"You are a text classifier."},{"role":"user","content":"Classify the question as Number, Location, Person, Description, Entity, or Abbreviation. Use one word.\nHere is the text: What is the capital of Denmark?"}]'

# bench N: sends N requests at the same time and waits for these N only.
# (A plain "wait" would also wait for the Ollama server, which never ends.)
bench() {
    local n=$1 pids="" start end tokens i
    rm -f "$JOB_TMP"/bench_*.json
    start=$(date +%s.%N)
    for i in $(seq "$n"); do
        curl -s --max-time 900 "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
            -d "{\"model\":\"$SERVED_MODEL\",$MESSAGES,\"temperature\":0,\"seed\":42,\"max_tokens\":$MAX_TOKENS}" \
            -o "$JOB_TMP/bench_$i.json" &
        pids="$pids $!"
    done
    # shellcheck disable=SC2086
    wait $pids
    end=$(date +%s.%N)
    tokens=$(cat "$JOB_TMP"/bench_*.json | grep -oE '"completion_tokens": *[0-9]+' | grep -oE '[0-9]+$' | awk '{x += $1} END {print x + 0}')
    awk -v n="$n" -v s="$start" -v e="$end" -v t="$tokens" 'BEGIN {
        d = e - s
        printf "BENCH %2d at the same time: %6.1f s, %5d tokens, %6.1f tokens/s, %5.2f requests/s\n", n, d, t, t / d, n / d
        printf "      at this speed: CB %.0f h, TREC %.0f h, DBPedia %.0f h, WoS %.0f h, RTE %.0f h (same request length assumed)\n",
            22500 / (n / d) / 3600, 45000 / (n / d) / 3600, 180000 / (n / d) / 3600, 180000 / (n / d) / 3600, 224100 / (n / d) / 3600
    }'
}

report_node
ensure_uv
ensure_eval_env
load_ollama
start_server "$SERVER_LOG"
ensure_model
load_model "$SERVER_LOG"

log "Warm-up request."
bench 1 >/dev/null
for n in 1 4 8 16; do
    (( n <= NUM_PARALLEL )) && bench "$n"
done

log "Reasoning off through Ollama's own API (/api/chat with think=false):"
curl -s --max-time 900 "http://127.0.0.1:$PORT/api/chat" \
    -d "{\"model\":\"$SERVED_MODEL\",\"think\":false,\"stream\":false,$MESSAGES,\"options\":{\"temperature\":0,\"seed\":42,\"num_predict\":$MAX_TOKENS}}" \
    -o "$JOB_TMP/think_off.json"
echo "THINK-OFF answer: $(grep -oE '"content": *"[^"]*"' "$JOB_TMP/think_off.json" || true)"
echo "THINK-OFF tokens written (eval_count): $(grep -oE '"eval_count": *[0-9]+' "$JOB_TMP/think_off.json" | grep -oE '[0-9]+$' || true)"
echo "THINK-OFF error text, if any: $(grep -oE '"error": *"[^"]*"' "$JOB_TMP/think_off.json" || true)"
log "Speed test finished."
