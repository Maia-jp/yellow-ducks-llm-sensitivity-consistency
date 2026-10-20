#!/usr/bin/env bash
set -euo pipefail

if [[ ${1:-} == --help ]]; then
    printf 'Usage: %s [ITU_USERNAME]\n' "$0"
    printf 'Start or reuse the shared HPC SSH connection. Default username: s25441.\n'
    printf 'Enter your password in the terminal. Keep the connection open for agent access.\n'
    exit 0
fi

if (( $# > 1 )); then
    printf 'Usage: %s [ITU_USERNAME]\n' "$0" >&2
    exit 2
fi

username=${1:-s25441}
socket="$HOME/.ssh/itu-hpc-control"

exec ssh -o ControlMaster=auto -S "$socket" "${username}@hpc.itu.dk"
