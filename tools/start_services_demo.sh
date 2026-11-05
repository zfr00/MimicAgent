#!/usr/bin/env bash
set -euo pipefail
TOOL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TOOL_LOG_DIR="${TOOL_DIR}/logs"
export PYTHON="${PYTHON:-${PY:-python3}}"
export PYTHONPATH="${TOOL_DIR}/..${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "$TOOL_LOG_DIR"

process_birth() {
    local stat
    local -a fields
    [[ -r "/proc/$1/stat" ]] || return 1
    stat="$(<"/proc/$1/stat")"
    read -r -a fields <<<"${stat##*) }"
    printf '%s\n' "${fields[19]}"
}

stop_service() {
    local name="$1" pid birth current pgid
    local pidfile="${TOOL_LOG_DIR}/${name}.pid"
    [[ -f "$pidfile" ]] || return 0
    read -r pid birth < "$pidfile" || true
    if [[ "$pid" =~ ^[1-9][0-9]*$ && "$birth" =~ ^[0-9]+$ ]] && ((pid > 1)); then
        current="$(process_birth "$pid" || true)"
        pgid="$(ps -o pgid= -p "$pid" 2>/dev/null || true)"
        if [[ -z "$current" || ( "$current" == "$birth" && "${pgid// /}" == "$pid" ) ]]; then
            kill -TERM -- "-$pid" 2>/dev/null || true
            for ((attempt=0; attempt<50; attempt++)); do
                kill -0 -- "-$pid" 2>/dev/null || break
                sleep .1
            done
            kill -KILL -- "-$pid" 2>/dev/null || true
        fi
    fi
    rm -f "$pidfile"
}

if [[ "${1:-}" == --stop ]]; then
    for name in executor fathom kg_backend; do
        stop_service "$name"
    done
    exit 0
fi
[[ $# == 0 ]] || { echo 'Usage: bash tools/start_services_demo.sh [--stop]' >&2; exit 2; }
: "${ARIS_TOOL_API_KEY:?Set ARIS_TOOL_API_KEY}"
export CURATORIAL_REPO_PATH="${CURATORIAL_REPO_PATH:-${TOOL_DIR}/../data/exemplars/curatorial_repo.jsonl}"
[[ -r "$CURATORIAL_REPO_PATH" ]] || { echo 'Unreadable CURATORIAL_REPO_PATH' >&2; exit 1; }
started=()
startup_complete=false
cleanup_startup() {
    if [[ "$startup_complete" == false ]]; then
        for ((index=${#started[@]}-1; index>=0; index--)); do
            stop_service "${started[index]}"
        done
    fi
}
trap cleanup_startup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_service() {
    local name="$1" pid birth current
    shift
    local pidfile="${TOOL_LOG_DIR}/${name}.pid"
    if [[ -f "$pidfile" ]]; then
        read -r pid birth < "$pidfile" || true
        current="$(process_birth "$pid" || true)"
        if [[ -n "$birth" && "$current" == "$birth" ]]; then
            echo "$name already running"
            return
        fi
    fi
    setsid nohup "$@" >"${TOOL_LOG_DIR}/${name}.log" 2>&1 &
    pid=$!
    birth="$(process_birth "$pid" || printf '0')"
    printf '%s %s\n' "$pid" "$birth" > "$pidfile"
    started+=("$name")
    echo "started $name"
}
if [[ "${START_FATHOM:-1}" == 1 ]]; then
    start_service fathom bash "${TOOL_DIR}/start_fathom.sh"
fi
if [[ "${START_KG_BACKEND:-1}" == 1 ]]; then
    start_service kg_backend bash "${TOOL_DIR}/start_kg.sh"
fi
start_service executor "$PYTHON" -m uvicorn tools.executor_server:app --host "${TOOL_HOST:-127.0.0.1}" --port 8000
"$PYTHON" - <<'PY'
import time
import requests
error = ''
with requests.Session() as session:
    session.trust_env = False
    deadline = time.monotonic() + 90
    while time.monotonic() < deadline:
        try:
            response = session.get('http://127.0.0.1:8000/health', timeout=12)
            response.raise_for_status()
            data = response.json()
            if data.get('status') == 'ok':
                print('Five tools ready at http://127.0.0.1:8000')
                break
            error = str(data)
        except (requests.RequestException, ValueError) as exc:
            error = str(exc)
        time.sleep(.5)
    else:
        raise SystemExit('Tool startup failed; inspect tools/logs/: ' + error)
PY
for name in "${started[@]}"; do
    read -r pid birth < "${TOOL_LOG_DIR}/${name}.pid"
    [[ "$(process_birth "$pid" || true)" == "$birth" ]] || {
        echo "Service exited during startup: $name" >&2
        exit 1
    }
done
startup_complete=true
