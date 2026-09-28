#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"
mkdir -p runs

PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi

log_path="$REPO_ROOT/runs/full1000_tmux_20260926.log"
status_path="$REPO_ROOT/runs/full1000_tmux.exitcode"
status_tmp="$REPO_ROOT/runs/full1000_tmux.exitcode.tmp"

printf 'running\n' > "$REPO_ROOT/runs/full1000_tmux.status"
"$PYTHON_BIN" scripts/run_pilot.py --limit 1000 2>&1 | tee "$log_path"
run_status=${PIPESTATUS[0]}
printf '%s\n' "$run_status" > "$status_tmp"
mv "$status_tmp" "$status_path"
printf 'Full run finished with exit code %s. Log: %s\n' "$run_status" "$log_path"
exit "$run_status"
