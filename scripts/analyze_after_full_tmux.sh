#!/usr/bin/env bash
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/venv/main/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi

exitcode_path="$REPO_ROOT/runs/full1000_tmux.exitcode"
run_status_path="$REPO_ROOT/runs/full1000_tmux.status"
analysis_status_path="$REPO_ROOT/runs/full1000_tmux.analysis_status"
log_path="$REPO_ROOT/runs/full1000_tmux_20260926.log"

printf 'waiting_for_full_run\n' > "$analysis_status_path"
while [[ ! -f "$exitcode_path" ]]; do
    sleep 30
done

run_status=$(<"$exitcode_path")
if [[ "$run_status" != "0" ]]; then
    printf 'run_failed_exit_%s\n' "$run_status" > "$run_status_path"
    printf 'Analysis skipped: full run exited with status %s.\n' "$run_status" | tee -a "$log_path"
    printf 'skipped_run_failed\n' > "$analysis_status_path"
    exit 0
fi

printf 'run_complete\n' > "$run_status_path"
printf 'analyzing\n' > "$analysis_status_path"
run_dir=$(find "$REPO_ROOT/runs" -mindepth 1 -maxdepth 1 -type d -name '*pilot1000' -printf '%T@ %p\n' | sort -nr | head -n1 | cut -d' ' -f2-)
if [[ -z "$run_dir" ]]; then
    printf 'analysis_failed_no_run_dir\n' > "$analysis_status_path"
    printf 'Analysis failed: no pilot1000 run directory found.\n' | tee -a "$log_path"
    exit 1
fi

printf 'Starting analysis for %s\n' "$run_dir" | tee -a "$log_path"
"$PYTHON_BIN" scripts/analyze.py "$run_dir" 2>&1 | tee -a "$log_path"
analysis_status=${PIPESTATUS[0]}
if [[ "$analysis_status" == "0" ]]; then
    printf 'analysis_complete\n' > "$analysis_status_path"
else
    printf 'analysis_failed_exit_%s\n' "$analysis_status" > "$analysis_status_path"
fi
printf 'Analysis finished with exit code %s.\n' "$analysis_status" | tee -a "$log_path"
exit "$analysis_status"
