#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash scripts/run_naacl_o4_gemma1.sh <o2-japanese-queue-pid>" >&2
  exit 2
fi
o2_pid="$1"
o2_manifest="results/naacl_experiments/o2_japanese/analysis/rq2_o2_japanese_metadata.json"
log_root="logs/naacl_experiments/20261012"
mkdir -p "$log_root"
queue_log="$log_root/O4-gemma1-queue.log"
exec > >(tee -a "$queue_log") 2>&1

log_event() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" "$*"
}

run_logged() {
  local log_name="$1"
  shift
  local log_path="$log_root/$log_name"
  local pipeline_status=()
  {
    printf 'START_UTC=%s\nCOMMAND=' "$(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    printf '%q ' "$@"
    printf '\n'
  } > "$log_path"
  set +e
  "$@" 2>&1 | tee -a "$log_path"
  pipeline_status=("${PIPESTATUS[@]}")
  set -e
  local command_status="${pipeline_status[0]}"
  printf 'COMMAND_EXIT=%s\nEND_UTC=%s\n' \
    "$command_status" "$(date -u '+%Y-%m-%d %H:%M:%S UTC')" | tee -a "$log_path"
  return "$command_status"
}

verify_o2() {
  /venv/main/bin/python - "$o2_manifest" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.exists():
    raise SystemExit(1)
d = json.loads(p.read_text())
if d.get("status") != "COMPLETE" or d.get("pair_order") != ["O2_Q06", "O2_G4_12"]:
    print(f"O2 analysis is incomplete: status={d.get('status')}")
    raise SystemExit(2)
print("O2 Japanese analysis metadata is COMPLETE")
PY
}

wait_for_o2() {
  if verify_o2; then
    return 0
  else
    local status="$?"
    if [[ "$status" -ne 1 ]]; then
      log_event "ERROR O2 analysis exists but is not complete"
      return 1
    fi
  fi
  local o2_args="$(ps -p "$o2_pid" -o args= || true)"
  if [[ "$o2_args" != *"run_naacl_o2_japanese.sh"* ]]; then
    log_event "ERROR O2 analysis is pending but queue PID $o2_pid is not alive"
    return 1
  fi
  log_event "waiting for O2 Japanese queue PID $o2_pid to finish"
  while true; do
    sleep 30
    if verify_o2; then
      return 0
    else
      local status="$?"
      if [[ "$status" -ne 1 ]]; then
        log_event "ERROR O2 analysis exists but is not complete"
        return 1
      fi
    fi
    if ! ps -p "$o2_pid" -o args= | grep -q "run_naacl_o2_japanese.sh"; then
      log_event "ERROR O2 queue exited before writing complete analysis metadata"
      return 1
    fi
  done
}

wait_for_o2
/venv/main/bin/python - <<'PY'
import json, pathlib
paths = [
    pathlib.Path("results/naacl_experiments/continuations/gemma12/common/run_metadata.json"),
    pathlib.Path("results/naacl_experiments/teacher_forced/generated/gemma12/gemma12/common/run_metadata.json"),
    pathlib.Path("results/naacl_experiments/aligned/kiwi/gemma12/common/run_metadata.json"),
]
for p in paths:
    d = json.loads(p.read_text())
    assert d.get("status") == "COMPLETE" and d.get("prompt_count") == 20000, str(p)
a = json.loads(pathlib.Path("results/naacl_experiments/provenance/tokenizer_audits/O4_G1_12.json").read_text())
assert a.get("compatible") is True
print("O4_PREREQUISITES_VALIDATED Gemma12 generation/TF/alignment and tokenizer audit")
PY

log_event "O2 complete; starting the Gemma1 draft against the frozen Gemma12 continuation"
if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ ! -s "$ROOT/hf_token" ]]; then
    log_event "ERROR local hf_token file is missing or empty"
    exit 1
  fi
  IFS= read -r HF_TOKEN < "$ROOT/hf_token" || [[ -n "${HF_TOKEN:-}" ]]
  if [[ -z "${HF_TOKEN:-}" ]]; then
    log_event "ERROR local hf_token file contains no token"
    exit 1
  fi
  export HF_TOKEN
fi
run_logged "O4-gemma1-download-plan.log" \
  /venv/main/bin/python -u scripts/download_models.py \
  --config configs/naacl_o4_gemma1.yaml --plan --model gemma1
run_logged "O4-gemma1-download.log" \
  timeout --foreground --signal=TERM --kill-after=10s 3600s \
  /venv/main/bin/python -u scripts/download_models.py \
  --config configs/naacl_o4_gemma1.yaml --model gemma1
run_logged "O4-gemma1-draft-TF.log" \
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  timeout --foreground --signal=TERM --kill-after=10s 14400s \
  /venv/main/bin/python -u scripts/tf_score.py \
  --config configs/naacl_o4_gemma1.yaml --model gemma1 --target gemma12 \
  --split common --role draft --batch-size 32 --chunk-positions 32
/venv/main/bin/python - <<'PY'
import json, pathlib
p = pathlib.Path("results/naacl_experiments/teacher_forced/generated/gemma12/gemma1/common/run_metadata.json")
d = json.loads(p.read_text())
rows = d.get("completed_shards", [])
assert d.get("status") == "COMPLETE" and d.get("role") == "draft"
assert d.get("prompt_count") == 20000 and len(rows) == 10
assert all(r.get("head_scale") == 1.0 and r.get("head_check_max_abs_error") == 0.0 for r in rows)
print("O4_DRAFT_TF_VALIDATED prompts=20000 shards=10 head_check=PASS")
PY
run_logged "O4-gemma1-simulate.log" \
  timeout --foreground --signal=TERM --kill-after=10s 3600s \
  /venv/main/bin/python -u scripts/simulate_sd.py \
  --config configs/naacl_o4_gemma1.yaml --pair O4_G1_12 --split common --block-sizes 4
/venv/main/bin/python - <<'PY'
import json, pathlib
p = pathlib.Path("results/naacl_experiments/simulated/O4_G1_12/common/run_metadata.json")
d = json.loads(p.read_text())
assert d.get("status") == "COMPLETE" and d.get("prompt_count") == 20000
assert d.get("block_sizes") == [4] and d.get("position_count", 0) > 0
print(f"O4_REPLAY_VALIDATED prompts=20000 positions={d['position_count']} k=4")
PY
run_logged "O4-gemma1-eviction.log" \
  timeout --foreground --signal=TERM --kill-after=10s 1800s \
  /venv/main/bin/python -u scripts/download_models.py \
  --config configs/naacl_o4_gemma1.yaml --evict-pair O4_G1_12
unset HF_TOKEN
run_logged "O4-gemma1-analysis.log" \
  timeout --foreground --signal=TERM --kill-after=10s 3600s \
  /venv/main/bin/python -u scripts/analyze_naacl_o4_gemma1.py \
  --config configs/naacl_o4_gemma1.yaml
/venv/main/bin/python - <<'PY'
import json, pathlib
p = pathlib.Path("results/naacl_experiments/analysis/rq2_o4_gemma1_metadata.json")
d = json.loads(p.read_text())
assert d.get("status") == "COMPLETE" and d.get("pair_id") == "O4_G1_12"
print("O4_ANALYSIS_VALIDATED status=COMPLETE pair=O4_G1_12")
PY
log_event "O4 Gemma1 queue finished successfully"
