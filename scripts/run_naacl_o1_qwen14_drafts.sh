#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash scripts/run_naacl_o1_qwen14_drafts.sh <target-self-tf-pid>" >&2
  exit 2
fi
target_pid="$1"
target_manifest="results/naacl_experiments/o1_qwen14/teacher_forced/generated/qwen14/qwen14/common/run_metadata.json"
log_root="logs/naacl_experiments/20261012"
mkdir -p "$log_root"
queue_log="$log_root/O1-qwen14-draft-queue.log"
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

verify_target_tf() {
  /venv/main/bin/python - "$target_manifest" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.exists():
    raise SystemExit(f"missing target self-TF manifest: {p}")
d = json.loads(p.read_text())
rows = d.get("completed_shards", [])
assert d.get("status") == "COMPLETE", f"target self-TF status={d.get('status')}"
assert d.get("prompt_count") == 20000, f"target prompt_count={d.get('prompt_count')}"
assert len(rows) == 10, f"target completed shards={len(rows)}"
assert sum(r.get("scored_token_count", 0) for r in rows) == 2459709
assert all(r.get("head_scale") == 1.0 and r.get("head_check_max_abs_error") == 0.0 for r in rows)
print("O1_TARGET_TF_VALIDATED prompts=20000 shards=10 tokens=2459709 head_check=PASS")
PY
}

verify_pair_tf() {
  local model="$1"
  /venv/main/bin/python - "$model" <<'PY'
import json, pathlib, sys
model = sys.argv[1]
p = pathlib.Path("results/naacl_experiments/o1_qwen14/teacher_forced/generated/qwen14") / model / "common/run_metadata.json"
if not p.exists():
    raise SystemExit(f"missing draft TF manifest: {p}")
d = json.loads(p.read_text())
rows = d.get("completed_shards", [])
assert d.get("status") == "COMPLETE", f"{model} TF status={d.get('status')}"
assert d.get("prompt_count") == 20000, f"{model} TF prompts={d.get('prompt_count')}"
assert len(rows) == 10, f"{model} TF shards={len(rows)}"
assert all(r.get("head_scale") == 1.0 and r.get("head_check_max_abs_error") == 0.0 for r in rows)
print(f"O1_DRAFT_TF_VALIDATED model={model} prompts=20000 shards=10 head_check=PASS")
PY
}

verify_simulation() {
  local pair="$1"
  /venv/main/bin/python - "$pair" <<'PY'
import json, pathlib, sys
pair = sys.argv[1]
p = pathlib.Path("results/naacl_experiments/o1_qwen14/simulated") / pair / "common/run_metadata.json"
if not p.exists():
    raise SystemExit(f"missing simulation manifest: {p}")
d = json.loads(p.read_text())
assert d.get("status") == "COMPLETE", f"{pair} replay status={d.get('status')}"
assert d.get("prompt_count") == 20000, f"{pair} replay prompts={d.get('prompt_count')}"
assert d.get("block_sizes") == [4], f"{pair} block sizes={d.get('block_sizes')}"
assert d.get("position_count", 0) > 0
print(f"O1_REPLAY_VALIDATED pair={pair} prompts=20000 positions={d['position_count']} k=4")
PY
}

if ! verify_target_tf 2>/dev/null; then
  target_args="$(ps -p "$target_pid" -o args= || true)"
  if [[ -z "$target_args" || "$target_args" != *"scripts/tf_score.py"* || \
        "$target_args" != *"--model qwen14"* || "$target_args" != *"--role target"* ]]; then
    log_event "ERROR target self-TF is neither complete nor running as PID $target_pid"
    exit 1
  fi
  log_event "waiting for target self-TF PID $target_pid"
  while ps -p "$target_pid" -o stat= | grep -qv '^[[:space:]]*Z'; do
    sleep 15
  done
  verify_target_tf
fi

log_event "target self-TF verified; evicting target and first draft weights to free model storage"
run_logged "O1-qwen14-pre-draft-eviction.log" \
  timeout --foreground --signal=TERM --kill-after=10s 1800s \
  /venv/main/bin/python -u scripts/download_models.py \
  --config configs/naacl_o1_qwen14.yaml --evict-pair O1_Q06

for spec in "O1_Q06 qwen06 q06" "O1_Q17 qwen17 q17" "O1_Q4 qwen4 q4"; do
  read -r pair model tag <<< "$spec"
  log_event "starting $pair draft=$model"
  run_logged "O1-qwen14-${tag}-download.log" \
    timeout --foreground --signal=TERM --kill-after=10s 3600s \
    /venv/main/bin/python -u scripts/download_models.py \
    --config configs/naacl_o1_qwen14.yaml --model "$model"
  run_logged "O1-qwen14-${tag}-TF.log" \
    env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    timeout --foreground --signal=TERM --kill-after=10s 14400s \
    /venv/main/bin/python -u scripts/tf_score.py \
    --config configs/naacl_o1_qwen14.yaml --model "$model" --target qwen14 \
    --split common --role draft --batch-size 32 --chunk-positions 32
  verify_pair_tf "$model"
  run_logged "O1-qwen14-${tag}-simulate.log" \
    timeout --foreground --signal=TERM --kill-after=10s 3600s \
    /venv/main/bin/python -u scripts/simulate_sd.py \
    --config configs/naacl_o1_qwen14.yaml --pair "$pair" --split common --block-sizes 4
  verify_simulation "$pair"
  run_logged "O1-qwen14-${tag}-eviction.log" \
    timeout --foreground --signal=TERM --kill-after=10s 1800s \
    /venv/main/bin/python -u scripts/download_models.py \
    --config configs/naacl_o1_qwen14.yaml --evict-pair "$pair"
done

log_event "all three draft TF and k=4 replay cells validated; running O1 analysis"
run_logged "O1-qwen14-analysis.log" \
  timeout --foreground --signal=TERM --kill-after=10s 3600s \
  /venv/main/bin/python -u scripts/analyze_naacl_o1_qwen14.py \
  --config configs/naacl_o1_qwen14.yaml
/venv/main/bin/python - <<'PY'
import json, pathlib
p = pathlib.Path("results/naacl_experiments/o1_qwen14/analysis/rq2_o1_qwen14_metadata.json")
d = json.loads(p.read_text())
assert d.get("status") == "COMPLETE", f"O1 analysis status={d.get('status')}"
assert d.get("pair_order") == ["O1_Q06", "O1_Q17", "O1_Q4"]
print("O1_ANALYSIS_VALIDATED status=COMPLETE pairs=3")
PY
log_event "O1 Qwen3-14B draft queue finished successfully"
