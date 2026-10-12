#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ $# -ne 1 || ! "$1" =~ ^[0-9]+$ ]]; then
  echo "Usage: bash scripts/run_naacl_o2_japanese.sh <o1-draft-queue-pid>" >&2
  exit 2
fi
o1_pid="$1"
o1_manifest="results/naacl_experiments/o1_qwen14/analysis/rq2_o1_qwen14_metadata.json"
log_root="logs/naacl_experiments/20261012"
mkdir -p "$log_root"
queue_log="$log_root/O2-japanese-queue.log"
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

verify_o1() {
  /venv/main/bin/python - "$o1_manifest" <<'PY'
import json, pathlib, sys
p = pathlib.Path(sys.argv[1])
if not p.exists():
    raise SystemExit(1)
d = json.loads(p.read_text())
if d.get("status") != "COMPLETE":
    print(f"O1 analysis status={d.get('status')}")
    raise SystemExit(2)
print("O1 analysis metadata is COMPLETE")
PY
}

wait_for_o1() {
  if verify_o1; then
    return 0
  else
    local status="$?"
    if [[ "$status" -ne 1 ]]; then
      log_event "ERROR O1 analysis exists but is not complete"
      return 1
    fi
  fi
  local o1_args="$(ps -p "$o1_pid" -o args= || true)"
  if [[ "$o1_args" != *"run_naacl_o1_qwen14_drafts.sh"* ]]; then
    log_event "ERROR O1 analysis is pending but queue PID $o1_pid is not alive"
    return 1
  fi
  log_event "waiting for O1 Qwen14 draft queue PID $o1_pid to finish"
  while true; do
    sleep 30
    if verify_o1; then
      return 0
    else
      local status="$?"
      if [[ "$status" -ne 1 ]]; then
        log_event "ERROR O1 analysis exists but is not complete"
        return 1
      fi
    fi
    if ! ps -p "$o1_pid" -o args= | grep -q "run_naacl_o1_qwen14_drafts.sh"; then
      log_event "ERROR O1 queue exited before writing complete analysis metadata"
      return 1
    fi
  done
}

verify_generation() {
  local model="$1"
  local path="results/naacl_experiments/o2_japanese/continuations/$model/common/run_metadata.json"
  /venv/main/bin/python - "$path" "$model" <<'PY'
import json, pathlib, sys
import pyarrow.parquet as pq
p, model = pathlib.Path(sys.argv[1]), sys.argv[2]
if not p.exists():
    raise SystemExit(f"missing generation manifest: {p}")
d = json.loads(p.read_text())
assert d.get("status") == "COMPLETE", f"{model} generation status={d.get('status')}"
assert d.get("model_key") == model, f"{model} generation model key mismatch"
assert d.get("prompt_count") == 5000, f"{model} generation prompts={d.get('prompt_count')}"
shards = sorted(p.parent.glob("shard-*.parquet"))
rows = [pq.ParquetFile(shard).metadata.num_rows for shard in shards]
assert rows == [2000, 2000, 1000], f"{model} generation shard rows={rows}"
print(f"O2_GENERATION_VALIDATED target={model} prompts=5000 shards=3")
PY
}

verify_alignment() {
  local target="$1"
  local path="results/naacl_experiments/o2_japanese/aligned/fugashi/$target/common/run_metadata.json"
  /venv/main/bin/python - "$path" "$target" <<'PY'
import json, pathlib, sys
p, target = pathlib.Path(sys.argv[1]), sys.argv[2]
if not p.exists():
    raise SystemExit(f"missing Fugashi alignment manifest: {p}")
d = json.loads(p.read_text())
assert d.get("status") == "COMPLETE", f"{target} alignment status={d.get('status')}"
assert d.get("analyzer") == "fugashi"
assert d.get("prompt_count") == 5000, f"{target} alignment prompts={d.get('prompt_count')}"
assert len(d.get("completed_shards", [])) == 3
print(f"O2_ALIGNMENT_VALIDATED target={target} prompts=5000 shards=3")
PY
}

verify_tf() {
  local target="$1"
  local model="$2"
  local role="$3"
  local path="results/naacl_experiments/o2_japanese/teacher_forced/generated/$target/$model/common/run_metadata.json"
  /venv/main/bin/python - "$path" "$model" "$role" <<'PY'
import json, pathlib, sys
p, model, role = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
if not p.exists():
    raise SystemExit(f"missing teacher-forced manifest: {p}")
d = json.loads(p.read_text())
rows = d.get("completed_shards", [])
assert d.get("status") == "COMPLETE", f"{model} TF status={d.get('status')}"
assert d.get("role") == role, f"{model} TF role={d.get('role')}"
assert d.get("prompt_count") == 5000, f"{model} TF prompts={d.get('prompt_count')}"
assert len(rows) == 3
assert all(r.get("head_scale") == 1.0 and r.get("head_check_max_abs_error") == 0.0 for r in rows)
print(f"O2_TF_VALIDATED model={model} role={role} prompts=5000 shards=3 head_check=PASS")
PY
}

verify_replay() {
  local pair="$1"
  local path="results/naacl_experiments/o2_japanese/simulated/$pair/common/run_metadata.json"
  /venv/main/bin/python - "$path" "$pair" <<'PY'
import json, pathlib, sys
p, pair = pathlib.Path(sys.argv[1]), sys.argv[2]
if not p.exists():
    raise SystemExit(f"missing replay manifest: {p}")
d = json.loads(p.read_text())
assert d.get("status") == "COMPLETE", f"{pair} replay status={d.get('status')}"
assert d.get("prompt_count") == 5000, f"{pair} replay prompts={d.get('prompt_count')}"
assert d.get("block_sizes") == [4], f"{pair} block sizes={d.get('block_sizes')}"
assert d.get("position_count", 0) > 0
print(f"O2_REPLAY_VALIDATED pair={pair} prompts=5000 positions={d['position_count']} k=4")
PY
}

wait_for_o1

log_event "O1 complete; beginning Japanese control O2_Q06"
run_logged "O2-Q06-download-plan.log" \
  /venv/main/bin/python -u scripts/download_models.py \
  --config configs/naacl_o2_japanese.yaml --plan --pair O2_Q06
run_logged "O2-Q06-download.log" \
  timeout --foreground --signal=TERM --kill-after=10s 3600s \
  /venv/main/bin/python -u scripts/download_models.py \
  --config configs/naacl_o2_japanese.yaml --pair O2_Q06
run_logged "O2-Q06-generation.log" \
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  timeout --foreground --signal=TERM --kill-after=10s 14400s \
  /venv/main/bin/python -u scripts/gen_target.py \
  --config configs/naacl_o2_japanese.yaml --target qwen4 --split common --batch-size 128
verify_generation qwen4
run_logged "O2-Q06-target-TF.log" \
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  timeout --foreground --signal=TERM --kill-after=10s 14400s \
  /venv/main/bin/python -u scripts/tf_score.py \
  --config configs/naacl_o2_japanese.yaml --model qwen4 --target qwen4 \
  --split common --role target --batch-size 32 --chunk-positions 32
verify_tf qwen4 qwen4 target
run_logged "O2-Q06-alignment.log" \
  timeout --foreground --signal=TERM --kill-after=10s 3600s \
  /venv/main/bin/python -u scripts/align_morphology.py \
  --config configs/naacl_o2_japanese.yaml --target qwen4 --split common --analyzer fugashi
verify_alignment qwen4
run_logged "O2-Q06-draft-TF.log" \
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  timeout --foreground --signal=TERM --kill-after=10s 14400s \
  /venv/main/bin/python -u scripts/tf_score.py \
  --config configs/naacl_o2_japanese.yaml --model qwen06 --target qwen4 \
  --split common --role draft --batch-size 32 --chunk-positions 32
verify_tf qwen4 qwen06 draft
run_logged "O2-Q06-simulate.log" \
  timeout --foreground --signal=TERM --kill-after=10s 3600s \
  /venv/main/bin/python -u scripts/simulate_sd.py \
  --config configs/naacl_o2_japanese.yaml --pair O2_Q06 --split common --block-sizes 4
verify_replay O2_Q06
run_logged "O2-Q06-eviction.log" \
  timeout --foreground --signal=TERM --kill-after=10s 1800s \
  /venv/main/bin/python -u scripts/download_models.py \
  --config configs/naacl_o2_japanese.yaml --evict-pair O2_Q06

log_event "O2_Q06 complete; loading the local Hugging Face token for gated Gemma runs"
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

run_logged "O2-G4_12-gemma12-download-plan.log" \
  /venv/main/bin/python -u scripts/download_models.py \
  --config configs/naacl_o2_japanese.yaml --plan --model gemma12
run_logged "O2-G4_12-gemma12-download.log" \
  timeout --foreground --signal=TERM --kill-after=10s 3600s \
  /venv/main/bin/python -u scripts/download_models.py \
  --config configs/naacl_o2_japanese.yaml --model gemma12
run_logged "O2-G4_12-generation.log" \
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  timeout --foreground --signal=TERM --kill-after=10s 14400s \
  /venv/main/bin/python -u scripts/gen_target.py \
  --config configs/naacl_o2_japanese.yaml --target gemma12 --split common --batch-size 48
verify_generation gemma12
run_logged "O2-G4_12-target-TF.log" \
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  timeout --foreground --signal=TERM --kill-after=10s 14400s \
  /venv/main/bin/python -u scripts/tf_score.py \
  --config configs/naacl_o2_japanese.yaml --model gemma12 --target gemma12 \
  --split common --role target --batch-size 32 --chunk-positions 32
verify_tf gemma12 gemma12 target
run_logged "O2-G4_12-alignment.log" \
  timeout --foreground --signal=TERM --kill-after=10s 3600s \
  /venv/main/bin/python -u scripts/align_morphology.py \
  --config configs/naacl_o2_japanese.yaml --target gemma12 --split common --analyzer fugashi
verify_alignment gemma12
run_logged "O2-G4_12-target-eviction.log" \
  timeout --foreground --signal=TERM --kill-after=10s 1800s \
  /venv/main/bin/python -u scripts/download_models.py \
  --config configs/naacl_o2_japanese.yaml --evict-pair O2_G4_12
run_logged "O2-G4_12-gemma4-download-plan.log" \
  /venv/main/bin/python -u scripts/download_models.py \
  --config configs/naacl_o2_japanese.yaml --plan --model gemma4
run_logged "O2-G4_12-gemma4-download.log" \
  timeout --foreground --signal=TERM --kill-after=10s 3600s \
  /venv/main/bin/python -u scripts/download_models.py \
  --config configs/naacl_o2_japanese.yaml --model gemma4
run_logged "O2-G4_12-draft-TF.log" \
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  timeout --foreground --signal=TERM --kill-after=10s 14400s \
  /venv/main/bin/python -u scripts/tf_score.py \
  --config configs/naacl_o2_japanese.yaml --model gemma4 --target gemma12 \
  --split common --role draft --batch-size 32 --chunk-positions 32
verify_tf gemma12 gemma4 draft
run_logged "O2-G4_12-simulate.log" \
  timeout --foreground --signal=TERM --kill-after=10s 3600s \
  /venv/main/bin/python -u scripts/simulate_sd.py \
  --config configs/naacl_o2_japanese.yaml --pair O2_G4_12 --split common --block-sizes 4
verify_replay O2_G4_12
run_logged "O2-G4_12-eviction.log" \
  timeout --foreground --signal=TERM --kill-after=10s 1800s \
  /venv/main/bin/python -u scripts/download_models.py \
  --config configs/naacl_o2_japanese.yaml --evict-pair O2_G4_12
unset HF_TOKEN

log_event "both Japanese pair replays validated; running O2 analysis"
run_logged "O2-japanese-analysis.log" \
  timeout --foreground --signal=TERM --kill-after=10s 3600s \
  /venv/main/bin/python -u scripts/analyze_naacl_o2_japanese.py \
  --config configs/naacl_o2_japanese.yaml
/venv/main/bin/python - <<'PY'
import json, pathlib
p = pathlib.Path("results/naacl_experiments/o2_japanese/analysis/rq2_o2_japanese_metadata.json")
d = json.loads(p.read_text())
assert d.get("status") == "COMPLETE", f"O2 analysis status={d.get('status')}"
assert d.get("pair_order") == ["O2_Q06", "O2_G4_12"]
print("O2_ANALYSIS_VALIDATED status=COMPLETE pairs=2")
PY
log_event "O2 Japanese control queue finished successfully"
