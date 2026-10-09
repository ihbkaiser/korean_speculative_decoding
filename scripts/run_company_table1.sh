#!/usr/bin/env bash
set -euo pipefail

# Company B200 launcher. Model snapshot paths live in
# configs/table1_pipeline.yaml under model_paths; no HF token or network is
# needed for the production-only path.
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
NUM_SHARDS="${NUM_SHARDS:-1}"
DEVICE="${DEVICE:-cuda}"
ALIGN_DEVICE="${ALIGN_DEVICE:-cpu}"
CONFIG="${TABLE1_CONFIG:-$REPO/configs/table1_pipeline.yaml}"
LOG_TAG="${TABLE1_LOG_TAG:-table1}"
PROGRESS_FLUSH_EVERY="${PROGRESS_FLUSH_EVERY:-64}"
PROGRESS_LOG_EVERY="${PROGRESS_LOG_EVERY:-100}"
PAIR="${1:-}"
batch_args=()
if [[ -n "${SD_BATCH_SIZE:-}" ]]; then
  batch_args=(--sd-batch-size "$SD_BATCH_SIZE")
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export PYTHONUNBUFFERED=1

if [[ ! -f "$REPO/data/prompts_40k.parquet" || ! -f "$REPO/metadata/dataset_revision.json" ]]; then
  echo "Missing frozen prompt pool or dataset metadata under $REPO" >&2
  exit 2
fi

run_pair() {
  local pair="$1"
  mkdir -p "$REPO/logs"
  "$PYTHON_BIN" "$REPO/scripts/table1_pipeline.py" \
    --config "$CONFIG" \
    --root "$REPO" \
    run-table1-main \
    --pair "$pair" \
    --num-shards "$NUM_SHARDS" \
    --device "$DEVICE" \
    --align-device "$ALIGN_DEVICE" \
    --progress-flush-every "$PROGRESS_FLUSH_EVERY" \
    --progress-log-every "$PROGRESS_LOG_EVERY" \
    "${batch_args[@]}" \
    2>&1 | tee -a "$REPO/logs/${LOG_TAG}_${pair}.log"
}

run_pair_sd_deferred_align() {
  local pair="$1"
  mkdir -p "$REPO/logs"
  "$PYTHON_BIN" "$REPO/scripts/table1_pipeline.py" \
    --config "$CONFIG" \
    --root "$REPO" \
    run-table1-main \
    --pair "$pair" \
    --num-shards "$NUM_SHARDS" \
    --device "$DEVICE" \
    --align-device "$ALIGN_DEVICE" \
    --skip-align \
    --progress-flush-every "$PROGRESS_FLUSH_EVERY" \
    --progress-log-every "$PROGRESS_LOG_EVERY" \
    "${batch_args[@]}" \
    2>&1 | tee -a "$REPO/logs/${LOG_TAG}_${pair}_sd.log"
}

run_pair_align() {
  local pair="$1"
  local shard_index
  mkdir -p "$REPO/logs"
  for ((shard_index = 0; shard_index < NUM_SHARDS; shard_index++)); do
    "$PYTHON_BIN" "$REPO/scripts/table1_pipeline.py" \
      --config "$CONFIG" \
      --root "$REPO" \
      align-morphology \
      --pair "$pair" \
      --shard-index "$shard_index" \
      --num-shards "$NUM_SHARDS" \
      --device "$ALIGN_DEVICE" \
      2>&1 | tee -a "$REPO/logs/${LOG_TAG}_${pair}_align.log"
  done
}

case "$PAIR" in
  Q1|Q2|Q3|M1|G1)
    run_pair "$PAIR"
    ;;
  all)
    alignment_pids=()
    for pair in Q1 Q2 Q3 M1 G1; do
      # Keep the single GPU busy with the next pair while CPU-only morphology
      # alignment for the previous pair runs in the background.
      run_pair_sd_deferred_align "$pair"
      run_pair_align "$pair" &
      alignment_pids+=("$!")
    done
    alignment_failed=0
    for pid in "${alignment_pids[@]}"; do
      if ! wait "$pid"; then
        echo "Morphology alignment process failed: pid=$pid" >&2
        alignment_failed=1
      fi
    done
    if [[ "$alignment_failed" -ne 0 ]]; then
      echo "At least one morphology alignment job failed; refusing to build Table 1." >&2
      exit 1
    fi

    "$PYTHON_BIN" "$REPO/scripts/table1_pipeline.py" \
      --config "$CONFIG" \
      --root "$REPO" build-table1 \
      2>&1 | tee -a "$REPO/logs/${LOG_TAG}_build.log"
    ;;
  *)
    echo "Usage: $0 {Q1|Q2|Q3|M1|G1|all}" >&2
    exit 2
    ;;
esac
