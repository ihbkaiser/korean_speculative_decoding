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
PAIR="${1:-}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"

if [[ ! -f "$REPO/data/prompts_40k.parquet" || ! -f "$REPO/metadata/dataset_revision.json" ]]; then
  echo "Missing frozen prompt pool or dataset metadata under $REPO" >&2
  exit 2
fi

run_pair() {
  local pair="$1"
  mkdir -p "$REPO/logs"
  "$PYTHON_BIN" "$REPO/scripts/table1_pipeline.py" \
    --config "$REPO/configs/table1_pipeline.yaml" \
    --root "$REPO" \
    run-table1-main \
    --pair "$pair" \
    --num-shards "$NUM_SHARDS" \
    --device "$DEVICE" \
    --align-device "$ALIGN_DEVICE" \
    2>&1 | tee "$REPO/logs/table1_${pair}.log"
}

case "$PAIR" in
  Q1|Q2|Q3|M1|G1)
    run_pair "$PAIR"
    ;;
  all)
    for pair in Q1 Q2 Q3 M1 G1; do
      run_pair "$pair"
    done
    "$PYTHON_BIN" "$REPO/scripts/table1_pipeline.py" \
      --config "$REPO/configs/table1_pipeline.yaml" \
      --root "$REPO" build-table1 \
      2>&1 | tee "$REPO/logs/table1_build.log"
    ;;
  *)
    echo "Usage: $0 {Q1|Q2|Q3|M1|G1|all}" >&2
    exit 2
    ;;
esac
