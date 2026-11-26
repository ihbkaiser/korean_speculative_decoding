#!/usr/bin/env bash
set -euo pipefail

cd /workspace/korean_speculative_decoding
OUT=results/rebuttal/instruct

/venv/main/bin/python -I scripts/gen_target.py \
  --config configs/rebuttal_instruct.yaml --target qwen4it --split common \
  --batch-size 48 --output-dir "$OUT" \
  > results/rebuttal/logs/t5_full_generate.log 2>&1
/venv/main/bin/python -I scripts/tf_score.py \
  --config configs/rebuttal_instruct.yaml --model qwen4it --target qwen4it \
  --split common --role target --output-dir "$OUT" \
  > results/rebuttal/logs/t5_full_target_tf.log 2>&1
/venv/main/bin/python -I scripts/tf_score.py \
  --config configs/rebuttal_instruct.yaml --model qwen06it --target qwen4it \
  --split common --role draft --output-dir "$OUT" \
  > results/rebuttal/logs/t5_full_draft_tf.log 2>&1
/venv/main/bin/python -I scripts/align_morphology.py \
  --config configs/rebuttal_instruct.yaml --target qwen4it --split common \
  --output-dir "$OUT" \
  > results/rebuttal/logs/t5_full_align.log 2>&1
/venv/main/bin/python -I scripts/simulate_sd.py \
  --config configs/rebuttal_instruct.yaml --pair IT_Q2 --split common \
  --block-sizes 1 2 4 8 16 --output-dir "$OUT" \
  > results/rebuttal/logs/t5_full_simulate.log 2>&1

/venv/main/bin/python -I scripts/rebuttal/analyze_instruct.py \
  --config configs/rebuttal_instruct.yaml --output-dir "$OUT/analysis" \
  > results/rebuttal/logs/t5_analysis.log 2>&1
/venv/main/bin/python -I scripts/download_models.py \
  --config configs/rebuttal_instruct.yaml --evict-pair IT_Q2 \
  > results/rebuttal/logs/t5_evict_pair.log 2>&1
