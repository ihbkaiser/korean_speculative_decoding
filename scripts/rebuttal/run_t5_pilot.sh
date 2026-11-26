#!/usr/bin/env bash
set -euo pipefail

cd /workspace/korean_speculative_decoding
T5_PILOT_OUT=results/rebuttal/instruct_pilot
/venv/main/bin/python -I scripts/gen_target.py \
  --config configs/rebuttal_instruct.yaml --target qwen4it --split common \
  --batch-size 48 --max-prompts 50 --output-dir "$T5_PILOT_OUT" \
  > results/rebuttal/logs/t5_pilot_generate.log 2>&1
/venv/main/bin/python -I scripts/tf_score.py \
  --config configs/rebuttal_instruct.yaml --model qwen4it --target qwen4it \
  --split common --role target --max-prompts 50 --output-dir "$T5_PILOT_OUT" \
  > results/rebuttal/logs/t5_pilot_target_tf.log 2>&1
/venv/main/bin/python -I scripts/tf_score.py \
  --config configs/rebuttal_instruct.yaml --model qwen06it --target qwen4it \
  --split common --role draft --max-prompts 50 --output-dir "$T5_PILOT_OUT" \
  > results/rebuttal/logs/t5_pilot_draft_tf.log 2>&1
/venv/main/bin/python -I scripts/align_morphology.py \
  --config configs/rebuttal_instruct.yaml --target qwen4it --split common \
  --max-prompts 50 --output-dir "$T5_PILOT_OUT" \
  > results/rebuttal/logs/t5_pilot_align.log 2>&1
/venv/main/bin/python -I scripts/simulate_sd.py \
  --config configs/rebuttal_instruct.yaml --pair IT_Q2 --split common \
  --block-sizes 1 2 4 8 16 --max-prompts 50 --output-dir "$T5_PILOT_OUT" \
  > results/rebuttal/logs/t5_pilot_simulate.log 2>&1
