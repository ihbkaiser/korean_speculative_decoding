#!/usr/bin/env bash
set -euo pipefail
cd /workspace/korean_speculative_decoding
export HF_TOKEN="$(< /workspace/hf_token)"
export CROSS_FAMILY_MODEL_DIR=/dev/shm/cross_family_models
/venv/main/bin/python -I scripts/rebuttal/score_cross_family_judge.py \
  --source-target qwen4 qwen8 --judge ministral8 --max-prompts 5000 --batch-size 8
/venv/main/bin/python -I scripts/rebuttal/score_cross_family_judge.py \
  --source-target ministral8 --judge qwen8 --max-prompts 5000 --batch-size 8
/venv/main/bin/python -I scripts/rebuttal/analyze_cross_family_judge.py \
  --max-prompts 5000
