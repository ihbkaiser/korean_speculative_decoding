#!/usr/bin/env bash
set -euo pipefail
cd /workspace/korean_speculative_decoding
export HF_TOKEN="$(< /workspace/hf_token)"
export CROSS_FAMILY_OUTPUT_ROOT=/workspace/korean_speculative_decoding/results/rebuttal/cross_family_attempts/gate_batch32
/venv/main/bin/python -I scripts/rebuttal/score_cross_family_judge.py \
  --source-target qwen4 --judge qwen4 --max-prompts 200 --batch-size 32
