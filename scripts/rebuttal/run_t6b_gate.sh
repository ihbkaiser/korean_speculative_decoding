#!/usr/bin/env bash
set -euo pipefail
cd /workspace/korean_speculative_decoding
export HF_TOKEN="$(< /workspace/hf_token)"
/venv/main/bin/python -I scripts/rebuttal/score_cross_family_judge.py \
  --source-target qwen4 --judge qwen4 --max-prompts 200 --batch-size 8
