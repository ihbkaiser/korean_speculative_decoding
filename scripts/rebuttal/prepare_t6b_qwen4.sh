#!/usr/bin/env bash
set -euo pipefail
cd /workspace/korean_speculative_decoding
export HF_TOKEN="$(< /workspace/hf_token)"
/venv/main/bin/python -I scripts/download_models.py \
  --config configs/rebuttal_instruct.yaml --evict-pair IT_Q2
/venv/main/bin/python -I scripts/download_models.py \
  --config configs/naacl_experiments.yaml --model qwen4 --smoke-load
