#!/usr/bin/env bash
set -euo pipefail

cd /workspace/korean_speculative_decoding
export HF_TOKEN="$(< /workspace/hf_token)"
/venv/main/bin/python -I scripts/download_models.py \
  --config configs/naacl_experiments.yaml --evict-model qwen4 \
  > results/rebuttal/logs/t5_evict_qwen4.log 2>&1
/venv/main/bin/python -I scripts/download_models.py \
  --config configs/rebuttal_instruct.yaml --model qwen06it --smoke-load \
  > results/rebuttal/logs/t5_download_qwen06it.log 2>&1
/venv/main/bin/python -I scripts/download_models.py \
  --config configs/rebuttal_instruct.yaml --model qwen4it --smoke-load \
  > results/rebuttal/logs/t5_download_qwen4it.log 2>&1
