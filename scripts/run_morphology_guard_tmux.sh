#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="/data/hoang/korean_speculative_decoding"
RUN_DIR="${REPO_ROOT}/runs/morphology_aware_boundary_guard"
cd "${REPO_ROOT}"
mkdir -p "${RUN_DIR}/logs"

# Required inventory immediately before the first model-loading/GPU job.
nvidia-smi -L | tee "${RUN_DIR}/logs/nvidia_smi_L_before_gpu_job.txt"

export CUDA_VISIBLE_DEVICES=3
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=true
export PYTHONUNBUFFERED=1

"${REPO_ROOT}/.venv/bin/python" -u scripts/run_morphology_guard.py --mode all --resume 2>&1 \
  | tee -a "${RUN_DIR}/logs/benchmark_tmux.log"
