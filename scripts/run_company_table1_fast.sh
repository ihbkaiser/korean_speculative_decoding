#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TABLE1_CONFIG="${TABLE1_CONFIG:-$REPO/configs/table1_fast_b200.yaml}"
export TABLE1_LOG_TAG="${TABLE1_LOG_TAG:-table1_fast_b200}"
export PROGRESS_LOG_EVERY="${PROGRESS_LOG_EVERY:-16}"
# Honor per-pair measured batch sizes in YAML; SD_BATCH_SIZE remains an override.
exec bash "$REPO/scripts/run_company_table1.sh" "${1:-Q1}" "${@:2}"
