#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# This entrypoint is fast-only: inherited shell settings must never select
# the legacy singleton decoder or its reference pre-generation pass.
export TABLE1_CONFIG="$REPO/configs/table1_fast_b200.yaml"
export TABLE1_REQUIRE_MICROBATCHED=1
export TABLE1_LOG_TAG="${TABLE1_LOG_TAG:-table1_fast_b200}"
export PROGRESS_LOG_EVERY="${PROGRESS_LOG_EVERY:-16}"
# Honor per-pair measured batch sizes in YAML; SD_BATCH_SIZE remains an override.
exec bash "$REPO/scripts/run_company_table1.sh" "${1:-Q1}" "${@:2}"
