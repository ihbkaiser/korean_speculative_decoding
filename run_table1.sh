#!/usr/bin/env bash
set -euo pipefail
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Selected workflow: BF16/SDPA microbatch target verification, no scalar reruns.
exec bash "$REPO/scripts/run_company_table1_fast.sh" "${1:-Q1}" "${@:2}"
