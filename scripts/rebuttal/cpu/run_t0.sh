#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../../.."
mkdir -p results/rebuttal/cpu results/rebuttal/logs

run_one() {
  local script="$1" output="$2"
  if [[ -s "$output" ]] && /venv/main/bin/python -I -c 'import json,sys; json.load(open(sys.argv[1]))' "$output"; then
    printf 'RESUME skip %s (valid output exists)\n' "$script"
    return
  fi
  rm -f "$output"
  printf 'RUN /venv/main/bin/python -I scripts/rebuttal/cpu/%s.py\n' "$script"
  /venv/main/bin/python -I "scripts/rebuttal/cpu/${script}.py"
  /venv/main/bin/python -I -c 'import json,sys; json.load(open(sys.argv[1]))' "$output"
}

run_one expA_policies results/rebuttal/cpu/expA_policies.json
run_one expA2_cost results/rebuttal/cpu/expA2_cost.json
run_one expB_thirdparty results/rebuttal/cpu/expB_thirdparty.json
run_one expC_gemma results/rebuttal/cpu/expC_gemma.json
run_one expC2_repetition results/rebuttal/cpu/expC2_repetition.json
