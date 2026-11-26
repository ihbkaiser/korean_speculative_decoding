#!/usr/bin/env bash
set -euo pipefail

cd /workspace/korean_speculative_decoding
export HF_TOKEN="$(< /workspace/hf_token)"
export CROSS_FAMILY_MODEL_DIR=/dev/shm/cross_family_models
export NAACL_MODEL_DIR=/dev/shm/cross_family_models
OUT=results/rebuttal/sampled_T1

for target in qwen4 qwen8; do
  /venv/main/bin/python -I scripts/rebuttal/gen_target_sampled.py \
    --target "$target" --temperature 1.0 --max-prompts 2000 --batch-size 48 \
    --output-root "$OUT" \
    > "results/rebuttal/logs/t8_sample_${target}.log" 2>&1
done

language_gate_status=0
/venv/main/bin/python -I scripts/rebuttal/analyze_sampled_gates.py \
  --sampled-root "$OUT" --language-only \
  > results/rebuttal/logs/t8_language_gate.log 2>&1 || language_gate_status=$?
if (( language_gate_status != 0 )); then
  echo "T8_STOPPED_AT_LANGUAGE_GATE exit=$language_gate_status" >> results/rebuttal/logs/t8_language_gate.log
  exit "$language_gate_status"
fi

for target in qwen4 qwen8; do
  /venv/main/bin/python -I scripts/align_morphology.py \
    --target "$target" --split common --max-prompts 2000 --output-dir "$OUT" \
    > "results/rebuttal/logs/t8_align_${target}.log" 2>&1
done

/venv/main/bin/python -I scripts/rebuttal/score_overlap.py \
  --pair Q2 --continuation sampled --max-prompts 2000 --batch-size 32 \
  --sampled-root "$OUT" \
  > results/rebuttal/logs/t8_score_Q2.log 2>&1
/venv/main/bin/python -I scripts/rebuttal/score_overlap.py \
  --pair Q5 --continuation sampled --max-prompts 2000 --batch-size 8 \
  --sampled-root "$OUT" \
  > results/rebuttal/logs/t8_score_Q5.log 2>&1
/venv/main/bin/python -I scripts/rebuttal/score_overlap.py \
  --pair Q5 --continuation generated --max-prompts 2000 --batch-size 32 \
  --streamed-models \
  > results/rebuttal/logs/t8_score_Q5_greedy_reference.log 2>&1

gate_status=0
/venv/main/bin/python -I scripts/rebuttal/analyze_sampled_gates.py \
  --sampled-root "$OUT" \
  > results/rebuttal/logs/t8_sampling_gates.log 2>&1 || gate_status=$?
/venv/main/bin/python -I scripts/rebuttal/analyze_overlap.py \
  --input-root results/rebuttal/overlap --output-dir results/rebuttal/overlap \
  --sampled-root "$OUT" \
  > results/rebuttal/logs/t8_analysis.log 2>&1
echo "T8_SAMPLING_GATES_EXIT=$gate_status" >> results/rebuttal/logs/t8_sampling_gates.log
