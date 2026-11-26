# Rebuttal experiment report

## Predictions recorded before experiments

- **T1:** The measured cost ratio will be well above the parameter ratio for small pairs. The margin rule will be faster than the fastest fixed block size when the measured ratio is between 0.1 and 0.5. Live tokens per target call will be within 3% of replay for `FixedK(4)` and somewhat below replay for adaptive rules.
- **T2:** Expected rejection will concentrate at target decision points, but less than under greedy verification. After matching on target margin, morpheme-onset/internal should be in 0.85–1.15 and fused/split at or below 1.0; word onsets should retain a small excess of 1.13–1.25.
- **T3:** Rescue rate will be higher at decision points than elsewhere. Branching only at draft decision points will keep most of the tokens-per-call gain of branching everywhere while using a minority of its extra verified nodes.
- **T4:** Disagreement will fall in condition B relative to A at every position type. The disagreement-by-margin curves will nearly coincide, matched ratios will stay near 1 in every condition, and condition and condition-by-position coefficients will be near zero after margin-bin fixed effects.

## Environment and baseline

Work branch: `rebuttal/experiments`. GPU: NVIDIA RTX A6000, CUDA 12.8; one GPU was idle at inspection. `/workspace` is not a persistent volume. A Hugging Face token file is available at `/workspace/hf_token`; its contents are not recorded in logs or metadata and will be used only for authorized model downloads.

Baseline command: `/venv/main/bin/python -m pytest -q` — **136 passed** after installing the declared dependencies. The first baseline invocation used system `pytest` and failed collection because that interpreter lacked torch, SciPy, and pandas; the repository environment was then provisioned and the baseline rerun successfully.

## T0. CPU analyses

Reference checks recorded before execution: Q2 replay at fixed `k=4` is expected to yield 3.45 tokens per target call; Q2 test-half policy values are expected to be 3.45 (`k=4`), 3.91 (entropy), 4.02 (top-1 probability), 4.03 (cumulative log-probability), 4.04 (margin), 4.06 (learned stopper), and 5.30 (oracle). Q5 judged by Qwen-4B is expected to have judge-margin AUC 0.886, boundary-label AUC 0.602, and morpheme-onset/internal ratio 2.21 raw and 0.96 after margin matching. Repeated four-gram rates are expected to be 78.8% for Gemma-12B and 42.9% for Qwen-4B. On the Q2 non-repeated subset, the expected matched ratios are 1.02, 1.27, and 0.95. These are comparison targets from the supplied experiment specification, not measured outcomes.

The checkout does not contain `paper/scripts/build_decision_point_tables.py` or an equivalent label builder, and it does not contain `paper/scripts/simulate_stopping.py`; the supplied onset re-derivation will therefore be retained and no paper-simulator curve comparison is available. The extended fixed-k grid through 16 is part of `expA2_cost.py`.

**Commands:** `bash scripts/rebuttal/cpu/run_t0.sh`, which runs `/venv/main/bin/python -I scripts/rebuttal/cpu/expA_policies.py`, `expA2_cost.py`, `expB_thirdparty.py`, `expC_gemma.py`, and `expC2_repetition.py` in that order. The wrapper validates each completed JSON and skips valid outputs on restart. The scripts read generated teacher-forced/alignment/simulation results; the analysis uses the common 20,000-document pool, with even `pool_index` for threshold selection and odd `pool_index` for held-out policy evaluation. `expB_thirdparty.py` also reads human continuations for Qwen3 pairs and M1; Gemma is generated-only. `expA_policies.py` uses candidate thresholds from the even documents and evaluates the odd half; replay block limits are 16. `expA2_cost.py` reports fixed-k results through 16 and cost-ratio sensitivity.

| Analysis | Measured result | Reference comparison |
|---|---:|---|
| Q2 held-out tokens/call: fixed k=4 | 3.4508 | Matches 3.45 |
| Q2 held-out: entropy / top-1 / cumulative log-prob / margin / learned stopper / oracle | 3.9050 / 4.0159 / 4.0279 / 4.0387 / 4.0576 / 5.3001 | Matches expected 3.91 / 4.02 / 4.03 / 4.04 / 4.06 / 5.30 |
| Q2 margin gain over k=4, document bootstrap | 17.04% [16.86%, 17.21%] | Computed from 500 test-document resamples |
| Q5, Qwen-4B judge margin AUC / boundary-label AUC | 0.886 / 0.602 | Matches both references |
| Q5 morpheme-onset/internal, raw → judge-margin matched | 2.21 → 0.96 | Matches both references |
| Repeated four-gram rate: Gemma-12B / Qwen-4B | 78.8% / 42.9% | Matches both references |
| Q2 non-repeated positions: morpheme/internal, word/internal, fused/split matched | 1.02 / 1.27 / 0.95 | Matches all references |

The fixed-k extension changes only the parameter-ratio rows for the Gemma pairs: G2 improves from 4.718 at k≤8 to 5.628 at k≤16, with selected k=16; G1 improves from 2.196 to 2.211, with selected k=11. The other ten pairs have identical k≤8 and k≤16 values at their parameter-ratio setting. All five rerun JSONs match their bundled reference JSONs exactly, so there are no differences larger than the shown precision. Neither paper helper script for alternative position labels nor the paper stopping simulator exists in this checkout; labels remain the supplied alignment-based re-derivation, and the simulator curve comparison was not possible. The supplied CPU policy output does not persist its dev-selected threshold values; threshold selection was still performed on the even documents, but those exact values are not recoverable from its JSON artifact.

**Prediction / outcome / verdict:** The supplied references were expected to reproduce; all five outputs match exactly — **held**. The Q2 replay gives 3.4508 tokens per target call at fixed k=4; the held-out margin rule reaches 4.0387, with a bootstrap gain of 17.04% [16.86%, 17.21%]. For Q5, the independent Qwen-4B margin predicts rejection better than the boundary label (AUC 0.886 vs 0.602), and the morpheme-onset/internal ratio falls from 2.21 raw to 0.96 after matching. However, Q2 word-onset/internal remains 1.27 after matching on the non-repeated subset, while fused/split is 0.95; the result therefore does not remove every position-type association.

## T1. Wall-clock speed

### T1a. Qwen, Ministral and Gemma latency probes

**Commands:** `/venv/main/bin/python -I scripts/rebuttal/measure_latency.py --models qwen06 qwen17 qwen4 qwen8 --prompt-target qwen8 --output-dir results/rebuttal/latency/qwen3`; the analogous Ministral command used `--models ministral3 ministral8 --prompt-target ministral8 --output-dir results/rebuttal/latency/ministral`. Gemma 4B/12B used `/venv/main/bin/python -I scripts/rebuttal/measure_latency.py --models gemma4 gemma12 --prompt-target gemma12 --output-dir results/rebuttal/latency/gemma3_large`; the O4 probe used `/venv/main/bin/python -I scripts/rebuttal/measure_latency.py --config configs/naacl_o4_gemma1.yaml --models gemma1 gemma12 --prompt-target gemma12 --output-dir results/rebuttal/latency/gemma1_probe`. The probes measured cached single-token calls at contexts 128 and 256, verify calls at 1/2/3/5/9/17 tokens, and cache crops by 1/4/16 positions. Qwen produced 44 rows; the Ministral, Gemma 4B/12B, and Gemma 1B/12B probes each produced 22, for 110 raw rows. `results/rebuttal/latency/latency.csv` contains 99 merged rows after deduplicating the repeated Gemma12 measurements; pair ratios are in `results/rebuttal/latency/cost_ratio.csv`.

| Pair | Measured `c` | Parameter `c` |
|---|---:|---:|
| G1 | 0.6647 | 0.3333 |
| M1 | 0.6885 | 0.3750 |
| O4_G1_12 (G2) | 0.4774 | 0.0833 |
| Q1 | 0.8540 | 0.3529 |
| Q2 | 0.6641 | 0.1500 |
| Q3 | 0.9008 | 0.5000 |
| Q4 | 0.6902 | 0.4250 |
| Q5 | 0.6585 | 0.0750 |
| Q6 | 0.6844 | 0.2125 |

**Prediction / outcome / verdict:** The measured-cost portion is held: all nine pair ratios exceed their parameter ratios. M1 is 0.6885 measured versus 0.3750 parameter-based, G1 is 0.6647 versus 0.3333, and Q5 has the largest gap (0.6585 vs 0.0750). Eight ratios are above the proposed 0.1–0.5 range; O4_G1_12 is inside it at 0.4774 despite its 0.0833 parameter ratio. Qwen and O4 wall-clock outcomes are reported below. O4_G1_12 is inside the proposed measured-cost range and its margin policy beats the fastest fixed policy. The T1c priority list does not include an M1 timing run. Each probe recorded an idle A6000 at launch and ran without overlapping GPU work.

### T1b. Adaptive speculative decoder

Implemented `src/rebuttal/adaptive_sd.py` with persistent KV caches, cache cropping on rejection, one target forward per verification block, and `FixedK`, margin, entropy, and top-1-probability draft stopping rules. Fixed-K keeps proposal IDs on-device until verification; adaptive policies read one stop predicate per drafted token. `greedy_target_only` provides the cached reference decoder. `tests/test_adaptive_sd.py` compares all policies with greedy generation, checks EOS and token budgets, verifies the accepted-prefix/correction-or-bonus block structure, and exercises a Transformers dynamic cache. The baseline full suite passed 136 tests. Final validation ran `/venv/main/bin/python -I -m pytest -q`: **153 passed in 13.11 s**.

### T1c. Q5 pilot

**Command:** `/venv/main/bin/python -I scripts/rebuttal/bench_wallclock.py --pair Q5 --pilot --output-root results/rebuttal/wallclock`. This used 50 ascending odd-index test documents, one repeat, and 10 warm-up documents per policy. Thresholds selected on dev were margin 2.5, entropy 0.7402, and top-1 probability 0.7648; replay selected fixed k=1. The full pilot table is in `results/rebuttal/wallclock/Q5/pilot/summary.md`.

| Policy | Tokens/s | Speedup vs target-only [95% CI] | Speedup vs fastest fixed | Live L/call vs replay L/call | Exact stored / live |
|---|---:|---:|---:|---:|---:|
| Target-only | 27.241 | 1.000 [1.000, 1.000] | 1.075 | 1.000 / 1.000 | 0.32 / 1.00 |
| Fixed k=1 | 25.329 | 0.930 [0.915, 0.945] | 1.000 | 1.783 / 1.782 | 0.30 / 0.32 |
| Fixed k=4 | 21.697 | 0.796 [0.752, 0.846] | 0.857 | 3.194 / 3.213 | 0.24 / 0.30 |
| Margin tuned | 27.445 | 1.007 [0.978, 1.040] | 1.084 [1.066, 1.102] | 2.851 / 2.798 | 0.38 / 0.30 |

The measured Q5 cost ratio is 0.6585, outside the predicted 0.1–0.5 range. Margin tuning was slightly faster than target-only in this pilot and faster than the fastest fixed policy, while its measured speedup (1.007) fell below the cost-model prediction (1.133). Fixed k=4 live tokens per call were within 1% of replay, but the tuned margin rule was slightly above replay rather than below it. Exact-match rates were low: margin tuning matched 38% of stored continuations and 30% of the live target-only outputs, so the all-document speedup remains provisional; on its exact-live subset the margin policy reached 1.077× target-only throughput with a 95% interval of [1.021, 1.140]. **Prediction / outcome / verdict:** the measured-cost-ratio prediction is held; the pilot’s speed direction is partly held, while the predicted replay direction and cost-model accuracy are not — **partly held**; the main-run outcome is reported below.

The pilot projects 6.07 hours for 200 documents × two repeats. Dropping fixed k=2 and k=6 reduces that projection to 4.73 hours; reducing to 150 documents brings it to 3.55 hours. The Q5 main run used 150 test documents with those two fixed policies omitted, following the specified cap rule.

### T1c. Q5 main timing

**Command:** `/venv/main/bin/python -I scripts/rebuttal/bench_wallclock.py --pair Q5 --output-root results/rebuttal/wallclock`. Using the pilot cap plan, this measured 150 odd-index test documents for two repeats. It omitted fixed k=2 and k=6, ran eight policies, and produced 2,400 policy/document/repeat rows. The cumulative timed policy runtime was 12,757.9 s (3.54 h); the run completed without reaching the four-hour cap. The per-policy results and document-bootstrap intervals are in `results/rebuttal/wallclock/Q5/summary.csv`.

| Policy | Exact stored / live | Speedup vs target-only [95% CI] | Speedup vs fastest fixed [95% CI] | Cost-model speedup |
|---|---:|---:|---:|---:|
| Target-only | 34.0% / 100% | 1.000 [1.000, 1.000] | 1.082 [1.070, 1.094] | 1.000 |
| Fixed k=1 | 36.0% / 35.3% | 0.924 [0.914, 0.934] | 1.000 [1.000, 1.000] | 1.076 |
| Fixed k=4 | 31.3% / 33.3% | 0.789 [0.766, 0.816] | 0.854 [0.834, 0.877] | 0.901 |
| Fixed k=8 | 35.3% / 37.3% | 0.582 [0.552, 0.610] | 0.629 [0.601, 0.658] | 0.665 |
| Entropy tuned | 36.7% / 33.3% | 0.989 [0.973, 1.006] | 1.070 [1.061, 1.080] | 1.137 |
| Top-1 tuned | 34.0% / 35.3% | 0.992 [0.976, 1.010] | 1.073 [1.061, 1.084] | 1.144 |
| Margin tuned | 38.0% / 34.0% | 0.998 [0.980, 1.018] | 1.080 [1.069, 1.092] | 1.143 |
| Margin, 1 nat | 35.3% / 34.7% | 0.928 [0.905, 0.952] | 1.004 [0.985, 1.023] | 1.068 |

On all documents, adaptive margin is statistically indistinguishable from target-only throughput, but is 8.0% faster than fixed k=1, the fastest fixed policy. Its live exact-match subset contains 51 documents (34%); on those documents it is 1.069× target-only. Fixed k=4's all-document tokens per target call are 3.190 versus 3.253 from replay (−1.9%), within the 3% validation tolerance. The measured cost ratio is 0.6585, above the predicted 0.1–0.5 range, and the cost model predicts 1.143× for margin tuning versus the measured 0.998×. The low live exact-match rates (33–37% for the adaptive and fixed policies) make the all-document comparison the more relevant wall-clock result; the exact-live subset is a sensitivity check.

**Prediction / outcome / verdict:** adaptive policies beat fixed k=1 in this run, as predicted, but none establishes an all-document speedup over target-only; the margin interval includes 1. The large measured cost ratio and overprediction by the cost model explain why replay gains did not translate into a clear end-to-end speedup. The fixed k=4 live replay check passes its 3% tolerance. **Partly held.**

### T1c. Q2 main timing

**Command:** `/venv/main/bin/python -I scripts/rebuttal/bench_wallclock.py --pair Q2 --output-root results/rebuttal/wallclock`. This used the same cap plan as Q5: 150 ascending odd-index test documents, two repeats, eight policies, with fixed k=2 and k=6 omitted. It produced 2,400 rows in 12,575.2 cumulative timed seconds (3.49 h), below the four-hour cap. The full policy table and document-bootstrap intervals are in `results/rebuttal/wallclock/Q2/summary.csv`.

| Policy | Exact stored / live | Speedup vs target-only [95% CI] | Speedup vs fastest fixed | Cost-model speedup |
|---|---:|---:|---:|---:|
| Target-only | 34.0% / 100% | 1.000 [1.000, 1.000] | 1.070 | 1.000 |
| Fixed k=1 | 38.0% / 34.7% | 0.935 [0.925, 0.945] | 1.000 | 1.088 |
| Fixed k=4 | 39.3% / 30.0% | 0.844 [0.823, 0.865] | 0.903 | 0.931 |
| Fixed k=8 | 33.3% / 30.0% | 0.636 [0.606, 0.665] | 0.680 | 0.701 |
| Entropy tuned | 34.0% / 38.7% | 1.004 [0.987, 1.021] | 1.075 | 1.150 |
| Top-1 tuned | 34.7% / 33.3% | 1.013 [0.996, 1.030] | 1.084 | 1.158 |
| Margin tuned | 34.7% / 38.7% | 1.023 [1.006, 1.041] | 1.095 | 1.161 |
| Margin, 1 nat | 37.3% / 36.0% | 0.955 [0.933, 0.979] | 1.022 | 1.090 |

Margin tuning is the only adaptive rule with an all-document confidence interval wholly above target-only (2.3% faster; 95% CI 0.6–4.1%). It is 9.5% faster than fixed k=1. On its live exact-match subset (58 documents), it reaches 1.066× target-only. Fixed k=4 produces 3.410 tokens per target call versus 3.378 in replay (+1.0%), within the 3% tolerance. The measured cost ratio is 0.6641, outside the predicted 0.1–0.5 range; the margin cost model predicts 1.161×, versus measured 1.023×.

**Prediction / outcome / verdict:** the Q2 margin interval is narrowly above 1, while Q5’s corresponding interval includes 1. For O4, whose measured cost ratio is within 0.1–0.5, the margin policy beats the fastest fixed policy with its 95% interval above 1. Fixed k=4 live/replay agreement is within tolerance for all three timed pairs; adaptive live token counts are not below replay in O4. The cost model overpredicts measured gains in all three. **Partly held across Q2, Q5, and O4.**

### O4_G1_12 (Gemma-3 1B → 12B) timing

**Prediction recorded before running:** O4's measured cost ratio is 0.4774, inside the 0.1–0.5 range, so the margin policy is predicted to beat the fastest fixed policy. Fixed k=4 live tokens per target call are predicted to stay within 3% of replay; adaptive live values may be somewhat lower.

**Scheduling pilot command:** `/venv/main/bin/python -I scripts/rebuttal/bench_wallclock.py --config configs/naacl_o4_gemma1.yaml --pair O4_G1_12 --pilot --output-root results/rebuttal/wallclock`. It used 50 odd-index test documents and one timed repeat, with 10 warm-up documents per policy. The run started at 05:54:36 UTC and completed at 06:49:42 UTC on the idle RTX A6000; 10 policies accumulated 2,743.4 timed seconds (45.7 min) and wrote 500 rows. Dev thresholds were margin 2.0 nats, entropy 0.6740, top-1 probability 0.7395, and fixed k=7.

| Pilot policy | Tokens/s | Speedup vs target-only (95% CI) | Speedup vs fastest fixed (95% CI) | Live / replay tokens per target call |
|---|---:|---:|---:|---:|
| Target-only | 15.818 | 1.000 | 0.644 [0.632, 0.655] | 1.000 / 1.000 |
| Fastest fixed, k=8 | 24.580 | 1.554 [1.527, 1.581] | 1.000 | 7.921 / 7.891 |
| Fixed k=4 | 23.520 | 1.487 [1.468, 1.507] | 0.957 [0.942, 0.970] | 4.696 / 4.685 |
| Margin, 2.0 nats | 26.194 | 1.656 [1.632, 1.680] | 1.066 [1.051, 1.081] | 10.596 / 10.339 |

The pilot supports the prediction that margin tuning beats the fastest fixed policy; the fixed-k=4 live/replay difference is 0.22%, within tolerance. Margin's cost model predicts 1.845× target-only versus 1.656× measured. Its exact-match rates are 86% stored and 80% live.

**Main command:** `/venv/main/bin/python -I scripts/rebuttal/bench_wallclock.py --config configs/naacl_o4_gemma1.yaml --pair O4_G1_12 --output-root results/rebuttal/wallclock`. It ran 150 ascending odd-index test documents × two repeats over eight policies, omitting fixed k=2 and k=6 as selected by the pilot-based cap planner. The main run started at 06:53:42 UTC and completed at 10:41:26 UTC on the idle A6000, with 2,400 rows and 13,187.6 cumulative timed seconds (3.66 h, within the four-hour cap). It used the same dev thresholds as the pilot: margin 2.0 nats, entropy 0.6740, top-1 0.7395, and fixed k=7.

| Main policy | Exact stored / live | Tokens/s | Speedup vs target-only [95% CI] | Speedup vs fastest fixed [95% CI] | Live / replay tokens per target call |
|---|---:|---:|---:|---:|---:|
| Target-only | 83.3% / 100% | 15.741 | 1.000 [1.000, 1.000] | 0.650 [0.641, 0.658] | 1.000 / 1.000 |
| Fastest fixed, k=8 | 82.7% / 80.7% | 24.227 | 1.539 [1.519, 1.560] | 1.000 | 7.814 / 7.811 |
| Fixed k=4 | 82.7% / 81.3% | 23.437 | 1.489 [1.477, 1.500] | 0.967 [0.959, 0.978] | 4.676 / 4.666 |
| Margin, 2.0 nats | 84.7% / 82.7% | 26.023 | 1.653 [1.636, 1.670] | 1.074 [1.059, 1.089] | 10.213 / 10.095 |

**Prediction / outcome / verdict:** Margin tuning beats the fastest fixed policy by 7.4%, with a paired document-bootstrap 95% interval excluding 1; this prediction is held. Fixed k=4 differs from replay by 0.22%, within the 3% tolerance. Margin's live token count is 1.2% above replay rather than below it, so the adaptive replay-direction prediction is not held. The cost model predicts 1.830× target-only versus 1.653× measured. Across the 150 documents, margin is the fastest policy at 26.023 tokens/s, and its stored/live exact-match rates are 84.7%/82.7%.

## T2. Speculative sampling overlap

### Qwen T2a scoring

**Commands:** `/venv/main/bin/python -I scripts/rebuttal/score_overlap.py --pair Q2 --continuation human --max-prompts 5000 --batch-size 32 --output-root results/rebuttal/overlap`; `/venv/main/bin/python -I scripts/rebuttal/score_overlap.py --pair Q5 --continuation human --max-prompts 5000 --batch-size 32 --output-root results/rebuttal/overlap`; and `/venv/main/bin/python -I scripts/rebuttal/score_overlap.py --pair Q2 --continuation generated --max-prompts 5000 --batch-size 32 --output-root results/rebuttal/overlap`. Each human run scored the first 5,000 common continuations (640,000 positions). Q2 generated scored 5,000 frozen greedy continuations (620,451 positions).

| Pair / continuation | Target argmax agreement | Draft argmax agreement | Target / draft median margin delta |
|---|---:|---:|---:|
| Q2 human | 100% | 100% | 0 / 0 |
| Q5 human | 100% | 100% | 0 / 0 |
| Q2 generated | 100% | 98.92% | 0 / 0 |

All mandatory consistency gates passed. The target and draft head scales were 1.0 in these runs.

### M1 T2a scoring

**Command:** `/venv/main/bin/python -I scripts/rebuttal/score_overlap.py --pair M1 --continuation human --max-prompts 5000 --batch-size 32 --output-root results/rebuttal/overlap`. It scored the first 5,000 complete common human continuations (640,000 positions) using the original length-sorted human reference batches.

| Model | Positions | Argmax agreement | Median absolute margin delta | Head scale |
|---|---:|---:|---:|---:|
| Ministral-8B target | 640,000 | 100% | 0 | 1.0 |
| Ministral-3B draft | 640,000 | 100% | 0 | 1.0 |

Both mandatory consistency gates passed. The generated run's progress label says `human` on shard writes, but its run identity and generated output path are correct; this is recorded in the run log.

The generated pass used `/venv/main/bin/python -I scripts/rebuttal/score_overlap.py --pair M1 --continuation generated --max-prompts 5000 --batch-size 32 --output-root results/rebuttal/overlap`, scoring 5,000 frozen greedy continuations (625,374 positions). Target agreement was 100%; draft agreement was 99.379% and both median absolute margin deltas were 0, so both passed the ≥98.5% gate. Head scales were 1.0. The scorer's per-output-shard progress messages label these rows “human,” but the run identity, generated output directory, and final completion marker identify the continuation correctly; this label-only issue is recorded in the run log.

### Gemma G1 T2a scoring

**Command:** `/venv/main/bin/python -I scripts/rebuttal/score_overlap.py --pair G1 --continuation generated --max-prompts 5000 --batch-size 32 --output-root results/rebuttal/overlap_g1_retry32`. The successful shards were copied to the canonical `results/rebuttal/overlap/G1/generated/` path. It scored 5,000 frozen generated continuations (638,163 positions); Gemma is generated-only as specified.

| Model | Positions | Argmax agreement | Median absolute margin delta | Head scale |
|---|---:|---:|---:|---:|
| Gemma-12B target | 638,163 | 100% | 0 | 1.0 |
| Gemma-4B draft | 638,163 | 100% | 0 | 1.0 |

The first full attempt used batch size 2 and failed the strict margin gate (both medians were 0.125), despite argmax agreement above 99.7%. A later batch-32 full retry disappeared without a scoring shard or traceback; the subsequent diagnostic found the Gemma checkpoints absent from `/dev/shm`. After restoring the pinned weights, a batch-32 diagnostic and the full retry passed both consistency gates with scales 1.0 and calibration errors 0. The token value was never recorded. These failures and recovery are detailed in `RUNLOG.md`.

### T2c bootstrap analysis

**Command:** `/venv/main/bin/python -I scripts/rebuttal/analyze_overlap.py --input-root results/rebuttal/overlap --output-dir results/rebuttal/overlap`. It analyzed the first 5,000 documents for Q2 human/generated (640,000 / 620,451 positions), Q5 human (640,000), M1 human/generated (640,000 / 625,374), and G1 generated (638,163), with 500 paired document-bootstrap resamples. The Q2/Q5 rows reuse the earlier completed analysis; M1/G1 were computed against their new shards with the same script, seed, alignments, and bootstrap count, then merged. Rejection is `1 − alpha`; T1 and T07 denote target sampling temperatures 1 and 0.7.

| Pair | Continuation | T | Target DP position share | Greedy mismatches at target DP | Expected rejection mass at target DP |
|---|---|---:|---:|---:|---:|
| Q2 | Human | 1 | 39.3% | 66.7% | 48.2% |
| Q2 | Human | 0.7 | 39.3% | 66.7% | 54.1% |
| Q2 | Generated | 1 | 21.4% | 71.2% | 44.2% |
| Q2 | Generated | 0.7 | 21.4% | 71.2% | 54.4% |
| Q5 | Human | 1 | 37.6% | 62.5% | 45.5% |
| Q5 | Human | 0.7 | 37.6% | 62.5% | 51.1% |
| M1 | Human | 1 | 44.7% | 81.4% | 57.0% |
| M1 | Human | 0.7 | 44.7% | 81.4% | 64.7% |
| M1 | Generated | 1 | 22.7% | 83.8% | 47.9% |
| M1 | Generated | 0.7 | 22.7% | 83.8% | 60.9% |
| G1 | Generated | 1 | 0.9% | 34.7% | 12.1% |
| G1 | Generated | 0.7 | 0.9% | 34.7% | 23.4% |

Expected rejection mass at target decision points exceeds the DP position share in every row, including G1, where only 0.9% of positions are DPs but 12.1–23.4% of rejection mass occurs there. Greedy disagreement is more concentrated in DPs in every row (34.7–83.8% of its mass); the paired bootstrap intervals for sampling-minus-greedy DP mass share exclude zero throughout (−11.2 to −36.0 percentage points for the added M1/G1 rows). Thus sampling rejection is enriched at decision points but consistently less concentrated there than greedy mismatch.

| Pair | Continuation | T | Mean rejection | Greedy disagreement | Lowest-margin-20% positions / rejection mass | Spearman(rejection, margin) |
|---|---|---:|---:|---:|---:|---:|
| Q2 | Human | 1 | 0.325 | 0.361 | 21.7% / 26.6% | −0.473 |
| Q2 | Human | 0.7 | 0.334 | 0.361 | 21.7% / 30.2% | −0.586 |
| Q2 | Generated | 1 | 0.170 | 0.161 | 21.3% / 44.0% | −0.842 |
| Q2 | Generated | 0.7 | 0.160 | 0.161 | 21.3% / 54.1% | −0.890 |
| Q5 | Human | 1 | 0.352 | 0.382 | 20.9% / 25.2% | −0.442 |
| Q5 | Human | 0.7 | 0.360 | 0.382 | 20.9% / 28.6% | −0.553 |
| M1 | Human | 1 | 0.230 | 0.265 | 23.6% / 30.2% | −0.551 |
| M1 | Human | 0.7 | 0.237 | 0.265 | 23.6% / 35.0% | −0.676 |
| M1 | Generated | 1 | 0.123 | 0.115 | 20.4% / 43.4% | −0.861 |
| M1 | Generated | 0.7 | 0.112 | 0.115 | 20.4% / 55.6% | −0.920 |
| G1 | Generated | 1 | 0.043 | 0.016 | 20.7% / 65.7% | −0.701 |
| G1 | Generated | 0.7 | 0.023 | 0.016 | 20.7% / 82.0% | −0.743 |

| Pair | Continuation | T | Morpheme/internal raw → margin / entropy matched | Word/internal raw → margin / entropy matched | Fused/split raw → margin / entropy matched |
|---|---|---:|---:|---:|---:|
| Q2 | Human | 1 | 1.268 → 0.900 / 0.860 | 1.319 → 0.954 / 0.901 | 1.121 → 0.899 / 0.918 |
| Q2 | Human | 0.7 | 1.390 → 0.892 / 0.852 | 1.572 → 1.019 / 0.920 | 1.155 → 0.872 / 0.934 |
| Q2 | Generated | 1 | 2.039 → 1.045 / 0.937 | 2.249 → 1.205 / 0.958 | 1.430 → 0.909 / 1.015 |
| Q2 | Generated | 0.7 | 2.270 → 1.035 / 0.914 | 2.695 → 1.257 / 0.923 | 1.533 → 0.918 / 1.098 |
| Q5 | Human | 1 | 1.278 → 0.925 / 0.885 | 1.339 → 0.989 / 0.941 | 1.122 → 0.920 / 0.938 |
| Q5 | Human | 0.7 | 1.400 → 0.914 / 0.871 | 1.592 → 1.052 / 0.955 | 1.156 → 0.890 / 0.945 |
| M1 | Human | 1 | 1.344 → 0.896 / 0.822 | 1.584 → 0.994 / 0.811 | 1.110 → 0.878 / 0.947 |
| M1 | Human | 0.7 | 1.483 → 0.888 / 0.821 | 1.903 → 1.028 / 0.791 | 1.138 → 0.853 / 0.984 |
| M1 | Generated | 1 | 1.945 → 1.077 / 0.891 | 2.456 → 1.277 / 0.859 | 1.353 → 0.909 / 1.050 |
| M1 | Generated | 0.7 | 2.144 → 1.061 / 0.858 | 2.920 → 1.291 / 0.785 | 1.431 → 0.899 / 1.129 |
| G1 | Generated | 1 | 1.083 → 0.952 / 1.079 | 2.120 → 1.214 / 1.157 | 1.119 → 0.744 / 0.902 |
| G1 | Generated | 0.7 | 1.181 → 1.042 / 1.226 | 2.513 → 1.266 / 1.205 | 1.070 → 0.638 / 0.828 |

Margin matching keeps morpheme/internal within 0.888–1.077 and fused/split at or below 0.920 across all pairs. Human-text word/internal ratios after margin matching are 0.954–1.052. Generated-text word/internal ratios are 1.205–1.291, with four of the six temperature/pair rows above 1.25; the predicted modest excess is present but can be larger than expected. Entropy-matched ratios and overlap confidence intervals are in `results/rebuttal/overlap/summary.csv`; decision-point-mass intervals are in `decision_point_comparison.csv`, and margin-decile curves are in `margin_deciles.csv`.

**Prediction / outcome / verdict:** expected rejection mass is enriched at target decision points but is less concentrated there than greedy disagreement for every pair and text type; this prediction is held. Margin matching brings morpheme/internal and fused/split into their expected ranges. Generated word/internal remains elevated across Q2, M1 and G1, sometimes beyond the predicted 1.25 ceiling, while human ratios stay near 1. **Partly held.**

## T3. Token tree

### Qwen T3a scoring

**Commands:** `/venv/main/bin/python -I scripts/rebuttal/score_draft_topk.py --pair Q2 --max-prompts 5000 --output-root results/rebuttal/draft_topk` and the same command with `--pair Q5`. The scorer followed the first 5,000 common generated continuations and replayed the frozen teacher-forcing reference batches (2,000 prompts per source shard, length-sorted). It stored the draft top-eight distribution on each target continuation position.

| Pair | Documents | Positions | Draft argmax agreement | Median absolute margin delta | Head scale |
|---|---:|---:|---:|---:|---:|
| Q2 | 5,000 | 620,451 | 100% | 0 | 1.0 |
| Q5 | 5,000 | 616,280 | 100% | 0 | 1.0 |

Both consistency gates passed. The per-pair metadata records the A6000 state and full `nvidia-smi` output at job start.

### Gemma G1 T3a scoring

**Command:** `NAACL_MODEL_DIR=/dev/shm/naacl_models PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /venv/main/bin/python -I scripts/rebuttal/score_draft_topk.py --pair G1 --max-prompts 5000 --output-root results/rebuttal/draft_topk`. It scored the first 5,000 frozen generated continuations for Gemma-12B and stored the Gemma-4B draft's top-eight distribution (638,163 positions).

| Pair | Documents | Positions | Draft argmax agreement | Median absolute margin delta | Head scale |
|---|---:|---:|---:|---:|---:|
| G1 | 5,000 | 638,163 | 100% | 0 | 1.0 |

The consistency gate passed; maximum head calibration error was zero. The run metadata records the A6000 state and full `nvidia-smi` output.

### M1 T3a scoring

**Command:** `/venv/main/bin/python -I scripts/rebuttal/score_draft_topk.py --pair M1 --max-prompts 5000 --output-root results/rebuttal/draft_topk`. It stored the Ministral-3B draft's top-eight distribution along the first 5,000 frozen generated continuations for Ministral-8B (625,374 positions).

| Pair | Documents | Positions | Draft argmax agreement | Median absolute margin delta | Head scale |
|---|---:|---:|---:|---:|---:|
| M1 | 5,000 | 625,374 | 100% | 0 | 1.0 |

The consistency gate passed. The per-pair metadata records the A6000 state and full `nvidia-smi` output at job start.

### Qwen, M1 and Gemma G1 T3b comb-tree replay

**Command:** `/venv/main/bin/python -I scripts/rebuttal/replay_tree.py --pair Q2 Q5 M1 G1 --output-dir results/rebuttal/tree`. For each pair, the first 5,000 generated continuations were scored (2,500 even-index dev and 2,500 odd-index test documents). Dev-tuned margin thresholds were 2.375 for Q2, 2.5 for Q5, 1.875 for M1, and 2.5 for G1. The replay evaluated fixed k=4 and these margin rules with widths 1/2/3/4/8. Intervals use 500 paired document-bootstrap resamples.

| Pair | Width-8 margin tokens/call, everywhere → draft-DP only | Gain retained [95% CI] | Extra nodes retained [95% CI] |
|---|---:|---:|---:|
| Q2 | 3.667 → 3.561 | 77.7% [76.9%, 78.5%] | 18.6% [18.2%, 19.1%] |
| Q5 | 3.424 → 3.310 | 75.4% [74.5%, 76.1%] | 20.3% [19.8%, 20.7%] |
| M1 | 4.080 → 4.026 | 86.8% [86.1%, 87.5%] | 18.2% [17.7%, 18.6%] |
| G1 | 10.792 → 10.617 | 52.3% [49.6%, 55.2%] | 1.4% [1.3%, 1.4%] |

At width 8, branching everywhere adds 19.38 / 17.77 / 22.03 / 69.97 verified nodes per target call for Q2 / Q5 / M1 / G1; restricting branching to draft decision points lowers this to 3.61 / 3.60 / 4.00 / 0.97. The corresponding no-tree margin baselines are 3.192 / 2.960 / 3.671 / 10.425 tokens per call. At width 8 under the margin rule with branching everywhere, rescue rates for target decision-point versus non-decision-point rejections are 0.825 / 0.796 / 0.916 / 0.547 for Q2 / Q5 / M1 / G1, versus 0.870 / 0.834 / 0.929 / 0.735 off decision points. For draft decision points versus non-decision points, the rates are 0.843 / 0.811 / 0.924 / 0.715, versus 0.822 / 0.801 / 0.884 / 0.624, respectively; rescue is consistently higher at draft decision points but lower at target decision points. The results CSV reports these rates for all widths and both branching modes, including paired document-bootstrap intervals. The frozen logs support a comb-tree replay only; sibling branches are not extended beyond the mismatch position, and these are simulated rather than live tree-decoding timings.

**Prediction / outcome / verdict:** restricting width-8 margin branching to draft decision points preserves 52.3–86.8% of the gain at 1.4–20.3% of the extra nodes across the four pairs. For G1, this retains 52.3% [49.6%, 55.2%] of the gain while using 1.4% [1.3%, 1.4%] of the extra nodes; the other pairs retain 75.4–86.8%. The predicted rescue concentration holds at draft decision points in all four pairs, but target decision-point rescue rates are lower than non-decision-point rates in all four. Gain/node is partly held because Gemma retains only about half the gain; the rescue-rate prediction is partly held.

## T4. Decision-strength intervention

### Q2 (Qwen3 0.6B → 4B)

**Commands:** the first pass, `/venv/main/bin/python -I scripts/rebuttal/score_copy_context.py --pair Q2 --max-prompts 2000 --batch-size 8 --output-root results/rebuttal/copy_context`, produced all shards but failed condition A consistency (qwen4 98.30%; qwen06 97.59%; qwen06 median margin delta 0.0625). A 100-document batch-32 diagnostic also failed. The cause was that condition A had been scored on only complete documents, which changed the teacher-forcing batch composition. The corrected run was `/venv/main/bin/python -I scripts/rebuttal/score_copy_context.py --pair Q2 --max-prompts 2000 --batch-size 32 --output-root results/rebuttal/copy_context_rebatch`; its A pass reconstructed the original common-set 2,000-prompt shards and length-grouped batches, then retained the first 2,000 complete continuations. Analysis: `/venv/main/bin/python -I scripts/rebuttal/analyze_copy_context.py --pair Q2`. The failed attempts are preserved under `results/rebuttal/copy_context/attempts/` and `results/rebuttal/copy_context_diag32/`.

The intervention uses the first 2,000 complete human continuations in ascending `pool_index` (256,000 token positions per condition). Position types come from the fixed human alignment. A is the original prompt; B prepends the same continuation; C prepends its shuffled decoded eojeols; D prepends the next document's continuation. The conditions keep the scored continuation tokens fixed.

| Condition | Disagreement | Mean target margin | Target decision-point share | Morph/internal raw → matched | Word/internal raw → matched | Fused/split raw → matched |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.359 | 2.404 | 0.390 | 1.536 → 0.878 | 1.830 → 1.019 | 1.179 → 0.845 |
| B | 0.001 | 9.873 | 0.001 | 5.471 → 3.616 | 3.609 → 2.523 | 1.089 → 0.621 |
| C | 0.262 | 3.692 | 0.221 | 3.480 → 1.070 | 8.001 → 1.823 | 1.547 → 0.909 |
| D | 0.366 | 2.403 | 0.388 | 1.538 → 0.884 | 1.840 → 1.038 | 1.210 → 0.874 |

The common-margin-curve largest / position-weighted mean absolute gaps versus A were B 0.3083 / 0.2209, C 0.0168 / 0.0127, and D 0.0097 / 0.0077 (11, 11, and 10 shared bins). The following table gives all condition and condition × position-type log-odds coefficients, with document-clustered 95% intervals, before and after adding margin-bin fixed effects. `fused`, `morph`, and `word` interactions are relative to internal positions.

| Term | Without margin-bin effects | With margin-bin effects |
|---|---:|---:|
| B | −7.134 [−7.639, −6.629] | −3.613 [−4.136, −3.090] |
| C | −1.671 [−1.714, −1.627] | −0.760 [−0.806, −0.714] |
| D | 0.021 [0.010, 0.032] | 0.025 [0.010, 0.041] |
| B × fused | 0.173 [−1.857, 2.203] | −0.143 [−2.180, 1.895] |
| C × fused | 1.116 [0.971, 1.261] | 0.450 [0.292, 0.607] |
| D × fused | −0.016 [−0.081, 0.049] | −0.008 [−0.085, 0.068] |
| B × morph | 1.084 [0.520, 1.648] | 1.392 [0.829, 1.954] |
| C × morph | 0.791 [0.743, 0.839] | 0.427 [0.374, 0.480] |
| D × morph | 0.007 [−0.011, 0.024] | 0.004 [−0.018, 0.026] |
| B × word | 0.373 [−0.195, 0.941] | 0.801 [0.236, 1.366] |
| C × word | 1.713 [1.667, 1.759] | 1.107 [1.057, 1.156] |
| D × word | 0.018 [0.003, 0.034] | 0.033 [0.012, 0.053] |

The paired A/B view and common curves are in the adjacent CSV outputs.

**Consistency:** condition A reproduces the stored human logs at 100% argmax agreement over 256,000 positions for each model, with median absolute top-two-margin delta 0. The GPU run recorded its A6000 state and `nvidia-smi` output in `run_metadata.json`.

**Prediction / outcome / verdict (Q2):** B lowers disagreement at each position type, as predicted, but the near-zero B disagreement comes with a large margin shift and large common-curve gap; B also leaves high matched position-type ratios because disagreements are rare. C and D have curves close to A, but C retains a large matched word-onset/internal ratio (1.823), and the margin-adjusted condition and interaction terms are not generally near zero. **Partly held.**

On Q2, copying the continuation into the context drives disagreement from 35.9% to 0.1%, while mean target margin rises from 2.40 to 9.87 nats and the decision-point share falls from 39.0% to 0.1%. The longer-context control D stays close to A (36.6% versus 35.9% disagreement), so the B effect is not explained by context length alone. Shuffling the continuation's eojeols in C reduces disagreement to 26.2%, yet its margin-matched word-onset/internal ratio remains 1.82 and its common-margin curve is close to A. This experiment changes decision strength, not morphology; in B both models can copy the continuation, so the near-zero disagreement is evidence about context predictability rather than boundary causality.

### M1 (Ministral-3 3B → 8B)

**Command:** `/venv/main/bin/python -I scripts/rebuttal/score_copy_context.py --pair M1 --max-prompts 2000 --batch-size 32 --output-root results/rebuttal/copy_context`; analysis: `/venv/main/bin/python -I scripts/rebuttal/analyze_copy_context.py --pair M1`. The sample is the first 2,000 complete common human continuations by ascending `pool_index` (256,000 positions per condition), with the same four context conditions and position alignments described for Q2.

| Condition | Disagreement | Mean target margin | Target decision-point share | Morph/internal raw → matched | Word/internal raw → matched | Fused/split raw → matched |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.264 | 1.933 | 0.446 | 1.580 → 0.858 | 2.180 → 0.990 | 1.163 → 0.852 |
| B | 0.002 | 8.058 | 0.002 | 4.257 → 2.367 | 2.021 → 0.928 | 2.804 → 0.717 |
| C | 0.210 | 3.138 | 0.263 | 3.499 → 1.071 | 12.339 → 1.831 | 1.439 → 0.862 |
| D | 0.266 | 1.924 | 0.447 | 1.586 → 0.865 | 2.165 → 0.992 | 1.151 → 0.843 |

Condition A reproduces stored human logs exactly for both models over all 256,000 positions (100% argmax agreement, median absolute margin delta 0). The common-margin-curve largest / position-weighted mean absolute gaps versus A are B 0.1167 / 0.0770, C 0.0673 / 0.0504, and D 0.0026 / 0.0020 (11 shared bins each). The next table gives all condition and condition × position-type log-odds coefficients, with document-clustered 95% intervals, before and after adding margin-bin fixed effects. Interactions are relative to internal positions.

| Term | Without margin-bin effects | With margin-bin effects |
|---|---:|---:|
| B | −5.902 [−6.259, −5.544] | −1.298 [−1.664, −0.932] |
| C | −1.748 [−1.804, −1.691] | −0.548 [−0.607, −0.489] |
| D | 0.009 [−0.006, 0.024] | 0.007 [−0.015, 0.028] |
| B × fused | 0.586 [−0.055, 1.226] | 0.284 [−0.351, 0.919] |
| C × fused | 1.279 [1.187, 1.372] | 0.571 [0.471, 0.670] |
| D × fused | −0.005 [−0.047, 0.038] | −0.016 [−0.067, 0.035] |
| B × morph | 0.874 [0.469, 1.280] | 0.753 [0.348, 1.159] |
| C × morph | 0.764 [0.699, 0.829] | 0.403 [0.333, 0.472] |
| D × morph | 0.006 [−0.015, 0.028] | 0.007 [−0.021, 0.035] |
| B × word | −0.332 [−0.758, 0.094] | −0.168 [−0.593, 0.258] |
| C × word | 1.954 [1.895, 2.012] | 1.057 [0.995, 1.119] |
| D × word | −0.008 [−0.027, 0.012] | 0.000 [−0.026, 0.026] |

The models, curves, gaps, and paired A/B view are also in the adjacent M1 CSV files.

**Prediction / outcome / verdict (M1):** B lowers disagreement across all four position types, and D again stays close to A. However, B produces a substantial margin shift and the largest curve gap; C also differs from A, and its margin-matched word-onset/internal ratio is 1.831. The condition and interaction terms therefore do not generally vanish after margin adjustment. **Partly held.**

For M1, copying the continuation reduces disagreement from 26.4% to 0.15%, while mean target margin rises from 1.93 to 8.06 nats and the decision-point share drops from 44.6% to 0.18%. D remains close to A at 26.6% disagreement, whereas shuffled C reaches 21.0% and retains a large matched word-onset/internal ratio (1.83). As on Q2, the manipulation changes decision strength rather than morphology, and in B both models can copy the continuation; its very low disagreement does not establish a boundary effect.

## Deviations and limitations

T1c was run for Q5, Q2, and the requested O4_G1_12 pair; no M1 timing pair is in the specified priority list. The optional O1 timing pair and sampled-continuation T2b extension were left out to finish and validate the required T0–T4 work; T5 was not started because the instructions say not to start it without a request. The T4 analyses use the alignment-derived position labels available in this checkout; the paper's label-building helper is absent. `analyze_copy_context.py` emitted Pandas join deprecation and statsmodels covariance specification warnings; the CSV outputs were written successfully. M1 scoring emitted the checkpoint's non-fatal tied-embedding mapping warning, and both models passed the condition-A consistency check. Gemma T2a's batch-2 run failed the exact median-margin gate; using the frozen batch size of 32 passed, after restoring the pinned weights. No Gemma human continuation was scored, as prohibited by the runbook.
