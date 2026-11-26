# Rebuttal experiment report

## Predictions recorded before experiments

- **T1:** The measured cost ratio will be well above the parameter ratio for small pairs. The margin rule will be faster than the fastest fixed block size when the measured ratio is between 0.1 and 0.5. Live tokens per target call will be within 3% of replay for `FixedK(4)` and somewhat below replay for adaptive rules.
- **T2:** Expected rejection will concentrate at target decision points, but less than under greedy verification. After matching on target margin, morpheme-onset/internal should be in 0.85–1.15 and fused/split at or below 1.0; word onsets should retain a small excess of 1.13–1.25.
- **T3:** Rescue rate will be higher at decision points than elsewhere. Branching only at draft decision points will keep most of the tokens-per-call gain of branching everywhere while using a minority of its extra verified nodes.
- **T4:** Disagreement will fall in condition B relative to A at every position type. The disagreement-by-margin curves will nearly coincide, matched ratios will stay near 1 in every condition, and condition and condition-by-position coefficients will be near zero after margin-bin fixed effects.
- **T0b:** The two CPU reruns will reproduce the supplied reference JSONs to the last reported digit: G1 morpheme/internal 1.34 raw, 1.33 after target-margin absolute-bin matching, and 0.96 after matching both margins; Q2 2.30 / 0.99 / 0.97; and 67.4% of Q2 rejections in the most-confident target-margin decile at target decision points.
- **T6-a:** The character-offset join will be valid across the two model families. The human-text cross-family judge will yield AUC 0.714 for Qwen3 0.6B→4B judged by Ministral-8B and 0.725 for Ministral 3B→8B judged by Qwen3-8B; judge-margin matching will reduce the listed position-type ratios near the supplied reference values.
- **T6-b:** On generated text, the cross-family judge margin will predict rejection with AUC at least 0.75 while the boundary label is at most 0.62. Judge-margin-matched morpheme/internal will be 0.85–1.15 and fused/split at most 1.00; word-onset/internal will retain a larger residual than the same-family judge (1.21–1.35), up to 1.45.
- **T5:** The instruction-tuned Qwen pair will show target-margin AUC above 0.85 and boundary-label AUC at or below 0.62; after matching both margins, morpheme/internal will be 0.85–1.15, fused/split at or below 1.05, and word/internal at or below 1.30. Margin stopping will gain at least 8% tokens per target call over fixed k=4.
- **T8:** On target-sampled text at temperature 1.0, target-margin-matched morpheme/internal will be 0.85–1.15, fused/split at most 1.00, and word/internal at most 1.30. Target decision points will hold at least 1.15 times their position share in expected-rejection mass, and expected rejection will fall monotonically across target-margin deciles.
- **T7:** Static compiled decoding will reduce the measured 0.6B/4B and 0.6B/8B cost ratios below 0.40. Margin stopping will be at least 1.3× faster than target-only and 7% faster than the fastest fixed k on both pairs; fixed k=4 will beat target-only, and live tokens per target call will be within 2% of replay.

## Environment and baseline

Work branch: `rebuttal/experiments`. GPU: NVIDIA RTX A6000, CUDA 12.8; one GPU was idle at inspection. `/workspace` is not a persistent volume. A Hugging Face token file is available at `/workspace/hf_token`; its contents are not recorded in logs or metadata and will be used only for authorized model downloads. The round-2 GPU experiments (T5, T6-b, T8 and any T7 work) use the NVIDIA GeForce RTX 3090 with driver 595.99.02, CUDA 13.2, torch 2.14.1+cu132 and Transformers 5.19.0. Round-1 wall-clock results remain those collected on the RTX A6000; the two rounds are reported separately and are not treated as directly comparable timings.

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

## T0b. CPU reference reruns

**Commands:** `/venv/main/bin/python -I scripts/rebuttal/cpu/expD_uniform_matching.py` and `/venv/main/bin/python -I scripts/rebuttal/cpu/expD2_confident_strata.py`, run sequentially in tmux session `t0b_retry`. Logs are `results/rebuttal/logs/t0b_uniform_retry1.log` and `results/rebuttal/logs/t0b_confident.log`; outputs are under `results/rebuttal/cpu/`. The first launch stopped at import because pandas was absent. I installed the repository analysis dependencies into `/venv/main` and reran; both scripts then completed. This is a CPU-only analysis on the RTX 3090 instance (AMD EPYC 7543 host; 6 vCPUs allocated).

| Pair / statistic | Measured | Reference | Difference |
|---|---:|---:|---:|
| G1 morpheme/internal, raw → target-margin absolute bins → both-margin absolute bins | 1.34 → 1.33 → 0.96 | 1.34 → 1.33 → 0.96 | 0 |
| Q2 morpheme/internal, raw → target-margin absolute bins → both-margin absolute bins | 2.30 → 0.99 → 0.97 | 2.30 → 0.99 → 0.97 | 0 |
| Q2 most-confident target-margin decile: rejections at target decision points | 67.4% | 67.4% | 0 pp |

Both output JSONs match their bundled references exactly (`expD_uniform_matching.json`: 12 rows; `expD2_confident_strata.json`: 14 rows). The prediction was held: the G1 and Q2 ratios reproduce at the reported precision, and 67.4% of Q2 rejections in the most-confident decile occur at target decision points. There were no reference differences beyond the last digit. The only deviation was the fixed environment missing pandas on the first attempt; installing the declared analysis stack resolved it without changing the scripts or inputs.

## T6-a. Cross-family margins on human text

**Offset audit:** `scripts/align_morphology.py:_human_references` tokenizes the source document with `add_special_tokens=False`, then takes the first 128 IDs as the prompt and the following IDs as the continuation. `src/table1_morphology.py:_make_prompt_alignment_rows` concatenates and decodes those sequences; `token_ids_to_character_offsets` maps offsets over that full decoded text from character zero, and the function copies the spans directly into `global_token_char_span`. The human path adds no chat wrapper, so both families' sequences start from the document's first token. `tokenize_with_offsets` retains the tokenizer's returned offsets unchanged; whitespace is stripped only from the separate morphology core text, not from the global character span. The offset join is therefore comparable for the shared source documents.

**Command:** `/venv/main/bin/python -I scripts/rebuttal/cpu/expE_crossfamily_human.py`, run on CPU in tmux session `t6a_human`; log `results/rebuttal/logs/t6a_crossfamily_human.log`. The script also computed supplementary rows for the other Qwen3 pairs from existing human logs; it downloaded no additional judges and never used Gemma human text.

| Pair | Shared-offset positions / coverage | AUC cross-family judge [95% CI] | AUC target margin [95% CI] | AUC same-family outside judge [95% CI] | AUC boundary [95% CI] |
|---|---:|---:|---:|---:|---:|
| Qwen3 0.6B→4B; Ministral-8B judge | 791,819 / 67.6% | 0.714 [0.712, 0.715] | 0.782 [0.781, 0.784] | Qwen3-8B: 0.745 [0.744, 0.747] | 0.583 [0.582, 0.585] |
| Ministral 3B→8B; Qwen3-8B judge | 799,662 / 70.5% | 0.725 [0.724, 0.727] | 0.817 [0.816, 0.819] | No third-party Ministral judge in the frozen human logs | 0.573 [0.572, 0.575] |

| Pair / judge | Ratio | Raw | Judge-margin deciles, matched [95% CI] | Judge-margin absolute bins, matched [95% CI] |
|---|---|---:|---:|---:|
| Qwen3 0.6B→4B / Ministral-8B | Morpheme/internal | 1.47 | 1.03 [1.02, 1.04] | 1.02 [1.01, 1.03] |
|  | Word/internal | 1.91 | 1.24 [1.22, 1.25] | 1.23 [1.22, 1.24] |
|  | Fused/split | 1.08 | 0.91 [0.90, 0.92] | 0.91 [0.90, 0.92] |
| Ministral 3B→8B / Qwen3-8B | Morpheme/internal | 1.54 | 0.92 [0.91, 0.93] | 0.92 [0.91, 0.93] |
|  | Word/internal | 2.09 | 1.11 [1.10, 1.12] | 1.11 [1.10, 1.13] |
|  | Fused/split | 1.14 | 1.00 [0.98, 1.01] | 1.00 [0.99, 1.01] |

The prediction was held: the cross-family margins have AUCs 0.714 and 0.725, while the boundary-label AUCs are 0.583 and 0.573. Matching on the cross-family judge margin reduces morpheme/internal to 1.03 and 0.92, and fused/split to 0.91 and 1.00; the word/internal ratios remain 1.24 and 1.11. The offset coverage and all original point estimates match `reference_out/expE_crossfamily_human.json` exactly at the stored precision; the added 500-resample document-bootstrap intervals and absolute-bin ratios are in `results/rebuttal/cpu/expE_crossfamily_human.json`. There are no differences from the reference values beyond the last digit. The only scope limitation is that the frozen human logs contain no third-party Ministral judge for M1.

## T6-b. Cross-family margins on generated text

**Commands:** Self-consistency gate: `/venv/main/bin/python -I scripts/rebuttal/score_cross_family_judge.py --source-target qwen4 --judge qwen4 --max-prompts 200 --batch-size 32` (the gate used the final corrected top-20 target tie-break procedure). Main scoring used `/venv/main/bin/python -I scripts/rebuttal/score_cross_family_judge.py --source-target qwen4 qwen8 --judge ministral8 --max-prompts 5000 --batch-size 8` followed by the same scorer with `--source-target ministral8 --judge qwen8 --max-prompts 5000 --batch-size 8`; CPU analysis: `/venv/main/bin/python -I scripts/rebuttal/analyze_cross_family_judge.py --max-prompts 5000`. Logs are `results/rebuttal/logs/t6b_self_gate_batch32_top20.log`, `t6b_main_retry3.log`, and `t6b_analysis_same_family.log`. Output and per-run metadata are under `results/rebuttal/cross_family/`. GPU: NVIDIA GeForce RTX 3090, driver 595.99.02, CUDA 13.2; torch 2.14.1+cu132, Transformers 5.19.0. The image’s runtime differs from the objective’s torch 2.11 version; these results record the installed runtime.

Pinned revisions: Qwen3-4B Base `906bfd4b4dc7f14ee4320094d8b41684abff8539`, Qwen3-8B Base `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`, Ministral-3-8B Base `d4883f9b36aa2e5d775730d3fdba3d30de51a8ef`. The source-boundary audit compared tokenizer-decoded continuation starts with `global_token_char_span[0]` on 100 documents: Qwen4 11,626/11,723 (99.17%), Qwen8 11,667/11,690 (99.80%), and Ministral8 12,377/12,377 (100%); the scorer used alignment-table offsets when valid. On the 200-document self gate, 23,142/25,179 continuation positions (91.91%) had canonical re-tokenized prefixes; target argmax agreement there was 99.0148%, above the 98.5% threshold. All judge LM-head checks passed at scale 1 with zero max absolute error.

| Pair | Joined coverage | Rejection all / joined | AUC cross-family judge | AUC target margin | AUC same-family judge | AUC boundary | Spearman target/judge |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3 0.6B→4B | 430,489/535,028 (80.46%) | 0.170/0.187 | 0.503 | 0.890 | 0.892 | 0.604 | 0.011 |
| Qwen3 0.6B→8B | 429,972/536,089 (80.21%) | 0.193/0.212 | 0.502 | 0.874 | 0.879 | 0.605 | 0.008 |
| Ministral 3B→8B | 512,461/529,609 (96.76%) | 0.121/0.124 | 0.526 | 0.913 | — | 0.589 | 0.065 |

The following ratios use joined k=4 verified proposals. Intervals are 95% document-bootstrap CIs (500 resamples); matching uses deciles or `expD_lib.EDGES` absolute bins. “Same-family” is Qwen3-1.7B for Q2 and Qwen3-4B for Q5; there is no same-family outside judge for M1.

| Pair | Ratio | Raw [95% CI] | Cross-family judge deciles [95% CI] | Cross-family judge absolute bins [95% CI] | Target margin deciles [95% CI] | Same-family deciles [95% CI] |
|---|---|---:|---:|---:|---:|---:|
| Q2 | morph/int | 2.119 [2.05, 2.20] | 2.119 [2.04, 2.20] | 2.120 [2.05, 2.19] | 0.994 [0.97, 1.02] | 0.985 [0.96, 1.01] |
| Q2 | word/int | 3.189 [3.09, 3.29] | 3.190 [3.10, 3.30] | 3.190 [3.09, 3.31] | 1.219 [1.19, 1.25] | 1.206 [1.18, 1.23] |
| Q2 | fused/split | 1.421 [1.37, 1.47] | 1.422 [1.37, 1.47] | 1.422 [1.37, 1.47] | 0.947 [0.92, 0.97] | 0.943 [0.92, 0.97] |
| Q5 | morph/int | 2.017 [1.96, 2.09] | 2.017 [1.96, 2.08] | 2.017 [1.95, 2.09] | 0.972 [0.95, 0.99] | 0.963 [0.94, 0.98] |
| Q5 | word/int | 3.060 [2.97, 3.15] | 3.060 [2.97, 3.16] | 3.060 [2.96, 3.17] | 1.205 [1.18, 1.23] | 1.189 [1.17, 1.22] |
| Q5 | fused/split | 1.401 [1.36, 1.45] | 1.402 [1.36, 1.45] | 1.401 [1.36, 1.44] | 0.973 [0.95, 1.00] | 0.948 [0.93, 0.97] |
| M1 | morph/int | 1.954 [1.88, 2.03] | 1.933 [1.86, 2.00] | 1.932 [1.86, 2.00] | 0.967 [0.94, 0.99] | — |
| M1 | word/int | 3.029 [2.93, 3.15] | 2.990 [2.88, 3.10] | 2.987 [2.87, 3.10] | 1.195 [1.17, 1.22] | — |
| M1 | fused/split | 1.526 [1.48, 1.58] | 1.522 [1.47, 1.57] | 1.517 [1.47, 1.57] | 0.947 [0.93, 0.97] | — |

**Prediction / outcome / verdict:** The prediction was not held for judge-margin discrimination or for the two primary matched ratios. The cross-family judge AUCs are 0.504, 0.502, and 0.526, far below 0.75, while boundary-label AUCs are 0.604, 0.605, and 0.589 (under 0.62). Judge-margin matched morph/internal remains 2.119, 2.017, and 1.933; fused/split is 1.422, 1.402, and 1.522, all outside the predicted ranges. Target-margin matching brings these ratios near 1, and same-family judges do too for morph/internal and fused/split in Q2/Q5; word/internal stays higher (1.206 and 1.189 under same-family matching).

This generated-text experiment therefore separates two claims: the boundary label remains a weak rejection predictor, but an unrelated-family margin is also a weak predictor and does not recover the target’s ranking. The strong target-margin AUCs (0.874–0.913) and same-family AUCs (0.879–0.892 on Q2/Q5) show that the weak cross-family result is not due to absent signal in the verified labels. The same-family word/internal point estimates (1.206 and 1.189) sit near or just below the anticipated 1.21 lower bound; their intervals [1.18, 1.23] and [1.17, 1.22] straddle it. Cross-family judge matching leaves the word/internal ratio essentially at its raw value. We kept the preregistered judge/pairs and did not retune; no Gemma or additional judges were used.


## T5. Instruction-tuned Qwen pair

**Prediction (recorded before the full run):** The instruction-tuned pair should retain target-margin AUC above 0.85 and boundary-label AUC at or below 0.62; after matching on both margins, morph/internal should be 0.85–1.15, fused/split at or below 1.05, and word/internal no higher than 1.30. If target decision points cover under 5% of positions, either-model low-margin positions should contain at least 65% of rejections. Margin stopping should gain at least 8% target calls over fixed k=4.

**Models and prompt pool:** Qwen/Qwen3-0.6B (instruction-tuned) `c1899de289a04d12100db370d81485cdf75e47ca`; Qwen/Qwen3-4B `1cfa9a7208912126459214e8b04321603b3df60c`. Both model cards identify chat/instruction checkpoints; tokenizer configs provide a chat template accepting `enable_thinking=False`. The ten-prompt tokenizer audit found identical vocabularies and identical rendered/input IDs. `data/rebuttal_instruct_5000.parquet` freezes the first 5,000 Q2 source prompts (pool indices 0–4,999), rendered as a Korean continuation request. The prompts have 5,000 distinct document IDs and 4,994 distinct rendered hashes: six rows repeat an identical rendered prompt. The instruct config explicitly allows identical hashes only when the underlying text is identical; the shared pool loader remains strict by default. All prompts pass the <512-token and template-token-ID assertions. The 50-document pilot completed generation, target/draft teacher forcing, alignment and simulation; ten decoded continuations were reviewed, with no reasoning block or second chat turn.

**Full commands:** `python -I scripts/gen_target.py --config configs/rebuttal_instruct.yaml --target qwen4it --split common --batch-size 48 --output-dir results/rebuttal/instruct`; `python -I scripts/tf_score.py --config configs/rebuttal_instruct.yaml --model qwen4it --target qwen4it --split common --role target --output-dir results/rebuttal/instruct`; the same `tf_score.py` command with `--model qwen06it --role draft`; `python -I scripts/align_morphology.py --config configs/rebuttal_instruct.yaml --target qwen4it --split common --output-dir results/rebuttal/instruct`; `python -I scripts/simulate_sd.py --config configs/rebuttal_instruct.yaml --pair IT_Q2 --split common --block-sizes 1 2 4 8 16 --output-dir results/rebuttal/instruct`; and `/venv/main/bin/python -I scripts/rebuttal/analyze_instruct.py --config configs/rebuttal_instruct.yaml --output-dir results/rebuttal/instruct/analysis`. The stages ran sequentially in tmux `t5_full` on the RTX 3090 (driver 595.99.02, CUDA 13.2, torch 2.14.1+cu132, Transformers 5.19.0); the full generation produced 5,000 rows in 796.8 seconds. Teacher-forced head checks passed at scale 1 with zero calibration error. Both instruction checkpoints were evicted after analysis; run logs and per-stage metadata are under `results/rebuttal/`.

| Continuation quality / gate | Instruct | Base Q2, same 5,000 docs |
|---|---:|---:|
| Length tokens: min / p10 / median / mean / p90 / max | 2 / 25.0 / 114.0 / 95.6 / 128.0 / 128 | 1 / 128.0 / 128.0 / 124.1 / 128.0 / 128 |
| EOS rate | 0.5966 | 0.0472 |
| Hangul share median / p10 | 0.7188 / 0.6071 | 0.6750 / 0.4594 |
| ALIGNED token share | 0.6741 | 0.7535 |
| Repeated 4-gram rate | 0.0512 | 0.4279 |
| Documents with ≥64 continuation tokens | 3,907/5,000 | 4,846/5,000 |

Instruct reasoning-block documents: 0; documents with a second chat turn: 0; median Hangul-share gate (≥0.6): **pass**. The 64-token subset is reported separately below.

| Pair / subset | Rejection | Target DP share / rejection mass | Lowest-20% rejection mass | AUC target / draft / boundary | Either margin <1: position share / rejection mass |
|---|---:|---:|---:|---:|---:|
| Instruct / all | 0.3808 | 0.1907 / 0.3782 | 0.4162 | 0.8446 / 0.8399 / 0.6233 | 0.3983 / 0.7390 |
| Instruct / ≥64 tokens | 0.3828 | 0.1931 / 0.3803 | 0.4183 | 0.8437 / 0.8381 / 0.6234 | 0.4018 / 0.7405 |
| Base Q2 / all | 0.1705 | 0.2252 / 0.7098 | 0.6571 | 0.8957 / 0.9103 / 0.5996 | 0.3114 / 0.9282 |
| Base Q2 / ≥64 tokens | 0.1701 | 0.2248 / 0.7102 | 0.6574 | 0.8960 / 0.9105 / 0.5997 | 0.3109 / 0.9283 |

Ratio values are document-bootstrap 95% CIs (500 resamples); matching uses absolute margin bins.

| Pair / subset | Ratio | Raw [95% CI] | Target margin [95% CI] | Draft margin [95% CI] | Both margins [95% CI] |
|---|---|---:|---:|---:|---:|
| Instruct / all | Morpheme/internal | 2.187 [2.14, 2.24] | 1.153 [1.13, 1.17] | 1.087 [1.07, 1.10] | 1.002 [0.99, 1.01] |
| Instruct / all | Word/internal | 3.228 [3.16, 3.29] | 1.499 [1.48, 1.52] | 1.543 [1.52, 1.56] | 1.300 [1.28, 1.32] |
| Instruct / all | Fused/split | 1.474 [1.45, 1.50] | 0.962 [0.95, 0.98] | 0.941 [0.93, 0.96] | 0.880 [0.87, 0.89] |
| Instruct / ≥64 tokens | Morpheme/internal | 2.181 [2.13, 2.23] | 1.150 [1.13, 1.17] | 1.086 [1.07, 1.10] | 1.000 [0.99, 1.01] |
| Instruct / ≥64 tokens | Word/internal | 3.218 [3.15, 3.29] | 1.497 [1.47, 1.52] | 1.544 [1.52, 1.57] | 1.298 [1.28, 1.32] |
| Instruct / ≥64 tokens | Fused/split | 1.466 [1.44, 1.49] | 0.960 [0.94, 0.98] | 0.940 [0.93, 0.95] | 0.878 [0.86, 0.89] |
| Base Q2 / all | Morpheme/internal | 2.244 [2.18, 2.32] | 0.975 [0.95, 1.00] | 0.982 [0.96, 1.00] | 0.957 [0.94, 0.98] |
| Base Q2 / all | Word/internal | 2.920 [2.83, 3.01] | 1.232 [1.21, 1.26] | 1.221 [1.20, 1.24] | 1.154 [1.13, 1.17] |
| Base Q2 / all | Fused/split | 1.605 [1.55, 1.66] | 0.940 [0.92, 0.96] | 0.940 [0.92, 0.96] | 0.928 [0.91, 0.95] |
| Base Q2 / ≥64 tokens | Morpheme/internal | 2.251 [2.18, 2.32] | 0.975 [0.95, 0.99] | 0.981 [0.96, 1.00] | 0.956 [0.94, 0.97] |
| Base Q2 / ≥64 tokens | Word/internal | 2.926 [2.84, 3.01] | 1.231 [1.21, 1.26] | 1.219 [1.20, 1.24] | 1.152 [1.13, 1.18] |
| Base Q2 / ≥64 tokens | Fused/split | 1.610 [1.56, 1.67] | 0.941 [0.91, 0.96] | 0.940 [0.91, 0.97] | 0.928 [0.91, 0.95] |

Policy replay uses even pool indices for threshold selection and odd indices for evaluation; gains are relative to fixed k=4 with paired document-bootstrap 95% CIs (500 resamples).

| Pair / subset | Policy | Threshold | Draft tokens/call | Tokens/call | Gain vs k=4 [95% CI]% |
|---|---|---:|---:|---:|---:|
| Instruct / all | fixed_k4 | — | 3.955 | 2.329 | 0.00 [0.00, 0.00] |
| Instruct / all | entropy | -2.732 | 3.984 | 2.398 | 2.98 [2.60, 3.38] |
| Instruct / all | top1_probability | -1.019 | 3.884 | 2.401 | 3.11 [2.72, 3.53] |
| Instruct / all | margin | 0.75 | 3.815 | 2.370 | 1.79 [1.41, 2.19] |
| Instruct / ≥64 tokens | fixed_k4 | — | 3.962 | 2.327 | 0.00 [0.00, 0.00] |
| Instruct / ≥64 tokens | entropy | -2.749 | 3.989 | 2.389 | 2.67 [2.26, 3.07] |
| Instruct / ≥64 tokens | top1_probability | -1.026 | 3.895 | 2.393 | 2.83 [2.46, 3.22] |
| Instruct / ≥64 tokens | margin | 0.75 | 3.810 | 2.361 | 1.47 [1.08, 1.90] |
| Base Q2 / all | fixed_k4 | — | 3.960 | 3.429 | 0.00 [0.00, 0.00] |
| Base Q2 / all | entropy | -2.614 | 4.010 | 3.897 | 13.65 [12.66, 14.50] |
| Base Q2 / all | top1_probability | -0.9017 | 3.860 | 3.949 | 15.19 [14.15, 16.30] |
| Base Q2 / all | margin | 1 | 3.963 | 4.016 | 17.12 [16.05, 18.26] |
| Base Q2 / ≥64 tokens | fixed_k4 | — | 3.961 | 3.432 | 0.00 [0.00, 0.00] |
| Base Q2 / ≥64 tokens | entropy | -2.611 | 4.013 | 3.901 | 13.68 [12.85, 14.60] |
| Base Q2 / ≥64 tokens | top1_probability | -0.8998 | 3.860 | 3.953 | 15.19 [14.17, 16.15] |
| Base Q2 / ≥64 tokens | margin | 1 | 3.969 | 4.022 | 17.21 [16.11, 18.31] |

**Prediction / outcome / verdict:** The all-document target-margin AUC is 0.8446, just below 0.85, and the boundary-label AUC is 0.6233, just above 0.62; both AUC predictions missed narrowly. Both-margin matched morpheme/internal (1.002) and fused/split (0.880) meet their ranges; word/internal is exactly 1.300 at the stated ceiling, with a 95% CI extending to 1.32. Target decision points cover 19.07% of positions, so the conditional prediction for pairs with under 5% decision points does not apply; positions where either margin is below 1 nat cover 39.83% of positions and contain 73.90% of rejections. Margin stopping gains 1.79% [1.41%, 2.19%] over fixed k=4, below the predicted 8%. Overall verdict: **partly held**—the position-type ratios largely disappear after matching both margins, while rejection discrimination and the expected adaptive-policy gain are weaker than predicted. Base Q2 on the same documents has target AUC 0.896, boundary AUC 0.600, matched ratios 0.957 / 1.154 / 0.928, and a 17.12% [16.05%, 18.26%] margin-policy gain. The instruct continuations also have much lower four-gram repetition (5.12% vs 42.79%) and lower aligned-token share (67.41% vs 75.35%) than base Q2. The only launch deviation was an initial pre-generation failure due to already-evicted weights; reloading resolved it, and the complete 5,000-document run was analyzed without a second pair or prompt setting.

## T8. Speculative sampling on sampled text

**Prediction (recorded before the run):** At temperature 1.0, target-margin-matched morph/internal should be 0.85–1.15, fused/split at most 1.00, word/internal at most 1.30; target decision points should carry at least 1.15 times their position share in expected-rejection mass; expected rejection should decrease monotonically over target-margin deciles.

**Command and setup:** `bash scripts/rebuttal/run_t8_sampled.sh` in tmux `t8_sampled`. It ran `/venv/main/bin/python -I scripts/rebuttal/gen_target_sampled.py --target qwen4 --temperature 1.0 --max-prompts 2000 --batch-size 48 --output-root results/rebuttal/sampled_T1` and the same command with `--target qwen8`, followed by `/venv/main/bin/python -I scripts/rebuttal/analyze_sampled_gates.py --sampled-root results/rebuttal/sampled_T1 --language-only`. Both 2,000-document generations completed on the RTX 3090 (driver 595.99.02; CUDA 13.2; torch 2.14.1+cu132; Transformers 5.19.0). Revisions are Qwen3-4B-Base `906bfd4b4dc7f14ee4320094d8b41603b3df60c` and Qwen3-8B-Base `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`. Sampling used seed 20261125 (per-shard seed = base seed + shard index), `do_sample=true`, temperature 1, `top_p=1.0`, `top_k=0` and the other truncation/filter controls disabled. Actual batch sizes, including OOM backoff, were 1, 2, 3, 4, 46, and 48 for both targets.

The language gate ran before alignment or overlap scoring. Hangul share is calculated per decoded continuation document; the gate requires median ≥0.60.

| Target / pair | Documents | Length tokens: min / p10 / median / mean / p90 / max | EOS rate | Hangul share: median / p10 | Korean gate |
|---|---:|---|---:|---:|---|
| Qwen3-4B / Q2 | 2,000 | 2 / 128 / 128 / 127.070 / 128 / 128 | 0.0160 | 0.4000 / 0.2900 | **Fail** |
| Qwen3-8B / Q5 | 2,000 | 9 / 128 / 128 / 127.5125 / 128 / 128 | 0.0095 | 0.5429 / 0.4027 | **Fail** |

The unfiltered-sampling check comparing sampled-token mean NLL to target mean entropy was **not run**: the prescribed Hangul gate failed for both pairs (median shares 0.4000 and 0.5429, both below 0.60), so the pipeline stopped before alignment, teacher-forced overlap scoring, and analysis. Length and EOS statistics are reported above; no acceptance or position-type estimates were produced.

| Prediction component | Outcome |
|---|---|
| Target-margin-matched morph/internal in 0.85–1.15 | Not assessed; stopped at Korean gate |
| Target-margin-matched fused/split ≤1.00 | Not assessed; stopped at Korean gate |
| Target-margin-matched word/internal ≤1.30 | Not assessed; stopped at Korean gate |
| Decision-point rejection-mass / position-share ratio ≥1.15 | Not assessed; stopped at Korean gate |
| Expected rejection falls monotonically across target-margin deciles | Not assessed; stopped at Korean gate |

**Outcome / verdict:** The sampled continuations fail the Korean-language quality gate for both required pairs, so T8 stops as specified and its speculative-acceptance predictions remain untested. The samples are heavily length-capped (median 128 tokens for both targets) and terminate with EOS on 1.60% of Qwen4 and 0.95% of Qwen8 documents. Although generation explicitly disabled top-k/top-p truncation, the empirical NLL-versus-entropy gate was not measured and is therefore not claimed to pass. No morphology alignment, overlap scoring, Q5 greedy comparator, T8 acceptance table, or optional temperature-0.7 run was produced; no language filter or prompt subset was introduced to rescue the gate.

## T7. Wall-clock with per-call overhead removed

The preregistered prediction is recorded above. T7-a completed; the Q2 static-compiled path failed the required target-only equivalence gate in both compile modes, so T7-c timing was not run.

### T7-a. Forward-overhead diagnosis

**Command:** `/venv/main/bin/python -I scripts/rebuttal/profile_forward_overhead.py --models qwen06 qwen4 qwen8 --output-dir results/rebuttal/wallclock_opt` (30 timed repeats and 8 warm-ups per operation). It used one fixed 128-token prompt and cached single-token, 5-token, and 17-token forwards. GPU: NVIDIA GeForce RTX 3090, driver 595.99.02, CUDA 13.2; torch 2.14.1+cu132 and Transformers 5.19.0. Host CPU: AMD EPYC 7543; `nproc` reports 6 CPUs available to the container (host-wide `lscpu` reports 128 logical CPUs). The GPU profiler sums only CUDA-device kernel events; host time is the synchronized median wall time minus that sum. Model layers and per-layer wall time are shown with each result.

| Model | Operation | Layers | Wall ms | Wall ms/layer | GPU kernel ms | Host ms | Host share |
|---|---|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | Cached 1 token | 28 | 22.19 | 0.792 | 5.50 | 16.69 | 75.2% |
| Qwen3-0.6B | Verify 5 tokens | 28 | 33.85 | 1.209 | 6.90 | 26.95 | 79.6% |
| Qwen3-0.6B | Verify 17 tokens | 28 | 32.92 | 1.176 | 7.81 | 25.10 | 76.3% |
| Qwen3-4B | Cached 1 token | 36 | 37.60 | 1.045 | 14.83 | 22.78 | 60.6% |
| Qwen3-4B | Verify 5 tokens | 36 | 43.59 | 1.211 | 16.27 | 27.31 | 62.7% |
| Qwen3-4B | Verify 17 tokens | 36 | 44.46 | 1.235 | 17.54 | 26.92 | 60.5% |
| Qwen3-8B | Cached 1 token | 36 | 39.73 | 1.104 | 23.39 | 16.34 | 41.1% |
| Qwen3-8B | Verify 5 tokens | 36 | 45.47 | 1.263 | 24.85 | 20.63 | 45.4% |
| Qwen3-8B | Verify 17 tokens | 36 | 44.50 | 1.236 | 28.08 | 16.43 | 36.9% |

| Pair | Target verify tokens | Wall draft/target ratio | GPU-kernel draft/target ratio |
|---|---:|---:|---:|
| Q2 (0.6B→4B) | 5 | 0.509 | 0.338 |
| Q2 (0.6B→4B) | 17 | 0.499 | 0.314 |
| Q5 (0.6B→8B) | 5 | 0.488 | 0.221 |
| Q5 (0.6B→8B) | 17 | 0.499 | 0.196 |

The old wall-clock ratios remain near 0.50 for both target widths, but the profiler's kernel-only ratios fall to 0.314 for Q2 and 0.196 for Q5 at 17-token verification. The 0.6B path spends 76–80% of measured time outside GPU kernels; the corresponding shares are 61–63% for 4B and 37–45% for 8B. This supports a substantial host-side per-call cost, while also showing that it is smaller as a fraction of the 8B forward. Deviations: the first probe stopped because this PyTorch version uses `self_device_time_total`; a preliminary rerun double-counted CPU operators and CUDA kernel events. Both attempts are preserved in `results/rebuttal/logs/`; the reported CSV is from the corrected CUDA-event-only pass. The installed torch 2.14.1 differs from the plan's torch 2.11, while Transformers 5.19.0 matches.

### T7-b. Static-cache compile gate and T7-c stop

**Command:** `/venv/main/bin/python -I scripts/rebuttal/validate_static_compiled.py --pairs Q2 Q5 --documents 50`, run in tmux `t7_equivalence_retry`; log `results/rebuttal/logs/t7_equivalence.log`. It tried `torch.compile(mode="reduce-overhead")` first and automatically retried with `mode="default"` after the first failure. Both models use bfloat16 and SDPA on the same RTX 3090. Pinned revisions: Qwen3-0.6B `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`, Qwen3-4B `906bfd4b4dc7f14ee4320094d8b41684abff8539`, and Qwen3-8B `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`. The gate stopped on Q2 target-only, before fixed-k, margin, speculative forward-count checks, or Q5.

| Pair / mode | Policy | Documents | Dynamic repeat exact | Static vs dynamic exact | `target_forwards == blocks` | Gate |
|---|---|---:|---:|---:|---|---|
| Q2 / reduce-overhead | Target-only | 50 | 1.000 | 0.660 | Not applicable | **Fail** |
| Q2 / default fallback | Target-only | 50 | 1.000 | 0.060 | Not applicable | **Fail** |

Short target-only diagnostics localized the discrepancy to the compiled path. On the first document (pool index 1), both eager static and compiled static matched dynamic exactly. On the first five documents (pool indices 1, 3, 5, 7, 9), eager static matched 5/5, while compiled static matched 3/5. The two compiled divergences began at token 11 for pool index 3 (dynamic ID 115, static ID 231) and token 15 for pool index 9 (dynamic ID 15, static ID 13); the 50-document default-mode run matched only 3/50. We did not measure logit margins at the divergence, so these are output mismatches without a claim that they are near-tie flips. Logs and per-attempt summaries/metadata are under `results/rebuttal/logs/t7_static_*` and `results/rebuttal/wallclock_opt/equivalence/attempts/`.

| Preregistered T7 prediction component | Outcome |
|---|---|
| Optimized 0.6B/4B and 0.6B/8B cost ratios below 0.40 | Not assessed; equivalence gate failed |
| Margin at least 1.3× faster than target-only on both pairs | Not assessed; no timing run |
| Margin at least 7% faster than the fastest fixed k | Not assessed; no timing run |
| Fixed k=4 faster than target-only | Not assessed; no timing run |
| Live tokens per target call within 2% of replay | Not assessed; no timing run |

**Prediction / outcome / verdict:** T7's overhead diagnosis is supported by T7-a: the measured wall ratios remain about 0.50, while kernel-only ratios fall to 0.314 (Q2) and 0.196 (Q5) for 17-token verification, with host overhead shares of 76.3% and 36.9% respectively. The optimized path did not pass equivalence: target-only exact match was 0.660 under `reduce-overhead` and 0.060 under `default`, against 1.000 dynamic repeat agreement. Eager static matched all five diagnostic documents, while compiled static matched three of five, which points to compiler-induced output differences on the checked examples; the cause of the full-set discrepancy remains unresolved. Because the gate failed, optimized latency, cost ratios, Q5 equivalence, speculative policies, and all T7-c predictions are unassessed; no timing was taken on a non-equivalent path. **Overall verdict: T7 prediction not held by the measured evidence; T7-c not completed.**

**Deviations:** The installed torch is 2.14.1 rather than the plan's 2.11; Transformers 5.19.0 matches. The first gate invocation exposed a validator model-key bug, which was fixed before retrying. The main 50-document run completed both prescribed compile-mode attempts; short one- and five-document diagnostics were then used to isolate the failure. The shared metadata destination had been reused by those diagnostics, so per-attempt metadata was reconstructed from the saved logs and captured separately; this is marked in each `run_metadata.json`.

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

T1c was run for Q5, Q2, and the requested O4_G1_12 pair; no M1 timing pair is in the specified priority list. The optional O1 timing pair and sampled-continuation T2b extension from round 1 were not run. In round 2, T8 stopped at its Korean-language gate, and T7-c stopped before timing because static-compiled decoding failed equivalence. The T4 analyses use the alignment-derived position labels available in this checkout; the paper's label-building helper is absent. `analyze_copy_context.py` emitted Pandas join deprecation and statsmodels covariance specification warnings; the CSV outputs were written successfully. M1 scoring emitted the checkpoint's non-fatal tied-embedding mapping warning, and both models passed the condition-A consistency check. Gemma T2a's batch-2 run failed the exact median-margin gate; using the frozen batch size of 32 passed, after restoring the pinned weights. No Gemma human continuation was scored, as prohibited by the runbook.

## Round 2 final validation

`/venv/main/bin/python -I -m pytest -q` passed: **157 passed in 10.42 s**. `git diff --check` and `py_compile` over the modified and new Python modules also passed. The final test log is `results/rebuttal/logs/pytest_round2_final.log`.
