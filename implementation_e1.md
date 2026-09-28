# E1 Token-Frequency Confounding Analysis Plan and Results

**Goal:** Determine whether token rarity explains the higher SD rejection of `CROSS_MORPHEME` tokens than `WITHIN_SPLIT` tokens in the existing 1,000-prompt run.

**Architecture:** Add a standalone E1 analysis module and CLI that join H2 token rows to token-ID occurrence counts from the locally cached Korean Wikipedia prompt corpus. Cache corpus counts with a content hash and resolved tokenizer revision, then fit the original H2 M3, linear and nonlinear frequency controls, a frequency-stratified robustness model, and a frequency interaction model. No generation or SD decoding is repeated.

**Files:** Add `src/e1_frequency.py` and `scripts/analyze_e1.py`; write this plan and all E1 output artifacts inside `runs/20260926T184145Z_pilot1000/`.

**Fixed analysis rules:** Reuse H2's eligible population (`sd_valid=True`, fragmentation categories `2,3,4,5,6,7,8+`, and the `CROSS_MORPHEME`/`WITHIN_SPLIT` contrast); keep `WITHIN_SPLIT` and fragmentation `2` as references; cluster standard errors by `prompt_id`; report two-sided tests; use the same M2 structural controls and M3 entropy controls; do not make causal claims.

## Steps

1. Inspect and load the existing H2 token table, H2 model formula, local Korean Wikipedia JSONL cache, and configured Qwen target tokenizer. Implement a deterministic tokenizer-frequency cache keyed by corpus content hash and tokenizer name/revision. Record document count, total token count, cache metadata, token count, relative frequency, vocabulary rank, log count, unseen indicator, and eojeol-surface counts where feasible.
2. Create `e1_frequency_table.parquet` by joining cached frequencies onto H2 rows by token ID. Refit M3 as an equivalence check; fit M4 with linear `log_token_count`, M5 with a cubic spline of log count, and M6 only if unseen tokens have adequate support. Also fit decile fixed-effect frequency control, a clustered adjusted rarity contrast, and exploratory morphology-by-frequency interaction.
3. Write class-level descriptive statistics, the M3–M6/robustness regression report, distribution and predicted-probability figures, and a numeric summary with the original M3 OR, frequency-controlled OR, coefficient change, CI check, robustness conclusion, and A–D classification.
4. Run the CLI on existing artifacts only. Verify that reproduced M3 is approximately OR 1.77; check output counts, cache identity, model specifications, and all requested files before transferring artifacts into the Git repository and pushing.

**Verification command:** `python scripts/analyze_e1.py runs/20260926T184145Z_pilot1000 --prompt-cache data/korean_wikipedia_20231101_ko.jsonl`

**Review focus:** tokenizer revision or corpus changes invalidate the cache; zero-count tokens require finite `log1p` values; ties can reduce available quantile bins; the decile robustness and spline models must use the same eligible rows as M3; predictions must average over a documented covariate sample; token frequency is estimated only from the local 1,000-document cache and is not an external corpus frequency.

## Results

The complete report and generated tables are in [`runs/20260926T184145Z_pilot1000/e1_frequency_control_summary.md`](runs/20260926T184145Z_pilot1000/e1_frequency_control_summary.md). The analysis used the saved run and H2 artifacts; speculative decoding was not rerun.

- **Classification: A — frequency does not explain the pooled H2 result.** The CROSS coefficient stayed positive under linear and nonlinear frequency adjustment, and the frequency-stratified estimate remained positive.
- Corpus coverage: 1,000 locally cached Korean Wikipedia texts, 3,425,737 Qwen tokenizer tokens. The primary set contained 73,143 tokens across 997 prompts; 86 tokens (0.118%) were unseen.
- Original M3: coefficient **0.5687**, OR **1.766** (95% CI 1.475–2.115), p=6.342e-10.
- M4, linear log frequency: coefficient **0.5594**, OR **1.750** (95% CI 1.457–2.101), p=2.107e-09.
- M5, nonlinear spline frequency: coefficient **0.6387**, OR **1.894** (95% CI 1.542–2.327), p=1.176e-09. The coefficient increased **12.3%** relative to M3; the confidence interval still excludes OR=1.
- Frequency-stratified robustness: OR **1.961** (95% CI 1.614–2.381), p=1.110e-11, using 44,158 tokens across 992 prompts in frequency deciles with both classes.
- M6 was not fit: only 86 analyzed tokens were unseen (28 prompts), which did not meet the prespecified support threshold.
- Exploratory interaction: conditional CROSS OR was 2.57 (95% CI 2.04–3.24) at the 10th frequency percentile, 1.36 (1.07–1.72) at the median, and 0.64 (0.45–0.93) at the 90th percentile. The pooled M5 result persists, while the interaction suggests the CROSS excess is concentrated among rarer tokens.

These results describe associations in this generated-continuation sample and do not establish causality.
