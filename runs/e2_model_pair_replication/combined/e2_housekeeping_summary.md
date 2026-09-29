# E2 housekeeping summary

Adjusted CROSS-minus-SPLIT differences (M2/M3/M5): P1 M2/M3/M5 +11.59/+5.44/+5.79 pp; P2 M2/M3/M5 +10.95/+5.15/+6.62 pp; P3 M2/M3/M5 +11.48/+5.36/+6.07 pp. The exact three-way eligible intersection is 993 prompts; common-prompt sensitivity is **unchanged** (maximum OR change 1.24%); historical P3 weight revisions remain unresolved.

**These checks do not change `E2 result: A — strong replication`.** All pair-specific adjusted differences remain positive and the common-prompt sensitivity does not materially change the pooled estimates. The unresolved P3 revisions limit exact model-level reproducibility, but are not by themselves evidence against the statistical replication.

## CONFIRMED

- All three pair-specific M2, M3, and M5 adjusted marginal rejection differences use the unchanged H2 eligible populations and prompt-clustered delta-method covariance.
- The common-prompt-only pooled analysis uses exactly 993 eligible prompt IDs present in P1, P2, and P3.
- Common-prompt pooled sensitivity classification: unchanged; classification uses effect magnitude and direction, not significance alone.

## EXPLORATORY

- Adjusted marginal probabilities are conditional associations averaged over each fitted sample's observed covariate distribution; they are not causal effects.
- The common-prompt restriction is a sensitivity analysis and does not replace the primary pooled estimates.

## FAILED / INCOMPLETE

- P3 historical model-weight revision recovery is unresolved for both draft and target. Local snapshots were created after the P3 run interval, and available tokenizer metadata does not identify model weights.
- No speculative decoding or teacher-forced scoring was rerun for this housekeeping analysis.

## HIGHEST VERIFIED RUNG

The completed E2 analysis and these housekeeping sensitivities are supported by the full 1,000-prompt P1/P2 artifacts, reused P3 artifacts, source summary CSV, and the additional AME/common-prompt outputs in this directory. The historical P3 weight identity remains outside what the artifacts can verify.

## EVIDENCE GAPS

- The original P3 0.6B and 4B weight revisions were not recorded; exact historical model identity is unavailable locally.
- AME intervals condition on the observed covariate rows and use clustered coefficient covariance; they do not resample the covariate distribution.

## RECOMMENDED NEXT

Before H3, record immutable model revisions in future run metadata. H3 can proceed without a P3 rerun if the unresolved historical revision is carried as a reproducibility limitation.

## Adjusted marginal effects

| Pair | Model | Adjusted P(CROSS) | Adjusted P(SPLIT) | AME percentage points (95% CI) | p |
|---|---|---:|---:|---:|---:|
| P1 | M2 | 24.24% | 12.65% | 11.59 (8.68–14.50) | 5.63e-15 |
| P1 | M3 | 18.23% | 12.79% | 5.44 (3.53–7.36) | 2.58e-08 |
| P1 | M5 | 18.56% | 12.77% | 5.79 (3.76–7.82) | 2.24e-08 |
| P2 | M2 | 25.01% | 14.06% | 10.95 (7.60–14.31) | 1.62e-10 |
| P2 | M3 | 19.35% | 14.20% | 5.15 (2.85–7.45) | 1.17e-05 |
| P2 | M5 | 20.75% | 14.13% | 6.62 (4.51–8.72) | 6.98e-10 |
| P3 | M2 | 29.64% | 18.16% | 11.48 (8.70–14.26) | 5.66e-16 |
| P3 | M3 | 23.70% | 18.34% | 5.36 (3.55–7.18) | 6.77e-09 |
| P3 | M5 | 24.37% | 18.30% | 6.07 (3.97–8.17) | 1.54e-08 |

## Common-prompt sensitivity

The old pooled count of 999 is the union of the eligible prompt sets, not the intersection. The exact three-way common eligible count is 993. See `e2_prompt_overlap_audit.md` and `e2_common_prompt_sensitivity.md` for exact IDs and estimates.

## Revision recovery

Neither the P3 0.6B draft weight revision nor the P3 4B target weight revision can be recovered reliably from the accessible local artifacts. Current cache snapshots and the E1 tokenizer revision are not attributed to P3 weights.

## Verification

- Existing pair-specific summary estimates were not rewritten by the housekeeping script.
- P1/P2/P3 token tables and frequency cache were reused; morphology labels were not changed.
- Only CPU-side regressions and artifact audits were run. No model weights were loaded; no speculative decoding or teacher-forced scoring was rerun.
- Counterfactual predictions modify only `morph_class`; uncertainty is computed with prompt-clustered delta-method covariance.
- Pooled common-prompt models use the original E2 M3/M5 formulas and prompt-clustered SEs.
