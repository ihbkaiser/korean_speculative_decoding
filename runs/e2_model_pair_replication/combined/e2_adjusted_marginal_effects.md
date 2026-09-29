# Adjusted marginal rejection differences

Each estimate averages two counterfactual predictions over the eligible rows for that pair and model, holding every observed covariate fixed and changing only `morph_class`. The difference is CROSS_MORPHEME minus WITHIN_SPLIT.

Intervals and two-sided p-values use a delta method with the fitted model's prompt-clustered sandwich covariance. These are adjusted marginal associations conditional on the observed covariate distribution, not causal effects. No multiplicity correction is applied.

| Pair | Model | N tokens / prompts | Adjusted P(CROSS) | Adjusted P(SPLIT) | Difference (percentage points, 95% CI) | p |
|---|---|---:|---:|---:|---:|---:|
| 0.6B → 1.7B | M2 | 73,001 / 995 | 24.24% (21.36%–27.12%) | 12.65% (12.17%–13.12%) | 11.59 (8.68–14.50) | 5.63e-15 |
| 0.6B → 1.7B | M3 | 73,001 / 995 | 18.23% (16.39%–20.07%) | 12.79% (12.45%–13.12%) | 5.44 (3.53–7.36) | 2.58e-08 |
| 0.6B → 1.7B | M5 | 73,001 / 995 | 18.56% (16.61%–20.52%) | 12.77% (12.44%–13.10%) | 5.79 (3.76–7.82) | 2.24e-08 |
| 1.7B → 4B | M2 | 73,257 / 997 | 25.01% (21.71%–28.32%) | 14.06% (13.53%–14.59%) | 10.95 (7.60–14.31) | 1.62e-10 |
| 1.7B → 4B | M3 | 73,257 / 997 | 19.35% (17.17%–21.52%) | 14.20% (13.83%–14.57%) | 5.15 (2.85–7.45) | 1.17e-05 |
| 1.7B → 4B | M5 | 73,257 / 997 | 20.75% (18.72%–22.77%) | 14.13% (13.78%–14.48%) | 6.62 (4.51–8.72) | 6.98e-10 |
| 0.6B → 4B | M2 | 73,143 / 997 | 29.64% (26.88%–32.40%) | 18.16% (17.50%–18.83%) | 11.48 (8.70–14.26) | 5.66e-16 |
| 0.6B → 4B | M3 | 73,143 / 997 | 23.70% (21.97%–25.44%) | 18.34% (17.95%–18.73%) | 5.36 (3.55–7.18) | 6.77e-09 |
| 0.6B → 4B | M5 | 73,143 / 997 | 24.37% (22.34%–26.40%) | 18.30% (17.91%–18.69%) | 6.07 (3.97–8.17) | 1.54e-08 |

The fitted odds-ratio term from each re-estimated model is included in the CSV for parity checks against the original M2/M3/M5 pair estimates.
