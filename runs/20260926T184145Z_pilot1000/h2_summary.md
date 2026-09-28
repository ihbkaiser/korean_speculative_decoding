# H2 analysis: cross-morpheme vs within-morpheme split tokens

Run: `20260926T184145Z_pilot1000`. This analysis reuses the saved 1,000-prompt run; speculative decoding was not rerun.

## Data and alignment

- Generated token rows reconstructed: **127,245** across **1,000 prompts**.
- SD-valid generated tokens: **127,245**; primary contrast tokens: **73,143** across **997 prompts**.
- Visible token spans with exact retokenized ID alignment: **127,097/127,232 (99.89%)**; **13** special tokens have no visible span.
- Ambiguous classification exclusions (`CROSS_EOJEOL` or `KIWI_COMPLEX`): **9,660/127,245 (7.59%)** of generated tokens.
- `CROSS_EOJEOL`: **0**; `KIWI_COMPLEX`: **9,660**.
- Ambiguity reasons: `no_nonwhitespace_character_span` 6,065; `overlapping_kiwi_morpheme_spans` 3,373; `special_token_without_visible_character_span` 13; `token_id_not_retokenized_exactly` 135; `token_span_crosses_prompt_boundary` 74.
- Visible generated IDs not mapped by exact re-tokenization and retained as `KIWI_COMPLEX`: **135**. Contextual round-trip was exact for **926/1,000 prompts**.
- Exact fragmentation levels are retained as `2, 3, 4, 5, 6, 7, 8+`; no `5+` pooling was used.

## Primary result

All regressions use two-sided Wald tests with standard errors clustered by `prompt_id`. The class coefficient is parameterized as CROSS relative to the WITHIN_SPLIT reference (the SPLIT coefficient is 0 by reference coding); `CROSS - SPLIT` is that same log-odds contrast.

| Model | CROSS coefficient (SPLIT ref.) | CROSS − SPLIT | OR (95% CI) | p | N tokens | N prompts |
|---|---:|---:|---:|---:|---:|---:|
| M1 | 0.355 (SPLIT=0) | 0.355 | 1.43 (1.3, 1.56) | 1.49e-14 | 73,143 | 997 |
| M2 | 0.743 (SPLIT=0) | 0.743 | 2.1 (1.79, 2.47) | 1.04e-19 | 73,143 | 997 |
| M3 | 0.569 (SPLIT=0) | 0.569 | 1.77 (1.47, 2.11) | 6.34e-10 | 73,143 | 997 |

| Model | CROSS − SPLIT coefficient | 95% CI | p | N tokens | N prompts |
|---|---:|---:|---:|---:|---:|
| Draft entropy | 0.389 | [0.278, 0.499] | 5.19e-12 | 73,143 | 997 |
| Draft-target entropy gap | -0.0585 | [-0.121, 0.00369] | 0.0652 | 73,143 | 997 |

## Exact-fragmentation rates

Rates below include SD-valid tokens only. Confidence intervals are percentile intervals from 2,000 bootstrap resamples of prompts with replacement.

| Fragmentation | EXACT | WITHIN_SPLIT | CROSS_MORPHEME |
|---|---:|---:|---:|
| 2 | 23.2% [22.1%, 24.4%], n=10,311 | 21.4% [19.3%, 23.6%], n=3,814 | 21.8% [19.2%, 24.8%], n=1,109 |
| 3 | 16.9% [16.1%, 17.8%], n=11,364 | 20.2% [19.2%, 21.3%], n=15,789 | 24.2% [21.4%, 27.5%], n=994 |
| 4 | 16.9% [15.8%, 18.0%], n=7,874 | 18.7% [17.7%, 19.7%], n=17,336 | 27.1% [24.1%, 30.1%], n=950 |
| 5 | 11.0% [10.0%, 12.0%], n=5,820 | 15.9% [14.9%, 16.8%], n=18,913 | 24.9% [19.9%, 30.3%], n=446 |
| 6 | 18.4% [16.5%, 20.3%], n=2,376 | 17.9% [16.5%, 19.3%], n=7,189 | 33.5% [27.5%, 40.6%], n=209 |
| 7 | 19.1% [15.3%, 23.4%], n=843 | 18.2% [15.7%, 21.1%], n=2,963 | 46.3% [29.4%, 67.9%], n=41 |
| 8+ | 19.6% [15.9%, 23.9%], n=958 | 19.4% [16.1%, 22.6%], n=3,339 | 37.3% [23.5%, 53.8%], n=51 |

## Result classification

**A. H2 supported in M2 and M3.** The positive CROSS-vs-SPLIT contrast remains statistically distinguishable from zero after structural and entropy adjustment.

## Secondary checks and limits

- Teacher-forced draft/target disagreement is included only as a sanity check; on 127,245 SD-valid rows with both outcomes, its token-level agreement with the SD rejection label is 99.88% (not an independent replication).
- Primary token-level rejection rate across CROSS and WITHIN_SPLIT rows: 18.71%.
- The entropy regressions estimate conditional associations; they do not establish a mechanism or causal mediation.
- The run uses the first 1,000 non-empty Korean Wikipedia articles in dataset order. Results describe this generated continuation sample and should not be read as causal or corpus-wide claims.

## Files

- `h2_token_table.parquet` — one row per saved generated token, with Kiwi surface/POS/span overlaps and SD/teacher outcomes.
- `h2_class_counts.csv`, `h2_fragmentation_summary.csv`, `h2_regression.txt`.
- `h2_rejection_by_morphology_fragmentation.png`, `h2_forest_cross_vs_split.png`.
