# E1: Token-frequency confounding analysis

Run: `20260926T184145Z_pilot1000`. The analysis reuses saved H2 outputs; speculative decoding was not rerun.

## Main conclusion

**A. Frequency does not explain H2.** CROSS remains higher than WITHIN_SPLIT after the nonlinear frequency spline and the supported frequency-stratified model.

- Original H2 M3 OR: **1.766** (95% CI 1.475–2.115; p=6.342e-10).
- Flexible-frequency M5 OR: **1.894** (95% CI 1.542–2.327; p=1.176e-09).
- CROSS coefficient change from M3 to M5: **12.3%**.
- M5 95% CI excludes OR=1: **yes**; frequency-stratified robustness CI excludes 1: **yes**.
- Conclusion under nonlinear frequency control: **remains positive and significant**.

| Model | CROSS − SPLIT coefficient | OR (95% CI) | p-value | N tokens | N prompts |
|---|---:|---:|---:|---:|---:|
| M3 | 0.5687 | 1.766 (1.475–2.115) | 6.342e-10 | 73,143 | 997 |
| M4 | 0.5594 | 1.75 (1.457–2.101) | 2.107e-09 | 73,143 | 997 |
| M5 | 0.6387 | 1.894 (1.542–2.327) | 1.176e-09 | 73,143 | 997 |
| M6 | NA | NA | NA | 73,143 | 997 |
| Frequency-stratified | 0.6732 | 1.961 (1.614–2.381) | 1.11e-11 | 44,158 | 992 |

## Corpus and coverage

- Corpus: Local Korean Wikipedia prompt cache (wikimedia/wikipedia, config 20231101.ko, train; first 1,000 non-empty articles).
- Texts: **1,000**; tokenizer tokens: **3,425,737**.
- Tokenizer: `Qwen/Qwen3-4B-Base` / `Qwen2Tokenizer`; resolved revision: `906bfd4b4dc7f14ee4320094d8b41684abff8539`.
- Tokenizer backend fingerprint, verified against H2 compatibility metadata: `41e00eccf531cffc2e562d38bdd879d41e5044ea279af5b73c6a32aabcc8fe04`.
- Primary analyzed H2 tokens unseen in the cache: **86/73,143 (0.118%)**.
- Cache: `e1_token_frequency_cache.parquet` and metadata `e1_token_frequency_cache_metadata.json`; corpus SHA256 `6f5a90e25f62ba84f49cfd84285aab17b6f91c6c3c6a06ebb4c129059d72563e`.

## Descriptive frequency check

| Morphology class | N | Median token count | Mean log1p count | Unseen | Rejection rate |
|---|---:|---:|---:|---:|---:|
| EXACT | 39,546 | 18,964 | 9.363 | 0.16% | 17.87% |
| WITHIN_SPLIT | 69,343 | 6,315 | 8.708 | 0.12% | 18.35% |
| CROSS_MORPHEME | 3,800 | 3,268 | 7.616 | 0.05% | 25.24% |

Raw CROSS − WITHIN_SPLIT mean log1p token count: **-1.0920**; median count ratio: **0.517**; unseen-rate difference: **-0.069 percentage points**.
Adjusted rarity contrast (structural controls and fragmentation FE): coefficient **1.0706**, 95% CI [0.6661, 1.4752], p=2.141e-07.

## Frequency-stratified robustness and interaction

- The frequency-stratified model retains bins with both classes (D1, D2, D3, D4, D5, D6; 44,158 tokens) and adds bin fixed effects to the M3 controls. Its OR is 1.961 (95% CI 1.614–2.381; p=1.11e-11).
- Exploratory class × log-frequency interaction: coefficient -0.3755, 95% CI [-0.4741, -0.2769], p=8.582e-14.
- At log-count p10/p50/p90, estimated conditional CROSS ORs are 2.572 [2.044, 3.236] / 1.357 [1.068, 1.725] / 0.644 [0.4466, 0.9286]. See the prediction plot for population-averaged predictions over 6,000 sampled observed covariate rows.
- Interaction indicates the CROSS excess is concentrated toward the rarer end: the conditional CROSS OR at p10 is 2.57 (95% CI 2.04–3.24), at p50 is 1.36 (1.07–1.72), and at p90 is 0.64 (0.45–0.93). Thus CROSS is not worse in the most frequent range in this exploratory interaction model. These results do not change the pooled M5 estimand and do not establish a causal mechanism.

## Scope

Frequency is estimated from the locally cached first 1,000 non-empty Korean Wikipedia articles, not a broad external corpus. Analyses are associations on this selected generated-continuation sample and do not imply causality.

## Outputs

- `e1_frequency_table.parquet` — H2 token rows plus token and eojeol frequency features.
- `e1_frequency_class_summary.csv`, `e1_frequency_regression.txt`.
- `e1_frequency_distribution_by_morph_class.png`, `e1_predicted_rejection_by_frequency.png`.
