# RQ6: Cross-target decision-point consistency

## Design and scope

The analysis evaluated 38,340 shared eojeol-onset prefixes from 1,917 of 2,000 considered FineWeb2 Korean documents. 83 documents were excluded because they had too few eligible eojeol onsets. Each target received the identical NFC text prefix at each document/character offset, then tokenized it independently. The primary decision-point definition is the exact bottom 20% of margins within each target on the common position set.

Prefixes contain 128–490 characters after snapping the nominal 256-character window back to an eojeol boundary (median 257). Per-model token-length summaries are in `summary.json`. Raw margins and the absolute margin <1 rule are secondary because next-token vocabularies and margin scales differ. This is a diagnostic comparison, not a causal test.

## Pairwise results

| Target pair | Family relation | Spearman rho (95% doc-bootstrap CI) | Bottom-20% Jaccard (95% CI) | Observed / independence overlap (%) | P(DP_B &#124; DP_A) | Overlap enrichment vs independence | Null Jaccard mean | Observed − null | Upper-tail p |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| qwen4 vs qwen8 | within Qwen3 | 0.488 [0.479, 0.497] | 0.207 [0.201, 0.214] | 6.85 / 4.00 | 0.342 | 1.712 | 0.122 | +0.084 | 0.001 |
| qwen4 vs qwen14 | within Qwen3 | 0.424 [0.414, 0.434] | 0.187 [0.180, 0.193] | 6.30 / 4.00 | 0.315 | 1.574 | 0.121 | +0.065 | 0.001 |
| qwen4 vs gemma12 | cross-family | 0.029 [0.019, 0.040] | 0.116 [0.110, 0.120] | 4.15 / 4.00 | 0.208 | 1.039 | 0.112 | +0.004 | 0.081 |
| qwen4 vs ministral8 | cross-family | 0.373 [0.362, 0.383] | 0.175 [0.168, 0.181] | 5.96 / 4.00 | 0.298 | 1.489 | 0.119 | +0.056 | 0.001 |
| qwen8 vs qwen14 | within Qwen3 | 0.462 [0.452, 0.472] | 0.200 [0.193, 0.207] | 6.66 / 4.00 | 0.333 | 1.664 | 0.121 | +0.078 | 0.001 |
| qwen8 vs gemma12 | cross-family | 0.027 [0.017, 0.037] | 0.111 [0.107, 0.117] | 4.00 / 4.00 | 0.200 | 1.000 | 0.110 | +0.001 | 0.385 |
| qwen8 vs ministral8 | cross-family | 0.375 [0.364, 0.385] | 0.178 [0.171, 0.183] | 6.04 / 4.00 | 0.302 | 1.509 | 0.119 | +0.058 | 0.001 |
| qwen14 vs gemma12 | cross-family | 0.022 [0.012, 0.032] | 0.116 [0.111, 0.121] | 4.16 / 4.00 | 0.208 | 1.040 | 0.112 | +0.004 | 0.045 |
| qwen14 vs ministral8 | cross-family | 0.380 [0.369, 0.391] | 0.175 [0.170, 0.183] | 5.97 / 4.00 | 0.298 | 1.492 | 0.120 | +0.055 | 0.001 |
| gemma12 vs ministral8 | cross-family | 0.027 [0.016, 0.037] | 0.113 [0.109, 0.119] | 4.07 / 4.00 | 0.203 | 1.017 | 0.112 | +0.001 | 0.312 |

Empirical permutation p-values are upper-tail, unadjusted across pairs. The overlap column gives observed and independence-reference percentages; P(DP_B | DP_A) is directional in the displayed pair order, and enrichment is observed overlap divided by the independence expectation. The within-document permutation is the main null reference; independence is only a descriptive baseline.

### Threshold sensitivity

| Bottom fraction | Mean pairwise Jaccard | Mean within-Qwen3 Jaccard | Mean cross-family Jaccard |
|---:|---:|---:|---:|
| 10% | 0.084 | 0.113 | 0.071 |
| 20% | 0.158 | 0.198 | 0.141 |
| 30% | 0.242 | 0.295 | 0.219 |

## Family and shared-position summaries

Mean Qwen3 within-family Jaccard is 0.198 (within-document permutation mean 0.122); mean cross-family Jaccard is 0.141 (permutation mean 0.115). The within-minus-cross document-bootstrap 95% CI is [0.052, 0.062]. Only Qwen3 has multiple targets in this set, so there is no within-Gemma or within-Ministral estimate.

The fraction marked by at least one target from each of the three families (Qwen3, Gemma 3, Ministral 3) is 0.024. Across positions, 15,194 are selected by zero targets, 12,516 by exactly one, 6,984 by exactly two, and 3,646 by three or more. `shared_target_buckets.csv` compares these groups with the permutation distribution; `shared_target_counts.csv` gives exact counts zero through five.

## Morphological comparison

Kiwi-derived morpheme-boundary and within-morpheme character offsets were matched within the same sampled eojeol (1,586 word pairs from 1,265 documents). Automatic alignment exclusions: kiwi_span_not_exact_cover: 909, no_internal_morpheme_boundary: 1025, no_within_morpheme_character_offset: 314. The comparison is associational.

| Outcome | Matched words | Boundary prevalence | Within-morpheme prevalence | Difference (boundary − internal) | Document-bootstrap 95% CI |
|---|---:|---:|---:|---:|---:|
| exactly_one_target | 1586 | 0.267 | 0.299 | -0.034 | [-0.066, -0.002] |
| shared_by_three_or_more | 1586 | 0.144 | 0.090 | +0.059 | [+0.034, +0.082] |
| cross_family_shared | 1586 | 0.030 | 0.026 | +0.005 | [-0.006, +0.015] |

## Interpretation for KORSpec

**Observed pattern: Outcome B.** The pattern is closest to Outcome B: Qwen3 within-family overlap is above its permutation mean, and the document-bootstrap interval places it above cross-family overlap. This supports family- or target-dependent decision-point locations in this model set.

Outcome A would support model-robust textual hotspots without establishing causal or intrinsic linguistic ambiguity. Outcome B indicates family dependence. Outcome C limits the existing RQ4 interpretation to draft failures conditional on a fixed target. If estimates do not meet a clean pattern, the report labels them mixed rather than forcing an outcome.

Figures A–D show normalized margin ranks, pairwise overlap, shared-target counts against the within-document permutation baseline, and the matched morphological comparison. Full CSVs, model revisions, per-model token lengths, exclusions, and output hashes are saved under this directory.

### Proposed RQ6 paragraph

On identical Korean text prefixes from 1,917 FineWeb2 documents (38,340 eojeol onsets), bottom-20% decision-point Jaccard averaged 0.198 within Qwen3 and 0.141 across families, against corresponding within-document permutation means of 0.122 and 0.115. The within-Qwen minus cross-family mean difference was +0.057 (document-bootstrap 95% CI [+0.052, +0.062]); selected pairwise overlaps included Qwen3-4B/8B 0.207, Qwen3-8B/Gemma3-12B 0.111, and Qwen3-8B/Ministral3-8B 0.178. The observed pattern is closest to Outcome B. These results describe normalized low-margin locations in the evaluated prefixes; they do not establish intrinsic linguistic ambiguity, and the paper's RQ4 remains conditional on a fixed target.
