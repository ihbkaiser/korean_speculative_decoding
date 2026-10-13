# RQ6 interim result: Qwen3-4B vs Qwen3-8B

## Scope

Two-model interim analysis; supports conclusions only about the Qwen3-4B/Qwen3-8B pair. No cross-family comparison is available.
This is a two-target, within-family result. It cannot determine whether low-margin decision points are shared across model families or generalize to other target architectures.

The analysis uses 38,340 identical Korean text prefixes at eojeol onsets from 1,917 FineWeb2 documents. Each model tokenized the same NFC character prefix independently. For each model, a decision point is one of the exact bottom 20% of next-token top-2 margins across the common positions.
Prefixes span 128–490 characters (median 257); per-model token counts and pinned revisions are recorded in `summary.json`.

## Main result

| Pair | Margin Spearman rho (95% document-bootstrap CI) | Bottom-20% Jaccard (95% document-bootstrap CI) | Shared positions | Fraction of all positions | Qwen4 DP positions shared with Qwen8 | Within-document null Jaccard mean [95% permutation interval] | Observed − null | Upper-tail permutation p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3-4B-Base vs Qwen/Qwen3-8B-Base | 0.488 [0.479, 0.497] | 0.207 [0.201, 0.214] | 2,625 | 6.85% | 34.2% | 0.122 [0.117, 0.127] | +0.084 | 0.001 |

The permutation null independently relocates each target’s selected positions within each document while preserving that document’s decision-point count. The reported p-value is an unadjusted upper-tail empirical value. The document bootstrap resamples documents, retaining each selected document’s positions.

## Threshold sensitivity

| Bottom fraction per model | Pairwise Jaccard | Positions shared | Shared fraction |
|---:|---:|---:|---:|
| 10% | 0.122 | 831 | 2.17% |
| 20% | 0.207 | 2,625 | 6.85% |
| 30% | 0.308 | 5,413 | 14.12% |

## Positions selected by both models

Among all common prefixes, 25,629 are selected by neither model, 10,086 by exactly one, and 2,625 by both. The permutation comparison for each count bucket is in `shared_target_buckets.csv`.

## Morphological comparison

Kiwi-derived boundary and within-morpheme offsets are matched within the same sampled eojeol (1,586 word pairs across 1,265 documents). Alignment exclusions: kiwi_span_not_exact_cover: 909, no_internal_morpheme_boundary: 1025, no_within_morpheme_character_offset: 314. Differences are descriptive associations.

| Outcome | Matched words | Boundary prevalence | Internal prevalence | Boundary − internal (95% document-bootstrap CI) |
|---|---:|---:|---:|---:|
| exactly_one_target | 1,586 | 0.254 | 0.205 | +0.049 [+0.020, +0.081] |
| shared_by_all_targets | 1,586 | 0.112 | 0.059 | +0.053 [+0.032, +0.075] |

## Interpretation and limitation

This interim result estimates consistency between two sizes in the Qwen3 family only. It does not support the planned cross-family comparison or the original multi-target outcome classification. Low-margin overlap is descriptive and does not establish intrinsic linguistic ambiguity or causality. The complete five-target RQ6 remains pending the other model runs.

### Proposed interim paragraph

On identical Korean text prefixes from 1,917 FineWeb2 documents (38,340 eojeol onsets), the bottom-20% decision-point sets for Qwen3-4B and Qwen3-8B had a Jaccard similarity of 0.207 (document-bootstrap 95% CI [0.201, 0.214]) and shared 2,625 positions (6.85% of all tested positions). Their margin ranks correlated at Spearman ρ = 0.488 (95% CI [0.479, 0.497]). The observed overlap was 0.207, compared with a within-document relocation null mean of 0.122 (95% permutation interval [0.117, 0.127], upper-tail p = 0.001). This interim comparison is limited to two Qwen3 targets and cannot establish cross-family generality.
