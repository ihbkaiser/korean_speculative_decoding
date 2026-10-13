# RQ6 interim result: Qwen3-4B vs Qwen3-8B

## Scope

Two-model interim analysis; supports conclusions only about the Qwen3-4B/Qwen3-8B pair. No cross-family comparison is available.
This is a two-target, within-family result. It cannot determine whether low-margin decision points are shared across model families or generalize to other target architectures.

The analysis uses 191,900 identical Korean text prefixes at eojeol onsets from 9,595 FineWeb2 documents. Each model tokenized the same NFC character prefix independently. For each model, a decision point is one of the exact bottom 20% of next-token top-2 margins across the common positions.
Prefixes span 128–873 characters (median 257); per-model token counts and pinned revisions are recorded in `summary.json`.

## Main result

| Pair | Margin Spearman rho (95% document-bootstrap CI) | Bottom-20% Jaccard (95% document-bootstrap CI) | Shared positions | Fraction of all positions | Qwen4 DP positions shared with Qwen8 | Within-document null Jaccard mean [95% permutation interval] | Observed − null | Upper-tail permutation p |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Qwen/Qwen3-4B-Base vs Qwen/Qwen3-8B-Base | 0.493 [0.489, 0.497] | 0.204 [0.201, 0.207] | 12,986 | 6.77% | 33.8% | 0.124 [0.121, 0.126] | +0.080 | 0.001 |

The permutation null independently relocates each target’s selected positions within each document while preserving that document’s decision-point count. The reported p-value is an unadjusted upper-tail empirical value. The document bootstrap resamples documents, retaining each selected document’s positions.

## Threshold sensitivity

| Bottom fraction per model | Pairwise Jaccard | Positions shared | Shared fraction |
|---:|---:|---:|---:|
| 10% | 0.123 | 4,194 | 2.19% |
| 20% | 0.204 | 12,986 | 6.77% |
| 30% | 0.307 | 27,025 | 14.08% |

## Positions selected by both models

Among all common prefixes, 128,126 are selected by neither model, 50,788 by exactly one, and 12,986 by both. The permutation comparison for each count bucket is in `shared_target_buckets.csv`.

## Morphological comparison

Kiwi-derived boundary and within-morpheme offsets are matched within the same sampled eojeol (7,999 word pairs across 6,330 documents). Alignment exclusions: kiwi_span_not_exact_cover: 4535, no_internal_morpheme_boundary: 5152, no_within_morpheme_character_offset: 1504. Differences are descriptive associations.

| Outcome | Matched words | Boundary prevalence | Internal prevalence | Boundary − internal (95% document-bootstrap CI) |
|---|---:|---:|---:|---:|
| exactly_one_target | 7,999 | 0.243 | 0.203 | +0.039 [+0.027, +0.052] |
| shared_by_all_targets | 7,999 | 0.109 | 0.068 | +0.041 [+0.032, +0.050] |

## Interpretation and limitation

This interim result estimates consistency between two sizes in the Qwen3 family only. It does not support the planned cross-family comparison or the original multi-target outcome classification. Low-margin overlap is descriptive and does not establish intrinsic linguistic ambiguity or causality. The complete five-target RQ6 remains pending the other model runs.

### Proposed interim paragraph

On identical Korean text prefixes from 9,595 FineWeb2 documents (191,900 eojeol onsets), the bottom-20% decision-point sets for Qwen3-4B and Qwen3-8B had a Jaccard similarity of 0.204 (document-bootstrap 95% CI [0.201, 0.207]) and shared 12,986 positions (6.77% of all tested positions). Their margin ranks correlated at Spearman ρ = 0.493 (95% CI [0.489, 0.497]). The observed overlap was 0.204, compared with a within-document relocation null mean of 0.124 (95% permutation interval [0.121, 0.126], upper-tail p = 0.001). This interim comparison is limited to two Qwen3 targets and cannot establish cross-family generality.
