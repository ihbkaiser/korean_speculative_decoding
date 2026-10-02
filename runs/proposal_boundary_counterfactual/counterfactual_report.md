# Paired N→P proposal counterfactuals

This offline intervention replays actual saved greedy speculative-decoding prefixes using the pinned P1/P2 draft and target models. At each N→P CROSS prefix, it compares the natural draft argmax with the highest-ranked valid alternative that does not cross N→P. It uses a unique nearest WITHIN_SPLIT proposal as a generic perturbation placebo; same-prompt matches are used when available, and cross-prompt fallback is used otherwise. Target and draft argmaxes are recomputed; any prefix where either fails to reproduce the saved event is excluded from causal contrasts.

Placebo matching: 1,322/1,375 contexts are matched within the same prompt; no placebo prefix is reused.

Primary estimand: paired change in immediate target-argmax agreement under the N→P-avoidance proposal policy. Specificity check: compare that change with the generic WITHIN_SPLIT placebo. This is a policy-level causal estimand; it does not identify an effect of morphology independent of candidate-token identity. It is not an end-to-end speed or quality experiment.

## Model replay audit

```json
{
  "NP_CROSS": {
    "n": 1375,
    "draft_replay_match": 1370,
    "target_replay_match": 1373,
    "saved_reject_label_match": 1368,
    "valid_counterfactual": 1368,
    "no_valid_alternative": 0,
    "model_replay_mismatch": 7
  },
  "WITHIN_SPLIT_PLACEBO": {
    "n": 1375,
    "draft_replay_match": 1374,
    "target_replay_match": 1370,
    "saved_reject_label_match": 1369,
    "valid_counterfactual": 1369,
    "no_valid_alternative": 0,
    "model_replay_mismatch": 6
  }
}
```

## Paired counterfactual estimates

| workload | pair | condition | metric | n_events | n_prompts | estimate | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| FLORES | P1 | NP_CROSS | baseline_accept | 355.0000 | 265.0000 | 0.6254 | 0.5634 | 0.6848 |
| FLORES | P1 | NP_CROSS | alternative_accept | 355.0000 | 265.0000 | 0.1099 | 0.0769 | 0.1459 |
| FLORES | P1 | NP_CROSS | delta_accept | 355.0000 | 265.0000 | -0.5155 | -0.5994 | -0.4310 |
| FLORES | P1 | WITHIN_SPLIT_PLACEBO | baseline_accept | 356.0000 | 264.0000 | 0.6994 | 0.6412 | 0.7527 |
| FLORES | P1 | WITHIN_SPLIT_PLACEBO | alternative_accept | 356.0000 | 264.0000 | 0.1124 | 0.0801 | 0.1490 |
| FLORES | P1 | WITHIN_SPLIT_PLACEBO | delta_accept | 356.0000 | 264.0000 | -0.5871 | -0.6630 | -0.5048 |
| FLORES | P2 | NP_CROSS | baseline_accept | 378.0000 | 291.0000 | 0.6164 | 0.5668 | 0.6658 |
| FLORES | P2 | NP_CROSS | alternative_accept | 378.0000 | 291.0000 | 0.1984 | 0.1586 | 0.2397 |
| FLORES | P2 | NP_CROSS | delta_accept | 378.0000 | 291.0000 | -0.4180 | -0.4973 | -0.3342 |
| FLORES | P2 | WITHIN_SPLIT_PLACEBO | baseline_accept | 378.0000 | 288.0000 | 0.7989 | 0.7594 | 0.8392 |
| FLORES | P2 | WITHIN_SPLIT_PLACEBO | alternative_accept | 378.0000 | 288.0000 | 0.0873 | 0.0593 | 0.1169 |
| FLORES | P2 | WITHIN_SPLIT_PLACEBO | delta_accept | 378.0000 | 288.0000 | -0.7116 | -0.7743 | -0.6474 |
| WIKIPEDIA | P1 | NP_CROSS | baseline_accept | 308.0000 | 137.0000 | 0.8117 | 0.7518 | 0.8629 |
| WIKIPEDIA | P1 | NP_CROSS | alternative_accept | 308.0000 | 137.0000 | 0.1266 | 0.0844 | 0.1765 |
| WIKIPEDIA | P1 | NP_CROSS | delta_accept | 308.0000 | 137.0000 | -0.6851 | -0.7781 | -0.5786 |
| WIKIPEDIA | P1 | WITHIN_SPLIT_PLACEBO | baseline_accept | 308.0000 | 137.0000 | 0.9123 | 0.8763 | 0.9425 |
| WIKIPEDIA | P1 | WITHIN_SPLIT_PLACEBO | alternative_accept | 308.0000 | 137.0000 | 0.0487 | 0.0264 | 0.0723 |
| WIKIPEDIA | P1 | WITHIN_SPLIT_PLACEBO | delta_accept | 308.0000 | 137.0000 | -0.8636 | -0.9141 | -0.8112 |
| WIKIPEDIA | P2 | NP_CROSS | baseline_accept | 327.0000 | 159.0000 | 0.7920 | 0.7292 | 0.8453 |
| WIKIPEDIA | P2 | NP_CROSS | alternative_accept | 327.0000 | 159.0000 | 0.1193 | 0.0784 | 0.1678 |
| WIKIPEDIA | P2 | NP_CROSS | delta_accept | 327.0000 | 159.0000 | -0.6728 | -0.7642 | -0.5672 |
| WIKIPEDIA | P2 | WITHIN_SPLIT_PLACEBO | baseline_accept | 327.0000 | 166.0000 | 0.8869 | 0.8501 | 0.9202 |
| WIKIPEDIA | P2 | WITHIN_SPLIT_PLACEBO | alternative_accept | 327.0000 | 166.0000 | 0.0581 | 0.0330 | 0.0870 |
| WIKIPEDIA | P2 | WITHIN_SPLIT_PLACEBO | delta_accept | 327.0000 | 166.0000 | -0.8287 | -0.8835 | -0.7710 |
| FLORES | P1+P2 | NP_MINUS_PLACEBO | delta_difference | 529.0000 | 407.0000 | 0.2124 | 0.1276 | 0.2938 |
| WIKIPEDIA | P1+P2 | NP_MINUS_PLACEBO | delta_difference | 292.0000 | 236.0000 | 0.2993 | 0.2076 | 0.3975 |
| FLORES | P1+P2 | NP_CROSS | delta_accept | 733.0000 | 422.0000 | -0.4652 | -0.5265 | -0.4017 |
| FLORES | P1+P2 | WITHIN_SPLIT_PLACEBO | delta_accept | 734.0000 | 426.0000 | -0.6512 | -0.6996 | -0.5994 |
| WIKIPEDIA | P1+P2 | NP_CROSS | delta_accept | 635.0000 | 239.0000 | -0.6787 | -0.7417 | -0.6071 |
| WIKIPEDIA | P1+P2 | WITHIN_SPLIT_PLACEBO | delta_accept | 635.0000 | 246.0000 | -0.8457 | -0.8813 | -0.8087 |

## Decision rule

The prespecified positive criterion is a positive pooled P1+P2 NP_CROSS delta-accept estimate with a prompt-clustered 95% CI above zero in both workloads and positive point estimates in both pairs, plus a positive pooled NP-minus-placebo specificity estimate in both workloads. Otherwise the result is negative or inconclusive for this intervention.

## Interpretation

The estimated change in immediate acceptance is negative for the N→P-avoidance policy in both datasets and both pairs: suppressing the natural N→P proposal and taking the next valid non-N→P draft candidate lowers agreement with the target. The specificity contrast is positive because this replacement is less harmful than the generic WITHIN_SPLIT top-candidate replacement, but both policy effects are negative. This does not support the claim that crossing itself causes rejection; it shows that candidate identity and its rank under the draft matter, and that avoiding N→P candidates is not a reliable way to reduce rejection.

Inputs are the existing validated full-block proposal projections; model revisions are pinned in `src/e2_model_pairs.py`. See the CSV and JSON outputs for every row and replay mismatch.
