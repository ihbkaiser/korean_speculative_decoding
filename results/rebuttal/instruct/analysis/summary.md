# T5: instruction-tuned Qwen pair

Model revisions: `Qwen/Qwen3-0.6B` @ `c1899de289a04d12100db370d81485cdf75e47ca`; `Qwen/Qwen3-4B` @ `1cfa9a7208912126459214e8b04321603b3df60c`.
The base comparison is Q2 on the same first 5,000 common documents.

## Sample quality and alignment gates

| Measure | Instruct | Base Q2 |
|---|---:|---:|
| Continuation length median (tokens) | 114.0000 | 128.0000 |
| Continuation length p10 (tokens) | 25.0000 | 128.0000 |
| EOS rate | 0.5966 | 0.0472 |
| Hangul share median | 0.7188 | 0.6750 |
| Hangul share p10 | 0.6071 | 0.4594 |
| ALIGNED token share | 0.6741 | 0.7535 |
| Repeated 4-gram rate | 0.0512 | 0.4279 |

Reasoning-block continuations: 0; second chat turn: 0; Hangul median gate: True.

## Core analysis on k=4-verified proposals

| Pair / subset | Rejection | DP positions | DP rejection share | Lowest 20% rejection share | AUC target | AUC draft | AUC boundary | Either-margin rejection share |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Instruct / all | 0.3808 | 0.1907 | 0.3782 | 0.4162 | 0.8446 | 0.8399 | 0.6233 | 0.7390 |
| Instruct / ≥64 tokens | 0.3828 | 0.1931 | 0.3803 | 0.4183 | 0.8437 | 0.8381 | 0.6234 | 0.7405 |
| Base Q2 / all | 0.1705 | 0.2252 | 0.7098 | 0.6571 | 0.8957 | 0.9103 | 0.5996 | 0.9282 |
| Base Q2 / ≥64 tokens | 0.1701 | 0.2248 | 0.7102 | 0.6574 | 0.8960 | 0.9105 | 0.5997 | 0.9283 |

## Policy replay (odd-index test documents)

Tokens per target call at the draft-token cost selected on even-index documents near fixed k=4.
Gains are relative to fixed k=4; intervals are paired document-bootstrap 95% intervals (500 resamples).

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
