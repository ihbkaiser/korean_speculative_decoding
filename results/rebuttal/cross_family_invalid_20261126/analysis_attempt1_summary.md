# T6-b: cross-family judge on generated text

All statistics use proposals verified under k=4 and the first 5,000 common documents.
Ratios are morpheme-onset/internal, word-onset/internal, and fused/split.
Matched rows use judge or target margin deciles, plus judge absolute bins from `expD_lib.EDGES`.
Intervals are 95% document-bootstrap intervals (500 resamples; seed 20261125).

| Pair | Shared-offset coverage | Rejection, all / joined | AUC judge | AUC target | AUC boundary | AUC same-family judge |
|---|---:|---:|---:|---:|---:|---:|
| Q2 | 0.805 | 0.170 / 0.187 | 0.503 | 0.890 | 0.604 | 0.892 |
| Q5 | 0.802 | 0.193 / 0.212 | 0.502 | 0.874 | 0.605 | 0.879 |
| M1 | 0.968 | 0.121 / 0.124 | 0.526 | 0.913 | 0.589 | — |

## Position-type ratios

### Q2 (qwen4 judged by ministral8)

| Contrast | Raw [95% CI] | Judge decile [95% CI] | Judge absolute bin [95% CI] | Target decile [95% CI] |
|---|---:|---:|---:|---:|
| morph/int | 2.119 [2.050, 2.200] | 2.119 [2.040, 2.200] | 2.120 [2.050, 2.190] | 0.994 [0.970, 1.020] |
| word/int | 3.189 [3.090, 3.290] | 3.190 [3.100, 3.300] | 3.190 [3.090, 3.310] | 1.219 [1.190, 1.250] |
| fused/split | 1.421 [1.370, 1.470] | 1.422 [1.370, 1.470] | 1.422 [1.370, 1.470] | 0.947 [0.920, 0.970] |

### Q5 (qwen8 judged by ministral8)

| Contrast | Raw [95% CI] | Judge decile [95% CI] | Judge absolute bin [95% CI] | Target decile [95% CI] |
|---|---:|---:|---:|---:|
| morph/int | 2.017 [1.960, 2.090] | 2.017 [1.960, 2.080] | 2.017 [1.950, 2.090] | 0.972 [0.950, 0.990] |
| word/int | 3.060 [2.970, 3.150] | 3.060 [2.970, 3.160] | 3.060 [2.960, 3.170] | 1.205 [1.180, 1.230] |
| fused/split | 1.401 [1.360, 1.450] | 1.402 [1.360, 1.450] | 1.401 [1.360, 1.440] | 0.973 [0.950, 1.000] |

### M1 (ministral8 judged by qwen8)

| Contrast | Raw [95% CI] | Judge decile [95% CI] | Judge absolute bin [95% CI] | Target decile [95% CI] |
|---|---:|---:|---:|---:|
| morph/int | 1.954 [1.880, 2.030] | 1.933 [1.860, 2.000] | 1.932 [1.860, 2.000] | 0.967 [0.940, 0.990] |
| word/int | 3.029 [2.930, 3.150] | 2.990 [2.880, 3.100] | 2.987 [2.870, 3.100] | 1.195 [1.170, 1.220] |
| fused/split | 1.526 [1.480, 1.580] | 1.522 [1.470, 1.570] | 1.517 [1.470, 1.570] | 0.947 [0.930, 0.970] |
