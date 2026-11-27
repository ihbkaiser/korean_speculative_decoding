# T6-b: cross-family judge on generated text

All statistics use proposals verified under k=4 and the first 5,000 frozen documents.
Ratios are morpheme-onset/internal, word-onset/internal, and fused/split.
Matched rows use judge or target margin deciles, plus judge absolute bins from `expD_lib.EDGES`.
Intervals are 95% document-bootstrap intervals (500 resamples; seed 20261125).

| Scope | Pair | Documents | Shared-offset positions / coverage | Rejection, all / joined | AUC judge | AUC target | AUC boundary | AUC same-family judge | Spearman target/judge |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| preregistered | Q2 | 5000 | 430,489/535,028 (0.805) | 0.170 / 0.187 | 0.850 | 0.890 | 0.604 | 0.892 | 0.821 |
| preregistered | Q5 | 5000 | 429,972/536,089 (0.802) | 0.193 / 0.212 | 0.847 | 0.874 | 0.605 | 0.879 | 0.808 |
| preregistered | M1 | 5000 | 512,461/529,609 (0.968) | 0.121 / 0.124 | 0.857 | 0.913 | 0.589 | — | 0.839 |
| secondary, not preregistered | Q4 | 5000 | 424,975/528,052 (0.805) | 0.133 / 0.146 | 0.854 | 0.905 | 0.602 | 0.910 | 0.823 |
| secondary, not preregistered | Q3 | 5000 | 418,629/521,752 (0.802) | 0.118 / 0.130 | 0.849 | 0.908 | 0.596 | 0.914 | 0.810 |
| secondary, not preregistered | Q6 | 5000 | 423,855/528,413 (0.802) | 0.150 / 0.166 | 0.851 | 0.894 | 0.603 | 0.892 | 0.810 |

## Offset coverage by position type

| Scope | Pair | Internal | Morpheme onset | Word onset |
|---|---|---:|---:|---:|
| preregistered | Q2 | 0.803 | 0.780 | 0.833 |
| preregistered | Q5 | 0.807 | 0.781 | 0.831 |
| preregistered | M1 | 0.930 | 0.988 | 0.988 |
| secondary, not preregistered | Q4 | 0.805 | 0.780 | 0.833 |
| secondary, not preregistered | Q3 | 0.808 | 0.781 | 0.831 |
| secondary, not preregistered | Q6 | 0.807 | 0.780 | 0.831 |

## Position-type ratios

### Q2 (qwen4 judged by ministral8; n=5000)

| Contrast | Raw [95% CI] | Cross-family decile [95% CI] | Cross-family absolute bin [95% CI] | Same-family decile [95% CI] | Target decile [95% CI] |
|---|---:|---:|---:|---:|---:|
| morph/int | 2.119 [2.050, 2.200] | 1.218 [1.190, 1.250] | 1.209 [1.180, 1.240] | 0.985 [0.960, 1.010] | 0.994 [0.970, 1.020] |
| word/int | 3.189 [3.090, 3.290] | 1.538 [1.500, 1.580] | 1.522 [1.490, 1.560] | 1.206 [1.180, 1.230] | 1.219 [1.190, 1.250] |
| fused/split | 1.421 [1.370, 1.470] | 1.077 [1.050, 1.110] | 1.079 [1.050, 1.110] | 0.943 [0.920, 0.970] | 0.947 [0.920, 0.970] |

### Q5 (qwen8 judged by ministral8; n=5000)

| Contrast | Raw [95% CI] | Cross-family decile [95% CI] | Cross-family absolute bin [95% CI] | Same-family decile [95% CI] | Target decile [95% CI] |
|---|---:|---:|---:|---:|---:|
| morph/int | 2.017 [1.960, 2.090] | 1.182 [1.160, 1.210] | 1.172 [1.140, 1.200] | 0.963 [0.940, 0.980] | 0.972 [0.950, 0.990] |
| word/int | 3.060 [2.970, 3.150] | 1.489 [1.450, 1.520] | 1.473 [1.440, 1.510] | 1.189 [1.170, 1.220] | 1.205 [1.180, 1.230] |
| fused/split | 1.401 [1.360, 1.450] | 1.073 [1.050, 1.100] | 1.076 [1.050, 1.100] | 0.948 [0.930, 0.970] | 0.973 [0.950, 1.000] |

### M1 (ministral8 judged by qwen8; n=5000)

| Contrast | Raw [95% CI] | Cross-family decile [95% CI] | Cross-family absolute bin [95% CI] | Same-family decile [95% CI] | Target decile [95% CI] |
|---|---:|---:|---:|---:|---:|
| morph/int | 1.954 [1.880, 2.030] | 0.888 [0.870, 0.910] | 0.889 [0.860, 0.910] | — | 0.967 [0.940, 0.990] |
| word/int | 3.029 [2.930, 3.150] | 1.154 [1.120, 1.180] | 1.153 [1.120, 1.180] | — | 1.195 [1.170, 1.220] |
| fused/split | 1.526 [1.480, 1.580] | 1.178 [1.140, 1.210] | 1.175 [1.140, 1.210] | — | 0.947 [0.930, 0.970] |

### Q4 (secondary, not preregistered) (qwen4 judged by ministral8; n=5000)

| Contrast | Raw [95% CI] | Cross-family decile [95% CI] | Cross-family absolute bin [95% CI] | Same-family decile [95% CI] | Target decile [95% CI] |
|---|---:|---:|---:|---:|---:|
| morph/int | 2.128 [2.050, 2.210] | 1.200 [1.170, 1.240] | 1.185 [1.150, 1.220] | 0.972 [0.950, 1.000] | 0.969 [0.940, 0.990] |
| word/int | 3.350 [3.240, 3.470] | 1.563 [1.520, 1.610] | 1.537 [1.490, 1.580] | 1.202 [1.180, 1.230] | 1.208 [1.180, 1.240] |
| fused/split | 1.336 [1.290, 1.400] | 1.018 [0.980, 1.050] | 1.023 [0.990, 1.060] | 0.894 [0.870, 0.920] | 0.891 [0.860, 0.920] |

### Q3 (secondary, not preregistered) (qwen8 judged by ministral8; n=5000)

| Contrast | Raw [95% CI] | Cross-family decile [95% CI] | Cross-family absolute bin [95% CI] | Same-family decile [95% CI] | Target decile [95% CI] |
|---|---:|---:|---:|---:|---:|
| morph/int | 1.964 [1.880, 2.050] | 1.111 [1.080, 1.150] | 1.097 [1.060, 1.130] | 0.900 [0.880, 0.930] | 0.887 [0.860, 0.910] |
| word/int | 3.194 [3.080, 3.320] | 1.462 [1.420, 1.500] | 1.435 [1.390, 1.480] | 1.118 [1.090, 1.140] | 1.116 [1.090, 1.140] |
| fused/split | 1.187 [1.130, 1.240] | 0.916 [0.880, 0.950] | 0.920 [0.890, 0.960] | 0.806 [0.780, 0.830] | 0.821 [0.790, 0.850] |

### Q6 (secondary, not preregistered) (qwen8 judged by ministral8; n=5000)

| Contrast | Raw [95% CI] | Cross-family decile [95% CI] | Cross-family absolute bin [95% CI] | Same-family decile [95% CI] | Target decile [95% CI] |
|---|---:|---:|---:|---:|---:|
| morph/int | 2.075 [2.000, 2.150] | 1.183 [1.150, 1.220] | 1.170 [1.140, 1.210] | 0.952 [0.930, 0.970] | 0.945 [0.920, 0.970] |
| word/int | 3.252 [3.150, 3.370] | 1.515 [1.480, 1.560] | 1.492 [1.450, 1.530] | 1.183 [1.160, 1.210] | 1.178 [1.150, 1.200] |
| fused/split | 1.287 [1.230, 1.340] | 0.989 [0.960, 1.020] | 0.994 [0.970, 1.020] | 0.861 [0.840, 0.890] | 0.883 [0.860, 0.910] |
