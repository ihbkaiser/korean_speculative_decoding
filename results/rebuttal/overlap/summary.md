# Speculative-sampling overlap

Expected rejection is `1 − alpha`; intervals are 95% document-bootstrap intervals (500 resamples).
Ratios compare rejection rates for position types and are directly standardized to the target-margin or target-entropy decile distribution.

| Pair | Text | T | Positions | Mean rejection | Greedy disagreement | DP share | DP rejection mass | Morph/internal raw → margin matched | Word/internal raw → margin matched | Fused/split raw → margin matched |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Q2 | human | T1 | 640000 | 0.325 | 0.361 | 0.393 | 0.482 | 1.268 → 0.900 | 1.319 → 0.954 | 1.121 → 0.899 |
| Q2 | human | T07 | 640000 | 0.334 | 0.361 | 0.393 | 0.541 | 1.390 → 0.892 | 1.572 → 1.019 | 1.155 → 0.872 |
| Q2 | generated | T1 | 620451 | 0.170 | 0.161 | 0.214 | 0.442 | 2.039 → 1.045 | 2.249 → 1.205 | 1.430 → 0.909 |
| Q2 | generated | T07 | 620451 | 0.160 | 0.161 | 0.214 | 0.544 | 2.270 → 1.035 | 2.695 → 1.257 | 1.533 → 0.918 |
| Q5 | human | T1 | 640000 | 0.352 | 0.382 | 0.376 | 0.455 | 1.278 → 0.925 | 1.339 → 0.989 | 1.122 → 0.920 |
| Q5 | human | T07 | 640000 | 0.360 | 0.382 | 0.376 | 0.511 | 1.400 → 0.914 | 1.592 → 1.052 | 1.156 → 0.890 |
| M1 | human | T1 | 640000 | 0.230 | 0.265 | 0.447 | 0.570 | 1.344 → 0.896 | 1.584 → 0.994 | 1.110 → 0.878 |
| M1 | human | T07 | 640000 | 0.237 | 0.265 | 0.447 | 0.647 | 1.483 → 0.888 | 1.903 → 1.028 | 1.138 → 0.853 |
| M1 | generated | T1 | 625374 | 0.123 | 0.115 | 0.227 | 0.479 | 1.945 → 1.077 | 2.456 → 1.277 | 1.353 → 0.909 |
| M1 | generated | T07 | 625374 | 0.112 | 0.115 | 0.227 | 0.609 | 2.144 → 1.061 | 2.920 → 1.291 | 1.431 → 0.899 |
| G1 | generated | T1 | 638163 | 0.043 | 0.016 | 0.009 | 0.121 | 1.083 → 0.952 | 2.120 → 1.214 | 1.119 → 0.744 |
| G1 | generated | T07 | 638163 | 0.023 | 0.016 | 0.009 | 0.234 | 1.181 → 1.042 | 2.513 → 1.266 | 1.070 → 0.638 |
