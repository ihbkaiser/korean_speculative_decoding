# Copy-context intervention

This intervention changes decision strength while keeping the scored human continuation tokens and their aligned position types fixed. It does not manipulate morphology. In condition B, both target and draft can copy the continuation from the preceding context.

| Pair | Condition | Positions | Disagreement | Mean target margin | Target DP share | Morph/internal raw → matched | Word/internal raw → matched | Fused/split raw → matched |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Q2 | A | 256000 | 0.359 | 2.404 | 0.390 | 1.536 → 0.878 | 1.830 → 1.019 | 1.179 → 0.845 |
| Q2 | B | 256000 | 0.001 | 9.873 | 0.001 | 5.471 → 3.616 | 3.609 → 2.523 | 1.089 → 0.621 |
| Q2 | C | 256000 | 0.262 | 3.692 | 0.221 | 3.480 → 1.070 | 8.001 → 1.823 | 1.547 → 0.909 |
| Q2 | D | 256000 | 0.366 | 2.403 | 0.388 | 1.538 → 0.884 | 1.840 → 1.038 | 1.210 → 0.874 |
| M1 | A | 256000 | 0.264 | 1.933 | 0.446 | 1.580 → 0.858 | 2.180 → 0.990 | 1.163 → 0.852 |
| M1 | B | 256000 | 0.002 | 8.058 | 0.002 | 4.257 → 2.367 | 2.021 → 0.928 | 2.804 → 0.717 |
| M1 | C | 256000 | 0.210 | 3.138 | 0.263 | 3.499 → 1.071 | 12.339 → 1.831 | 1.439 → 0.862 |
| M1 | D | 256000 | 0.266 | 1.924 | 0.447 | 1.586 → 0.865 | 2.165 → 0.992 | 1.151 → 0.843 |

## Common margin-curve gaps versus A

| Pair | Condition | Largest absolute gap | Position-weighted mean absolute gap | Bins compared |
|---|---|---:|---:|---:|
| Q2 | B | 0.3083 | 0.2209 | 11 |
| Q2 | C | 0.0168 | 0.0127 | 11 |
| Q2 | D | 0.0097 | 0.0077 | 10 |
| M1 | B | 0.1167 | 0.0770 | 11 |
| M1 | C | 0.0673 | 0.0504 | 11 |
| M1 | D | 0.0026 | 0.0020 | 11 |

## Logistic condition and interaction coefficients

Coefficients are log odds with document-clustered standard errors; models are shown with and without margin-bin fixed effects. See `logistic.csv` for intervals and p-values.

| Pair | Margin-bin fixed effects | Coefficient | Estimate | Clustered SE | p-value |
|---|---|---|---:|---:|---:|
| Q2 | False | C(condition, Treatment(reference='A'))[T.B] | -7.1338 | 0.2576 | 8.06e-169 |
| Q2 | False | C(condition, Treatment(reference='A'))[T.C] | -1.6707 | 0.0222 | 0 |
| Q2 | False | C(condition, Treatment(reference='A'))[T.D] | 0.0208 | 0.0057 | 0.0002779 |
| Q2 | False | C(condition, Treatment(reference='A'))[T.B]:C(ptype, Treatment(reference='internal'))[T.fused] | 0.1732 | 1.0357 | 0.8672 |
| Q2 | False | C(condition, Treatment(reference='A'))[T.C]:C(ptype, Treatment(reference='internal'))[T.fused] | 1.1161 | 0.0740 | 2.045e-51 |
| Q2 | False | C(condition, Treatment(reference='A'))[T.D]:C(ptype, Treatment(reference='internal'))[T.fused] | -0.0159 | 0.0330 | 0.63 |
| Q2 | False | C(condition, Treatment(reference='A'))[T.B]:C(ptype, Treatment(reference='internal'))[T.morph_onset] | 1.0838 | 0.2879 | 0.0001667 |
| Q2 | False | C(condition, Treatment(reference='A'))[T.C]:C(ptype, Treatment(reference='internal'))[T.morph_onset] | 0.7913 | 0.0245 | 2.899e-229 |
| Q2 | False | C(condition, Treatment(reference='A'))[T.D]:C(ptype, Treatment(reference='internal'))[T.morph_onset] | 0.0065 | 0.0088 | 0.4575 |
| Q2 | False | C(condition, Treatment(reference='A'))[T.B]:C(ptype, Treatment(reference='internal'))[T.word_onset] | 0.3729 | 0.2898 | 0.1983 |
| Q2 | False | C(condition, Treatment(reference='A'))[T.C]:C(ptype, Treatment(reference='internal'))[T.word_onset] | 1.7128 | 0.0233 | 0 |
| Q2 | False | C(condition, Treatment(reference='A'))[T.D]:C(ptype, Treatment(reference='internal'))[T.word_onset] | 0.0182 | 0.0079 | 0.02123 |
| Q2 | True | C(condition, Treatment(reference='A'))[T.B] | -3.6130 | 0.2666 | 7.694e-42 |
| Q2 | True | C(condition, Treatment(reference='A'))[T.C] | -0.7603 | 0.0235 | 1.082e-228 |
| Q2 | True | C(condition, Treatment(reference='A'))[T.D] | 0.0254 | 0.0080 | 0.001596 |
| Q2 | True | C(condition, Treatment(reference='A'))[T.B]:C(ptype, Treatment(reference='internal'))[T.fused] | -0.1426 | 1.0396 | 0.8909 |
| Q2 | True | C(condition, Treatment(reference='A'))[T.C]:C(ptype, Treatment(reference='internal'))[T.fused] | 0.4497 | 0.0803 | 2.132e-08 |
| Q2 | True | C(condition, Treatment(reference='A'))[T.D]:C(ptype, Treatment(reference='internal'))[T.fused] | -0.0083 | 0.0391 | 0.8323 |
| Q2 | True | C(condition, Treatment(reference='A'))[T.B]:C(ptype, Treatment(reference='internal'))[T.morph_onset] | 1.3918 | 0.2869 | 1.228e-06 |
| Q2 | True | C(condition, Treatment(reference='A'))[T.C]:C(ptype, Treatment(reference='internal'))[T.morph_onset] | 0.4269 | 0.0269 | 6.823e-57 |
| Q2 | True | C(condition, Treatment(reference='A'))[T.D]:C(ptype, Treatment(reference='internal'))[T.morph_onset] | 0.0037 | 0.0111 | 0.742 |
| Q2 | True | C(condition, Treatment(reference='A'))[T.B]:C(ptype, Treatment(reference='internal'))[T.word_onset] | 0.8013 | 0.2882 | 0.005426 |
| Q2 | True | C(condition, Treatment(reference='A'))[T.C]:C(ptype, Treatment(reference='internal'))[T.word_onset] | 1.1066 | 0.0251 | 0 |
| Q2 | True | C(condition, Treatment(reference='A'))[T.D]:C(ptype, Treatment(reference='internal'))[T.word_onset] | 0.0327 | 0.0104 | 0.001714 |
| M1 | False | C(condition, Treatment(reference='A'))[T.B] | -5.9018 | 0.1824 | 8.892e-230 |
| M1 | False | C(condition, Treatment(reference='A'))[T.C] | -1.7476 | 0.0290 | 0 |
| M1 | False | C(condition, Treatment(reference='A'))[T.D] | 0.0093 | 0.0078 | 0.231 |
| M1 | False | C(condition, Treatment(reference='A'))[T.B]:C(ptype, Treatment(reference='internal'))[T.fused] | 0.5857 | 0.3267 | 0.07302 |
| M1 | False | C(condition, Treatment(reference='A'))[T.C]:C(ptype, Treatment(reference='internal'))[T.fused] | 1.2793 | 0.0473 | 6.131e-161 |
| M1 | False | C(condition, Treatment(reference='A'))[T.D]:C(ptype, Treatment(reference='internal'))[T.fused] | -0.0046 | 0.0217 | 0.832 |
| M1 | False | C(condition, Treatment(reference='A'))[T.B]:C(ptype, Treatment(reference='internal'))[T.morph_onset] | 0.8744 | 0.2071 | 2.419e-05 |
| M1 | False | C(condition, Treatment(reference='A'))[T.C]:C(ptype, Treatment(reference='internal'))[T.morph_onset] | 0.7640 | 0.0330 | 1.532e-118 |
| M1 | False | C(condition, Treatment(reference='A'))[T.D]:C(ptype, Treatment(reference='internal'))[T.morph_onset] | 0.0064 | 0.0111 | 0.5657 |
| M1 | False | C(condition, Treatment(reference='A'))[T.B]:C(ptype, Treatment(reference='internal'))[T.word_onset] | -0.3318 | 0.2174 | 0.1269 |
| M1 | False | C(condition, Treatment(reference='A'))[T.C]:C(ptype, Treatment(reference='internal'))[T.word_onset] | 1.9536 | 0.0300 | 0 |
| M1 | False | C(condition, Treatment(reference='A'))[T.D]:C(ptype, Treatment(reference='internal'))[T.word_onset] | -0.0077 | 0.0101 | 0.445 |
| M1 | True | C(condition, Treatment(reference='A'))[T.B] | -1.2981 | 0.1868 | 3.623e-12 |
| M1 | True | C(condition, Treatment(reference='A'))[T.C] | -0.5478 | 0.0301 | 3.751e-74 |
| M1 | True | C(condition, Treatment(reference='A'))[T.D] | 0.0069 | 0.0110 | 0.5313 |
| M1 | True | C(condition, Treatment(reference='A'))[T.B]:C(ptype, Treatment(reference='internal'))[T.fused] | 0.2841 | 0.3238 | 0.3804 |
| M1 | True | C(condition, Treatment(reference='A'))[T.C]:C(ptype, Treatment(reference='internal'))[T.fused] | 0.5707 | 0.0507 | 2.083e-29 |
| M1 | True | C(condition, Treatment(reference='A'))[T.D]:C(ptype, Treatment(reference='internal'))[T.fused] | -0.0159 | 0.0260 | 0.5405 |
| M1 | True | C(condition, Treatment(reference='A'))[T.B]:C(ptype, Treatment(reference='internal'))[T.morph_onset] | 0.7534 | 0.2071 | 0.0002744 |
| M1 | True | C(condition, Treatment(reference='A'))[T.C]:C(ptype, Treatment(reference='internal'))[T.morph_onset] | 0.4027 | 0.0354 | 4.654e-30 |
| M1 | True | C(condition, Treatment(reference='A'))[T.D]:C(ptype, Treatment(reference='internal'))[T.morph_onset] | 0.0071 | 0.0144 | 0.621 |
| M1 | True | C(condition, Treatment(reference='A'))[T.B]:C(ptype, Treatment(reference='internal'))[T.word_onset] | -0.1678 | 0.2170 | 0.4394 |
| M1 | True | C(condition, Treatment(reference='A'))[T.C]:C(ptype, Treatment(reference='internal'))[T.word_onset] | 1.0568 | 0.0316 | 5.999e-245 |
| M1 | True | C(condition, Treatment(reference='A'))[T.D]:C(ptype, Treatment(reference='internal'))[T.word_onset] | 0.0002 | 0.0132 | 0.9859 |
