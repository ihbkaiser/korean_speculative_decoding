# H3 Korean morphological boundary-type decomposition

H3 result: A — structured grammatical concentration, with a reproducible positive NOMINAL_TO_PARTICLE penalty.

## Preserved H2 exclusions

| pair | generated_token_rows | span_exact_false_all | exact_retokenization_failures_non_special | cross_eojeol_excluded | kiwi_complex_excluded | sd_valid_false | fragmentation_outside_primary_bins | morph_class_outside_primary_classes | h2_eligible_tokens | h2_eligible_prompts |
| ---- | -------------------- | -------------------- | ----------------------------------------- | --------------------- | --------------------- | -------------- | ---------------------------------- | ----------------------------------- | ------------------ | ------------------- |
| P1   | 127118               | 348                  | 338                                       | 0                     | 10145                 | 0              | 11851                              | 53597                               | 73001              | 995                 |
| P2   | 127245               | 153                  | 140                                       | 0                     | 9698                  | 0              | 11633                              | 53438                               | 73257              | 997                 |
| P3   | 127245               | 148                  | 135                                       | 0                     | 9660                  | 0              | 11669                              | 53549                               | 73143              | 997                 |

## Taxonomy support

# H3 boundary taxonomy audit

The eligibility and H2 class definitions are unchanged. H3a/H3b require single-boundary CROSS tokens; CROSS rows with zero reconstructed overlap transitions are flagged as unresolved, while multi-boundary rows are retained for exploratory analysis.

## Cross-boundary reconstruction

| pair   | all_cross_tokens | single_boundary_cross | multi_boundary_cross | cross_boundary_unresolved | percent_single_of_cross |
| ------ | ---------------- | --------------------- | -------------------- | ------------------------- | ----------------------- |
| P1     | 3454             | 2969                  | 485                  | 0                         | 85.96                   |
| P2     | 3762             | 3158                  | 604                  | 0                         | 83.94                   |
| P3     | 3800             | 3167                  | 633                  | 0                         | 83.34                   |
| POOLED | 11016            | 9294                  | 1722                 | 0                         | 84.37                   |

## Environment support

| pair   | environment         | n_cross | n_cross_prompts | percent_of_single_boundary_cross | raw_rejection | median_fragmentation | median_token_frequency | mean_draft_entropy | support_warning |
| ------ | ------------------- | ------- | --------------- | -------------------------------- | ------------- | -------------------- | ---------------------- | ------------------ | --------------- |
| P1     | NOMINAL_TO_PARTICLE | 212     | 87              | 7.140                            | 0.264         | 2.000                | 593.000                | 0.899              | false           |
| P1     | PREDICATE_TO_ENDING | 988     | 349             | 33.277                           | 0.168         | 3.000                | 4781.000               | 1.370              | false           |
| P1     | ENDING_TO_ENDING    | 159     | 90              | 5.355                            | 0.082         | 3.000                | 3431.000               | 1.202              | false           |
| P1     | LEXICAL_TO_LEXICAL  | 33      | 16              | 1.111                            | 0.242         | 4.000                | 1066.000               | 0.967              | true            |
| P1     | OTHER               | 1577    | 434             | 53.116                           | 0.229         | 3.000                | 2418.000               | 1.435              | false           |
| P2     | NOMINAL_TO_PARTICLE | 212     | 97              | 6.713                            | 0.241         | 3.000                | 593.000                | 0.940              | false           |
| P2     | PREDICATE_TO_ENDING | 958     | 345             | 30.336                           | 0.175         | 3.000                | 4781.000               | 1.304              | false           |
| P2     | ENDING_TO_ENDING    | 137     | 91              | 4.338                            | 0.161         | 3.000                | 3431.000               | 1.097              | false           |
| P2     | LEXICAL_TO_LEXICAL  | 51      | 27              | 1.615                            | 0.196         | 4.000                | 1066.000               | 1.157              | true            |
| P2     | OTHER               | 1800    | 514             | 56.998                           | 0.207         | 3.000                | 2718.000               | 1.409              | false           |
| P3     | NOMINAL_TO_PARTICLE | 218     | 98              | 6.883                            | 0.376         | 2.000                | 586.000                | 1.086              | false           |
| P3     | PREDICATE_TO_ENDING | 978     | 349             | 30.881                           | 0.200         | 3.000                | 4781.000               | 1.439              | false           |
| P3     | ENDING_TO_ENDING    | 130     | 83              | 4.105                            | 0.138         | 3.000                | 3431.000               | 1.223              | false           |
| P3     | LEXICAL_TO_LEXICAL  | 47      | 27              | 1.484                            | 0.298         | 4.000                | 1066.000               | 1.331              | true            |
| P3     | OTHER               | 1794    | 519             | 56.647                           | 0.283         | 3.000                | 2718.000               | 1.553              | false           |
| POOLED | NOMINAL_TO_PARTICLE | 642     | 155             | 6.908                            | 0.294         | 2.000                | 593.000                | 0.976              | false           |
| POOLED | PREDICATE_TO_ENDING | 2924    | 461             | 31.461                           | 0.181         | 3.000                | 4781.000               | 1.371              | false           |
| POOLED | ENDING_TO_ENDING    | 426     | 161             | 4.584                            | 0.124         | 3.000                | 3431.000               | 1.175              | false           |
| POOLED | LEXICAL_TO_LEXICAL  | 131     | 43              | 1.410                            | 0.244         | 4.000                | 1066.000               | 1.171              | true            |
| POOLED | OTHER               | 5171    | 656             | 55.638                           | 0.240         | 3.000                | 2718.000               | 1.467              | false           |

Categories with fewer than 100 CROSS tokens or 50 CROSS prompts are flagged and retained without merging. SPLIT support is audited separately in the H3B_ALL_CLASSES rows of h3_boundary_counts.csv.

## Frequent fine-POS transitions (top five per taxonomy class)

| pair | crossed_fine_pos_sequence | n_cross | n_prompts | raw_rejection | support_threshold_met | coarse_boundary_type |
| ---- | ------------------------- | ------- | --------- | ------------- | --------------------- | -------------------- |
| P1   | EP→EF                     | 149     | 87        | 0.074         | true                  | ENDING_TO_ENDING     |
| P1   | EP→ETM                    | 10      | 5         | 0.200         | false                 | ENDING_TO_ENDING     |
| P1   | NNB→XSN                   | 22      | 9         | 0.182         | false                 | LEXICAL_TO_LEXICAL   |
| P1   | NNG→VCP                   | 5       | 2         | 0.200         | false                 | LEXICAL_TO_LEXICAL   |
| P1   | XSN→VCP                   | 2       | 1         | 0.500         | false                 | LEXICAL_TO_LEXICAL   |
| P1   | MM→NNB                    | 1       | 1         | 1.000         | false                 | LEXICAL_TO_LEXICAL   |
| P1   | NNB→VCP                   | 1       | 1         | 1.000         | false                 | LEXICAL_TO_LEXICAL   |
| P1   | NNG→JKO                   | 82      | 36        | 0.207         | false                 | NOMINAL_TO_PARTICLE  |
| P1   | NNG→JX                    | 51      | 24        | 0.275         | false                 | NOMINAL_TO_PARTICLE  |
| P1   | NNG→JKS                   | 32      | 15        | 0.375         | false                 | NOMINAL_TO_PARTICLE  |
| P1   | NNG→JKG                   | 18      | 7         | 0.167         | false                 | NOMINAL_TO_PARTICLE  |
| P1   | NNB→JKB                   | 14      | 9         | 0.357         | false                 | NOMINAL_TO_PARTICLE  |
| P1   | XSV→EC                    | 406     | 186       | 0.271         | true                  | OTHER                |
| P1   | XSV→ETM                   | 377     | 123       | 0.133         | true                  | OTHER                |
| P1   | JKB→JX                    | 116     | 45        | 0.310         | false                 | OTHER                |
| P1   | XSV→EF                    | 94      | 32        | 0.319         | false                 | OTHER                |
| P1   | SSO→SL                    | 75      | 30        | 0.173         | false                 | OTHER                |
| P1   | VCP→EF                    | 389     | 159       | 0.159         | true                  | PREDICATE_TO_ENDING  |
| P1   | VA→EF                     | 120     | 67        | 0.075         | true                  | PREDICATE_TO_ENDING  |
| P1   | VX→EF                     | 119     | 73        | 0.076         | true                  | PREDICATE_TO_ENDING  |
| P1   | VV→EC                     | 92      | 56        | 0.228         | false                 | PREDICATE_TO_ENDING  |
| P1   | VCP→EC                    | 58      | 40        | 0.362         | false                 | PREDICATE_TO_ENDING  |
| P2   | EP→EF                     | 135     | 89        | 0.156         | true                  | ENDING_TO_ENDING     |
| P2   | EC→EC                     | 1       | 1         | 1.000         | false                 | ENDING_TO_ENDING     |
| P2   | EP→ETM                    | 1       | 1         | 0.000         | false                 | ENDING_TO_ENDING     |
| P2   | NNB→XSN                   | 24      | 9         | 0.167         | false                 | LEXICAL_TO_LEXICAL   |
| P2   | NNG→NNG                   | 9       | 6         | 0.111         | false                 | LEXICAL_TO_LEXICAL   |
| P2   | NNB→VCP                   | 7       | 4         | 0.429         | false                 | LEXICAL_TO_LEXICAL   |
| P2   | XSN→VCP                   | 5       | 2         | 0.200         | false                 | LEXICAL_TO_LEXICAL   |
| P2   | NNG→VCP                   | 2       | 2         | 0.000         | false                 | LEXICAL_TO_LEXICAL   |
| P2   | NNG→JKO                   | 72      | 41        | 0.278         | false                 | NOMINAL_TO_PARTICLE  |
| P2   | NNG→JX                    | 51      | 26        | 0.196         | false                 | NOMINAL_TO_PARTICLE  |
| P2   | NNG→JKG                   | 28      | 11        | 0.286         | false                 | NOMINAL_TO_PARTICLE  |
| P2   | NNG→JKS                   | 27      | 14        | 0.111         | false                 | NOMINAL_TO_PARTICLE  |
| P2   | NNP→JX                    | 12      | 8         | 0.417         | false                 | NOMINAL_TO_PARTICLE  |
| P2   | XSV→EC                    | 514     | 245       | 0.249         | true                  | OTHER                |
| P2   | XSV→ETM                   | 436     | 145       | 0.110         | true                  | OTHER                |
| P2   | JKB→JX                    | 172     | 63        | 0.233         | true                  | OTHER                |
| P2   | XSV→EF                    | 155     | 53        | 0.181         | true                  | OTHER                |
| P2   | SSO→SL                    | 79      | 34        | 0.278         | false                 | OTHER                |
| P2   | VCP→EF                    | 357     | 165       | 0.196         | true                  | PREDICATE_TO_ENDING  |
| P2   | VX→EF                     | 149     | 81        | 0.087         | true                  | PREDICATE_TO_ENDING  |
| P2   | VA→EF                     | 96      | 61        | 0.052         | false                 | PREDICATE_TO_ENDING  |
| P2   | VV→EC                     | 92      | 50        | 0.141         | false                 | PREDICATE_TO_ENDING  |
| P2   | VCP→EC                    | 60      | 37        | 0.250         | false                 | PREDICATE_TO_ENDING  |
| P3   | EP→EF                     | 128     | 81        | 0.125         | true                  | ENDING_TO_ENDING     |
| P3   | EC→EC                     | 1       | 1         | 1.000         | false                 | ENDING_TO_ENDING     |
| P3   | EP→ETM                    | 1       | 1         | 1.000         | false                 | ENDING_TO_ENDING     |
| P3   | NNB→XSN                   | 24      | 10        | 0.292         | false                 | LEXICAL_TO_LEXICAL   |
| P3   | NNG→NNG                   | 9       | 6         | 0.111         | false                 | LEXICAL_TO_LEXICAL   |
| P3   | XSN→VCP                   | 5       | 2         | 0.400         | false                 | LEXICAL_TO_LEXICAL   |
| P3   | NNB→VCP                   | 3       | 3         | 0.667         | false                 | LEXICAL_TO_LEXICAL   |
| P3   | NNG→VCP                   | 2       | 2         | 0.000         | false                 | LEXICAL_TO_LEXICAL   |
| P3   | NNG→JKO                   | 74      | 40        | 0.270         | false                 | NOMINAL_TO_PARTICLE  |
| P3   | NNG→JX                    | 55      | 26        | 0.382         | false                 | NOMINAL_TO_PARTICLE  |
| P3   | NNG→JKG                   | 28      | 11        | 0.286         | false                 | NOMINAL_TO_PARTICLE  |
| P3   | NNG→JKS                   | 27      | 15        | 0.519         | false                 | NOMINAL_TO_PARTICLE  |
| P3   | NNP→JX                    | 12      | 9         | 0.667         | false                 | NOMINAL_TO_PARTICLE  |
| P3   | XSV→EC                    | 521     | 247       | 0.347         | true                  | OTHER                |
| P3   | XSV→ETM                   | 436     | 148       | 0.142         | true                  | OTHER                |
| P3   | JKB→JX                    | 166     | 61        | 0.307         | true                  | OTHER                |
| P3   | XSV→EF                    | 152     | 54        | 0.349         | true                  | OTHER                |
| P3   | SSO→SL                    | 73      | 31        | 0.370         | false                 | OTHER                |
| P3   | VCP→EF                    | 363     | 167       | 0.193         | true                  | PREDICATE_TO_ENDING  |
| P3   | VX→EF                     | 146     | 79        | 0.089         | true                  | PREDICATE_TO_ENDING  |
| P3   | VA→EF                     | 98      | 61        | 0.112         | false                 | PREDICATE_TO_ENDING  |
| P3   | VV→EC                     | 90      | 50        | 0.233         | false                 | PREDICATE_TO_ENDING  |
| P3   | VCP→EC                    | 60      | 35        | 0.333         | false                 | PREDICATE_TO_ENDING  |


## H3a: CROSS-only boundary-type analysis

| pair | environment         | n_cross | n_prompts | raw_rejection | adjusted_P_95CI    | reference | omnibus_p |
| ---- | ------------------- | ------- | --------- | ------------- | ------------------ | --------- | --------- |
| P1   | NOMINAL_TO_PARTICLE | 212     | 87        | 0.2642        | 34.9% [28.7, 41.1] | OTHER     | <0.0001   |
| P1   | PREDICATE_TO_ENDING | 988     | 349       | 0.1680        | 17.9% [15.3, 20.6] | OTHER     | <0.0001   |
| P1   | ENDING_TO_ENDING    | 159     | 90        | 0.0818        | 9.7% [4.7, 14.8]   | OTHER     | <0.0001   |
| P1   | LEXICAL_TO_LEXICAL  | 33      | 16        | 0.2424        | 30.4% [16.9, 43.8] | OTHER     | <0.0001   |
| P1   | OTHER               | 1577    | 434       | 0.2289        | 20.8% [18.9, 22.8] | OTHER     | <0.0001   |
| P2   | NOMINAL_TO_PARTICLE | 212     | 97        | 0.2406        | 29.7% [21.0, 38.5] | OTHER     | 0.077     |
| P2   | PREDICATE_TO_ENDING | 958     | 345       | 0.1754        | 20.2% [17.2, 23.2] | OTHER     | 0.077     |
| P2   | ENDING_TO_ENDING    | 137     | 91        | 0.1606        | 20.7% [14.3, 27.0] | OTHER     | 0.077     |
| P2   | LEXICAL_TO_LEXICAL  | 51      | 27        | 0.1961        | 20.5% [11.2, 29.9] | OTHER     | 0.077     |
| P2   | OTHER               | 1800    | 514       | 0.2067        | 18.4% [16.6, 20.2] | OTHER     | 0.077     |
| P3   | NOMINAL_TO_PARTICLE | 218     | 98        | 0.3761        | 43.0% [35.0, 50.9] | OTHER     | <0.0001   |
| P3   | PREDICATE_TO_ENDING | 978     | 349       | 0.2004        | 21.7% [19.0, 24.4] | OTHER     | <0.0001   |
| P3   | ENDING_TO_ENDING    | 130     | 83        | 0.1385        | 18.2% [11.4, 25.0] | OTHER     | <0.0001   |
| P3   | LEXICAL_TO_LEXICAL  | 47      | 27        | 0.2979        | 31.9% [21.1, 42.6] | OTHER     | <0.0001   |
| P3   | OTHER               | 1794    | 519       | 0.2826        | 26.0% [24.0, 28.0] | OTHER     | <0.0001   |

H3a adjusted probabilities standardize every boundary category over the same pair-specific CROSS-only covariate distribution. The lexical reference was below the fixed 100-token/50-prompt support rule in all pairs, so OTHER was used as the computational reference in each H3a fit; all category probabilities are reported.

## H3b: raw CROSS/SPLIT rejection by environment

| pair   | environment         | n_cross | n_split | raw_CROSS/SPLIT | support_warning | split_support_warning |
| ------ | ------------------- | ------- | ------- | --------------- | --------------- | --------------------- |
| P1     | NOMINAL_TO_PARTICLE | 212     | 22211   | 26.4% / 13.2%   | false           | false                 |
| P1     | PREDICATE_TO_ENDING | 988     | 1221    | 16.8% / 18.5%   | false           | false                 |
| P1     | ENDING_TO_ENDING    | 159     | 10      | 8.2% / 40.0%    | false           | true                  |
| P1     | LEXICAL_TO_LEXICAL  | 33      | 24373   | 24.2% / 13.4%   | true            | false                 |
| P1     | OTHER               | 1577    | 21732   | 22.9% / 11.4%   | false           | false                 |
| P2     | NOMINAL_TO_PARTICLE | 212     | 22014   | 24.1% / 14.4%   | false           | false                 |
| P2     | PREDICATE_TO_ENDING | 958     | 1246    | 17.5% / 15.2%   | false           | false                 |
| P2     | ENDING_TO_ENDING    | 137     | 33      | 16.1% / 33.3%   | false           | true                  |
| P2     | LEXICAL_TO_LEXICAL  | 51      | 25021   | 19.6% / 14.2%   | true            | false                 |
| P2     | OTHER               | 1800    | 21181   | 20.7% / 14.1%   | false           | false                 |
| P3     | NOMINAL_TO_PARTICLE | 218     | 21829   | 37.6% / 18.6%   | false           | false                 |
| P3     | PREDICATE_TO_ENDING | 978     | 1271    | 20.0% / 21.7%   | false           | false                 |
| P3     | ENDING_TO_ENDING    | 130     | 30      | 13.8% / 46.7%   | false           | true                  |
| P3     | LEXICAL_TO_LEXICAL  | 47      | 24849   | 29.8% / 18.1%   | true            | false                 |
| P3     | OTHER               | 1794    | 21364   | 28.3% / 18.1%   | false           | false                 |
| POOLED | NOMINAL_TO_PARTICLE | 642     | 66054   | 29.4% / 15.4%   | false           | false                 |
| POOLED | PREDICATE_TO_ENDING | 2924    | 3738    | 18.1% / 18.5%   | false           | false                 |
| POOLED | ENDING_TO_ENDING    | 426     | 73      | 12.4% / 39.7%   | false           | true                  |
| POOLED | LEXICAL_TO_LEXICAL  | 131     | 74243   | 24.4% / 15.2%   | true            | false                 |
| POOLED | OTHER               | 5171    | 64277   | 24.0% / 14.5%   | false           | false                 |

## H3b: pair-specific adjusted CROSS − SPLIT differences

| pair | environment         | n_cross | n_split | n_prompts | adjusted_P_cross_split | AME_pp_95CI          | p_value | p_Holm  | support_warning | estimable |
| ---- | ------------------- | ------- | ------- | --------- | ---------------------- | -------------------- | ------- | ------- | --------------- | --------- |
| P1   | NOMINAL_TO_PARTICLE | 212     | 22211   | 879       | 34.0% / 13.1%          | +20.9 [+15.4, +26.4] | <0.0001 | <0.0001 | false           | true      |
| P1   | PREDICATE_TO_ENDING | 988     | 1221    | 432       | 16.2% / 19.2%          | -2.9 [-6.4, +0.6]    | 0.1     | 0.1     | false           | true      |
| P1   | ENDING_TO_ENDING    | 159     | 10      | 95        | 8.0% / 44.4%           | -36.4 [-58.7, -14.1] | 0.0014  | 0.0041  | true            | true      |
| P1   | LEXICAL_TO_LEXICAL  | 33      | 24373   | 908       | 25.9% / 13.4%          | +12.5 [+1.4, +23.7]  | 0.028   | 0.056   | true            | true      |
| P1   | OTHER               | 1577    | 21732   | 937       | 17.9% / 11.4%          | +6.5 [+4.4, +8.5]    | <0.0001 | <0.0001 | false           | true      |
| P2   | NOMINAL_TO_PARTICLE | 212     | 22014   | 905       | 33.0% / 14.4%          | +18.7 [+12.4, +25.0] | <0.0001 | <0.0001 | false           | true      |
| P2   | PREDICATE_TO_ENDING | 958     | 1246    | 452       | 18.1% / 14.7%          | +3.4 [+0.1, +6.7]    | 0.043   | 0.085   | false           | true      |
| P2   | ENDING_TO_ENDING    | 137     | 33      | 104       | 16.0% / 33.0%          | -17.0 [-32.0, -2.1]  | 0.026   | 0.077   | true            | true      |
| P2   | LEXICAL_TO_LEXICAL  | 51      | 25021   | 936       | 20.1% / 14.2%          | +6.0 [-1.1, +13.0]   | 0.096   | 0.096   | true            | true      |
| P2   | OTHER               | 1800    | 21181   | 952       | 19.5% / 14.0%          | +5.5 [+3.4, +7.6]    | <0.0001 | <0.0001 | false           | true      |
| P3   | NOMINAL_TO_PARTICLE | 218     | 21829   | 907       | 43.2% / 18.5%          | +24.7 [+18.4, +31.0] | <0.0001 | <0.0001 | false           | true      |
| P3   | PREDICATE_TO_ENDING | 978     | 1271    | 454       | 19.8% / 22.1%          | -2.3 [-5.7, +1.1]    | 0.19    | 0.33    | false           | true      |
| P3   | ENDING_TO_ENDING    | 130     | 30      | 96        | 13.2% / 50.1%          | -36.9 [-50.9, -23.0] | <0.0001 | <0.0001 | true            | true      |
| P3   | LEXICAL_TO_LEXICAL  | 47      | 24849   | 935       | 24.7% / 18.1%          | +6.6 [-2.7, +15.9]   | 0.17    | 0.33    | true            | true      |
| P3   | OTHER               | 1794    | 21364   | 954       | 24.4% / 18.2%          | +6.3 [+4.1, +8.4]    | <0.0001 | <0.0001 | false           | true      |

## Pair-level interaction tests

| pair | H3b interaction Wald p | H3a boundary Wald p | H3b converged | H3b design rank/columns |
| ---- | ---------------------- | ------------------- | ------------- | ----------------------- |
| P1   | <0.0001                | <0.0001             | true          | 29/29                   |
| P2   | <0.0001                | 0.077               | true          | 29/29                   |
| P3   | <0.0001                | <0.0001             | true          | 29/29                   |

## Pooled and common-prompt sensitivity

The exact three-way intersection contains **993 eligible prompts** (the E2 audit's 999 value is the union). Common-prompt sensitivity is minor numerical change; maximum absolute AME shift is 0.95 pp, with no sign reversal. All-population interaction p=<0.0001; common-993 interaction p=<0.0001.

| environment         | difference_pp_common_minus_all | all_p   | common_p | all_prompt_AME_pp | common_993_AME_pp |
| ------------------- | ------------------------------ | ------- | -------- | ----------------- | ----------------- |
| NOMINAL_TO_PARTICLE | -0.9522                        | <0.0001 | <0.0001  | 21.5572           | 20.6050           |
| PREDICATE_TO_ENDING | 0.0401                         | 0.66    | 0.69     | -0.5139           | -0.4737           |
| ENDING_TO_ENDING    | 0.0494                         | <0.0001 | <0.0001  | -29.3598          | -29.3104          |
| LEXICAL_TO_LEXICAL  | 0.0438                         | 0.0086  | 0.0084   | 7.4415            | 7.4854            |
| OTHER               | 0.0497                         | <0.0001 | <0.0001  | 6.1018            | 6.1515            |

### Construction-level environment sensitivity

NOMINAL_TO_PARTICLE sensitivity AMEs: P1 +21.4 [+15.2, +27.7] pp, P2 +18.8 [+12.6, +25.1] pp, P3 +24.1 [+17.7, +30.5] pp. Ending→Ending is non-estimable because the construction-level class contains no single-boundary CROSS tokens in any pair; its SPLIT-only rows remain included in fitting the other estimable terms.

## Contribution and exploratory analyses

| pair   | environment         | n_single_boundary_cross | share_of_cross_tokens | adjusted_excess_rejection | contribution_score | normalized_contribution_percent |
| ------ | ------------------- | ----------------------- | --------------------- | ------------------------- | ------------------ | ------------------------------- |
| P1     | NOMINAL_TO_PARTICLE | 212                     | 0.0714                | 0.2092                    | 0.0149             | 69.1404                         |
| P1     | PREDICATE_TO_ENDING | 988                     | 0.3328                | -0.0291                   | -0.0097            | -44.8734                        |
| P1     | ENDING_TO_ENDING    | 159                     | 0.0536                | -0.3640                   | -0.0195            | -90.2124                        |
| P1     | LEXICAL_TO_LEXICAL  | 33                      | 0.0111                | 0.1254                    | 0.0014             | 6.4505                          |
| P1     | OTHER               | 1577                    | 0.5312                | 0.0649                    | 0.0345             | 159.4950                        |
| P2     | NOMINAL_TO_PARTICLE | 212                     | 0.0671                | 0.1867                    | 0.0125             | 26.2238                         |
| P2     | PREDICATE_TO_ENDING | 958                     | 0.3034                | 0.0341                    | 0.0103             | 21.6399                         |
| P2     | ENDING_TO_ENDING    | 137                     | 0.0434                | -0.1701                   | -0.0074            | -15.4399                        |
| P2     | LEXICAL_TO_LEXICAL  | 51                      | 0.0161                | 0.0596                    | 0.0010             | 2.0131                          |
| P2     | OTHER               | 1800                    | 0.5700                | 0.0550                    | 0.0313             | 65.5631                         |
| P3     | NOMINAL_TO_PARTICLE | 218                     | 0.0688                | 0.2472                    | 0.0170             | 54.4643                         |
| P3     | PREDICATE_TO_ENDING | 978                     | 0.3088                | -0.0227                   | -0.0070            | -22.4746                        |
| P3     | ENDING_TO_ENDING    | 130                     | 0.0410                | -0.3692                   | -0.0152            | -48.5223                        |
| P3     | LEXICAL_TO_LEXICAL  | 47                      | 0.0148                | 0.0659                    | 0.0010             | 3.1307                          |
| P3     | OTHER               | 1794                    | 0.5665                | 0.0625                    | 0.0354             | 113.4019                        |
| POOLED | NOMINAL_TO_PARTICLE | 642                     | 0.0691                | 0.2156                    | 0.0149             | 42.7715                         |
| POOLED | PREDICATE_TO_ENDING | 2924                    | 0.3146                | -0.0051                   | -0.0016            | -4.6436                         |
| POOLED | ENDING_TO_ENDING    | 426                     | 0.0458                | -0.2936                   | -0.0135            | -38.6536                        |
| POOLED | LEXICAL_TO_LEXICAL  | 131                     | 0.0141                | 0.0744                    | 0.0010             | 3.0127                          |
| POOLED | OTHER               | 5171                    | 0.5564                | 0.0610                    | 0.0339             | 97.5130                         |

In the pooled descriptive contribution score, NOMINAL_TO_PARTICLE prevalence is 6.9% with adjusted excess +21.6 pp (score +1.49 pp); OTHER is 55.6% with +6.1 pp (score +3.39 pp). These scores are descriptive and use signed category excesses.

### Multi-boundary CROSS

| pair | boundary_count_band | n_cross | n_prompts | raw_rejection | omnibus_boundary_count_p | adjusted_rejection | ci_low | ci_high |
| ---- | ------------------- | ------- | --------- | ------------- | ------------------------ | ------------------ | ------ | ------- |
| P1   | 1                   | 2969    | 602       | 0.2034        | 0.2902                   | 0.2062             | 0.1906 | 0.2217  |
| P1   | 2                   | 485     | 195       | 0.1897        | 0.2902                   | 0.1775             | 0.1338 | 0.2212  |
| P1   | 3+                  | 0       | 0         | —             | 0.2902                   | —                  | —      | —       |
| P2   | 1                   | 3158    | 667       | 0.1973        | 0.3603                   | 0.1896             | 0.1750 | 0.2042  |
| P2   | 2                   | 604     | 253       | 0.1722        | 0.3603                   | 0.2134             | 0.1679 | 0.2588  |
| P2   | 3+                  | 0       | 0         | —             | 0.3603                   | —                  | —      | —       |
| P3   | 1                   | 3167    | 661       | 0.2580        | 0.5739                   | 0.2497             | 0.2336 | 0.2658  |
| P3   | 2                   | 633     | 256       | 0.2243        | 0.5739                   | 0.2657             | 0.2182 | 0.3131  |
| P3   | 3+                  | 0       | 0         | —             | 0.5739                   | —                  | —      | —       |

### Fine-grained POS transitions (exploratory; fixed support threshold)

| pair | transition | n_cross | n_prompts | raw_rejection | exploratory | adjusted_rejection | ci_low | ci_high |
| ---- | ---------- | ------- | --------- | ------------- | ----------- | ------------------ | ------ | ------- |
| P1   | EP→EF      | 149     | 87        | 0.0738        | true        | 0.0990             | 0.0372 | 0.1607  |
| P1   | VA→EF      | 120     | 67        | 0.0750        | true        | 0.1271             | 0.0000 | 0.2564  |
| P1   | VCP→EF     | 389     | 159       | 0.1594        | true        | 0.1576             | 0.1022 | 0.2130  |
| P1   | VX→EF      | 119     | 73        | 0.0756        | true        | 0.0885             | 0.0000 | 0.2278  |
| P1   | XSV→EC     | 406     | 186       | 0.2709        | true        | 0.1852             | 0.1400 | 0.2304  |
| P1   | XSV→ETM    | 377     | 123       | 0.1326        | true        | 0.1763             | 0.0983 | 0.2544  |
| P2   | EP→EF      | 135     | 89        | 0.1556        | true        | 0.2433             | 0.1615 | 0.3252  |
| P2   | JKB→JX     | 172     | 63        | 0.2326        | true        | 0.1189             | 0.0457 | 0.1920  |
| P2   | VCP→EF     | 357     | 165       | 0.1961        | true        | 0.2246             | 0.1504 | 0.2988  |
| P2   | VX→EF      | 149     | 81        | 0.0872        | true        | 0.0393             | 0.0000 | 0.1228  |
| P2   | XSV→EC     | 514     | 245       | 0.2490        | true        | 0.2195             | 0.1605 | 0.2785  |
| P2   | XSV→EF     | 155     | 53        | 0.1806        | true        | 0.1284             | 0.0682 | 0.1886  |
| P2   | XSV→ETM    | 436     | 145       | 0.1101        | true        | 0.2172             | 0.1315 | 0.3029  |
| P3   | EP→EF      | 128     | 81        | 0.1250        | true        | 0.1950             | 0.1062 | 0.2837  |
| P3   | JKB→JX     | 166     | 61        | 0.3072        | true        | 0.1685             | 0.0801 | 0.2568  |
| P3   | VCP→EF     | 363     | 167       | 0.1928        | true        | 0.1998             | 0.1319 | 0.2676  |
| P3   | VX→EF      | 146     | 79        | 0.0890        | true        | 0.0403             | 0.0000 | 0.1295  |
| P3   | XSV→EC     | 521     | 247       | 0.3474        | true        | 0.2914             | 0.2360 | 0.3467  |
| P3   | XSV→EF     | 152     | 54        | 0.3487        | true        | 0.3028             | 0.2226 | 0.3830  |
| P3   | XSV→ETM    | 436     | 148       | 0.1422        | true        | 0.2959             | 0.2149 | 0.3769  |

## Method and limitations

Adjusted probabilities and AMEs use the analyzed covariate distribution and change only morphology class; they are adjusted associations, not causal effects. Confidence intervals and two-sided p-values use the model covariance clustered by prompt and the delta method. Holm adjustment is within each pair across five planned environment contrasts. The eojeol-construction sensitivity uses the broader construction rule and marks empty class-by-environment cells non-estimable. All saved H2 overlapping morphemes matched the reconstructed local eojeol sequence by surface, POS, and span; the number of internal boundaries from the two views differed in 19 P1 and 47 each in P2/P3 CROSS rows, so H2 overlap transitions define primary CROSS counts while the full eojeol sequence defines SPLIT adjacency. Lexical CROSS support is low in all pairs, and Ending→Ending has only 10–33 SPLIT tokens, so those contrasts are imprecise/unstable. All primary model matrices were full rank and converged; H3a and multiboundary fits also converged. Exploratory fine-POS models converged after optimizer fallback, but coefficients and support-threshold estimates remain exploratory.

No speculative decoding or target generation was rerun. Saved teacher-forced values were reused for entropy covariates; H3 did not run teacher-forced scoring.

Large-coefficient warning (|β|>10) appears only in the exploratory fine-POS models: P2 fine_pos_model (max |β|=11.90), P3 fine_pos_model (max |β|=14.12); interpret those sparse transition estimates cautiously.

## Paper-ready interpretation

Across P1, P2, and P3, the adjusted rejection difference at NOMINAL_TO_PARTICLE is +20.9, +18.7, and +24.7 percentage points, with positive pair-specific estimates and strong support after within-pair Holm correction. The pooled and 993-common-prompt analyses preserve the positive nominal-to-particle contrast, while the pooled interaction remains strong. Predicate-to-ending estimates are near zero, and Ending-to-Ending estimates are negative with sparse SPLIT cells, so the evidence supports a specific nominal attachment concentration rather than a uniform increase at all grammatical boundaries. The results describe adjusted associations and do not establish that morphology causes token rejection.

## Pair result details

# P1 H3 results (0.6B → 1.7B)

H3b interaction omnibus p=<0.0001; H3a boundary omnibus p=<0.0001.

## H3b planned environment contrasts

| environment         | n_cross | n_split | n_prompts | adjusted_p_cross | adjusted_p_split | AME_pp_95CI          | p_value | p_holm_within_pair | support_warning | estimable |
| ------------------- | ------- | ------- | --------- | ---------------- | ---------------- | -------------------- | ------- | ------------------ | --------------- | --------- |
| NOMINAL_TO_PARTICLE | 212     | 22211   | 879       | 0.3403           | 0.1311           | +20.9 [+15.4, +26.4] | <0.0001 | <0.0001            | false           | true      |
| PREDICATE_TO_ENDING | 988     | 1221    | 432       | 0.1624           | 0.1916           | -2.9 [-6.4, +0.6]    | 0.1     | 0.1                | false           | true      |
| ENDING_TO_ENDING    | 159     | 10      | 95        | 0.0799           | 0.4440           | -36.4 [-58.7, -14.1] | 0.0014  | 0.0041             | true            | true      |
| LEXICAL_TO_LEXICAL  | 33      | 24373   | 908       | 0.2591           | 0.1337           | +12.5 [+1.4, +23.7]  | 0.028   | 0.056              | true            | true      |
| OTHER               | 1577    | 21732   | 937       | 0.1794           | 0.1145           | +6.5 [+4.4, +8.5]    | <0.0001 | <0.0001            | false           | true      |

## H3a CROSS-only adjusted rejection

| environment         | n_cross | n_prompts | raw_rejection | adjusted_rejection | adjusted_ci_low | adjusted_ci_high | omnibus_boundary_p |
| ------------------- | ------- | --------- | ------------- | ------------------ | --------------- | ---------------- | ------------------ |
| NOMINAL_TO_PARTICLE | 212     | 87        | 0.2642        | 0.3489             | 0.2867          | 0.4111           | <0.0001            |
| PREDICATE_TO_ENDING | 988     | 349       | 0.1680        | 0.1791             | 0.1526          | 0.2056           | <0.0001            |
| ENDING_TO_ENDING    | 159     | 90        | 0.0818        | 0.0970             | 0.0465          | 0.1475           | <0.0001            |
| LEXICAL_TO_LEXICAL  | 33      | 16        | 0.2424        | 0.3035             | 0.1688          | 0.4383           | <0.0001            |
| OTHER               | 1577    | 434       | 0.2289        | 0.2082             | 0.1887          | 0.2277           | <0.0001            |

# P2 H3 results (1.7B → 4B)

H3b interaction omnibus p=<0.0001; H3a boundary omnibus p=0.077.

## H3b planned environment contrasts

| environment         | n_cross | n_split | n_prompts | adjusted_p_cross | adjusted_p_split | AME_pp_95CI          | p_value | p_holm_within_pair | support_warning | estimable |
| ------------------- | ------- | ------- | --------- | ---------------- | ---------------- | -------------------- | ------- | ------------------ | --------------- | --------- |
| NOMINAL_TO_PARTICLE | 212     | 22014   | 905       | 0.3303           | 0.1436           | +18.7 [+12.4, +25.0] | <0.0001 | <0.0001            | false           | true      |
| PREDICATE_TO_ENDING | 958     | 1246    | 452       | 0.1812           | 0.1471           | +3.4 [+0.1, +6.7]    | 0.043   | 0.085              | false           | true      |
| ENDING_TO_ENDING    | 137     | 33      | 104       | 0.1597           | 0.3298           | -17.0 [-32.0, -2.1]  | 0.026   | 0.077              | true            | true      |
| LEXICAL_TO_LEXICAL  | 51      | 25021   | 936       | 0.2012           | 0.1416           | +6.0 [-1.1, +13.0]   | 0.096   | 0.096              | true            | true      |
| OTHER               | 1800    | 21181   | 952       | 0.1954           | 0.1404           | +5.5 [+3.4, +7.6]    | <0.0001 | <0.0001            | false           | true      |

## H3a CROSS-only adjusted rejection

| environment         | n_cross | n_prompts | raw_rejection | adjusted_rejection | adjusted_ci_low | adjusted_ci_high | omnibus_boundary_p |
| ------------------- | ------- | --------- | ------------- | ------------------ | --------------- | ---------------- | ------------------ |
| NOMINAL_TO_PARTICLE | 212     | 97        | 0.2406        | 0.2972             | 0.2098          | 0.3846           | 0.077              |
| PREDICATE_TO_ENDING | 958     | 345       | 0.1754        | 0.2018             | 0.1719          | 0.2318           | 0.077              |
| ENDING_TO_ENDING    | 137     | 91        | 0.1606        | 0.2066             | 0.1427          | 0.2705           | 0.077              |
| LEXICAL_TO_LEXICAL  | 51      | 27        | 0.1961        | 0.2054             | 0.1123          | 0.2985           | 0.077              |
| OTHER               | 1800    | 514       | 0.2067        | 0.1839             | 0.1662          | 0.2015           | 0.077              |

# P3 H3 results (0.6B → 4B)

H3b interaction omnibus p=<0.0001; H3a boundary omnibus p=<0.0001.

## H3b planned environment contrasts

| environment         | n_cross | n_split | n_prompts | adjusted_p_cross | adjusted_p_split | AME_pp_95CI          | p_value | p_holm_within_pair | support_warning | estimable |
| ------------------- | ------- | ------- | --------- | ---------------- | ---------------- | -------------------- | ------- | ------------------ | --------------- | --------- |
| NOMINAL_TO_PARTICLE | 218     | 21829   | 907       | 0.4323           | 0.1852           | +24.7 [+18.4, +31.0] | <0.0001 | <0.0001            | false           | true      |
| PREDICATE_TO_ENDING | 978     | 1271    | 454       | 0.1978           | 0.2206           | -2.3 [-5.7, +1.1]    | 0.19    | 0.33               | false           | true      |
| ENDING_TO_ENDING    | 130     | 30      | 96        | 0.1316           | 0.5009           | -36.9 [-50.9, -23.0] | <0.0001 | <0.0001            | true            | true      |
| LEXICAL_TO_LEXICAL  | 47      | 24849   | 935       | 0.2471           | 0.1812           | +6.6 [-2.7, +15.9]   | 0.17    | 0.33               | true            | true      |
| OTHER               | 1794    | 21364   | 954       | 0.2442           | 0.1817           | +6.3 [+4.1, +8.4]    | <0.0001 | <0.0001            | false           | true      |

## H3a CROSS-only adjusted rejection

| environment         | n_cross | n_prompts | raw_rejection | adjusted_rejection | adjusted_ci_low | adjusted_ci_high | omnibus_boundary_p |
| ------------------- | ------- | --------- | ------------- | ------------------ | --------------- | ---------------- | ------------------ |
| NOMINAL_TO_PARTICLE | 218     | 98        | 0.3761        | 0.4295             | 0.3504          | 0.5086           | <0.0001            |
| PREDICATE_TO_ENDING | 978     | 349       | 0.2004        | 0.2169             | 0.1897          | 0.2441           | <0.0001            |
| ENDING_TO_ENDING    | 130     | 83        | 0.1385        | 0.1823             | 0.1141          | 0.2505           | <0.0001            |
| LEXICAL_TO_LEXICAL  | 47      | 27        | 0.2979        | 0.3185             | 0.2107          | 0.4263           | <0.0001            |
| OTHER               | 1794    | 519       | 0.2826        | 0.2600             | 0.2403          | 0.2797           | <0.0001            |
