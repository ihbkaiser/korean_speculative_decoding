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
