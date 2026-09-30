# H3 boundary taxonomy audit

The eligibility and H2 class definitions are unchanged. H3a/H3b require single-boundary CROSS tokens; CROSS rows with zero reconstructed overlap transitions are flagged as unresolved, while multi-boundary rows are retained for exploratory analysis.

## Cross-boundary reconstruction

| pair   | all_cross_tokens | single_boundary_cross | multi_boundary_cross | cross_boundary_unresolved | percent_single_of_cross |
| ------ | ---------------- | --------------------- | -------------------- | ------------------------- | ----------------------- |
| P1     | 2356             | 2275                  | 81                   | 0                         | 96.56                   |
| P2     | 2138             | 2052                  | 86                   | 0                         | 95.98                   |
| P3     | 2138             | 2052                  | 86                   | 0                         | 95.98                   |
| POOLED | 6632             | 6379                  | 253                  | 0                         | 96.19                   |

## Environment support

| pair   | environment         | n_cross | n_cross_prompts | percent_of_single_boundary_cross | raw_rejection | median_fragmentation | median_token_frequency | mean_draft_entropy | support_warning |
| ------ | ------------------- | ------- | --------------- | -------------------------------- | ------------- | -------------------- | ---------------------- | ------------------ | --------------- |
| P1     | NOMINAL_TO_PARTICLE | 239     | 172             | 10.505                           | 0.544         | 2.000                | 572.000                | 1.251              | false           |
| P1     | PREDICATE_TO_ENDING | 628     | 439             | 27.604                           | 0.390         | 2.000                | 439.000                | 1.581              | false           |
| P1     | ENDING_TO_ENDING    | 70      | 70              | 3.077                            | 0.529         | 3.000                | 79.000                 | 1.452              | true            |
| P1     | LEXICAL_TO_LEXICAL  | 75      | 55              | 3.297                            | 0.253         | 2.000                | 11.000                 | 1.138              | true            |
| P1     | OTHER               | 1263    | 653             | 55.516                           | 0.372         | 3.000                | 1896.000               | 1.518              | false           |
| P2     | NOMINAL_TO_PARTICLE | 176     | 148             | 8.577                            | 0.375         | 2.000                | 536.000                | 0.928              | false           |
| P2     | PREDICATE_TO_ENDING | 494     | 383             | 24.074                           | 0.261         | 2.000                | 367.000                | 1.206              | false           |
| P2     | ENDING_TO_ENDING    | 51      | 48              | 2.485                            | 0.275         | 3.000                | 10.000                 | 0.990              | true            |
| P2     | LEXICAL_TO_LEXICAL  | 139     | 125             | 6.774                            | 0.094         | 2.000                | 1.000                  | 0.445              | false           |
| P2     | OTHER               | 1192    | 629             | 58.090                           | 0.291         | 3.000                | 1896.000               | 1.157              | false           |
| P3     | NOMINAL_TO_PARTICLE | 176     | 148             | 8.577                            | 0.648         | 2.000                | 536.000                | 1.292              | false           |
| P3     | PREDICATE_TO_ENDING | 494     | 383             | 24.074                           | 0.352         | 2.000                | 367.000                | 1.511              | false           |
| P3     | ENDING_TO_ENDING    | 51      | 48              | 2.485                            | 0.373         | 3.000                | 10.000                 | 1.222              | true            |
| P3     | LEXICAL_TO_LEXICAL  | 139     | 125             | 6.774                            | 0.180         | 2.000                | 1.000                  | 0.564              | false           |
| P3     | OTHER               | 1192    | 629             | 58.090                           | 0.393         | 3.000                | 1896.000               | 1.484              | false           |
| POOLED | NOMINAL_TO_PARTICLE | 591     | 246             | 9.265                            | 0.525         | 2.000                | 567.000                | 1.167              | false           |
| POOLED | PREDICATE_TO_ENDING | 1616    | 556             | 25.333                           | 0.339         | 2.000                | 372.000                | 1.445              | false           |
| POOLED | ENDING_TO_ENDING    | 172     | 93              | 2.696                            | 0.407         | 3.000                | 10.000                 | 1.247              | false           |
| POOLED | LEXICAL_TO_LEXICAL  | 353     | 162             | 5.534                            | 0.161         | 2.000                | 1.000                  | 0.639              | false           |
| POOLED | OTHER               | 3647    | 784             | 57.172                           | 0.352         | 3.000                | 1896.000               | 1.389              | false           |

Categories with fewer than 100 CROSS tokens or 50 CROSS prompts are flagged and retained without merging. SPLIT support is audited separately in the H3B_ALL_CLASSES rows of h3_boundary_counts.csv.

## Frequent fine-POS transitions (top five per taxonomy class)

| pair | crossed_fine_pos_sequence | n_cross | n_prompts | raw_rejection | support_threshold_met | coarse_boundary_type |
| ---- | ------------------------- | ------- | --------- | ------------- | --------------------- | -------------------- |
| P1   | EP→EF                     | 61      | 61        | 0.475         | false                 | ENDING_TO_ENDING     |
| P1   | EP→ETM                    | 5       | 5         | 0.800         | false                 | ENDING_TO_ENDING     |
| P1   | EP→EC                     | 4       | 4         | 1.000         | false                 | ENDING_TO_ENDING     |
| P1   | SL→SL                     | 43      | 25        | 0.023         | false                 | LEXICAL_TO_LEXICAL   |
| P1   | NNG→XSN                   | 11      | 11        | 0.727         | false                 | LEXICAL_TO_LEXICAL   |
| P1   | XSN→VCP                   | 9       | 9         | 0.556         | false                 | LEXICAL_TO_LEXICAL   |
| P1   | NNB→VCP                   | 4       | 4         | 0.500         | false                 | LEXICAL_TO_LEXICAL   |
| P1   | NNG→NNG                   | 4       | 3         | 0.250         | false                 | LEXICAL_TO_LEXICAL   |
| P1   | NNG→JKO                   | 112     | 77        | 0.473         | true                  | NOMINAL_TO_PARTICLE  |
| P1   | NNG→JKS                   | 43      | 40        | 0.651         | false                 | NOMINAL_TO_PARTICLE  |
| P1   | NNG→JX                    | 32      | 31        | 0.719         | false                 | NOMINAL_TO_PARTICLE  |
| P1   | NNB→JKB                   | 24      | 19        | 0.250         | false                 | NOMINAL_TO_PARTICLE  |
| P1   | NNG→JKB                   | 9       | 9         | 0.778         | false                 | NOMINAL_TO_PARTICLE  |
| P1   | XSV→EC                    | 288     | 227       | 0.378         | true                  | OTHER                |
| P1   | XSV→ETM                   | 209     | 161       | 0.383         | true                  | OTHER                |
| P1   | XSN→JX                    | 127     | 109       | 0.268         | true                  | OTHER                |
| P1   | XSN→JKS                   | 92      | 82        | 0.467         | false                 | OTHER                |
| P1   | XSN→JKB                   | 89      | 79        | 0.202         | false                 | OTHER                |
| P1   | VA→EF                     | 115     | 103       | 0.270         | true                  | PREDICATE_TO_ENDING  |
| P1   | VX→EF                     | 94      | 87        | 0.319         | false                 | PREDICATE_TO_ENDING  |
| P1   | VV→EC                     | 81      | 58        | 0.358         | false                 | PREDICATE_TO_ENDING  |
| P1   | VCP→EC                    | 61      | 56        | 0.525         | false                 | PREDICATE_TO_ENDING  |
| P1   | VX→EC                     | 59      | 54        | 0.492         | false                 | PREDICATE_TO_ENDING  |
| P2   | EP→EF                     | 44      | 43        | 0.273         | false                 | ENDING_TO_ENDING     |
| P2   | EP→EC                     | 5       | 5         | 0.000         | false                 | ENDING_TO_ENDING     |
| P2   | EP→ETM                    | 2       | 2         | 1.000         | false                 | ENDING_TO_ENDING     |
| P2   | SL→SL                     | 101     | 90        | 0.010         | true                  | LEXICAL_TO_LEXICAL   |
| P2   | XSN→VCP                   | 15      | 15        | 0.400         | false                 | LEXICAL_TO_LEXICAL   |
| P2   | NNG→XSN                   | 9       | 9         | 0.333         | false                 | LEXICAL_TO_LEXICAL   |
| P2   | NNB→VCP                   | 6       | 3         | 0.500         | false                 | LEXICAL_TO_LEXICAL   |
| P2   | NNB→XSN                   | 3       | 3         | 0.000         | false                 | LEXICAL_TO_LEXICAL   |
| P2   | NNG→JKO                   | 75      | 63        | 0.400         | false                 | NOMINAL_TO_PARTICLE  |
| P2   | NNG→JX                    | 38      | 34        | 0.368         | false                 | NOMINAL_TO_PARTICLE  |
| P2   | NNG→JKS                   | 29      | 28        | 0.207         | false                 | NOMINAL_TO_PARTICLE  |
| P2   | NNB→JKB                   | 8       | 8         | 0.125         | false                 | NOMINAL_TO_PARTICLE  |
| P2   | NNG→JKG                   | 7       | 7         | 0.429         | false                 | NOMINAL_TO_PARTICLE  |
| P2   | XSV→EC                    | 251     | 210       | 0.327         | true                  | OTHER                |
| P2   | XSV→ETM                   | 155     | 134       | 0.271         | true                  | OTHER                |
| P2   | XSN→JX                    | 121     | 116       | 0.289         | true                  | OTHER                |
| P2   | XSN→JKB                   | 98      | 93        | 0.194         | false                 | OTHER                |
| P2   | XSN→JKS                   | 87      | 84        | 0.356         | false                 | OTHER                |
| P2   | VA→EF                     | 106     | 99        | 0.075         | true                  | PREDICATE_TO_ENDING  |
| P2   | VX→EF                     | 82      | 79        | 0.280         | false                 | PREDICATE_TO_ENDING  |
| P2   | VV→EC                     | 73      | 68        | 0.315         | false                 | PREDICATE_TO_ENDING  |
| P2   | VX→EC                     | 48      | 46        | 0.417         | false                 | PREDICATE_TO_ENDING  |
| P2   | VCP→EC                    | 35      | 35        | 0.314         | false                 | PREDICATE_TO_ENDING  |
| P3   | EP→EF                     | 44      | 43        | 0.364         | false                 | ENDING_TO_ENDING     |
| P3   | EP→EC                     | 5       | 5         | 0.400         | false                 | ENDING_TO_ENDING     |
| P3   | EP→ETM                    | 2       | 2         | 0.500         | false                 | ENDING_TO_ENDING     |
| P3   | SL→SL                     | 101     | 90        | 0.010         | true                  | LEXICAL_TO_LEXICAL   |
| P3   | XSN→VCP                   | 15      | 15        | 0.733         | false                 | LEXICAL_TO_LEXICAL   |
| P3   | NNG→XSN                   | 9       | 9         | 0.778         | false                 | LEXICAL_TO_LEXICAL   |
| P3   | NNB→VCP                   | 6       | 3         | 0.500         | false                 | LEXICAL_TO_LEXICAL   |
| P3   | NNB→XSN                   | 3       | 3         | 0.333         | false                 | LEXICAL_TO_LEXICAL   |
| P3   | NNG→JKO                   | 75      | 63        | 0.640         | false                 | NOMINAL_TO_PARTICLE  |
| P3   | NNG→JX                    | 38      | 34        | 0.684         | false                 | NOMINAL_TO_PARTICLE  |
| P3   | NNG→JKS                   | 29      | 28        | 0.655         | false                 | NOMINAL_TO_PARTICLE  |
| P3   | NNB→JKB                   | 8       | 8         | 0.125         | false                 | NOMINAL_TO_PARTICLE  |
| P3   | NNG→JKG                   | 7       | 7         | 0.857         | false                 | NOMINAL_TO_PARTICLE  |
| P3   | XSV→EC                    | 251     | 210       | 0.414         | true                  | OTHER                |
| P3   | XSV→ETM                   | 155     | 134       | 0.400         | true                  | OTHER                |
| P3   | XSN→JX                    | 121     | 116       | 0.322         | true                  | OTHER                |
| P3   | XSN→JKB                   | 98      | 93        | 0.235         | false                 | OTHER                |
| P3   | XSN→JKS                   | 87      | 84        | 0.483         | false                 | OTHER                |
| P3   | VA→EF                     | 106     | 99        | 0.198         | true                  | PREDICATE_TO_ENDING  |
| P3   | VX→EF                     | 82      | 79        | 0.378         | false                 | PREDICATE_TO_ENDING  |
| P3   | VV→EC                     | 73      | 68        | 0.425         | false                 | PREDICATE_TO_ENDING  |
| P3   | VX→EC                     | 48      | 46        | 0.479         | false                 | PREDICATE_TO_ENDING  |
| P3   | VCP→EC                    | 35      | 35        | 0.514         | false                 | PREDICATE_TO_ENDING  |
