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
