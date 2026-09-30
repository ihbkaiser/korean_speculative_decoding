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
