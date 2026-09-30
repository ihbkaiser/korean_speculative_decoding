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
