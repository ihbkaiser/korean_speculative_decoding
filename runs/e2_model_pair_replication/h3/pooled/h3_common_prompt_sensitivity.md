# Common-prompt H3 sensitivity

The pooled H3 model used the exact three-way eligible-prompt intersection of 993 IDs from the E2 overlap audit. The interaction model was fit on the ordinary pooled eligible population and on the restricted common-prompt rows; both use identical formulas and prompt-clustered covariance.

Classification: **minor numerical change** (predefined thresholds: unchanged if maximum absolute environment AME shift ≤0.5 pp; minor if >0.5 and ≤1.5 pp; material if >1.5 pp or any sign reversal).

All eligible prompts: N=217,679 rows, 999 clusters; interaction omnibus p=9.536399409507805e-45.
Common 993: N=216,839 rows, 993 clusters; interaction omnibus p=4.1103025243820895e-44.

| environment         | difference_pp_common_minus_all | all_p   | common_p | all_prompt_AME_pp | common_993_AME_pp |
| ------------------- | ------------------------------ | ------- | -------- | ----------------- | ----------------- |
| NOMINAL_TO_PARTICLE | -0.9522                        | <0.0001 | <0.0001  | 21.5572           | 20.6050           |
| PREDICATE_TO_ENDING | 0.0401                         | 0.66    | 0.69     | -0.5139           | -0.4737           |
| ENDING_TO_ENDING    | 0.0494                         | <0.0001 | <0.0001  | -29.3598          | -29.3104          |
| LEXICAL_TO_LEXICAL  | 0.0438                         | 0.0086  | 0.0084   | 7.4415            | 7.4854            |
| OTHER               | 0.0497                         | <0.0001 | <0.0001  | 6.1018            | 6.1515            |

These pooled estimates are secondary and do not replace pair-specific replication.
