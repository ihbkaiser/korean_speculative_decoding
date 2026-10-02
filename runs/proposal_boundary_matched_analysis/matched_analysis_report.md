# Prompt- and rank-matched N→P proposal analysis

## Experiment card

- **Question:** Among actual Korean draft proposals near nominal→particle structure, does an N→P boundary-crossing proposal have a higher target-rejection rate than a proposal that stays within one morpheme?
- **Hypothesis:** N→P CROSS proposals retain higher rejection after matching on prompt, model pair, exact proposal slot and exact candidate-token character length, then minimizing observed difficulty/position/frequency differences.
- **Baseline / variant:** WITHIN_SPLIT proposal near an N→P boundary / valid one-boundary N→P CROSS proposal.
- **Primary metric:** Matched difference in rejection probability (CROSS minus WITHIN_SPLIT), with prompt-cluster bootstrap 95% CI.
- **Decision rule:** Direction is consistent with the hypothesis only if both dataset CIs are above zero, both P1/P2 point estimates are positive within each dataset, and every primary covariate has absolute SMD ≤ 0.10. This is evidence of a robust conditional association, not a morphology-only causal effect.
- **Data / split:** Saved actual greedy SD proposals for Wiki and English→Korean FLORES, P1/P2; proposals after the first rejection are already excluded by the validated source population.
- **Cheapest rung / budget:** CPU reanalysis only; no new generation or GPU inference.
- **Exploratory-only:** The slot-only sensitivity match and any claim that morphology itself causes rejection.

## Matching design

Primary matches are 1:1 without replacement within workload × model pair × prompt × proposal slot × exact candidate-token character length. Within each exact stratum, the Hungarian minimum-distance assignment uses standardized draft/target entropy, target fragmentation (as a mismatch penalty), relative and generation position, eojeol character length, candidate-token frequency, and first/last-token flags. A strict-fragment sensitivity additionally matches target fragmentation exactly; a broader sensitivity omits exact length and fragmentation matching. Outcomes are not used in matching. Confidence intervals resample prompts as clusters.

Eligible rows: 1,374 CROSS and 36,428 WITHIN_SPLIT. Primary matched pairs: 73; control reuse: 0. Strict-fragment matched pairs: 336. Broader sensitivity matched pairs: 952; control reuse: 0.

## Result

The primary same-length/same-slot match estimates +6.8 pp (95% CI -10.0 to +25.6) on Wiki and +10.3 pp (95% CI -12.0 to +34.6) on FLORES, from 44 and 29 matched proposal pairs. Maximum primary absolute SMDs are 2.81 on Wiki and 2.02 on FLORES. The prespecified criterion is not met: both primary CIs include zero and covariate balance remains inadequate.

## Matched rejection estimates

| analysis | workload | pair | n_matched | n_prompts | cross_reject_rate | within_reject_rate | risk_difference | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| PRIMARY_SAME_LENGTH_SLOT | WIKIPEDIA | P1 | 19 | 14 | 0.3684 | 0.1053 | 0.2632 | -0.0455 | 0.6250 |
| PRIMARY_SAME_LENGTH_SLOT | WIKIPEDIA | P2 | 25 | 17 | 0.1600 | 0.2400 | -0.0800 | -0.2609 | 0.1154 |
| PRIMARY_SAME_LENGTH_SLOT | WIKIPEDIA | P1+P2 | 44 | 30 | 0.2500 | 0.1818 | 0.0682 | -0.1000 | 0.2564 |
| PRIMARY_SAME_LENGTH_SLOT | FLORES | P1 | 13 | 10 | 0.2308 | 0.1538 | 0.0769 | -0.3000 | 0.5000 |
| PRIMARY_SAME_LENGTH_SLOT | FLORES | P2 | 16 | 16 | 0.3750 | 0.2500 | 0.1250 | -0.1875 | 0.4375 |
| PRIMARY_SAME_LENGTH_SLOT | FLORES | P1+P2 | 29 | 25 | 0.3103 | 0.2069 | 0.1034 | -0.1200 | 0.3462 |
| STRICT_FRAGMENT_BIN | WIKIPEDIA | P1 | 95 | 55 | 0.2316 | 0.2632 | -0.0316 | -0.1573 | 0.1059 |
| STRICT_FRAGMENT_BIN | WIKIPEDIA | P2 | 87 | 60 | 0.3103 | 0.2069 | 0.1034 | -0.0238 | 0.2360 |
| STRICT_FRAGMENT_BIN | WIKIPEDIA | P1+P2 | 182 | 99 | 0.2692 | 0.2363 | 0.0330 | -0.0517 | 0.1226 |
| STRICT_FRAGMENT_BIN | FLORES | P1 | 66 | 56 | 0.4697 | 0.4394 | 0.0303 | -0.1343 | 0.2063 |
| STRICT_FRAGMENT_BIN | FLORES | P2 | 88 | 84 | 0.5909 | 0.3068 | 0.2841 | 0.1176 | 0.4388 |
| STRICT_FRAGMENT_BIN | FLORES | P1+P2 | 154 | 129 | 0.5390 | 0.3636 | 0.1753 | 0.0600 | 0.2914 |
| SENSITIVITY_SLOT_ONLY | WIKIPEDIA | P1 | 227 | 119 | 0.2026 | 0.1586 | 0.0441 | -0.0345 | 0.1306 |
| SENSITIVITY_SLOT_ONLY | WIKIPEDIA | P2 | 242 | 141 | 0.2149 | 0.1322 | 0.0826 | 0.0202 | 0.1480 |
| SENSITIVITY_SLOT_ONLY | WIKIPEDIA | P1+P2 | 469 | 216 | 0.2090 | 0.1450 | 0.0640 | 0.0154 | 0.1171 |
| SENSITIVITY_SLOT_ONLY | FLORES | P1 | 235 | 181 | 0.3872 | 0.3362 | 0.0511 | -0.0282 | 0.1319 |
| SENSITIVITY_SLOT_ONLY | FLORES | P2 | 248 | 211 | 0.4234 | 0.2298 | 0.1935 | 0.1070 | 0.2776 |
| SENSITIVITY_SLOT_ONLY | FLORES | P1+P2 | 483 | 322 | 0.4058 | 0.2816 | 0.1242 | 0.0639 | 0.1872 |

## Primary covariate balance

| analysis | workload | covariate | cross_mean | within_mean | smd | abs_smd |
| --- | --- | --- | --- | --- | --- | --- |
| PRIMARY_SAME_LENGTH_SLOT | WIKIPEDIA | draft_entropy | 1.1645 | 1.4283 | -0.1794 | 0.1794 |
| PRIMARY_SAME_LENGTH_SLOT | WIKIPEDIA | target_entropy | 0.9193 | 1.4600 | -0.3984 | 0.3984 |
| PRIMARY_SAME_LENGTH_SLOT | WIKIPEDIA | relative_position | 0.5871 | 0.4080 | 0.6128 | 0.6128 |
| PRIMARY_SAME_LENGTH_SLOT | WIKIPEDIA | generation_position | 59.3409 | 60.2273 | -0.0241 | 0.0241 |
| PRIMARY_SAME_LENGTH_SLOT | WIKIPEDIA | candidate_token_char_length | 2.0227 | 2.0227 | 0.0000 | 0.0000 |
| PRIMARY_SAME_LENGTH_SLOT | WIKIPEDIA | candidate_eojeol_char_length | 2.9091 | 5.2500 | -2.0649 | 2.0649 |
| PRIMARY_SAME_LENGTH_SLOT | WIKIPEDIA | log_candidate_token_count | 6.4839 | 7.3794 | -0.7426 | 0.7426 |
| PRIMARY_SAME_LENGTH_SLOT | WIKIPEDIA | first_token | 0.3636 | 0.2955 | 0.1655 | 0.1655 |
| PRIMARY_SAME_LENGTH_SLOT | WIKIPEDIA | last_token | 0.7727 | 0.0682 | 2.8087 | 2.8087 |
| PRIMARY_SAME_LENGTH_SLOT | FLORES | draft_entropy | 1.4947 | 1.6452 | -0.1121 | 0.1121 |
| PRIMARY_SAME_LENGTH_SLOT | FLORES | target_entropy | 1.1260 | 1.5091 | -0.3270 | 0.3270 |
| PRIMARY_SAME_LENGTH_SLOT | FLORES | relative_position | 0.4874 | 0.2971 | 0.6081 | 0.6081 |
| PRIMARY_SAME_LENGTH_SLOT | FLORES | generation_position | 37.3448 | 28.6897 | 0.3745 | 0.3745 |
| PRIMARY_SAME_LENGTH_SLOT | FLORES | candidate_token_char_length | 2.0000 | 2.0000 | 0.0000 | 0.0000 |
| PRIMARY_SAME_LENGTH_SLOT | FLORES | candidate_eojeol_char_length | 2.7931 | 4.8966 | -2.0247 | 2.0247 |
| PRIMARY_SAME_LENGTH_SLOT | FLORES | log_candidate_token_count | 6.3708 | 6.3838 | -0.0101 | 0.0101 |
| PRIMARY_SAME_LENGTH_SLOT | FLORES | first_token | 0.4483 | 0.4483 | 0.0000 | 0.0000 |
| PRIMARY_SAME_LENGTH_SLOT | FLORES | last_token | 0.7586 | 0.1034 | 1.9781 | 1.9781 |

## Interpretation limits

Matching controls observed context, proposal rank, target fragmentation, and measured proposal difficulty. Draft candidate identity remains different between the matched proposals, and unmeasured token-level/contextual features may remain. Therefore this analysis can test whether the earlier association survives a stricter comparison, but it cannot establish that crossing a morpheme boundary itself causes rejection. The separate proposal-suppression intervention estimates a policy effect and should not be interpreted as this token-level causal effect.
