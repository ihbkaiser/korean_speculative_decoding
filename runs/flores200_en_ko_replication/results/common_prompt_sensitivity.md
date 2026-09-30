# FLORES common-prompt sensitivity

The exact three-way intersection is **1,010 prompt IDs** from the eligible H3 tables (see `audits/common_prompt_intersection.json`). The pooled H3 model was refit on that intersection with the identical formula and prompt-clustered covariance.
Nominal→Particle pooled AME shift: +0.01 pp maximum absolute shift; sign reversal: False; classification: **unchanged**.

| environment | ame_all | ame_ci_low_all | ame_ci_high_all | ame_common | ame_ci_low_common | ame_ci_high_common | ame_change_pp |
| --- | --- | --- | --- | --- | --- | --- | --- |
| NOMINAL_TO_PARTICLE | 0.2556 | 0.2171 | 0.2941 | 0.2555 | 0.2170 | 0.2940 | -0.0075 |

Pair-specific estimates are primary. This pooled common-prompt refit is sensitivity analysis only.
