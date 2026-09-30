# FLORES population audit

Every non-empty official English–Korean devtest row was retained. The Korean reference was not used for prompt construction, decoding, alignment, inclusion, or labels.

| pair | prompts | generated_tokens | sd_valid_output_tokens | sd_invalid_output_tokens | invalidated_sd_proposals | target_sd_exact_prompt_count | target_sd_mismatch_prompt_count | fallback_target_prompts | eligible_h2_tokens | eligible_h2_prompts | cross_morpheme | within_split | cross_eojeol_excluded | kiwi_complex_excluded | exact | teacher_forced_rows | retokenization_exact_prompts | retokenization_fallback_prompts | visible_non_special_tokens | exact_span_rows | alignment_success_percent_visible | alignment_failures | cross_kiwi_exclusion_percent | eligible_cross_split_tokens_in_h3 | eligible_cross_split_prompts_in_h3 | n_to_p_cross | n_to_p_split |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| P1 | 1012.000 | 50454.000 | 50454.000 | 0.000 | 32910.000 | 1012.000 | 0.000 | 0.000 | 26828.000 | 1010.000 | 3111.000 | 24472.000 | 0.000 | 4225.000 | 18646.000 | 50454.000 | 967.000 | 45.000 | 49508.000 | 49411.000 | 99.804 | 97.000 | 8.374 | 26828.000 | 1010.000 | 239.000 | 9322.000 |
| P2 | 1012.000 | 46523.000 | 46523.000 | 0.000 | 19850.000 | 1012.000 | 0.000 | 0.000 | 23901.000 | 1012.000 | 2752.000 | 21764.000 | 0.000 | 3870.000 | 18137.000 | 46523.000 | 994.000 | 18.000 | 45532.000 | 45493.000 | 99.914 | 39.000 | 8.318 | 23901.000 | 1012.000 | 176.000 | 7902.000 |
| P3 | 1012.000 | 46523.000 | 46523.000 | 0.000 | 30430.000 | 1012.000 | 0.000 | 0.000 | 23901.000 | 1012.000 | 2752.000 | 21764.000 | 0.000 | 3870.000 | 18137.000 | 46523.000 | 994.000 | 18.000 | 45532.000 | 45493.000 | 99.914 | 39.000 | 8.318 | 23901.000 | 1012.000 | 176.000 | 7902.000 |

`sd_invalid_output_tokens` counts output positions with no valid SD proposal decision; invalidated later proposals are counted separately in `invalidated_sd_proposals`. Exact target/SD continuation mismatch prompts must be zero. Morphology exclusion rates and class counts are in `population_audit.csv`; detailed spans are in the H2-compatible `token_tables/` files.

## Generated-language audit

| pair | outputs | predominantly_korean | hangul_share_mean | hangul_share_median | predominantly_korean_percent |
| --- | --- | --- | --- | --- | --- |
| P1 | 1012.000 | 973.000 | 0.944 | 1.000 | 96.146 |
| P2 | 1012.000 | 858.000 | 0.828 | 1.000 | 84.783 |
| P3 | 1012.000 | 858.000 | 0.828 | 1.000 | 84.783 |

Predominantly Korean uses the frozen descriptive threshold of at least 50% Hangul among Unicode letters. No output was excluded on this measure or on translation quality.
