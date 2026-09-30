# H4 geometry audit

Exact token/morpheme character intersections define coverage. Categories use the mutually exclusive precedence in implementation_h4.md. No geometry types were merged after outcome inspection.

| pair | geometry_class                    | n_cross | n_prompts | raw_rejection | mean_draft_entropy | median_token_frequency | nominal_fraction_mean | nominal_fraction_median | particle_fraction_mean | particle_fraction_median | nominal_chars_mean | nominal_chars_median | nominal_chars_q25 | nominal_chars_q75 | particle_chars_mean | particle_chars_median | particle_chars_q25 | particle_chars_q75 | share_of_pair_cross |
| ---- | --------------------------------- | ------- | --------- | ------------- | ------------------ | ---------------------- | --------------------- | ----------------------- | ---------------------- | ------------------------ | ------------------ | -------------------- | ----------------- | ----------------- | ------------------- | --------------------- | ------------------ | ------------------ | ------------------- |
| P1   | FULL_NOMINAL_PLUS_FULL_PARTICLE   | 21      | 14        | 0.524         | 1.559              | 995.000                | 1.000                 | 1.000                   | 1.000                  | 1.000                    | 1.524              | 2.000                | 1.000             | 2.000             | 1.190               | 1.000                 | 1.000              | 1.000              | 0.100               |
| P1   | NOMINAL_SUFFIX_PLUS_FULL_PARTICLE | 190     | 76        | 0.237         | 0.825              | 584.000                | 0.450                 | 0.500                   | 1.000                  | 1.000                    | 1.000              | 1.000                | 1.000             | 1.000             | 1.032               | 1.000                 | 1.000              | 1.000              | 0.900               |
| P2   | FULL_NOMINAL_PLUS_FULL_PARTICLE   | 8       | 7         | 0.125         | 1.715              | 718.000                | 1.000                 | 1.000                   | 1.000                  | 1.000                    | 1.250              | 1.000                | 1.000             | 1.250             | 1.375               | 1.000                 | 1.000              | 2.000              | 0.041               |
| P2   | NOMINAL_SUFFIX_PLUS_FULL_PARTICLE | 189     | 83        | 0.228         | 0.876              | 593.000                | 0.425                 | 0.500                   | 1.000                  | 1.000                    | 1.000              | 1.000                | 1.000             | 1.000             | 1.005               | 1.000                 | 1.000              | 1.000              | 0.959               |
| P3   | FULL_NOMINAL_PLUS_FULL_PARTICLE   | 7       | 6         | 0.571         | 2.707              | 718.000                | 1.000                 | 1.000                   | 1.000                  | 1.000                    | 1.143              | 1.000                | 1.000             | 1.000             | 1.429               | 1.000                 | 1.000              | 2.000              | 0.035               |
| P3   | NOMINAL_SUFFIX_PLUS_FULL_PARTICLE | 194     | 85        | 0.345         | 0.989              | 578.000                | 0.427                 | 0.500                   | 1.000                  | 1.000                    | 1.000              | 1.000                | 1.000             | 1.000             | 1.005               | 1.000                 | 1.000              | 1.000              | 0.965               |

## Continuous distributions by pair

Distributions below use CROSS rows only.

| pair | metric                    | n_cross | mean  | sd    | min   | q25   | median | q75   | max   |
| ---- | ------------------------- | ------- | ----- | ----- | ----- | ----- | ------ | ----- | ----- |
| P1   | nominal_chars_in_token    | 211     | 1.052 | 0.223 | 1.000 | 1.000 | 1.000  | 1.000 | 2.000 |
| P1   | particle_chars_in_token   | 211     | 1.047 | 0.213 | 1.000 | 1.000 | 1.000  | 1.000 | 2.000 |
| P1   | nominal_total_chars       | 211     | 2.237 | 0.570 | 1.000 | 2.000 | 2.000  | 3.000 | 4.000 |
| P1   | particle_total_chars      | 211     | 1.047 | 0.213 | 1.000 | 1.000 | 1.000  | 1.000 | 2.000 |
| P1   | nominal_fraction_covered  | 211     | 0.504 | 0.182 | 0.250 | 0.333 | 0.500  | 0.500 | 1.000 |
| P1   | particle_fraction_covered | 211     | 1.000 | 0.000 | 1.000 | 1.000 | 1.000  | 1.000 | 1.000 |
| P2   | nominal_chars_in_token    | 197     | 1.010 | 0.101 | 1.000 | 1.000 | 1.000  | 1.000 | 2.000 |
| P2   | particle_chars_in_token   | 197     | 1.020 | 0.141 | 1.000 | 1.000 | 1.000  | 1.000 | 2.000 |
| P2   | nominal_total_chars       | 197     | 2.411 | 0.596 | 1.000 | 2.000 | 2.000  | 3.000 | 4.000 |
| P2   | particle_total_chars      | 197     | 1.020 | 0.141 | 1.000 | 1.000 | 1.000  | 1.000 | 2.000 |
| P2   | nominal_fraction_covered  | 197     | 0.449 | 0.142 | 0.250 | 0.333 | 0.500  | 0.500 | 1.000 |
| P2   | particle_fraction_covered | 197     | 1.000 | 0.000 | 1.000 | 1.000 | 1.000  | 1.000 | 1.000 |
| P3   | nominal_chars_in_token    | 201     | 1.005 | 0.071 | 1.000 | 1.000 | 1.000  | 1.000 | 2.000 |
| P3   | particle_chars_in_token   | 201     | 1.020 | 0.140 | 1.000 | 1.000 | 1.000  | 1.000 | 2.000 |
| P3   | nominal_total_chars       | 201     | 2.408 | 0.594 | 1.000 | 2.000 | 2.000  | 3.000 | 4.000 |
| P3   | particle_total_chars      | 201     | 1.020 | 0.140 | 1.000 | 1.000 | 1.000  | 1.000 | 2.000 |
| P3   | nominal_fraction_covered  | 201     | 0.447 | 0.135 | 0.250 | 0.333 | 0.500  | 0.500 | 1.000 |
| P3   | particle_fraction_covered | 201     | 1.000 | 0.000 | 1.000 | 1.000 | 1.000  | 1.000 | 1.000 |

## Category counts and rates

| pair | geometry_class                    | n_cross | n_prompts | raw_rejection | mean_draft_entropy | median_token_frequency | nominal_fraction_mean | nominal_fraction_median | particle_fraction_mean | particle_fraction_median | nominal_chars_mean | nominal_chars_median | nominal_chars_q25 | nominal_chars_q75 | particle_chars_mean | particle_chars_median | particle_chars_q25 | particle_chars_q75 | share_of_pair_cross |
| ---- | --------------------------------- | ------- | --------- | ------------- | ------------------ | ---------------------- | --------------------- | ----------------------- | ---------------------- | ------------------------ | ------------------ | -------------------- | ----------------- | ----------------- | ------------------- | --------------------- | ------------------ | ------------------ | ------------------- |
| P1   | FULL_NOMINAL_PLUS_FULL_PARTICLE   | 21      | 14        | 0.524         | 1.559              | 995.000                | 1.000                 | 1.000                   | 1.000                  | 1.000                    | 1.524              | 2.000                | 1.000             | 2.000             | 1.190               | 1.000                 | 1.000              | 1.000              | 0.100               |
| P1   | NOMINAL_SUFFIX_PLUS_FULL_PARTICLE | 190     | 76        | 0.237         | 0.825              | 584.000                | 0.450                 | 0.500                   | 1.000                  | 1.000                    | 1.000              | 1.000                | 1.000             | 1.000             | 1.032               | 1.000                 | 1.000              | 1.000              | 0.900               |
| P2   | FULL_NOMINAL_PLUS_FULL_PARTICLE   | 8       | 7         | 0.125         | 1.715              | 718.000                | 1.000                 | 1.000                   | 1.000                  | 1.000                    | 1.250              | 1.000                | 1.000             | 1.250             | 1.375               | 1.000                 | 1.000              | 2.000              | 0.041               |
| P2   | NOMINAL_SUFFIX_PLUS_FULL_PARTICLE | 189     | 83        | 0.228         | 0.876              | 593.000                | 0.425                 | 0.500                   | 1.000                  | 1.000                    | 1.000              | 1.000                | 1.000             | 1.000             | 1.005               | 1.000                 | 1.000              | 1.000              | 0.959               |
| P3   | FULL_NOMINAL_PLUS_FULL_PARTICLE   | 7       | 6         | 0.571         | 2.707              | 718.000                | 1.000                 | 1.000                   | 1.000                  | 1.000                    | 1.143              | 1.000                | 1.000             | 1.000             | 1.429               | 1.000                 | 1.000              | 2.000              | 0.035               |
| P3   | NOMINAL_SUFFIX_PLUS_FULL_PARTICLE | 194     | 85        | 0.345         | 0.989              | 578.000                | 0.427                 | 0.500                   | 1.000                  | 1.000                    | 1.000              | 1.000                | 1.000             | 1.000             | 1.005               | 1.000                 | 1.000              | 1.000              | 0.965               |
