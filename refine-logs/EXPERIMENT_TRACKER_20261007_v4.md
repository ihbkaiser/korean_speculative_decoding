# Deadline experiment tracker v4 — 07–10 October 2026

Timezone: Asia/Bangkok (UTC+7). One implementer and no Korean annotator confirmed. All jobs remain proposed/not submitted. See DEADLINE_EXPERIMENT_PLAN.md for dated milestones, method specification, capacity gates and interpretation limits.

Core analysis47,929 SD trajectories; full analysis68,011. Separate full target references47,373; primary runtime33,000 if AR reuse is valid; full timed comparisons66,900 including3090. These are proposed counts, not completed results.

| Run | Resource | Priority | Card / pair | Planned count / unit | Latest finish UTC+7 | Status |
|---|---|---|---|---|---|---|
| D00 | LEAD | P0 | data_freeze / all |  manifest | 2026-10-07T12:00+07:00 | PLANNED_NOT_SUBMITTED |
| H00 | LEAD_3090_B200 | P0 | E00 / device_adapter_cache_verifier |  preflight | 2026-10-07T16:00+07:00 | PLANNED_NOT_SUBMITTED |
| H01 | B200_A_B | P0 | E00 / all_registered_pairs |  pilot | 2026-10-07T18:00+07:00 | PLANNED_NOT_SUBMITTED |
| AN01 | 3090_CPU | P0 | E01_automatic_only / actual_candidates_context_sensitivity | 600 automatic_audit_items | 2026-10-09T12:00+07:00 | PLANNED_NOT_GOLD_VALIDATION |
| HUM01 | HUMAN_TWO_KOREAN_ANNOTATORS | P2 | E01_gold / actual_candidates | 600 items_each_annotator |  | DEFERRED_NO_ANNOTATOR |
| C01 | B200_A | P1 | E02_E09 / Q2 | 11547 analysis_SD | 2026-10-09T16:00+07:00 | PLANNED_NOT_SUBMITTED |
| C02 | B200_A | P1 | E02 / Q1 | 6547 analysis_SD | 2026-10-09T16:00+07:00 | PLANNED_NOT_SUBMITTED |
| C03 | B200_A | P1 | E02 / Q3 | 6547 analysis_SD | 2026-10-09T16:00+07:00 | PLANNED_NOT_SUBMITTED |
| C04 | B200_A | P1 | E04 / Q4 | 2000 analysis_SD | 2026-10-09T16:00+07:00 | PLANNED_NOT_SUBMITTED |
| C05 | B200_A | P1 | E04 / Q5 | 2000 analysis_SD | 2026-10-09T16:00+07:00 | PLANNED_NOT_SUBMITTED |
| C06 | B200_A | P1 | E04 / Q6 | 2000 analysis_SD | 2026-10-09T16:00+07:00 | PLANNED_NOT_SUBMITTED |
| C07 | B200_B | P1 | E05 / G1 | 3547 analysis_SD | 2026-10-09T16:00+07:00 | PLANNED_NOT_SUBMITTED |
| C08 | B200_B | P1 | E06 / QI1 | 1550 analysis_SD | 2026-10-09T16:00+07:00 | PLANNED_NOT_SUBMITTED |
| C09 | B200_B | P1 | E06 / GI1 | 1550 analysis_SD | 2026-10-09T16:00+07:00 | PLANNED_NOT_SUBMITTED |
| C10 | B200_B | P1 | L01 / L01a | 3547 analysis_SD | 2026-10-09T16:00+07:00 | PLANNED_NOT_SUBMITTED |
| C11 | B200_B | P1 | L01 / L01b | 3547 analysis_SD | 2026-10-09T16:00+07:00 | PLANNED_NOT_SUBMITTED |
| C12 | B200_B | P1 | L02 / L02a | 3547 analysis_SD | 2026-10-09T16:00+07:00 | PLANNED_NOT_SUBMITTED |
| X01 | B200_A | P2 | L03 / L03_7 | 2997 analysis_SD | 2026-10-09T16:00+07:00 | PLANNED_CAPACITY_GATED |
| X02 | B200_A | P2 | L03 / L03_14 | 2997 analysis_SD | 2026-10-09T16:00+07:00 | PLANNED_CAPACITY_GATED |
| X03 | B200_B | P2 | L03 / L03_32 | 2997 analysis_SD | 2026-10-09T16:00+07:00 | PLANNED_CAPACITY_GATED |
| X04 | B200_B | P2 | L03 / L03_72 | 2997 analysis_SD | 2026-10-09T16:00+07:00 | PLANNED_CAPACITY_GATED |
| X05 | B200_B | P2 | L02 / L02b | 3547 analysis_SD | 2026-10-09T16:00+07:00 | PLANNED_CAPACITY_GATED |
| X06 | B200_B | P2 | L03 / L03_7draft32 | 2997 analysis_SD | 2026-10-09T16:00+07:00 | PLANNED_CAPACITY_GATED |
| X07 | B200_B | P2 | L04 / L04_4draft32it | 1550 analysis_SD | 2026-10-09T16:00+07:00 | PLANNED_CAPACITY_GATED |
| REF_CORE | B200_A_B | P1 | E00 / unique_target_AR | 33835 reference_AR | 2026-10-09T16:00+07:00 | PLANNED_NOT_SUBMITTED |
| REF_EXTRA | B200_A_B | P2 | E00 / unique_target_AR | 13538 reference_AR | 2026-10-09T16:00+07:00 | PLANNED_CAPACITY_GATED |
| CPU01 | 3090_CPU | P1 | E03_E07_E08_E09 / analysis_controls | 0 reuse_only | 2026-10-10T18:00+07:00 | PLANNED_NOT_SUBMITTED |
| MADS01 | LEAD_3090 | P1 | E14 / predictor_online_callback |  implementation | 2026-10-09T12:00+07:00 | PLANNED_NOT_SUBMITTED |
| AN02 | 3090_CPU | P2 | E10_exploratory / natural_examples_and_frames | 1200 exploratory_frames | 2026-10-09T12:00+07:00 | PLANNED_EXPLORATORY_NOT_GOLD |
| SCORE01 | B200_A_B | P1 | E04_E10_E12 / same_prefix_and_exploratory_frame_distribution |  scoring_positions | 2026-10-09T16:00+07:00 | PLANNED_NOT_SUBMITTED |
| ROB01 | B200_A | P1 | E11 / Q2_Q5_K_grid | 8000 robustness_SD | 2026-10-09T16:00+07:00 | PLANNED_NOT_SUBMITTED |
| ROB02 | B200_A | P1 | E11 / Q5_context_output_grid | 1500 robustness_SD | 2026-10-09T16:00+07:00 | PLANNED_NOT_SUBMITTED |
| ROB03 | B200_A | P1 | E11 / Q2_Q5_FP16 | 256 robustness_SD | 2026-10-09T16:00+07:00 | PLANNED_NOT_SUBMITTED |
| DEV01 | B200_EXCLUSIVE_HOST | P1 | E13 / shortlist_and_validate_six_policies | 8400 dev_timed_run | 2026-10-09T12:00+07:00 | PLANNED_NOT_SUBMITTED |
| RT01 | B200_B_EXCLUSIVE_HOST | P1 | E14 / Q4_Q5_six_policies | 33000 final_timed_run_if_AR_reuse | 2026-10-10T15:00+07:00 | PLANNED_NOT_SUBMITTED |
| AB01 | B200_B_EXCLUSIVE_HOST | P1 | E15 / Q4_five_extra_variants | 7500 final_timed_run | 2026-10-10T18:00+07:00 | PLANNED_NOT_SUBMITTED |
| RT02 | B200_B_EXCLUSIVE_HOST | P2 | L04 / IT32_six_policies | 14400 final_timed_run | 2026-10-10T18:00+07:00 | PLANNED_CAPACITY_GATED |
| RT03A | 3090_EXCLUSIVE_LOCAL | P2 | L05 / Q3_four_policies | 6000 final_timed_run | 2026-10-10T18:00+07:00 | PLANNED_CAPACITY_GATED |
| RT03B | B200_B_EXCLUSIVE_HOST | P2 | L05 / Q3_four_policies | 6000 final_timed_run | 2026-10-10T18:00+07:00 | PLANNED_CAPACITY_GATED |
| SUP01 | B200_A_B | P3 | L06 / Q2_Q5_L01a_L02a | 20000 extra_analysis_SD | 2026-10-09T16:00+07:00 | PLANNED_CAPACITY_GATED |
| REP01 | 3090_CPU_LEAD | P1 | E16_tables_figures_audit / all_completed_results |  report | 2026-10-10T23:59+07:00 | PLANNED_NOT_SUBMITTED |

Human gold validation is DEFERRED_NO_ANNOTATOR. Automated context/offset agreement and exploratory frames do not substitute for linguistic gold labels. Pipeline correctness and runtime are still evaluated. No GPU-time performance estimates have been measured on the company host.
