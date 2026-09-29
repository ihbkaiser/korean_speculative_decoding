# E2 result: A — strong replication.

Both new pairs show the prespecified positive CROSS association under M3 and M5, and the pooled frequency-controlled interaction model keeps all pair contrasts positive. P3's saved H2 M3 OR is 1.766; the pooled frequency-controlled interaction is reported as exploratory.

This report describes conditional associations in generated continuations; it does not claim that cross-morpheme tokenization causally increases rejection.

## Pair-specific results

| Draft → Target | N tokens | Raw CROSS rejection | Raw SPLIT rejection | M1 OR (95% CI) | M2 OR (95% CI) | M3 OR (95% CI) | M3 p | Frequency-controlled M5 OR (95% CI) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.6B → 1.7B | 73,001 | 20.15% (n=3,454) | 12.78% (n=69,547) | 1.609 (1.450–1.784) | 2.527 (2.081–3.069) | 2.002 (1.604–2.499) | 8.6e-10 | 2.086 (1.653–2.633) |
| 1.7B → 4B | 73,257 | 19.32% (n=3,762) | 14.27% (n=69,495) | 1.378 (1.246–1.524) | 2.236 (1.811–2.761) | 1.827 (1.426–2.342) | 1.92e-06 | 2.133 (1.718–2.649) |
| 0.6B → 4B | 73,143 | 25.24% (n=3,800) | 18.35% (n=69,343) | 1.426 (1.303–1.561) | 2.102 (1.791–2.467) | 1.766 (1.475–2.115) | 6.34e-10 | 1.894 (1.542–2.327) |

M1 controls exact fragmentation fixed effects; M2 adds H2 structural controls; M3 adds full-sequence draft and target entropy. M4 adds linear log token frequency. M5 adds the E1 cubic spline frequency control. All pair-specific logit models use two-sided Wald tests and standard errors clustered by prompt_id.

## Tokenizer, prompts, and correctness
- Shared tokenizer: **True**; all three vocab maps, special-token setup, tokenizer class, backend hashes, and ID/offset probes are documented in `tokenizer_compatibility.json`.
- Prompt source: `/data/hoang/korean_speculative_decoding/data/korean_wikipedia_20231101_ko.jsonl`; ordered prompt-set SHA256: `7ee8810367bbf808818eb30613d3f2e18837285d62bbe14c211241104c34a7d9`; count: 1,000; max prompt/generated lengths: 128/128.
- P1/P2 speculative output matched target greedy for every prompt: **True** (1,000 prompts per pair). Singleton reference fallbacks: P1 0; P2 160. P2 uses P3's saved 4B references except those explicitly recorded fallbacks.
- P3 was reused from `/data/hoang/korean_speculative_decoding/runs/20260926T184145Z_pilot1000`; no P3 speculative-decoding rerun occurred. P3 saved primary N and M1/M5 estimate checks: `{"e1_m5_recomputed_or": 1.8940513365460867, "e1_m5_saved_or": 1.8940513365460867, "h2_m1_recomputed_or": 1.425912530466511, "h2_m1_saved_or": 1.425912530466511, "h2_primary_tokens_recomputed": 73143, "h2_primary_tokens_saved": 73143}`.
- P3 cached-path vs saved full-sequence FP16 entropy diagnostic: {"disagreement_match_count": 127093, "disagreement_match_percentage": 99.88054540453456, "draft_argmax_match_count_cached_vs_full_sequence": 127141, "draft_argmax_match_percentage_cached_vs_full_sequence": 99.91826790836575, "draft_entropy_correlation_cached_vs_full_sequence": 0.9999978370702036, "draft_entropy_max_absolute_difference_cached_vs_full_sequence": 0.047119855880737305, "draft_entropy_mean_absolute_difference_cached_vs_full_sequence": 0.0015542601279563478, "rows": 127245, "sd_valid_cached_vs_full_sequence_rows": 127245, "target_argmax_match_count_cached_vs_full_sequence": 127078, "target_argmax_match_percentage_cached_vs_full_sequence": 99.86875712208732, "target_entropy_correlation_cached_vs_full_sequence": 0.9999942837788769, "target_entropy_max_absolute_difference_cached_vs_full_sequence": 0.9521242352202535, "target_entropy_mean_absolute_difference_cached_vs_full_sequence": 0.0021397853697806214}.
- E1 frequency cache identity: tokenizer `Qwen/Qwen3-4B-Base` revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`, backend SHA256 `41e00eccf531cffc2e562d38bdd879d41e5044ea279af5b73c6a32aabcc8fe04`; exact ID map verified shared before join.

## Alignment and exclusions
| Pair | Generated tokens | Aligned visible token spans | CROSS_EOJEOL | KIWI_COMPLEX | Retokenization failures | Ambiguous exclusions |
|---|---:|---:|---:|---:|---:|---:|
| P1 | 127,118 | 126,770 | 0 | 10,145 | 338 | 7.98% |
| P2 | 127,245 | 127,092 | 0 | 9,698 | 140 | 7.62% |
| P3 | 127,245 | 127,097 | 0 | 9,660 | 135 | 7.59% |

The `EXACT`, `WITHIN_SPLIT`, `CROSS_MORPHEME`, `CROSS_EOJEOL`, and `KIWI_COMPLEX` labels use the existing H2 span classifier without changes. Primary models exclude CROSS_EOJEOL/KIWI_COMPLEX and require `sd_valid=True` and fragmentation 2,3,4,5,6,7,8+.

## Pooled model-pair analysis
- Pooled M3 interaction model: N=219,401 tokens; 999 prompts in the eligible-pair union, not the 993-prompt three-way intersection; model-pair × morphology joint interaction p=0.001234.
- Common-prompt sensitivity: the exact eligible three-way intersection contains 993 prompts; pooled M3/M5 classification is **unchanged** (see `e2_common_prompt_sensitivity.md`).
- Pooled frequency-controlled M5 interaction model: N=219,401; pair contrasts: `{"P1": {"beta": 0.8730745903399685, "odds_ratio": 2.3942609200085148, "or_ci_high": 2.851940378422173, "or_ci_low": 2.0100298717505094, "p_value": 1.3369379064547938e-22, "std_error": 0.08924745245510128}, "P2": {"beta": 0.6099282648836689, "odds_ratio": 1.840299379956396, "or_ci_high": 2.185307071976143, "or_ci_low": 1.549760146433493, "p_value": 3.4685831068426334e-12, "std_error": 0.08766739294035436}, "P3": {"beta": 0.6624452871597943, "odds_ratio": 1.9395292464489804, "or_ci_high": 2.2983157336649103, "or_ci_low": 1.6367523585771218, "p_value": 2.014542051515685e-14, "std_error": 0.08659759071904208}}`.
- Pair interactions assess heterogeneity only. Three model pairs do not support a capacity-scaling law.

## Runtime, GPU, and limits
- Device: physical GPU 7, exposed as `cuda:0`; GPU model `NVIDIA A100-SXM4-80GB`; CUDA runtime `12.8`; PyTorch `2.11.0+cu128`; Transformers `5.17.0`; attention backend `sdpa`; dtype FP16.
- E2 process-local peak allocated/reserved memory: 14.49/14.66 GiB. GPU7 device-level monitor summary: `{"active_monitored_hours": 6.811333964722222, "first_sample_utc": "2026-09-28T18:23:52.894594+00:00", "last_sample_utc": "2026-09-29T06:10:07.286997+00:00", "max_gpu_utilization_pct": 100.0, "max_memory_used_mib": 39639.0, "mean_gpu_utilization_pct": 65.39478186484175, "monitor_windows": 4, "samples": 2338, "wall_span_hours": 11.770664556388889}`.
- Monitored pipeline time across all resumed run windows: 6.81 hours; first-to-last monitor span including pauses: 11.77 hours across 4 windows. The 3.14-hour `elapsed_seconds` value is only the final resumed invocation; model loading and unmonitored gaps are outside the summed monitor intervals. Teacher forcing uses full-sequence FP16 and SD rejection labels remain cached-path labels.
- GPU utilization and total device memory are board-level measurements, not E2-only. A concurrent external process was observed on physical GPU 7 during the run (26,691 MiB at one check); E2 allocator peaks above are process-local.
- P3's exact historical 0.6B draft and 4B target weight revisions remain unresolved in local artifacts; the later E1 4B tokenizer revision does not establish P3 weight identity. See `p3_revision_recovery.md`.
- No generated artifacts were silently excluded after target/speculative parity checks; prompt-level resume checkpoints and `progress.json` files are retained.

## Files
- `e2_model_pair_summary.csv`, `e2_model_pair_regression.txt`.
- `e2_forest_m3.png`, `e2_forest_frequency_controlled.png`.
- Pair folders contain `sd_events.parquet`, `continuations.parquet`, `teacher_forced_tokens.parquet`, `token_table.parquet`, `alignment_audit.csv`, `regression.txt`, and `summary.md`.
- P3 source artifacts are referenced at `runs/20260926T184145Z_pilot1000/` and are not duplicated.
