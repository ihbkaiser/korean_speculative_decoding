# H2 Analysis Implementation Plan

**Goal:** Analyze the existing 1,000-prompt run for rejection differences between cross-morpheme and within-morpheme split tokens, controlling for exact fragmentation and structural/entropy covariates.

**Architecture:** Add a dedicated H2 pipeline that reconstructs continuation token spans from saved generated IDs and the prompt cache, joins the saved SD and teacher-forced event records, and writes a token-level parquet plus regression tables, CSVs, plots, and an evidence-based summary into the requested run directory. Decoding is not rerun.

**Files:** Add `src/h2_analysis.py` and `scripts/analyze_h2.py`; save this plan at `implementation_h2.md`; write all requested result artifacts under `/workspace/runs/20260926T184145Z_pilot1000/`.

**Constraints:** Preserve exact fragmentation levels `2, 3, 4, 5, 6, 7, 8+`; classify ambiguous cross-eojeol and Kiwi/span cases separately and exclude them from the primary contrast; cluster logistic-regression standard errors by `prompt_id`; use two-sided inference and prompt-level bootstrap intervals; treat teacher disagreement as a secondary sanity check; make no causal claim.

## Steps

1. Implement reconstruction and token-table creation from `reference_outputs.jsonl`, the matching prompt cache, `eojeols.parquet`, `sd_events.parquet`, and `teacher_forced_tokens.parquet`. Retokenize with the configured target tokenizer, verify exact token-ID matches where offsets are exact, preserve non-roundtripping spans as explicit ambiguity exclusions, run Kiwi on continuation text, and write `h2_token_table.parquet` with the requested columns.
2. Implement exact-fragmentation summaries, M1–M3 clustered logistic models, entropy models, ambiguity/class counts, and the teacher-disagreement sanity check. Report CROSS relative to SPLIT, confidence intervals, two-sided p-values, and token/prompt counts.
3. Generate the class-by-fragmentation rejection-rate figure with prompt-level bootstrap 95% intervals, plus a forest plot of pooled and fragmentation-specific CROSS-vs-SPLIT odds ratios. Write the required CSVs, regression report, and numeric `h2_summary.md`, including excluded ambiguity percentage and A–D result classification.
4. Run the H2 script against the existing run only. Verify token-ID alignment assertions, table schemas/counts, model outputs, figure/CSV presence, and consistency of reported counts and estimates with saved outputs.

**Verification command:** `python scripts/analyze_h2.py /workspace/runs/20260926T184145Z_pilot1000 --prompt-cache /workspace/data/korean_wikipedia_20231101_ko.jsonl`

**Known review focus:** terminal EOS/special tokens have no visible character span; byte-fallback decoding can prevent exact re-tokenization; Kiwi may emit overlapping or zero-width spans; valid SD proposals must exclude invalidated proposals; eojeol fragmentation must retain `8+` without folding categories 5–7 together.
