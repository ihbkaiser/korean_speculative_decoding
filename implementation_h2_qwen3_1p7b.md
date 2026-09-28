# H2 target-size follow-up: Qwen3-0.6B → Qwen3-1.7B

## Experiment card

- **Question:** Does `CROSS_MORPHEME` retain a higher SD rejection rate than `WITHIN_SPLIT` when the target is Qwen3-1.7B?
- **Hypothesis:** At equal fragmentation and structural controls, CROSS has a positive rejection log-odds contrast against WITHIN_SPLIT.
- **Baseline:** Saved run `20260926T184145Z_pilot1000`, Qwen3-0.6B-Base draft → Qwen3-4B-Base target; seed 3090, FP16, greedy decoding, K=4, 1,000 fixed prompts, max prompt/generation lengths 128.
- **Variant:** Change only the target to `Qwen/Qwen3-1.7B-Base`; keep the draft, prompts, seed, precision, decoding parameters, and H2 analysis definitions fixed.
- **Primary metric:** M2 CROSS-vs-WITHIN_SPLIT log-odds coefficient from the H2 logistic model with exact fragmentation fixed effects and structural controls; prompt-clustered SE, two-sided 95% CI.
- **Decision rule:** Evidence for CROSS > SPLIT in this run if the M2 coefficient is positive and its two-sided 95% CI excludes zero. M1 and M3 are reported for comparability; M3 adds draft and target entropy.
- **Guardrails:** Require tokenizer compatibility and exact target-greedy/speculative output ID parity for all prompts. Report any missing prompts or token-alignment ambiguity.
- **Data and seed:** Reuse the existing cached first 1,000 non-empty Korean Wikipedia prompts and fixed seed 3090. No new corpus or prompt sampling.
- **Budget:** First run a 20-prompt compatibility rung; if it passes, run the full 1,000 prompts. Stop the full run at 9 hours, based on the original run's recorded 8.59-hour runtime on the same RTX 3090.
- **Exploratory-only:** Compare effect size to the existing 4B-target result. This single target-size contrast does not establish a causal effect of model size.

## Execution plan

1. Use the existing H2 pipeline and add a separate config/run ID; do not overwrite the original 4B run.
2. Run the 20-prompt rung to check model loading, tokenizer identity, memory use, and exact greedy parity.
3. If those checks pass, run the same pipeline over all 1,000 prompts, then apply the saved H2 analyzer and write a model-pair comparison report.
4. Verify token counts, prompt coverage, parity, primary model estimates, and report/figure outputs. Keep the run, config, code, and summary in this repository.

## Execution status

The 20-prompt compatibility rung is complete in `runs/20260928T102719Z_q3_06b_17b_pilot20/`.

- Tokenizer compatibility passed: vocabulary, added vocabulary, special tokens, and serialized tokenizer backends match.
- Target-greedy and speculative output token IDs matched for all 20 prompts.
- Runtime was 458.2 seconds; peak allocated VRAM was 4.55 GB (6.00 GB reserved) on an RTX 3090.
- Prompt ID 19 used the tokenizer round-trip fallback for character alignment; retain it as an explicit alignment ambiguity in subsequent reporting.
- The 1,000-prompt run and H2 analysis remain pending.
