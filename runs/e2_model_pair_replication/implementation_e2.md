# E2 implementation plan: robustness across Korean model pairs

## Objective and fixed question

Estimate the H2 association between `CROSS_MORPHEME` and `WITHIN_SPLIT` speculative-decoding rejection for two new Qwen3 pairs, then compare them with the existing P3 artifacts. The primary contrast is CROSS relative to WITHIN_SPLIT, with two-sided Wald inference and standard errors clustered by the shared source `prompt_id`. This is an association study; the three-pair comparison will not be presented as a capacity-scaling law.

## 1. Model IDs and revisions

Model commits were resolved from Hugging Face `main` on 2026-09-28 and will be passed as explicit `revision` values to every tokenizer/config/model load:

| Pair | Draft | Draft revision | Target | Target revision |
|---|---|---|---|---|
| P1 | `Qwen/Qwen3-0.6B-Base` | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | `Qwen/Qwen3-1.7B-Base` | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` |
| P2 | `Qwen/Qwen3-1.7B-Base` | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` | `Qwen/Qwen3-4B-Base` | `906bfd4b4dc7f14ee4320094d8b41684abff8539` |
| P3 | `Qwen/Qwen3-0.6B-Base` | not recorded in the P3 run metadata | `Qwen/Qwen3-4B-Base` | `906bfd4b4dc7f14ee4320094d8b41684abff8539` (matches the E1 tokenizer fingerprint and current Hub `main`) |

P3 is the saved run at `runs/20260926T184145Z_pilot1000/`. Its target tokenizer commit is documented in `e1_token_frequency_cache_metadata.json`; the model run metadata did not save model commit hashes. I will not rerun P3. The missing historical P3 draft commit will remain an explicit provenance limitation.

## 2. Tokenizer compatibility check

Before loading model weights, load the three pinned tokenizers and compare tokenizer class/fast status, full token-to-ID mapping and its SHA256, vocabulary sizes, added tokens, special token strings and IDs (including BOS/EOS/PAD), backend serialization SHA256, and exposed normalizer/pre-tokenizer configuration. Run the existing Korean/English/newline offset probes from `src.models.compare_tokenizers` and additional shared probes. Write per-model identities and every comparison to `tokenizer_compatibility.json`. Only mark `shared_tokenizer=true` if all checks pass; otherwise stop before GPU work and adapt pair-specific tokenization explicitly.

The P3 E1 frequency cache may be reused only if the verified shared vocabulary maps token IDs identically and its saved target backend hash matches the pinned 4B tokenizer.

## 3. Prompt source and ordering

Use `data/korean_wikipedia_20231101_ko.jsonl`, the same first 1,000 non-empty `wikimedia/wikipedia` / `20231101.ko` / `train` articles in file order. Preserve `source_index` as `prompt_id`; do not resample. Encode each prompt once with the shared tokenizer using `src.data.encode_prompt` semantics (`add_special_tokens=False`, truncate to 128). Save an ordered SHA256 per prompt over its source index, title, text, and encoded IDs, plus a combined ordered-set hash. Generation is greedy, batch-independent, at most 128 new tokens, EOS `151643`, and K=4.

## 4. P3 reuse

Use P3 artifacts by reference from `runs/20260926T184145Z_pilot1000/`: `reference_outputs.jsonl`, `sd_events.parquet`, `h2_token_table.parquet`, `teacher_forced_tokens.parquet`, `e1_token_frequency_cache.parquet`, and its cache metadata. Do not copy these large files or rerun P3. Reproduce its H2 M1–M3 and E1 M4–M5 estimates from the source Parquet files as a check. For pooled comparisons, use P3’s saved token rows and saved full-sequence teacher-forced entropies; report the original P3 estimates as recorded/reproduced and separately identify any harmonized pooled estimate.

P2 reuses the P3 4B target continuations and still performs 1.7B→4B speculative decoding on all 1,000 prompts. Verify exact target/speculative IDs prompt-by-prompt. The pinned target revision and tokenizer fingerprint must match P3; if a preflight output comparison contradicts that, generate the P2 target references for the mismatching prompts with the pinned 4B target and record that fallback without rerunning P3 SD.

## 5. Fast execution architecture

- Restrict every CUDA process to physical GPU 7 with `CUDA_VISIBLE_DEVICES=7`; assert that the process sees exactly one device and its `cuda:0` name is NVIDIA A100 80GB. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `TOKENIZERS_PARALLELISM=true`, FP16, `eval()`, `inference_mode()`, `use_cache=True`, and TF32 matmul/cuDNN flags.
- Prefer FlashAttention 2 only if installed and its parity benchmark passes; otherwise use SDPA. Do not use eager attention.
- Load the 0.6B, 1.7B, and 4B models once with `low_cpu_mem_usage=True`, keep them resident if measured memory remains comfortable, and use device `cuda:0` only.
- Pass A uses exact incremental KV-cache greedy target and draft decoding with K=4. It logs only proposals, valid target decisions, acceptance/rejection, invalidation, output IDs, positions, and parity. It does not calculate full-vocabulary entropy, run hypothetical target forwards for invalidated proposals, call `synchronize`, or write per-token records in the hot loop. Python-visible token transfers remain only where an autoregressive decision must feed the next model call.
- P1 target references use an efficient batched greedy path only if a first-32-prompt comparison against batch-size-1 cached decoding is exactly identical; otherwise use the verified singleton path. P2 uses saved P3 target IDs.
- Pass B scores prompt-plus-reference sequences with batched, right-padded, length-bucketed model forwards. It extracts argmax and entropy in GPU batches and transfers the resulting arrays once per batch. Probe batch sizes 1/4/8/16 on the first 32 prompts, verify exact per-position argmax/disagreement parity against singleton scoring on at least eight, then select the fastest batch that is stable within A100 80GB memory. These are teacher-forced metrics, not replacement SD rejection labels.
- Save each completed prompt chunk atomically to checkpoint Parquet, then atomically update stage progress. A prompt is marked complete only after its output, events, and parity result are durable. Teacher-forced scoring and morphology each have independent completed-prompt checkpoints.

## 6. 32-prompt performance benchmark

Benchmark the first 32 prompts before the full run. Compare SDPA with FlashAttention 2 when the latter is installed; benchmark the supported teacher-forcing batch sizes 1/4/8/16. The existing custom SD implementation is single-prompt and cache-dependent, so do not rewrite it merely to claim SD batch-size comparisons. Verify target ID parity, proposal IDs, valid accept/reject decisions, and stopping positions against the singleton reference for at least eight prompts for each optimized decoding configuration. Record backend, stage/batch size, prompts/sec, generated tokens/sec, wall time, peak allocated/reserved VRAM, and exact parity in `performance/e2_performance_benchmark.csv` and `performance/e2_performance.md`.

## 7. Correctness and morphology

For each new pair, require exact equality between target greedy and speculative output token IDs for every prompt. Keep a row for every prompt and write failures with first differing position before stopping; do not exclude failures silently. Reconstruct H2 token rows from the final target continuation with the existing `_span_tokenizer_ids`, `_classify_token`, `_fragmentation_bin`, tokenizer offset, whitespace, and Kiwi span semantics. Save exact retokenization audits and counts of total generated tokens, aligned tokens, `CROSS_EOJEOL`, `KIWI_COMPLEX`, retokenization failures, and percentage excluded. Preserve the existing five labels and all H2 eligibility rules.

## 8. Statistical models

Reuse the formula helpers/protocol in `src.h2_analysis` and `src.e1_frequency`:

- Eligible rows: `sd_valid=True`, fragmentation bins `2,3,4,5,6,7,8+`, class in `{WITHIN_SPLIT,CROSS_MORPHEME}`.
- M1: rejection ~ morphology + fragmentation fixed effects.
- M2: M1 + relative position, first/last token, proposal slot, generation position, token character length, eojeol character length.
- M3: M2 + draft and target entropy from Pass B (P3 full-sequence scores are reused and compared with its saved cached-path entropy values).
- Frequency robustness: join E1 token counts by token ID only after the cache/tokenizer identity check; fit M4 with linear `log_token_count` and M5 with the exact E1 cubic spline `bs(log_token_count, df=4, degree=3, include_intercept=False)`.
- Report CROSS−SPLIT coefficients, ORs, 95% CIs, two-sided p-values, eligible N, prompt N, and raw class rejection rates for each pair. Use prompt-clustered covariance with the existing small-sample correction.
- Pooled exploratory model: pair fixed effects, morphology, pair×morphology interaction, M2 structural controls and entropy controls; cluster on the shared `prompt_id`. Use interactions only to assess heterogeneity.

## 9. Outputs

Create `runs/e2_model_pair_replication/` with `implementation_e2.md`, `config.json`, `environment.txt`, `tokenizer_compatibility.json`, `prompt_hashes.json`, and `performance/`. Save P1 and P2 `sd_events.parquet`, `continuations.parquet`, `teacher_forced_tokens.parquet`, `token_table.parquet`, `alignment_audit.csv`, `regression.txt`, and `summary.md`; store checkpoints under each pair directory. P3 is referenced by path. In `combined/`, write `e2_model_pair_summary.csv`, `e2_model_pair_regression.txt`, `e2_model_pair_replication.md`, `e2_forest_m3.png`, and `e2_forest_frequency_controlled.png`.

The CLI will accept `--run-dir`, `--prompt-cache`, `--num-prompts`, `--max-prompt-tokens`, `--max-new-tokens`, `--speculative-k`, `--dtype`, `--resume`, and benchmark controls. The requested invocation pins `CUDA_VISIBLE_DEVICES=7`; runtime checks reject an unmasked or wrong GPU.
