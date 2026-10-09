# Selected no-fallback workflow: batch 256 and five-pair B200 smoke

Date: 2026-10-10, Asia/Bangkok. User explicitly selected BF16/SDPA,
target-block verification without scalar reference generation or fallback.
No full 20,000/40,000-prompt experiment was started. All remote apps ended.

## Batch 128 vs 256: Q1, same 512 real prompts

Maximum input/output lengths: 128/128 tokens; greedy, speculative K=4.
One warmup per batch size; no scalar diagnostics or old baseline reruns.

| Batch | Measured seconds | Prompts/s | Generated tokens/s | Peak GPU allocation |
|---|---:|---:|---:|---:|
| 128 | 36.325 | 14.095 | 1,745.9 | 12.91 GiB |
| 256 | 30.293 | 16.901 | 2,097.7 | 21.47 GiB |

Batch 256 provided about **20% higher prompt throughput**, or 16.6% less
elapsed time, in this trial. Settings, model snapshots and input prompt IDs
were the same; generated sequences can differ across batch shapes under the
selected numerical contract. There was no independent scalar equality check.

This is a decode benchmark with audit statistics, not checkpoint/alignment IO.
Its linear 20,000-prompt estimate is 19.7 minutes for Q1 decoding only.
First/remaining batch durations were 17.573/12.718 seconds at B=256; B=128
durations were 10.946, 10.797, 7.447, 7.133 seconds. One sweep, no confidence
interval; more prompts and repeated measurements could change the estimate.

Evidence: `modal_b200_q1_b256.json` and `.log`.
[Benchmark app](https://modal.com/apps/tungkieu6868/main/ap-f9uA9apXnCmoJg7WYKwmlS).

## Five-pair smoke: PASS

Each pair processed the same first **256 frozen-pool prompts**, at actual
batch size 256, at most 128 generated tokens. Checks covered generation,
resumable JSONL, reference/event Parquet, finite entropy/logprob/rank data,
accepted-token consistency, morphology invariants and Table 1 output/joins.
All five pairs finished with zero scalar fallback prompts.

| Pair | Smoke | SD + checkpoint/Parquet seconds | Prompts/s | Linear SD minutes / 20,000 |
|---|---|---:|---:|---:|
| Q1 | PASS | 21.522 | 11.895 | 28.0 |
| Q2 | PASS | 25.021 | 10.231 | 32.6 |
| Q3 | PASS | 35.332 | 7.245 | 46.0 |
| M1 | PASS | 27.477 | 9.317 | 35.8 |
| G1 | PASS | 25.013 | 10.235 | 32.6 |

These SD timings include first-use overhead and checkpoint/final Parquet
writes, but not model load, morphology or final table aggregation. They are
more conservative than the warm Q1 decode benchmark. Five SD jobs used one
B200 each; rates above are per pair/device, not combined multi-GPU throughput.

CPU morphology including tokenizer/Kiwi setup took 14.599, 14.435, 6.240,
7.170 and 8.443 seconds respectively. CPU aggregation took 3.627, 4.809,
3.218, 3.270 and 3.145 seconds. CPU jobs ran concurrently, so these small-run
times include setup/contention and should not be extrapolated as guaranteed
full-run costs. In particular, table joins and final materialization can have
different memory/IO costs at 20,000 prompts.

The default experiment still has Q2's original **40,000** prompts; the other
four pairs have 20,000. The table above normalizes every estimate to 20,000
for comparison; it does not silently change Q2's experiment definition.

## What the smoke found and how it was completed

Initial smoke attempts intentionally failed rather than switch decoding paths:

1. Transformers 5 selected `MistralCommonBackend` for M1 despite `use_fast=True`.
   It did not provide the offsets required by analysis. The selected M1 config
   now explicitly loads the pinned Rust `tokenizer.json` with
   `fix_mistral_regex=true`. SD, tokenizer audit and morphology use this same
   setting, recorded in resume identity. A Korean probe returned the same IDs
   with both backends; broader tokenizer equivalence was not asserted.
2. Multi-token Parquet arrays triggered an ambiguous NumPy truth-value error
   in generated-token counting. Counting now handles arrays without a boolean
   coercion. Regression test reproduced the real failure before the fix.
3. Excluded whitespace/prompt-fragment tokens were stored with an empty local
   span although their decoded span was nonempty. They now retain their actual
   surface as an explicitly unmatched/excluded coordinate frame. Visibility,
   exclusion reason and denominator are retained; invariants remain strict.
4. Kiwi emitted virtual zero-width morphemes, such as an implicit `이/VCP`.
   These rows were already `ambiguous_morphology_span` exclusions. Raw analysis
   is now retained in `raw_morpheme_*`; unavailable canonical morphology spans
   are empty. This does not make excluded rows eligible or change their counts.

After the M1 loader fix, **all five B200 SD jobs completed**. The later fixes
were CPU-only: existing completed SD checkpoints were reused without changing
or regenerating continuation tokens. Final morphology/table smoke passed for
all five; this is staged end-to-end validation, not a fresh GPU rerun after
every CPU serialization correction.

Evidence: `modal_b200_smoke5_b256_pass.json` and `.log`.
The failed attempts are retained in `modal_b200_smoke5_b256{,_fixed}.json/log`
and `modal_b200_smoke5_b256_cpu_before_fix.json/log`. Diagnostic artifacts:
`modal_m1_tokenizer_inspect.json` and `modal_alignment_anomalies.json`.

- [Completed B200 SD source app](https://modal.com/apps/tungkieu6868/main/ap-taSQx13O8BaIXbhwqfrtI5)
- [Passing CPU alignment/table app](https://modal.com/apps/tungkieu6868/main/ap-tj4sd0xlmkax6Qt6CRacJL)

## Locked workflow and commands

`configs/table1_fast_b200.yaml`: batch 256 for all five pairs, BF16 + SDPA,
K=4, target-block verification, no independent scalar reference pass or scalar
fallback. `run_table1.sh` selects this workflow. TQDM and checkpoint resume
remain enabled. Strict historical output stays separate in `runs/table1`.

Only when the user chooses to start a full run:

```bash
bash run_table1.sh Q1
# All pairs, with their configured original prompt counts:
bash run_table1.sh all
```

Bounded reproduction, no full run:

```bash
modal run --profile tungkieu6868 scripts/modal_table1_speed.py --stage benchmark --pair Q1 --prompts 512 --batch-sizes 128,256 --label q1_b256_recheck
modal run --profile tungkieu6868 scripts/modal_table1_speed.py --stage smoke-all --prompts 256 --batch-sizes 256 --label smoke5_b256_recheck
```

Use a fresh label to preserve prior artifacts. Production-path smoke stages reject
more than 512 prompts; the later B2048 decode-only benchmark permits up to 2,048.
All stages cap output at 128 tokens. No full-run Modal stage exists. Changing
batch size cannot silently resume incompatible B=128 fast checkpoints.

## Runtime, billing and limits

The Q1 benchmark reported NVIDIA B200, Torch 2.14.1+cu130, CUDA 13.0 and
Transformers 5.19.0. Five-pair smoke used the same Modal image and pinned model
snapshots listed in `configs/table1_pipeline.yaml`; no model download was needed.
Attention: SDPA; `use_remove_padding=false`; no explicit FA2 backend. Compute
capability and per-pair peak memory were not captured before the initial CPU
stage failures; no per-pair VRAM estimate is claimed beyond Q1's measured peak.

Profile/workspace: `tungkieu6868`. Billing interval 2026-10-01 to 2026-11-01
exclusive, calendar-month default rather than a verified invoice-cycle anchor.
Reported usage for these benchmark/smoke/debug apps was about **$2.99**; total
workspace usage was $27.5722 including unrelated jobs. The latest reported
app costs were $0.2052 benchmark, $1.1987 first smoke, $1.5275 second smoke,
$0.0359/$0.0194 CPU finish attempts and $0.0017 CPU diagnostics. Billing can
lag and precedes credits/reservations; remaining credits and a monthly hard
limit were unavailable. Dashboard budgets were not inspected. No GPU job is
left running by this task.

Highest evidence: real-model B200 bounded smoke, with five-pair CPU table
completion. Full dataset stability, full-run RAM/IO behavior and independent
scalar equality remain unverified. The selected mode does not claim exact
reproduction of FP32/scalar output; metadata retains that distinction.

Backend reference: [official Transformers tokenizer documentation](https://huggingface.co/docs/transformers/main/fast_tokenizers#backends).

## Change/instrumentation log

| File | Change | Purpose |
|---|---|---|
| `scripts/modal_table1_speed.py` | Modified | B=256 sweep, five-pair bounded smoke, finite artifact validation, CPU-only checkpoint postprocessing, JSON/log evidence; scalar diagnostics off by default |
| `src/table1_smoke.py` | Created | Reference/event integrity, finite statistics and accepted-token consistency checks |
| `src/models.py` | Modified | Explicit Rust tokenizer loader for offset-based M1 analysis |
| `src/table1_runner.py` | Modified | Use selected tokenizer backend consistently and record it in fast resume identity |
| `src/table1_morphology.py` | Modified | Array-safe counts, valid excluded token spans, preserved raw virtual-morpheme analysis |
| `configs/table1_fast_b200.yaml` | Modified | Select B=256 and M1 tokenizer settings; mark bounded smoke status |
| `run_table1.sh` | Created | Main no-fallback workflow launcher |
| `tests/test_table1_smoke.py` | Created | Artifact corruption/contract/NaN tests and explicit tokenizer-offset test |
| `tests/test_table1_morphology.py` | Modified | Array, unmatched-token and zero-width Kiwi regressions, preserving exclusions |
| `tests/test_table1_fast_runner.py` | Modified | Selected config checks |
| `README.md`, `profile_output/README.md` | Modified | Selected workflow and bounded reproduction commands |
| `profile_output/modal_b200_*`, diagnostics and test log | Created/updated | Raw evidence, failed attempts and this report |

Final local verification: **98 passed, 3 skipped**, shell syntax and Python
compilation checks passed. Benchmark-only helpers are independent of production
model forwards; they can be removed later without removing TQDM or the decoder.
