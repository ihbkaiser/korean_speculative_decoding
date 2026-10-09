# Q1: real B200 batch 2048 trial

Date: 2026-10-10, Asia/Bangkok. Bounded experiment requested by the user;
no full dataset job or independent scalar reference/fallback was run.
Production defaults remain batch 256.

## Result

Batch 2048 completed with `PYTORCH_ALLOC_CONF=expandable_segments:True`.
The default allocator attempt failed with OOM during the timed 128-token
decode, despite passing the four-token warmup.

Both variants below used the expandable allocator, the same first 2,048 frozen
Korean prompt IDs, max input/output 128/128, greedy K=4, BF16 and SDPA.
There was one warmup and one timed sweep per batch size.

| Batch | Timed seconds | Prompts/s | Tokens/s | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|---:|
| 256 | 115.034 | 17.803 | 2,222.9 | 21.465 | 25.326 |
| 2048 | 112.084 | 18.272 | 2,281.4 | 141.293 | 171.297 |

Actual B256 batches: eight batches of 256; actual B2048: one batch of 2,048.
Both generated 255,706 tokens. Proposal-event counts were 273,061 and 273,063;
equal total output lengths are not a claim of identical token sequences.

Measured prompt throughput was **2.63% higher**, or 2.56% less elapsed time,
at B2048. This single-run difference may be within run-to-run variability;
no confidence interval or repeatability claim is made.

Torch reported 178.351 GiB device memory. B2048's peak reserved memory was
**96.04%** of that capacity. Reserved includes cached blocks, not only live
tensors, and is not an `nvidia-smi`/total-process peak measurement. With little
remaining device capacity and only a small observed throughput improvement,
batch 2048 has not replaced the five-pair-validated B256 production default.

Linear Q1 decode-only estimates for 20,000 prompts: B256 **18.72 minutes**;
B2048 **18.24 minutes**. Neither includes model loading, checkpoint/Parquet,
CPU morphology, final table materialization or a larger-dataset stability test.

## Default allocator failure

Same prompt set and settings, before the allocator change:

- B256 completed in 115.102 seconds, 17.793 prompts/s, allocated/reserved
  peaks 21.468/29.174 GiB.
- B2048 passed warmup but failed during the timed decode. Peak allocated was
  141.293 GiB; peak reserved was 175.906 GiB.
- The exception requested 9.27 GiB with 6.94 GiB free, reporting 132.02 GiB
  live PyTorch allocation and 38.60 GiB reserved-but-unallocated at that point.
- No batch reduction or scalar rerun occurred. The B256 result was an explicit
  comparison variant, not an OOM fallback.

The retry used the same decoder and batch sizes, changing only the opt-in
allocator setting in a fresh Modal app. Its success is consistent with memory
fragmentation contributing to the first failure; no allocator/Nsight trace
was collected to establish a complete memory-causality profile.
See [official PyTorch memory-management documentation](https://docs.pytorch.org/docs/stable/notes/cuda#optimizing-memory-usage-with-pytorch-alloc-conf).

## Environment and timing

NVIDIA B200; capability 10.0; Torch 2.14.1+cu130, CUDA 13.0;
Transformers 5.19.0; SDPA, no explicit flash-attn backend,
`use_remove_padding=false`. Model snapshots:

- Draft Qwen3-0.6B-Base: `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`.
- Target Qwen3-1.7B-Base: `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
- Prompt-ID SHA256: `3b29d15e44e9e52b8a3dbbad9d194bf96bb2c7d487dc458aba2bd20eeb056152`.

Timing synchronizes CUDA and includes generation, proposal-audit statistics,
Python event collection and progress overhead. Model setup and warmup are
excluded. B256 warmup: 4.931 seconds; timed block durations: 19.239, 12.836,
13.102, 13.110, 16.015, 13.594, 12.788, 14.346 seconds. Mean block: 14.379;
mean excluding first: 13.685 seconds. B2048 warmup: 6.509 seconds; single
timed block: 112.080 seconds; excluding-first mean is unavailable.

## Evidence and reproduction

- Default allocator: `modal_b200_q1_b2048.json`, full log
  `modal_b200_q1_b2048_utf8.log`;
  [completed app](https://modal.com/apps/tungkieu6868/main/ap-zH0lCYvcY4zKqISLItcFuU).
- Expandable allocator: `modal_b200_q1_b2048_expandable.json` and `.log`;
  [completed app](https://modal.com/apps/tungkieu6868/main/ap-tcgC2uBLEctudmjcb1mL8n).
- An initial local Windows CLI Unicode error is retained in
  `modal_b200_q1_b2048.log`; the UTF-8 rerun launched the real first trial.

Bounded Bash reproduction; use a fresh label:

```bash
TABLE1_TRIAL_ALLOC_CONF=expandable_segments:True modal run --profile tungkieu6868 scripts/modal_table1_speed.py --stage benchmark --pair Q1 --prompts 2048 --baseline-prompts 0 --diagnostic-prompts 0 --max-new-tokens 128 --batch-sizes 256,2048 --label q1_b2048_recheck --output profile_output/modal_b200_q1_b2048_recheck.json
```

On Windows, also set `PYTHONUTF8=1` and `PYTHONIOENCODING=utf-8` for the CLI.
`TABLE1_TRIAL_ALLOC_CONF` configures the Modal benchmark image; it does not
set the allocator in company/production launchers.

Highest evidence: real Q1 decode/audit benchmark completed at B2048, with
bounded actual prompts. Not validated: other four pairs at B2048, B2048's
checkpoint/alignment/table pipeline, full datasets, independent scalar parity.
The earlier five-pair B256 pipeline smoke remains the production-path evidence.

## Local checks, changes and billing

Local verification: **106 passed, 3 skipped**, Python compilation and diff
whitespace checks passed. Test log: `table1_b2048_tests.log`.

- `src/table1_smoke.py`: tested trial guard permits at most 2,048 prompts only
  for benchmarks, keeps other trial stages at 512 and rejects batches larger
  than their prompt set (avoiding false B2048 labels on smaller actual batches).
- `tests/test_table1_smoke.py`: eight additional trial-boundary tests, observed
  failing before implementation and passing afterward.
- `scripts/modal_table1_speed.py`: apply guard locally/remotely, record actual
  batch sizes, warmup/allocated/reserved memory, device capacity, capability,
  detailed OOM phase and opt-in allocator environment; production model
  forwards and selected config unchanged.
- README files: updated bounds and links; prior reports/logs preserved.

Profile/workspace `tungkieu6868`; interval 2026-10-01 to 2026-11-01 exclusive,
calendar-month default, not an invoice-cycle anchor confirmed by the user.
Pre/post raw billing snapshots were saved separately. Latest CLI billing still
reported $27.5722 workspace-wide usage, excluding both new apps due to billing
lag; their actual costs remain **pending**, not zero. Each app has a conservative
$3 estimate in the local ledger; both are terminal, not active. Monthly budget,
remaining credits and hard limit are unavailable; dashboard budgets were not
inspected. No GPU app is left running by this trial.
