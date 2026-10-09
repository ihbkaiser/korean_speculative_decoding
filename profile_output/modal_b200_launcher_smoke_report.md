# Real launcher smoke on Modal B200: PASS

Date: 2026-10-10, Asia/Bangkok. Production source base: `09d5e3e`.
This was a bounded real-model test, not mock inference or a full Table 1 run.

## Scope and evidence

One NVIDIA B200 ran the actual `bash run_table1.sh all --output-dir ...`
launcher, sequential GPU pairs with overlapping CPU morphology, twice.
Each pair used the same first 256 frozen Korean prompts, batch 256, BF16/SDPA,
greedy K=4, max input/output 128/128. Only an isolated fixture copy changed
prompt counts and mapped pinned model paths to the existing Modal cache.
The repository's production split counts remain 20k/40k/20k/20k/20k.

An inherited `TABLE1_CONFIG` pointing at the strict config was deliberately
injected. The launcher still printed `FAST_ONLY` and used `microbatched` with
`batched_native`. No independent scalar reference pass, scalar repair,
decoder retry or automatic batch reduction ran. Scalar equality is not claimed.

| Pair | Status | SD + checkpoint/Parquet seconds | Prompts/s | Proposal events |
|---|---|---:|---:|---:|
| Q1 | PASS | 24.829 | 10.311 | 34,356 |
| Q2 | PASS | 24.908 | 10.278 | 37,331 |
| Q3 | PASS | 30.227 | 8.469 | 32,691 |
| M1 | PASS | 22.834 | 11.211 | 33,504 |
| G1 | PASS | 24.681 | 10.372 | 27,587 |

All five reference/event artifacts passed prompt uniqueness/count, finite
entropy/logprob/rank, accepted-token consistency, and no-scalar-reference checks.
Every pair completed CPU alignment; the final five-row smoke table was COMPLETE.
The output directory contained spaces; logs, checkpoints, runtime manifests and
results stayed under the requested output root, not the staged repository runs.

Initial launcher wall time: **289.175 s**, including model/tokenizer startup,
GPU inference, CPU alignment and aggregation. Resume: **66.254 s** including
Python startup, metadata and table rebuilding. All five SD and alignment stages
returned `already_complete`. SHA256 of each full `progress.jsonl` was unchanged;
no prompt was regenerated. There was no weight-loading pass on resume.

These are single cold-batch smoke timings, not a warm steady-state benchmark.
Only one measured batch exists per pair, so no mean excluding the first batch,
confidence interval, or full-run ETA is asserted. GPU peak allocation was not
collected across the separate launcher child processes. SDPA was selected, with
no explicit flash-attn backend and no remove-padding optimization enabled.

## Runtime and artifacts

Python 3.11.12; Torch 2.14.1+cu130; CUDA 13.0; Transformers 5.19.0;
Kiwi 0.24.0; B200 capability 10.0; driver 580.95.05; reported GPU memory
183,359 MiB. No special allocator setting. Pinned snapshot names match the
production config. Source file SHA256 values are in the summary because the
isolated Modal source upload does not contain `.git`.

- [Passing app](https://modal.com/apps/tungkieu6868/main/ap-PqtOVoqdpr20SRjfJPPpvx)
- `modal_b200_launcher_smoke_20261010_v2.json`: machine-readable summary.
- `modal_b200_launcher_smoke_initial.log`: raw UTF-8 launcher console log.
- `modal_b200_launcher_smoke_resume.log`: raw UTF-8 resume console log.
- `modal_b200_launcher_environment.json`: allowlisted runtime provenance.
- `modal_b200_launcher_smoke_20261010_v2.log`: complete local Modal CLI capture.
- Persistent volume: `korean-speculative-decoding-table1-artifacts`.
- Volume path: `/benchmarks/launcher_no_fallback_20261010_v2/`.
- Within it, `output with spaces/` retains all raw progress/events/references,
  alignment, launch history, per-stage logs and table artifacts, without sampling.

## Failures found, resolved and preserved

The first local CLI attempt could not render Unicode under the Windows console
encoding. UTF-8 mode resolved that operational issue; its empty app was stopped.
The first GPU attempt stopped before model loading because the Modal source
allowlist omitted root-level `run_table1.sh`. The snapshot helper now includes
it, and a regression test checks the actual snapshot. That failed app/log was
preserved; the passing app is a fresh invocation of all five pairs, not reuse
of a partially successful GPU trial.

M1 emitted a tied-weights/config warning; inference and alignment completed.
No model weights, tokenizer contract or production inference logic were changed
to silence it. A separate Windows stand-in test issue used process IDs as file
names, allowing PID reuse to overwrite a synthetic call record; UUID file names
fixed that test harness issue. This did not affect production checkpoints.

## Billing and cleanup

Profile/workspace: `tungkieu6868`. Billing interval 2026-10-01 through
2026-11-01 exclusive, calendar-month default, not a verified invoice-cycle anchor.
Latest CLI-reported cost: passing app **$0.5265**, failed GPU source-upload test
**$0.0089**, total **$0.5354**. Workspace month usage reported **$29.1783**, including
unrelated jobs. Billing can lag and precedes credits/reservations; remaining
credits, monthly hard budget and dashboard budget controls were unavailable.
Raw billing JSON is kept locally and excluded from Git. All three apps created
by this task are stopped with zero containers/tasks remaining.

Local verification after the source-snapshot fix: **128 passed, 3 skipped**.
Highest verified evidence: bounded real-model B200 end-to-end launcher smoke
plus completed-checkpoint resume. Full-dataset RAM/IO behavior, recovery from
real mid-batch GPU preemption, and independent scalar parity remain unverified
by this smoke (partial-batch interruption remains covered by local tests).

## Reproduction (bounded)

```bash
modal run --profile tungkieu6868 scripts/modal_table1_speed.py --stage launcher-smoke --prompts 256 --batch-sizes 256 --max-new-tokens 128 --label new_unique_launcher_label --output profile_output/modal_launcher_smoke.json
```

Use a fresh label. The smoke stage rejects more than 512 prompts and does not
expose a full-run route. The runbook documents the production command separately.
