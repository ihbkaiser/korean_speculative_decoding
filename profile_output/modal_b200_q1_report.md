# Q1: bounded Modal B200 trial and 20,000-prompt estimate

Completed 2026-10-10 (Asia/Bangkok). No full run was started. Only Q1 was
measured; these estimates must not be reused for Q2/Q3/M1/G1 or summed as if
they described the whole five-pair Table 1.

## Measured result

The production-path pilot processed **256 actual Korean prompts**, at most
128 input and 128 output tokens, speculative K=4, microbatch B=128. It wrote
resumable JSONL checkpoints, references/events Parquet, and morphology output.

- SD + checkpoints + final Parquet: **21.903 s**, **11.688 prompts/s**.
- SD including tokenizer/model setup: 29.246 s.
- Morphology including tokenizer/Kiwi setup: 8.923 s; 31,516 aligned rows.
- Peak GPU allocation: 12.913 GiB; checkpoint JSONL: 35,742,260 bytes.
- Linear SD estimate: `20,000 / 11.688 = 1,711 s`, or **28.5 minutes**.
- Linear alignment estimate including repeated setup: **11.6 minutes**.
- Planning estimate for SD + alignment: **about 35–45 minutes**. This is a
  rough allowance, not a confidence interval or guaranteed completion time.

The sum of the two literal linear extrapolations is 40.1 minutes. Alignment
includes fixed setup that is not repeated every 256 prompts in a full run.
Model setup, final aggregation and company shared-filesystem behavior are not
reliably predicted by this small trial. Checkpoints alone extrapolate to
about 2.8 GB; host RAM and large-run final materialization were not profiled.

## Batch sweep

Each fast variant decoded the same first 128 frozen-pool prompts. The old
production baseline used only the first 8, to bound test time. There were no
fallbacks on those 8 baseline prompts; this does not estimate the long-run
fallback frequency. Times below exclude checkpoint/morphology and model load.

| Variant | Batch | Sample | Seconds | Prompts/s | Linear SD time / 20,000 |
|---|---:|---:|---:|---:|---:|
| Current FP32/eager + reference pass | 1 | 8 | 38.586 | 0.207 | 26.8 hours |
| New BF16/SDPA | 8 | 128 | 88.168 | 1.452 | 229.6 min |
| New BF16/SDPA | 16 | 128 | 50.043 | 2.558 | 130.3 min |
| New BF16/SDPA | 32 | 128 | 27.378 | 4.675 | 71.3 min |
| New BF16/SDPA | 64 | 128 | 14.720 | 8.695 | 38.3 min |
| New BF16/SDPA | 128 | 128 | 9.806 | 13.053 | 25.5 min |

B=128 is the fastest **tested** setting, not a global optimum. B=256/512,
CUDA graphs, compilation, FlashAttention-specific builds and other engines
were not tested. The second trial confirms B=128 through the production IO
path on 256 prompts rather than just a single decode batch.

The baseline comparison changes decoder, batching, reference policy, dtype
and attention backend. It is not a controlled attribution of speedup to the
new decoder alone. Per-batch durations, memory and scalar diagnostics are in
the raw JSON, with no repetitions or statistical confidence interval.

## Correctness limitation — important for Table 1

At B=128, **4 of the 8 independently checked continuations differed from
BF16/SDPA scalar target generation** (indices 3, 5, 6, 7). Other batch sizes
also differed. The check was diagnostic only: no scalar fallback was run.
This trial does not prove that the mismatches are harmless numerical noise.

Every emitted token is selected using target block logits, but this is not
independent scalar-parity validation and is not an exact reproduction of the
old FP32/eager run. Bonus tokens also change proposal boundaries. Metadata
marks this explicitly; outputs and checkpoints are isolated from `runs/table1`.
Do not mix these results with the strict baseline or describe the fast trial
as scalar-exact. A strict scientific replacement still needs a declared
validation contract and a broader parity/integrity evaluation.

## Runtime and provenance

- GPU: NVIDIA B200. PyTorch 2.14.1+cu130; CUDA 13.0; Transformers 5.19.0.
- Fast path: BF16, SDPA, `use_remove_padding=false`; no explicit FA2 backend.
- Compute capability and FlashAttention package version were not collected.
- Q1 draft: Qwen3-0.6B-Base, commit
  `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`.
- Q1 target: Qwen3-1.7B-Base, commit
  `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`.
- Sources: `configs/table1_pipeline.yaml` baseline;
  `configs/table1_fast_b200.yaml` pilot, overridden to 256 prompts, B=128.
- Data: existing Modal frozen `workspace/data/prompts_40k.parquet`, first
  128/256 prompts of Q1's common_20k split. No new sampling/download.
- [Sweep app](https://modal.com/apps/tungkieu6868/main/ap-CXEvEt5sqBfFneTGaAViKA)
  and [pilot app](https://modal.com/apps/tungkieu6868/main/ap-5dm1ayXK5w8UJT3eCoTYe0).
- Raw evidence: `modal_b200_q1_initial.json`, `modal_b200_q1_pilot.json`,
  matching `.log` files. Both apps completed and stopped.

Modal billing refresh reported $0.6469 for the sweep and $0.1160 for the pilot
(about $0.76 combined, before any credits/reservations). Profile/workspace:
`tungkieu6868`. Reporting interval: 2026-10-01 through 2026-11-01 exclusive,
calendar-month default, not a verified invoice-cycle anchor. Total reported
workspace usage in that interval was $24.5838, including unrelated prior jobs.
These are possibly lagging usage figures, not final invoices. Remaining credits,
monthly budget, hard limit and dashboard budget controls were unavailable or
not inspected; no remaining-credit claim is made. Local run estimates were
$3 and $1 respectively; the ledger was completed using the reported run costs.

## TQDM and resume

SD bars show completed prompts, prompts/s and ETA; fast progress updates after
each checkpointed microbatch. Resume initializes at the completed prompt
count rather than zero. Target references and morphology have separate bars.
The pilot log contains the real output, including:

```text
Q1 SD (B=128): 100%|██████████| 256/256 [00:21<00:00, 11.78prompt/s]
Q1 morphology: 100%|██████████| 256/256 [00:03<00:00, 67.47prompt/s]
```

The CLI rejects trials above 512 prompts or 128 output tokens, accepts only
inspect/benchmark/pilot stages, and has no full-run stage. Q1's separate fast
YAML now selects measured B=128; other pairs retain unmeasured conservative B=8.

## Instrumentation and change log

| File | Change | Purpose |
|---|---|---|
| `scripts/modal_table1_speed.py` | Created | Bounded B200 sweep/pilot, isolated frozen source upload, runtime identity, checkpoint IO timing, JSON metrics and full log markers |
| `src/table1_runner.py` | Modified | TQDM for references and both SD paths; resume count, checkpoint-aware microbatch updates |
| `src/table1_morphology.py` | Modified | Prompt-level morphology TQDM |
| `requirements.txt` | Modified | Explicit TQDM dependency |
| `tests/test_table1_fast_runner.py` | Modified | Progress/resume regression and measured per-pair config checks |
| `configs/table1_fast_b200.yaml` | Modified | Q1 B=128 only; no batch increase for untested pairs |
| `scripts/run_company_table1_fast.sh` | Modified | Honor per-pair YAML batches unless explicitly overridden |
| `README.md`, `profile_output/README.md` | Modified | Bounded test commands, progress behavior and report pointer |
| `profile_output/modal_b200_q1_{initial,pilot}.{json,log}` | Created | Real B200 raw evidence |
| `profile_output/modal_b200_q1_report.md` | Created | Method, estimates, limitations and changelog |

No profiler hooks were inserted into model forwards or tight CUDA loops.
Verification: full local suite passed (88 passed, 3 skipped), plus actual
B200 generation/checkpoint/alignment pilot. A 20,000-prompt run and scientific
equivalence to the strict baseline remain unverified. Benchmark-only helpers
can be removed independently; production progress bars can be kept.
