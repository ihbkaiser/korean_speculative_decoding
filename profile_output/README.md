SD performance measurements

Latest: [Q1 batch 2048 and allocator trial](modal_b200_q1_b2048_report.md).
B2048 needed an opt-in allocator setting and only improved measured throughput
by 2.6%, so the production default remains B256.

Five-pair validation: [batch 256 and B200 smoke](modal_b200_b256_smoke5_report.md).
All five pairs passed bounded SD + alignment + table output checks, with no
scalar fallback. Earlier: [Q1 batch-128 trial](modal_b200_q1_report.md).
This includes real-model throughput, checkpoint/Parquet overhead, morphology,
TQDM evidence and the important scalar-parity limitations. No full run was started.

Historical local SD performance check, 2026-10-09

`table1_fast_synthetic_rtx3050ti.json` was generated with:

```bash
py -3.12 scripts/benchmark_table1_fast.py --synthetic --prompts 16 --max-new-tokens 32 --batch-sizes 1 4 8 16 --output profile_output/table1_fast_synthetic_rtx3050ti.json
```

Hardware: RTX 3050 Ti Laptop, 4 GB; PyTorch 2.3.1+cu121 and Transformers
4.44.2. Models are randomly initialized tiny Llamas (2 layers, hidden size 128,
vocabulary 4096), FP32. This workload measures launch, Python, audit collection
and synchronization overhead. It is not a model-quality evaluation or a B200
throughput estimate. Windows PyTorch reported no FlashAttention build.

| Decoder | Prompt batch | Tokens/s | Exact scalar outputs |
|---|---:|---:|---:|
| Previous cached block verifier | 1 | 26.8 | 16/16 |
| New microbatch verifier | 1 | 40.0 | 16/16 |
| New microbatch verifier | 4 | 153.2 | 16/16 |
| New microbatch verifier | 8 | 308.0 | 16/16 |
| New microbatch verifier | 16 | 553.3 | 16/16 |

Timing includes audit statistics but excludes loading, scalar reference
generation, checkpoint serialization and morphology alignment. The previous
decoder is timed with block verification enabled, without its production
reference pass or scalar retries. Outputs are checked against an independent
scalar reference once; differences are reported rather than repaired. Results
come from one run, with no confidence interval.

Source inspection found repeated `.item()` synchronizations for each proposal
statistic, a full reference-generation pass before progress, and full-prompt
scalar retries. The new decoder batches vocabulary statistics and prompt
generation, uses one target call per round, adds bonus tokens, and maintains
each row's accepted KV prefix. These are code-level findings rather than an
Nsight profile of the remote B200.

To measure the actual models on B200:

```bash
python3 scripts/benchmark_table1_fast.py --pair Q1 --prompts 16 --max-new-tokens 32 --batch-sizes 4 8 16
```

Files created or changed for performance measurement:

| File | Change | Purpose |
|---|---|---|
| `scripts/benchmark_table1_fast.py` | Created | Standalone synchronized benchmark, scalar comparison and peak VRAM measurements |
| `profile_output/table1_fast_synthetic_rtx3050ti.json` | Created | Raw local measurements and runtime identity |
| `profile_output/README.md` | Created | Method, limitations and measurement changelog |
| `README.md` | Modified | Commands for the local smoke check and B200 measurements |

No measurement hooks or profiler timers were inserted into model forward
methods. Production progress reports elapsed time, prompts/s and ETA after
completed microbatches.
