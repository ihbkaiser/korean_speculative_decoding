# Invalid T6-b cross-family judge run (2026-11-26)

These generated-text judge outputs and their analysis summaries are quarantined because the shard-streaming loader left non-persistent buffers uninitialized. The rerun uses the standard loader and has separate outputs under `results/rebuttal/cross_family/`.

| Stored judge output | Entropy, mean (nats) | Top-2 margin, median | `model_storage_mode` |
|---|---:|---:|---|
| `qwen4/ministral8` | 11.26 | 0.172 | `streamed_shards_to_cpu` |
| `qwen8/ministral8` | 11.26 | 0.172 | `streamed_shards_to_cpu` |
| `ministral8/qwen8` | 5.81 | 0.375 | `streamed_shards_to_cpu` |
| `qwen4/qwen4` (self-gate; retained in the active results folder) | 1.77 | 2.0 | `local_checkpoint` |

The invalid runs are preserved as recorded. The Qwen3-4B self-gate and all T6-a human-text outputs remain in their original locations.
