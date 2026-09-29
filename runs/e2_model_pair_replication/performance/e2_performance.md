# E2 performance benchmark

Benchmarked on the first 32 source prompts, with the three FP16 models resident on physical GPU 7 (mapped to `cuda:0`).
The SD decoder is sequential and cache dependent; it was not rewritten for cross-prompt batching. Teacher-forced scoring is the batched stage.

Selected target-only generation batch size: **1**.
Selected teacher-forcing batch size: **1**.
Attention backend: **sdpa**; FlashAttention 2 installed: **False**.

| Pair | Stage | Backend | Batch | Prompts/s | Generated tokens/s | Wall s | Peak allocated GB | Peak reserved GB | Exact parity |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| P1 | target_reference | sdpa | 1 | 0.295 | 37.78 | 108.42 | 11.87 | 11.89 | True |
| P1 | target_reference | sdpa | 4 | 1.064 | 136.18 | 30.08 | 12.13 | 12.15 | False |
| P1 | target_reference | sdpa | 8 | 2.129 | 272.57 | 15.03 | 12.44 | 12.54 | False |
| P1 | target_reference | sdpa | 16 | 4.479 | 562.14 | 7.14 | 13.06 | 13.19 | False |
| P1 | teacher_forced | sdpa | 1 | 16.522 | 2114.87 | 1.94 | 11.98 | 13.19 | True |
| P1 | teacher_forced | sdpa | 4 | 57.523 | 7362.89 | 0.56 | 12.48 | 13.19 | False |
| P1 | teacher_forced | sdpa | 8 | 78.333 | 10026.65 | 0.41 | 13.15 | 13.24 | False |
| P1 | teacher_forced | sdpa | 16 | 82.487 | 10558.32 | 0.39 | 14.49 | 14.61 | False |
| P2 | target_revision_validation | sdpa | 1 | 0.232 | 29.33 | 138.12 | 11.87 | 14.61 | False |
| P1 | speculative_decode | sdpa | 1 | 0.117 | 14.96 | 273.87 | 11.97 | 14.65 | True |
| P2 | speculative_decode | sdpa | 1 | 0.097 | 12.29 | 329.66 | 11.98 | 14.66 | True |

Parity means target continuation token IDs and stopping positions match the singleton cached reference for target generation; for teacher forcing, both draft and target argmax IDs and disagreement flags match the batch-size-1 full-sequence reference. SD rows verify exact output IDs, proposal IDs, valid acceptance/rejection decisions, and stopping position against target greedy.

Full-token-vocabulary entropy is reduced in chunks of eight sequence positions during Pass B. No entropy, full-vocabulary softmax, per-token file write, or CUDA synchronization is performed inside the sequential SD hot loop.
