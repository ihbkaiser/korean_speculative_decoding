# Proposal-boundary data expansion experiment

- **Question:** Does the higher rejection rate for Korean N→P boundary-crossing draft proposals replicate on additional Wiki and FLORES data after comparing proposals with similar prompt, model pair, proposal slot, token length, and model uncertainty?
- **Hypothesis:** The N→P CROSS vs WITHIN_SPLIT rejection difference remains positive in both workloads after matched comparison.
- **Baseline / variant:** Valid WITHIN_SPLIT proposals nearest to an N→P boundary / valid single-boundary N→P CROSS proposals.
- **Primary metric:** Prompt-cluster bootstrap risk difference in rejection probability, CROSS minus WITHIN_SPLIT, on the combined old+new traces.
- **Decision rule:** Treat the predictive association as replicated only if both workload CIs exclude zero in the positive direction, P1 and P2 point estimates are positive in each workload, matching has adequate common support, and all prespecified covariate absolute SMDs are ≤0.10.
- **Data:** 1,000 additional nonempty articles immediately following the original 1,000 from Wikimedia Wikipedia `20231101.ko/train` at immutable revision `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`; all 997 FLORES-200 English→Korean `dev` rows from the already-hashed original release archive. FLORES devtest remains the original sample and is not reused.
- **Models / decoding:** P1 Qwen3-0.6B→1.7B and P2 Qwen3-1.7B→4B, pinned revisions from `src/e2_model_pairs.py`; greedy, K=4, FP16, SDPA, up to 128 new tokens. Each pair runs separately on the available RTX A4000.
- **Measurement audit:** Independently compare the first 32 target-only continuations per workload and pair with speculative outputs; save all valid accepted proposals and the first rejection, with draft/target entropy at the decision prefix. Exclude invalidated suffix proposals.
- **Seed / resume:** Seed 3090; deterministic source order; prompt-level append-only checkpoints every 10 prompts.
- **Budget:** Correctness pilot of 32 prompts/workload/pair, then complete the 1,000+997 prompt expansion if parity passes; wall-clock ceiling 24 hours for the full SD trace run.
- **Exploratory-only:** This expanded observational trace study can validate prediction/association. It cannot establish that morphology itself causes rejection because candidate token identity and linguistic context are not randomized.

## Pilot outcome and execution decision (2026-10-01)

- The first 32 prompts in each workload × pair cell passed independent target-greedy parity: **128/128 exact output matches; zero failures**.
- Per-cell pilot elapsed times were 743 s (Wiki/P1), 354 s (FLORES/P1), 867 s (Wiki/P2), and 346 s (FLORES/P2), including the parity reference generation.
- Scaling those measured times to 1,000 Wiki and 997 FLORES prompts per pair projects about 20 h, within the predeclared 24 h ceiling. The full source sets are therefore being collected; no sample cap is applied.
- Pilot traces are retained as the first 32 rows of each append-only trace. The full run resumes from them and records all remaining prompts in source order.
