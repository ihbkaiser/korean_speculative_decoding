# Hardware execution plan v3 — RTX 3090 + two B200 on one host

Date: 2026-10-06. User confirmed: two B200 on the same host, available long term. This is an execution design, not a report of completed jobs. No company machine has been inspected or accessed. The v2 cards E00–E18 remain defined in `NAACL_EXPERIMENT_BLUEPRINT.md`; this document assigns them and adds L01–L06. The accompanying YAML/CSV are planning manifests, not accepted runner input.

## 1. Decision and paper scope

Use the 3090 workstation for small-pair pilots, trace correctness, data curation and CPU analysis. Use B200-A for the canonical Qwen3-Base confirmation/predictor corpus and capacity grid. Use B200-B for independent-family replication and large targets, then reserve it for runtime measurements. Keep one draft/target pair on one GPU by default: tensor parallel size 1. Two B200 should run two independent research jobs when memory permits.

The hardware supports a stronger paper without changing its two-analysis structure:

1. Analysis I: actual-proposal disagreement, directional/local morphology, stability across data, draft/target scale and model families.
2. Analysis II: label validity, comparable contexts and incremental prediction beyond confidence/frequency/identity; linguistic diagnostics.
3. Short operational implication: a cost-aware draft-length controller, only if incremental information and measured runtime headroom exist.

Large models strengthen external validity; they do not repair target-side labels, leakage or weak controls. MoE versus dense, Base versus IT, different families/tokenizers and different GPU architectures are separate comparison axes. No universal or causal scale law follows from this grid.

## 2. Machines and common configuration

| Resource | Default responsibilities | Models kept resident | Scheduling rule |
|---|---|---|---|
| RTX 3090 workstation, nominal 24 GB | H00/H01 small pilots; E01 annotation preparation; E03/E07/E08/E09 CPU analyses; E16 reuse; L05 hardware comparison | Q1/Q2/Q3 only for the primary assignment | One GPU job; CPU analysis can proceed if it does not disturb a timing run |
| B200-A | E02 canonical corpus; E04 Qwen capacity; E10/E11/E12 diagnostics; E13 engine/cost pilot | One Qwen draft/target pair, usually targets 1.7/4/8B | Model-target grouped queue, TP=1 |
| B200-B | E05/G1; E06 IT; L01 Qwen MoE; L02 Gemma27; optional L03 dense large/L04 IT32; E14/E15 runtime | One pair, targets 12/27/30/32/72B | TP=1; exclusive benchmark slots |
| Human annotators | E01 and validated E10 frames | No GPU dependency | Two Korean annotators; label accepted and rejected candidates |

Do not assume this company host has DGX memory, CPU RAM or NVLink solely because it contains B200 GPUs. NVIDIA's DGX specification is 1,440 GB across eight GPUs (180 GB/device); use that as a conservative full-device planning reference and inspect actual total/free memory, partitioning and topology. [NVIDIA specification](https://www.nvidia.com/en-us/data-center/dgx-b200/).

Common confirmation configuration: BF16 weights and inference; SDPA initially; batch size 1 for realized speculative trajectories; greedy; K=4; max_new_tokens=256; normal EOS; TP=1; no weight quantization or CPU offload. Keep old FP16/128-token results as discovery. Freeze package/container versions, source commit, tokenizer/weight revisions, dataset IDs, prompt IDs and GPU UUID in every artifact. Blackwell needs a supported software build; PyTorch documents Blackwell support and CUDA 12.8 wheels from 2.7. Use a validated, pinned build rather than the repo's unpinned package ranges. [PyTorch announcement](https://pytorch.org/blog/pytorch-2-7/).

For raw continuation, use the same 600-Unicode-character prefix, cut at whitespace and shortened consistently if any compared tokenizer exceeds 512 tokens. F translation cap=256 input tokens. IT cap=512; summarization cap=2048. Qwen IT must use the native template with `enable_thinking=False`, recorded in the artifact. Base continuation uses raw text. Confirm token ID/vocabulary/special-token compatibility within every proposed pair; compatibility is a gate, not an assumption from model names.

## 3. H00: runner and hardware preflight — first work item

The existing scripts are not launch-ready for this allocation:

| File | Current restriction | Required change before execution |
|---|---|---|
| `src/e2_model_pairs.py` | GPU7, A100 >=78 GiB; FP16 enforced | Explicit GPU UUID/device selection; generic memory/name validation; model registry and dtype argument |
| `scripts/run_morphology_guard.py` | GPU3, A100; FP16 | Same; replace full-block/sequential prototype for runtime claims |
| `scripts/collect_boundary_expansion_traces.py` | GPU0, RTX A4000; FP16 | Generalized device and precision selection |

Implement a new runner adapter or refactor these checks explicitly. Do not blindly remove device safeguards or relabel old output. Keep masking to a selected UUID and validate it at startup. No runnable new CLI is asserted in this plan.

H00 outputs and pass conditions:

- Hardware/software manifest: GPU UUID/name, total/free VRAM, partition/MIG state, driver, CUDA/PyTorch, host CPU/RAM, local storage, topology. Record the two B200 as A/B by UUID, not guessed physical indices.
- Immutable model/tokenizer revisions; Gemma access and text-only adapter; Qwen MoE adapter. CPU/tokenizer preflight must precede weight downloads.
- Efficient draft KV cache and one batched target verification per round, correct positions/masks, cache rollback, EOS/length handling. The present sequential verifier is a correctness scaffold, not an acceleration baseline.
- Record actual proposed token IDs/text, proposal-relative features and accepted/rejected/invalidated state. Proposals after the first failed token in a block are invalidated, not additional rejection outcomes.
- Online features use accepted history plus the current candidate, with no future block or target-derived entropy/fragmentation. Offline full-block morphology is retained as a labeled sensitivity analysis.
- Paired engine correctness on pilot prompts: token/EOS/stop parity with target AR on the SAME device/backend/dtype. Inspect every divergence before scaling; measure borderline logits and numerical effects rather than changing the contract silently.
- Cross-device audit Q1/Q2/Q3: 128 fixed prefixes, same checkpoints/dtype/software. Compare target argmax and labels on shared prefixes; record near ties. Corpus differences across hardware must be assessed before mixing shards. Keep the main confirmation on B200 to avoid an unmeasured hardware confound.

H01 pilot for each pair: v2 E00's 64 W +32 F +32 N, or 64 IT +32 summaries for an IT-only pair. These are development/pilot prompts, excluded from held-out confirmation. Add 8 warmup +32 timed development prompts per relevant workload; repeat if dispersion is too large. Measure target-only and instrumented SD seconds/prompt, output tokens, accepted tokens/round, peak VRAM, morphology CPU cost and bytes/trace. Do not forecast total hours from FLOPS or activated parameter count.

## 4. Dataset freeze and baseline allocation

W: new Wikipedia 10,000 source documents, excluding discovery/expansion articles, grouped deduplication, seed 20261006. Train4,000/dev1,000/test5,000; nested scale subset test2,000. F: original FLORES dev997 only if suitable for the declared confirmation status; audit/disclose prior inspection. Earlier devtest1012 is discovery. N: XL-Sum Korean official test550 for news continuation; official validation550 is development. I: KoAlpaca question-only test1,000 and dev300, grouped by source URL. S: summarization on the same XL-Sum test550; this is a second task, not 550 additional independent source documents.

| Job | Owner | Pair / cells | New speculative trajectories | Cards and purpose |
|---|---|---|---:|---|
| A01 | B200-A | Q2 = Qwen3-1.7B-Base ->4B-Base; W train4k/dev1k/test5k +F997 +N550 | 11,547 | E02/E09; canonical evidence and frozen predictor |
| A02 | B200-A | Q1 = Qwen3-0.6B-Base ->1.7B-Base; Wtest5k +F997 +N550 | 6,547 | E02; small target |
| A03 | B200-A | Q3 = Qwen3-0.6B-Base ->4B-Base; same W/F/N | 6,547 | E02; clean pinned replacement for ambiguous historical P3 |
| A04 | B200-A | Q4 = 0.6B-Base ->8B-Base; shared Wtest2k | 2,000 | E04; target scale |
| A05 | B200-A | Q5 = 1.7B-Base ->8B-Base; same Wtest2k | 2,000 | E04; fixed draft bridge |
| A06 | B200-A | Q6 = 4B-Base ->8B-Base; same Wtest2k | 2,000 | E04; fixed target/draft strength |
| B01 | B200-B | G1 = Gemma3-4b-pt ->12b-pt; Wtest2k +F997 +N550 | 3,547 | E05; different family/tokenizer |
| B02 | B200-B | QI1 = Qwen3-1.7B ->4B; I1k +S550, nonthinking | 1,550 | E06; IT/task robustness |
| B03 | B200-B | GI1 = Gemma3-4b-it ->12b-it; I1k +S550 | 1,550 | E06; same |

Baseline total=37,288 SD trajectories (A=30,641; B=6,647), before target-reference runs, pilots, diagnostics and timing repeats. Maximum generated OUTPUT tokens=9,545,728 at cap256, not an estimate of total processed tokens. Actual work includes draft tokens, verification, prefixes, retries and shorter EOS outputs. B200-A can be relieved by B200-B after family preflight if its queue becomes the bottleneck; use disjoint locked shards and keep all main generation on B200.

On the 3090, E03 nested data-scale analysis, E07 overlap/matching, E08 train-only frequency/identity, E09 logistic/HistGB predictors and document bootstrap are CPU jobs. No large-model fine-tuning is needed for E09. Save CPU resources on B200 host for the GPU queues. Do not call CPU tasks GPU experiments.

## 5. Additional large-model cards

All large cards inherit actual-proposal labels, group-level split/CI, common raw prefixes and proposal-side controls from v2. Run them irrespective of effect sign after technical gates; do not promote only models with a positive morphology effect. Report incompatibility/coverage failures and all completed registered cells.

### L01 — Does the signature persist with a large Qwen3 Base MoE target?

- Models: `Qwen/Qwen3-1.7B-Base -> Qwen/Qwen3-30B-A3B-Base` (L01a), and `Qwen/Qwen3-4B-Base -> Qwen/Qwen3-30B-A3B-Base` (L01b).
- Owner: B200-B, TP1. Same target grouped to reduce reloads.
- Data: shared Wtest2k +F997 +N550 per pair =7,094 trajectories for both. Required large-model extension for this hardware plan.
- Compare L01a with Q2/Q5 at fixed 1.7B draft, and L01b with Q6 at fixed 4B draft. Add shared-prefix scoring on 1,000 registered natural prefixes to separate target-generated context changes from next-token agreement.
- Primary: actual N->P/CROSS versus SPLIT adjusted risk difference, 95% document-cluster CI, exposed candidate count, distinct documents and nominal/particle identities. Also report ALL-boundary distributions and descriptive burden: frequency multiplied by excess disagreement.
- Secondary: target AR Korean output/morphology coverage, overall draft agreement and confidence-conditioned effect. MoE architecture is different; interpret as a new operating point, not a clean dense size-only effect.
- Negative: effect attenuation or disappearance limits the claim; do not drop the row. Insufficient support yields imprecision, not a null mechanism conclusion.
- Artifact: T1 large-target panel and appendix MoE table.
- Official model has 30.5B total/3.3B active parameters. BF16 memory is based on TOTAL resident parameters (~61 GB of weights), not 3.3B. [Model card](https://huggingface.co/Qwen/Qwen3-30B-A3B-Base).

### L02 — Independent-family large target, with an optional fixed-target draft contrast

- Models: `google/gemma-3-4b-pt -> google/gemma-3-27b-pt` (L02a, required extension); optional `google/gemma-3-12b-pt -> google/gemma-3-27b-pt` (L02b).
- Owner: B200-B, TP1; text-only validated adapter. Same 2,000W +997F +550N per pair. Required L02a=3,547 trajectories; optional L02b adds3,547.
- Compare G1 versus L02a at fixed 4B draft and L02a versus L02b at fixed 27B target. Report same-prefix diagnostic and coverage like L01.
- Primary success: a measured effect with interpretable interval across a genuinely different model/tokenizer family; the precise direction can differ. A predeclared broad all-boundary analysis prevents a single rare N->P event from carrying the entire result.
- Artifact: T1 family/scale panel; appendix draft-strength result if L02b is run.
- These are pretrained variants. Nominal paired BF16 weights: 4+27 ~62 GB; 12+27 ~78 GB. Actual multimodal/text adapter memory must be profiled. [Official Gemma27 PT card](https://huggingface.co/google/gemma-3-27b-pt).

### L03 — Dense Base target scale through 72B, fixed draft

- Models: `Qwen/Qwen2.5-1.5B` as draft; targets `Qwen/Qwen2.5-7B`, `Qwen/Qwen2.5-14B`, `Qwen/Qwen2.5-32B`, `Qwen/Qwen2.5-72B`. These IDs without `-Instruct` are the Base series. Optional 7B draft ->32B target for a second draft-strength check.
- Owner: B200-B after required main/family/large cards. Priority P2; worth doing with sustained access, but it does not block the two-analysis paper.
- Data: shared Wtest2k +F997 per pair, 4x2,997=11,988 trajectories. News550 can be a separately registered follow-up; do not silently add it to the count.
- Same raw prompts and 1,000 shared-prefix positions across all four targets. Primary is an interaction/trend estimate with uncertainty, supported by the per-cell effects/counts, not significance counting. Confidence distributions and tokenizer compatibility are explicit controls. Same generation, labeler and feature code.
- Plot adjusted morphology-associated disagreement against target capacity at fixed draft; also plot overall agreement. This addresses whether stronger targets remove or amplify the signature within a dense Base series. Qwen2.5 is a second model generation, not an independent vendor family.
- 72B gate: actual card reports72.7B; target weights ~145.4 GB BF16 plus ~3GB draft, then buffers/cache. Start batch1/input<=512/output256, no offload/quantization. Fits are estimates; proceed only if peak-memory pilot retains usable headroom. Stream teacher-forced logits in chunks (start16/32), retaining selected summaries rather than every full-vocabulary tensor. Never store all per-prefix logits by default.
- If 72B does not fit one GPU, use TP2 only after topology/backend/cache parity validation, pause both independent queues and label TP2 separately. A TP2 cell is analysis-only unless every compared timing baseline uses the same topology. Do not substitute a quantized checkpoint into the BF16 series without a separate precision comparison.
- Artifact: appendix scale curve, or a compact main figure if the trend is substantive. [Official72B](https://huggingface.co/Qwen/Qwen2.5-72B), [32B](https://huggingface.co/Qwen/Qwen2.5-32B), [14B](https://huggingface.co/Qwen/Qwen2.5-14B), [7B](https://huggingface.co/Qwen/Qwen2.5-7B), [1.5B](https://huggingface.co/Qwen/Qwen2.5-1.5B).

### L04 — A 32B instruction target for the short method section

- Pair: `Qwen/Qwen3-4B -> Qwen/Qwen3-32B`, native chat, nonthinking, greedy; optional8B draft ->32B target held separate.
- Owner: B200-B. Priority P2. First I1k +S550=1,550 analysis trajectories for 4B draft. This is a post-trained target, not `Qwen3-32B-Base`. [Official card](https://huggingface.co/Qwen/Qwen3-32B).
- If E09 offers incremental information and E13 has real cost headroom, benchmark on distinct frozen I500 +S300, three counterbalanced warm repeats, exactly the same policies/engine/hardware as E14. Tune on Idev300 and XL-Sum validation, never final test.
- Method metrics: end-to-end generation latency and speedup relative to target AR, relative gain over strongest tuned confidence-only scheduler, accepted tokens/round, peak VRAM, morphology overhead and zero token/EOS/stop divergence under the declared greedy contract.
- Why useful: a practical larger dense target may provide a different cost balance than an 8B target. Do not infer runtime improvement from the weight ratio; profile it. Transfer Q2 predictor zero-shot as a clearly named test; any adapted IT predictor must use separate IT train/dev and be reported separately.
- Negative runtime result is publishable as a bounded operational implication; do not expand to training a new drafter merely to obtain a positive result.

### L05 — Hardware dependence of the controller benefit

- Pair: Q3 (0.6B-Base ->4B-Base), fits the assigned3090 small-pair regime. Same frozen W500 prompts on3090 and B200; greedy output256, batch1, three counterbalanced repeats.
- Owner:3090 plus an exclusive B200-B slot; conditional on working cache/batched verifier and a useful controller.
- Policies: AR, best fixed K, strongest confidence-only scheduler, confidence+morphology. Include entropy baseline if it is stronger on development. Same functionality/backend/dtype across policies on each device.
- Report speedup with each device's OWN AR baseline, and method gain over each device's confidence baseline; do not compare raw3090 latency with B200 latency as a method gain. Same-prefix numerical audit first.
- Include the full CPU parser/scheduler cost. A faster B200 can make CPU morphology overhead proportionally more important; the direction of controller benefit is an empirical question. CPU platform differs too, so call this system-level robustness rather than isolating GPU architecture causally.
- Artifact: hardware appendix table, or a limitation paragraph if negative.

### L06 — Efficient reference/scoring batches and uncertainty-driven support expansion

- Owner: B200-A or B after mandatory corpus. This card improves evidence efficiency and effective sample size; it is not an extra speed claim.
- Target-only generation/teacher-forced scoring: test batches1/4/8/16 with32 registered prefixes; include32 only if memory and argmax/label parity pass. Keep realized primary SD trajectories batch1. Batched reference scoring does not reproduce the speculative visited-prefix distribution by itself.
- Data support: after dev/pilot, estimate exposed N->P documents, not only tokens. Historical event prevalence is roughly0.24–0.82% across audited Wiki/FLORES cells;5k outputs may still have few independent exposed documents. Use document-cluster bootstrap and report ESS/coverage.
- If support/pilot CI precision predicts inadequate confirmation, PREDECLARE an additional independent Wtest5k source-document tranche before inspecting final test effects; apply it to a fixed subset Q2/Q5/L01a/L02a, not only a favorable cell. Adds4x5k=20k trajectories. Set a development-based precision goal (e.g. CI half-width about5pp for the selected risk difference), estimate via cluster resampling; no guarantee from nominal prompt count. Do not stop sampling when p<.05.
- Controlled particle/frequency frames (E10) supplement linguistic understanding but do not replace natural-data confirmation or count as additional independent documents.

## 6. Runtime controller design and gates

Keep the method small: predict current-candidate rejection risk from draft confidence/margin/entropy and optionally online morphology, then choose whether to continue drafting within a cap8 or16. Current-candidate morphology can only decide whether to add the next candidate; the already generated candidate's cost is accounted for. It cannot undo its own drafting cost. Target verification and correction remain unchanged. No token blacklist and no target/future information.

Development: E13 on200W +200F development prompts for Q4/Q5, or registered large-target development prompts; profile actual draft, batched verification, cache and CPU costs. First compute an oracle WITHIN THE SAME policy class/cost model to estimate headroom. Oracle results are diagnostic, not deployable speedups.

E09 predictors: train Q2 Wtrain4k, select on Wdev1k; logistic C in[.01,.1,1,10,100], no class weights; HistGB max_iter200/max_leaf_nodes15/lr.05/min_samples_leaf50/l2=1. Fit confidence-only and confidence+morphology with equal tuning budget; freeze before test. Use log loss and Brier as primary, AUPRC/risk coverage secondary, paired document-cluster intervals. Token lookup and train-only identity/frequency controls are required. An effect association alone is insufficient for a method claim.

E14 final: Q4/Q5 on500W +500F, three counterbalanced repeats. Compare AR, best tuned fixed K in{1,2,4,8,16}, entropy/SVIP-style stopping, strongest confidence-only stopping, confidence+morphology. Threshold tuning on development only; same verified cache/backend and policy functionality. Include wall-clock morphology CPU cost, CUDA synchronization at measurement boundaries, model reload/warmup conventions, order and peak VRAM. Report natural EOS behavior and end-to-end generation latency excluding model loading but including prompt prefill. Hardware/environment is held fixed within each comparison.

Gates: E00 correctness -> E01 measurement validity -> E09 incremental risk information -> E13 measured headroom -> E14 final runtime -> E15 focused ablations. If runtime fails, retain the two-analysis paper with a short cost-aware negative implication. Do not claim lossless stochastic sampling from greedy parity. E12 exact sampling is a distinct optional implementation with proper residual distribution and distributional validation.

Because both B200 share a host, reserve benchmark slots with the other GPU idle initially. Record stable CPU load, GPU clocks/power and thermals; prefetch data/model weights, avoid downloads, trace compression and heavy parser queues during measurements. If independent jobs must continue, test co-run interference and record it rather than assuming isolation. Use disjoint CPU sets only after inspecting NUMA/topology. Avoid memory-filling parallel GPU processes solely to maximize utilization during latency measurements.

## 7. Concrete schedule by milestone, not guessed calendar hours

| Phase | RTX3090 workstation | B200-A | B200-B | Exit artifact |
|---|---|---|---|---|
| S0 | Freeze data/exclusions; audit proposal labels; small-pair adapter tests; annotation prep | H00/H01 Q2/Q5; canonical backend/cache preflight | H00/H01 Gemma/MoE; access, memory and tokenizer checks | Revisions, manifests, pilots; ETA measured |
| S1 | E01 annotation underway; CPU feature/schema validation | A01 train/dev first; then frozen test/F/N | B01 G1 replication | T1 first rows; E09 locked configs |
| S2 | E03/E07/E08/E09 CPU from completed shards | A02/A03 then A04–A06; shared-prefix scores | L01a/b and L02a; group by target to reduce reloads | Complete baseline Qwen/Gemma/large T1; scale figure |
| S3 | Human-audited E10 frames; frequency/identity-held-out reports | E10/E11/E12 focused diagnostics; E13 engine costs | B02/B03 IT; then L03 dense scale if bandwidth/time allow | Analysis II T2; diagnostic appendix |
| S4 | L05 only if controller gates pass; otherwise paper-ready figures | Finish analysis; idle during shared-host timing | Exclusive E14/E15; optional L04 large IT32/runtime | T3 or documented negative feasibility |
| S5 | Final claim/artifact audit, tables and narrative | L06 support expansion only if preregistered; resolve missing cells | Optional L02b, L03 7->32 or remaining checks | Frozen results, provenance and limitations |

Run sequence can change to reduce model reloads after all prerequisites pass. Record the change. Do not postpone independent-family replication until the end. If one B200 becomes unavailable, B200-A absorbs mandatory B jobs after main Qwen completion; keep L03/L04/L02b optional and use3090 for additional small-pair development, not unvalidated mixed-hardware main shards.

Recommended finite workload: baseline37,288 +required large L01a/b/L02a10,641 =47,929 trajectories. Optional L03 four dense targets adds11,988, L04 adds1,550: total61,467. Optional L02b adds3,547 and L03 7->32 adds2,997: maximum listed analysis corpus68,011 BEFORE separate support expansion, references, pilots, diagnostics and repeated timing. These counts are not all mandatory for NAACL. Main contribution remains more credible if measurement/prediction are strong than if every optional scale row is run.

ETA after pilots: sum over cells N_cell * measured seconds/prompt, with separate target-reference, scoring and analysis costs. For a reference shared by multiple drafts, generate it once per target/revision/dtype/backend/prompt set and reuse only after cache/output validation. Use realized token-length distribution rather than assuming256 tokens everywhere. Schedule to the slower of the two GPU queues, adjusting for shared-host benchmark reservations. Publish an interval from workload-stratified pilot measurements, not a single unsupported hour figure.

## 8. Storage, queue and reproducibility

- Shard by200 prompt IDs; last shard may be smaller. One writer per shard; a status/lock manifest handles ownership and retries. Final merge asserts exact prompt coverage, no duplicate event IDs and matching revisions/config hashes.
- Namespace artifacts by card/pair/data split/prompt shard/precision/backend/GPU UUID. Write completion metadata after content is flushed, using atomic finalization where supported. A failed/partial shard is never marked complete.
- Share company read-only model cache and immutable dataset snapshots. Prefetch before benchmark slots; store outputs on local NVMe, copy summaries to workstation. Company data access policies remain whatever the existing host requires; no external upload is part of this plan.
- Keep only candidate logits/confidence, selected target diagnostics and needed traces. Full vocabulary distributions are only for the registered overlap diagnostic in chunks. Measure bytes/trace in H01 and compute storage ETA before full rollout.
- Reserve memory from actual pilot peak, not weight sum; logging/teacher-forced full logits can dominate temporary allocations. TP2 introduces another correctness/latency condition and must be labeled.
- Job queue CSV records proposed ownership/dependencies; it is not a submitted scheduler queue. YAML does not imply unsupported command flags exist.

## 9. Paper-facing outputs and what not to expand

T1: actual-proposal boundary signature; rows Q1/Q2/Q3, 8B draft grid, G1, L01/L02; show raw/adjusted effects, CI, candidate/document support and coverage. Large-model panels may be compacted to appendix.

T2: held-out predictive utility beyond confidence and token identity/frequency; main Q2 test and frozen F/N transfer; add Q5/L01a/L02a zero-shot transfers with clearly different target/corpus distributions. Main metrics log loss/Brier plus relevant risk coverage.

T3: strongest tuned SD baselines versus morphology controller, actual wall-clock and overhead, parity; larger32B IT/hardware robustness only if run. If no practical gain, replace the positive-method narrative with measured cost/headroom and limits.

Figures: data scale0.5k/1k/2k/5k; fixed-draft/fixed-target capacity; incremental risk coverage; parser overhead versus useful accepted tokens. Dense7/14/32/72 curve is an appendix candidate. Avoid parameter count being the story by itself.

Do not add235B BF16 to this two-device plan: approximately470GB of weights exceeds two180GB planning devices even before cache. FP8/INT4 would add a precision/engine confound and is not needed for this paper. Do not start full draft distillation, a new training objective or a giant cross-lingual benchmark until the two-analysis claims are validated. Negative signs and poor overlap are findings, not criteria for hiding runs.
