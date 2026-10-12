# KORSpec A6000 experiment plan

Oct 11, 2026 · @ihb

The paper becomes a pure analysis paper built on five research questions, with no new method; future directions go in the discussion. The A6000 48GB runs until 12:00 on Oct 13, 2026 (about 41 GPU-hours). To fit, live speculative decoding is replaced by target greedy generation plus teacher-forced scoring, which reproduces greedy SD rejections exactly for any block size. The saved compute goes to generality and validity: all five frozen pairs, a Qwen draft x target grid, a second domain and human-written text.

**Revision (18:50, Oct 11).** The method (BADS) and everything that only served it are removed: the step-cost benchmark, policy replay and the live wall-clock check. Added in their place: a quantified cost of fusion for SD (RQ5), a particle confusion analysis, a human-written-text replication, and human validation of the morphology labels.

## 1. The key change: simulate SD exactly instead of running it

Under greedy SD, a draft proposal at position t is only verified if every earlier proposal in its block was accepted. Each of those accepted proposals equals the target's token, so the draft always predicts position t from the target's own greedy prefix. Its proposal is therefore exactly the draft's greedy prediction when teacher-forced on the target continuation.

Two cheap passes give everything the live decoder gave:

1. **Target pass:** batched greedy generation (HF `generate`, 128 new tokens), then one teacher-forced forward to log the target's entropy, top-5 tokens and margin at every position.
2. **Draft pass:** one teacher-forced forward over prompt + target continuation, logging the draft argmax, its log-probability, entropy and top-2 margin at every position.

A CPU simulator then walks each continuation with block size k: it accepts while draft argmax equals the target token, marks the first mismatch as a rejection, and drops the rest of the block. This gives the eligible proposals and rejection labels for k = 4, and for any other k, at no extra GPU cost.

**Why this is safe to claim.** The discovery study already found SD and teacher-forced labels agree at 99.88% of positions. On the A6000 we re-run the repo's real cached SD decoder on 1,000 Q2 prompts and 500 G1 prompts and report label agreement. The paper says "computed exactly by teacher forcing and validated against live SD"; residual disagreement comes from numerical near-ties and is reported, not hidden.

**What it buys.** Batched autoregressive generation of a target is far faster than the batched SD decoder, and one target continuation serves every draft paired with it. Three Qwen targets (1.7B, 4B, 8B) and three drafts (0.6B, 1.7B, 4B) give six Qwen pairs for the price of three generations.

## 2. Paper design: five research questions

Reviewers accept analysis papers that state their questions up front, show the measurement is valid, show the finding holds beyond one setting, rule out the obvious confounds and quantify why it matters. Each question below is one section of the paper with one or two exhibits.

| RQ | Question | Main exhibits | Experiments | If the result disagrees |
| --- | --- | --- | --- | --- |
| RQ1 Where? | Does scalar token–morpheme alignment predict rejection, and if not, which direction and which boundaries do? | Tab. 2 (F1 split into fusion vs splitting), Fig. 2 (boundary landscape) | Q2 on 40k prompts | 1−F1 positive at scale: report it; the landscape still localizes the effect |
| RQ2 How general? | Does the pattern hold across draft size, target size, model family, domain, and generated vs human-written text? | Fig. 3 (family heatmap + Qwen 3 x 3 grid), Tab. 3 (stability) | 6 Qwen pairs, Gemma, Ministral, Wikipedia-ko, human-written text | Weak cross-family agreement: claim tokenizer-dependent risk with shared top boundaries |
| RQ3 Why? | Is the cost from fused tokens bundling a lexical prediction with a grammatical choice? | Fig. 5 (affix locus, near-tie rank, draft entropy) + new particle confusion matrix | TF logs; draft-vs-target particle pairs; draft-size trend from the grid | Affix share does not track risk: fall back to a token-geometry account |
| RQ4 Is it morphology? | Do frequency, a few token types, arbitrary cut positions or analyzer errors explain it? | Fig. 6, Tab. 4 | Token-ID removal, pseudo-boundaries, frequency variants, proposal-side labels, second analyzer, 300-item human check | Pseudo-null overlaps the real effect: drop the morphology-specific claim |
| RQ5 How much does it cost SD? | What share of rejections and wasted draft work comes from fusion, and how much accepted length is lost? | New Fig. 7 (cost decomposition), block-size panel | Simulator: rejection and waste shares; counterfactual accepted length if fused tokens were rejected at the WITHIN rate; k = 1–16 | Small cost: frame as a diagnostic finding and say so |

There is no method section. Former Section 8 becomes **Implications and future directions**: tokenizer design that keeps particles separate, boundary-aware adaptive draft length, distilling drafts on particle decisions, and testing other agglutinative languages. The discovery-study substitution result (overriding the draft's token hurts) stays in the appendix as evidence for which direction is promising.

## 3. GPU jobs

Core jobs total about 17 GPU-hours on conservative estimates, leaving about 20 hours for slow-downs, reruns and the optional tier. All hours are guesses for an A6000 at BF16/SDPA, 128-token prompts and 128 new tokens; the timing pilot at G0 replaces them. Prompt sets are the frozen FineWeb2 splits already in the repo (common 20k, extra 20k).

| ID | Job | Prompts | Batch | Est. hours | Serves | Tier |
| --- | --- | --- | --- | --- | --- | --- |
| V0 | Live SD validation with the repo's cached decoder: Q2 0.6B→4B and G1 4B→12B | 1,000 + 500 | 64 / 32 | 1.5 | Validity of the simulator | Core |
| G-Q4 | Qwen3-4B-Base greedy generation | 40k (all) | 128 | 1.9 | RQ1, RQ3–5 | Core |
| G-Q1.7 | Qwen3-1.7B-Base greedy generation | 20k common | 192 | 0.6 | RQ2 grid | Core |
| G-Q8 | Qwen3-8B-Base greedy generation | 20k common | 96 | 1.6 | RQ2 grid | Core |
| G-G12 | Gemma-3-12B-pt greedy generation | 20k common | 48 | 2.8 | RQ2 family | Core |
| G-M8 | Ministral-3-8B-Base greedy generation | 20k common | 96 | 1.6 | RQ2 family | Core |
| TF-Qwen | Drafts 0.6B on 1.7B/4B/8B continuations, 1.7B on 4B/8B, 4B on 8B; targets on their own continuations | 20–40k each | 64 | 2.3 | All RQs | Core |
| TF-G | Gemma-3-4B-pt draft and 12B target on 12B continuations | 20k | 32 | 0.9 | RQ2 | Core |
| TF-M | Ministral-3-3B draft and 8B target on 8B continuations | 20k | 64 | 0.6 | RQ2 | Core |
| H | Human-written text: TF of all drafts and targets on the documents' real next 128 tokens (common docs long enough, about 10k) | about 10k | 64 | 1.5 | RQ2 (generated vs human text) | Core |
| W | Second domain, Korean Wikipedia 5k: generation and TF for Q2 and G1 | 5k | as above | 1.5 | RQ2 domain | Core-lite |
| O1 | Qwen3-14B-Base target (20k gen + TF of all Qwen drafts): extends the grid to 3 x 4 | 20k | 48 | 4.5 | RQ2 scale | Optional |
| O2 | Japanese control, FineWeb2 `jpn_Jpan` 5k with Qwen 0.6B→4B and Gemma 4B→12B: does topic は vs subject が fusion show the same risk? Needs a fugashi/UniDic aligner | 5k | as above | 1.5 | Discussion: beyond Korean | Optional |
| O3 | FLORES-200 en→ko (1,012) for all targets | 1,012 | as above | 0.5 | RQ2 domain | Optional |
| O4 | Gemma-3-1B-pt as a second Gemma draft (tokenizer audit first) | 20k | 64 | 0.3 | RQ2 family | Optional |

Pair naming in the paper: Q1 = 0.6B→1.7B, Q2 = 0.6B→4B, Q3 = 4B→8B as before; the grid adds 1.7B→4B, 0.6B→8B and 1.7B→8B.

## 4. Schedule (Indochina time)

The GPU never idles: one queue runs jobs back to back, and CPU alignment of each finished set starts in a second process. Everything must be copied off the machine by Tue 11:30.

| When | GPU queue | You / CPU | Gate at end |
| --- | --- | --- | --- |
| Sun 18:30–20:30 | Download models (Qwen first, then Gemma, Ministral; about 90 GB). Smoke-load each model once | Write `gen_target.py`, `tf_score.py`, `simulate_sd.py` (Section 6). Freeze the analysis plan (this doc + a commit hash) before any result | — |
| Sun 20:30–21:30 | Timing pilot: 200 prompts per target gen and TF pass. V0 on the first 100 Q2 prompts | Run the simulator on the 100 prompts; compare with V0 labels | **G0 (amended):** agreement ≥ 95% and every job's ETA known; fix tier |
| Sun 21:30–Mon 04:00 | G-Q4 (40k), G-Q1.7, G-Q8, then TF-Qwen | Kiwi alignment of each continuation set as it lands | — |
| Mon 04:00–09:00 | G-G12, TF-G | RQ1: Tab. 2 and Fig. 2 on Q2. Draw the 300-item human-validation sample | **G1 (09:00):** RQ1 result checked against Section 2's last column; Qwen grid complete |
| Mon 09:00–13:00 | G-M8, TF-M, rest of V0 (1,000 Q2 + 500 G1) | RQ3 diagnostics and particle confusion matrix; send the validation sheet to the annotator | — |
| Mon 13:00–16:00 | H (human-written text), then W (Wikipedia-ko) | RQ2 heatmap, grid and stability; RQ5 cost decomposition | **G2 (16:00):** all core jobs done, else apply cut order |
| Mon 16:00–Tue 09:00 | Optional tier in order O3, O1, O2, O4; reruns of any failed shard first | RQ4 falsification battery and second analyzer; update paper tables and figures | — |
| Tue 09:00–11:30 | Buffer only; no new long jobs after 09:00 | Checksum and copy all outputs (continuations, TF logs, V0, timings) off the machine | **G3 (11:30):** artifacts safe; GPU released at 12:00 |

All analysis after Tue 12:00 is CPU-only and can continue until the submission deadline.

**Post-freeze G0 amendment (2026-10-11).** After observing 98.11% shared-context
rejection-label agreement on the 100-prompt Q2 pilot, the user accepted that
result and set the operational G0 agreement threshold to 95%. This is a
post-result amendment; preserve and report the measured 98.11% value. The V0
full-sample plan (1,000 Q2 and 500 G1 prompts) is unchanged.

## 5. CPU analyses and the paper artifact each fills

Every artifact in the current draft can be produced from three inputs: the continuations, the TF logs and the Kiwi alignment. Order follows when inputs land, so the core story can be checked at G1.

| RQ | Paper artifact | Analysis | Needs | Earliest |
| --- | --- | --- | --- | --- |
| Setup | Table 1 | Counts, eligibility, class shares, simulated k=4 rejection rate per pair; V0 agreement in Section 4 text | All gens + TF, V0 | Mon 13:00 (Qwen rows Mon 04:00) |
| RQ1 | Table 2a | Logistic AME of 1−F1, 1−Recall (fusion), 1−Precision (splitting) with fragmentation fixed effects and full controls | Q2 | Mon 04:00 |
| RQ1 | Table 2b | Global CROSS vs WITHIN AME per pair | Each pair | As each pair lands |
| RQ1 | Fig. 2, Table 7 | Boundary landscape: class x boundary interaction, 200/200 support, BH q-values | Q2 | Mon 05:00 |
| RQ2 | Fig. 3 | (a) boundary x pair heatmap on common 20k; (b) Qwen draft x target grid of the N→P and global effects | All pairs | Mon 13:00 |
| RQ2 | Table 3 | Rank/sign/top-5 stability for every pair of settings, including Wikipedia-ko and human-written text as extra columns | All pairs, W, H | Mon 16:00 |
| RQ3 | Fig. 5 | Affix-locus share per boundary; target rank of the rejected draft token; draft entropy at rejection | Q2 TF logs | Mon 05:00 |
| RQ3 | New Fig. 5d / table | Particle confusion matrix at rejected N→P fusions: draft particle vs target particle (topic 은/는, subject 이/가, object 을/를, 도, 만, adverbial). Predicts topic/subject swaps dominate | Q2 TF logs + Kiwi on draft tokens | Mon 06:00 |
| RQ3 | Text + grid | Does a larger draft shrink the N→P penalty faster than the global one? Slope of AME against draft size per target | Qwen grid | Mon 13:00 |
| RQ4 | Fig. 6a | Remove top 10/25/50/100 CROSS token IDs per boundary, refit | Q2 | Mon 06:00 |
| RQ4 | Fig. 6b | 100 pseudo-boundary draws preserving boundary count per eojeol | Q2 | Mon 06:00 (about 1–2 CPU-hours) |
| RQ4 | Table 4 | Frequency variants, multi-boundary tokens, strict alignment, top-50 removal, proposal-side labels, MeCab-ko analyzer, common 20k | Q2 | Mon 16:00 |
| RQ4 | Appendix table | Human validation: 300 tokens stratified by class (EXACT/WITHIN/CROSS) x outcome, N→P oversampled; annotator marks whether Kiwi's boundary and the token class are correct. Report accuracy per stratum (and Cohen's kappa if two annotators) | Annotator, about 3 hours | Tue |
| RQ5 | New Fig. 7 | Share of all rejections and of all wasted draft tokens at CROSS positions, by boundary family; counterfactual mean accepted length per target call if CROSS tokens were rejected at the WITHIN rate of their boundary (bootstrap CI) | Simulator | Mon 13:00 |
| RQ5 | Fig. 7b | Same quantities for k = 1, 2, 4, 8, 16 | Simulator | Mon 13:00 |

Frequency covariate: count tokens per tokenizer on a FineWeb2 Korean slice disjoint from the 40k pool (CPU, start Sun night). The human validation is the single cheapest credibility gain. If no native Korean annotator can be found by Monday noon, report MeCab-ko agreement as automatic validation instead and state the gap in Limitations.

## 6. Engineering checklist

Three new scripts, each built from functions already in the repo (`greedy_generate_batch` in `src/speculative_decoding.py`, `score_teacher_forced_batch` in `src/e2_model_pairs.py`, the Table 1 morphology aligner). Write all outputs as resumable Parquet shards of 2,000 prompts.

- [ ] **`gen_target.py`**: left-padded batched greedy `generate`, `max_new_tokens=128`, `do_sample=False`, BF16, SDPA, stop at EOS. Save `doc_id, prompt_ids, continuation_ids, eos_hit`. Reuse the repo's prompt truncation so prompts match the frozen splits.
- [ ] **`tf_score.py`**: one forward over prompt + continuation with an attention mask. Per continuation position, from the draft: `argmax_id, logp_argmax, logp_target_token, entropy, top2_margin`; from the target on its own continuation: `entropy, top20_ids, top20_logp`. A `--continuation human` flag scores the document's real next 128 tokens instead (job H).
- [ ] **`simulate_sd.py`** (CPU): for block size k, accept while `draft_argmax == target_token`, mark the first mismatch rejected, drop the rest of the block, no proposal at bonus positions. One row per eligible proposal with `slot, block_id, rejected, invalidated_after`. Also computes the RQ5 counterfactual accepted length.
- [ ] **Alignment adapter:** feed `continuation_ids` to the existing Kiwi alignment as the target path, so all class and boundary columns match the current schema. Add a MeCab-ko variant behind a flag.
- [ ] **Validation join:** V0 live-SD labels vs simulated labels on (`doc_id`, position); agreement overall and at CROSS positions.
- [ ] **Annotation sheet:** 300 stratified items with eojeol, token span, Kiwi segmentation and class; columns for the annotator's judgement.

**Memory settings on 48 GB.**

- Never materialise full-sequence logits in FP32. For Gemma's 262k vocabulary, apply the LM head to chunks of 32 positions and reduce to top-20, entropy and the needed log-probabilities on the GPU before moving to CPU.
- Use `logits_to_keep=1` in generation (the default in recent `transformers`).
- Batch sizes in Section 3 are starting points. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and halve the batch on OOM rather than retrying blindly; log the batch size used.
- Load Gemma-3-4B/12B-pt through the same text-only path the repo validated in the B200 smoke, and Ministral with the pinned Rust `tokenizer.json` (`fix_mistral_regex=true`), exactly as the smoke did.

**Provenance to save for the paper:** model revisions, `transformers`/`torch`/CUDA versions, GPU name, batch size per job, wall time per job, and the V0 agreement report.

## 7. Cut order and what must never be cut

Apply cuts at G0 or G2 only, decided from ETAs and never from results, and list every cut in the paper's appendix.

**Cut first, in this order:**

1. Optional tier: O4 Gemma-1B, O2 Japanese, O1 Qwen3-14B, O3 FLORES.
2. W second domain: shrink from 5k to 2k prompts, then drop (H already covers "not just generated text").
3. M1 and G1 to the first 10k common prompts (keep both families).
4. Q3 (8B target) to 10k common prompts; the 0.6B→8B and 1.7B→8B grid cells shrink with it.
5. H human-written text to 5k documents.

**Never cut:**

- Q2 on all 40k prompts (the boundary landscape depends on its support).
- At least one non-Qwen family on 10k or more prompts. Gemma is the priority because its tokenizer fuses particles most often.
- At least two draft sizes and two target sizes in the Qwen grid.
- V0 validation on at least 500 Q2 prompts: without it the exact-simulation claim is unsupported.
- Copying artifacts off the machine by Tue 11:30.

## 8. Edits to the paper draft

These changes keep the paper's argument and make its methods match what will actually be run.

- **Title and abstract:** drop every mention of BADS and speedup. The last abstract sentence becomes the cost finding from RQ5 (share of rejections and accepted length lost to fusion).
- **Introduction:** state RQ1–RQ5 explicitly as a numbered list. Contributions become: directional alignment framework; boundary-risk landscape with generality across scale, family, domain and text source; choice-bundling evidence; validity and falsification battery; quantified cost for SD. Mention the released logs and simulator as a resource.
- **Section 3:** add the exact-simulation argument (Section 1 of this plan) as a short paragraph; it doubles as a methodological contribution others can reuse.
- **Section 4, Decoding:** greedy SD outcomes are computed from the target's greedy continuation and the draft's teacher-forced predictions, and agree with live SD on V0's prompts at the measured rate. Remove the unqualified "100% of prompts" parity claim. Hardware: one RTX A6000 48GB.
- **Section 5 (RQ1–RQ2):** add the Qwen draft x target grid as Fig. 3b, and Wikipedia-ko and human-written text as extra columns of the stability table.
- **Section 6 (RQ3):** add the particle confusion matrix and the draft-size trend.
- **Section 7 (RQ4):** add the MeCab-ko row to Table 4 and the human-validation summary (details in the appendix).
- **New Section 8 (RQ5), cost for SD:** cost decomposition figure and block-size panel. No runtime table.
- **Section 9, Implications and future directions:** tokenizer design, boundary-aware adaptive drafting, draft distillation on particle decisions, other agglutinative languages (Japanese if O2 ran). Keep claims hedged: these are directions, not results.
- **Limitations:** observational design; automatic morphology with a 300-item human check; greedy decoding only; base models; no wall-clock measurements.
- **Appendix:** completed, cut and skipped jobs with reasons, plus provenance (Section 6).

## 9. Execution log (2026-10-11)

Run logs and resumable outputs are being written under
`logs/naacl_experiments/20261011/` and `results/naacl_experiments/`.
The execution host has an NVIDIA RTX A6000, Python 3.12.14, PyTorch
2.11.0+cu128, Transformers 5.19.0, and CUDA runtime 12.8. Model weights are
staged in `/dev/shm/naacl_models`; the filesystem has about 27 GiB free. There
is no mounted persistent volume in this session, and `/workspace` is ephemeral.
Copy the results, provenance, and logs off-host before the instance is recycled.

The requested generation targets are run as complete target units: greedy
generation, target self teacher-forced scoring, then Kiwi alignment. Each
target's configured pair is evicted after those jobs finish, preserving its
tokenizer/config files. For Qwen, the eviction commands are `--evict-pair Q2`
(4B), `--evict-pair Q1` (1.7B), and `--evict-pair Q5` (8B). Ministral uses
`--evict-pair M1` after its target unit. The full command lines and per-command
logs are kept in the dated log directory. Job status below reflects actual
execution, not estimates.

| Job | Target revision | Planned command sequence | Status | Evidence |
| --- | --- | --- | --- | --- |
| G-Q4 | `906bfd4b4dc7f14ee4320094d8b41684abff8539` | `gen_target.py --target qwen4 --split all`; target self-score; Kiwi alignment; evict Q2 | Complete | Stage manifests and dated command logs |
| G-Q1.7 | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` | `gen_target.py --target qwen17 --split common`; target self-score; Kiwi alignment; evict Q1 | Complete | Dated command logs and output manifests |
| G-Q8 | `49e3418fbbbca6ecbdf9608b4d22e5a407081db4` | `gen_target.py --target qwen8 --split common`; `tf_score.py --model qwen8 --target qwen8 --split common --role target`; `align_morphology.py --target qwen8 --split common`; evict Q5 | Complete | Dated command logs and output manifests |
| G-G12 | `295efb63d01a7017928f273a94ebb86105c9526f` | Gemma-12B generation, target self-score, alignment; evict G1 | Complete | Dated command logs and complete generation, TF, and alignment manifests; `hf_token` access verified |
| G-M8 | `d4883f9b36aa2e5d775730d3fdba3d30de51a8ef` | Download M1; generation, target self-score, alignment; evict M1 | Complete | Dated logs and complete generation, TF, and alignment manifests; M1 evicted |

G-G12 requires access to `google/gemma-3-12b-pt`. On 2026-10-11 the local
`hf_token` file was found and verified against the pinned revision
`295efb63d01a7017928f273a94ebb86105c9526f`; access is granted. The credential
is read from the file only when downloading and is not written to logs or this
record. The Qwen8 target unit has completed and Q5 was evicted. The Gemma-12B
target-only download completed within the storage limit described below.

### G-Q4 run record

- Generation command: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout --foreground --signal=TERM --kill-after=10s 10800s /venv/main/bin/python -u scripts/gen_target.py --target qwen4 --split all --output-dir results/naacl_experiments`.
- Generation stdout/stderr: `logs/naacl_experiments/20261011/G-Q4-generation.log`.
- Requested batch 128, 40,000 prompts, 128 new tokens, BF16/SDPA. The model loaded at pinned revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`; the manifest records 40,000 prompts, 20 completed shards, actual batch sizes from 1 through 128, and 4,158.3 seconds wall time. Generation completed successfully at about 16:51 UTC; generation logs and manifest preserve the exact completion.
- Target self-score command: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout --foreground --signal=TERM --kill-after=10s 7200s /venv/main/bin/python -u scripts/tf_score.py --model qwen4 --target qwen4 --split all --role target --output-dir results/naacl_experiments`. Log: `logs/naacl_experiments/20261011/G-Q4-target-TF.log`.
- Alignment command: `timeout --foreground --signal=TERM --kill-after=10s 3600s /venv/main/bin/python -u scripts/align_morphology.py --target qwen4 --split all --output-dir results/naacl_experiments`. Log: `logs/naacl_experiments/20261011/G-Q4-alignment.log`.
- Target score and alignment started at 16:52 UTC. The score model loaded at the same pinned revision and reported `TF_HEAD_CHECK_OK scale=1 max_abs_error=0`. At 17:06 UTC, all 20 alignment shards completed successfully: 40,000 prompts, 4,945,226 aligned rows. The alignment log file was created at 16:52:11 and finalized at 17:06:47 (about 14m35s elapsed); the alignment manifest has `status=COMPLETE`.
- Target self-score completed all 20 shards with 40,000 prompts and 4,945,226 token positions. Every shard recorded `head_scale=1.0` and `head_check_max_abs_error=0.0`; summed shard scoring time was 1,641.96 seconds, and the command log ran from 16:52:11 to 17:19:48. The score manifest has `status=COMPLETE`.
- Q2 eviction completed at about 17:20 UTC. `download_models.py --evict-pair Q2` removed the qwen06 and qwen4 weight shards while retaining tokenizer/config files. `/dev/shm` free space rose to 13 GiB. Eviction log: `logs/naacl_experiments/20261011/Q2-eviction.log`.

### G-Q1.7 run record

- Generation command: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout --foreground --signal=TERM --kill-after=10s 3600s /venv/main/bin/python -u scripts/gen_target.py --target qwen17 --split common --output-dir results/naacl_experiments`.
- Generation stdout/stderr: `logs/naacl_experiments/20261011/G-Q1.7-generation.log`.
- Started at about 17:20 UTC on pinned revision `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`, requested batch 192, common split (20,000 prompts), 128 new tokens, BF16/SDPA. The completed run metadata records 10 shards, actual batch sizes from 1 through 192, prompt-pool SHA-256 `d0d316155380aac0d6aa3ade8bddd93f96579a7c88c750167c9d012162fa72eb`, and `wall_seconds=1841.43` (30m41s). All 20,000 prompts completed successfully; the log finalized at 17:51:53 UTC.
- Target self-score command: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout --foreground --signal=TERM --kill-after=10s 7200s /venv/main/bin/python -u scripts/tf_score.py --model qwen17 --target qwen17 --split common --role target --output-dir results/naacl_experiments`. Log: `logs/naacl_experiments/20261011/G-Q1.7-target-TF.log`.
- Alignment command: `timeout --foreground --signal=TERM --kill-after=10s 3600s /venv/main/bin/python -u scripts/align_morphology.py --target qwen17 --split common --output-dir results/naacl_experiments`. Log: `logs/naacl_experiments/20261011/G-Q1.7-alignment.log`.
- Target self-score and Kiwi alignment started at 17:52:10 UTC. Alignment completed all 20,000 prompts across 10 shards, producing 2,485,747 aligned rows; the alignment manifest reports `status=COMPLETE`. Its log ran from 17:52:10 to 17:59:35. Target self-score completed all 10 shards at 18:01:35: 20,000 prompts, 2,485,747 token positions, `head_scale=1.0` and `head_check_max_abs_error=0.0` for every shard. Summed shard scoring time was 553.23 seconds; the score manifest reports `status=COMPLETE`.
- Q1 eviction completed at 18:02 UTC. `download_models.py --evict-pair Q1` removed qwen06 and qwen17 weight shards while retaining tokenizer/config files. The eviction log is `logs/naacl_experiments/20261011/Q1-eviction.log`; `/dev/shm` free space rose to 16 GiB and Qwen8 remains staged.

### G-Q8 run record

- Generation command: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout --foreground --signal=TERM --kill-after=10s 10800s /venv/main/bin/python -u scripts/gen_target.py --target qwen8 --split common --output-dir results/naacl_experiments`.
- Generation stdout/stderr: `logs/naacl_experiments/20261011/G-Q8-generation.log`.
- Started at 18:02:42 UTC on pinned revision `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`, requested batch 96, common split (20,000 prompts), 128 new tokens, BF16/SDPA. All 20,000 prompts completed in 2,676.28 seconds (44m36s) across 10 shards. The generation manifest reports `status=COMPLETE`; the log finalized at 18:48:05 UTC.
- Target self-score command: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout --foreground --signal=TERM --kill-after=10s 7200s /venv/main/bin/python -u scripts/tf_score.py --model qwen8 --target qwen8 --split common --role target --output-dir results/naacl_experiments`. Log: `logs/naacl_experiments/20261011/G-Q8-target-TF.log`.
- Alignment command: `timeout --foreground --signal=TERM --kill-after=10s 3600s /venv/main/bin/python -u scripts/align_morphology.py --target qwen8 --split common --output-dir results/naacl_experiments`. Log: `logs/naacl_experiments/20261011/G-Q8-alignment.log`.
- Target self-score and Kiwi alignment started at 18:48:43 UTC. Alignment completed all 20,000 prompts across 10 shards, producing 2,465,034 aligned rows; its manifest reports `status=COMPLETE` and the log ran to 18:56:15 UTC. Target self-score completed all 10 shards by 19:08:37: 20,000 prompts and 2,465,034 token positions, `head_scale=1.0` and `head_check_max_abs_error=0.0` for every shard. Summed shard scoring time was 1,179.13 seconds; the score manifest reports `status=COMPLETE`.
- Q5 eviction completed after the target unit. `download_models.py --evict-pair Q5` removed the qwen06 and qwen8 weight files while retaining tokenizer/config files. Log: `logs/naacl_experiments/20261011/Q5-eviction.log`.

### G-G12 run record

- Hugging Face access to `google/gemma-3-12b-pt` at revision `295efb63d01a7017928f273a94ebb86105c9526f` was verified using the ignored local `hf_token` file; no token contents are logged or written here.
- `download_models.py --plan --pair G1` reported 30.71 GiB for Gemma-4B plus Gemma-12B against 30.96 GiB free. The downloader requires an additional 1 GiB for metadata, so the full pair cannot fit in `/dev/shm`. The planned target-only generation, target self-score, and alignment require Gemma-12B only; its 22.70 GiB checkpoint fits with the required reserve. The 4B draft is therefore not staged for this target-only unit.
- Target download command: `HF_TOKEN` was read from the ignored `hf_token` file into the process environment, then `/venv/main/bin/python -u scripts/download_models.py --model gemma12` was run with a two-hour timeout. Log: `logs/naacl_experiments/20261011/G-G12-download.log`. No credential value is recorded.
- The target download completed in 3m40s, pinned to revision `295efb63d01a7017928f273a94ebb86105c9526f`, with five indexed weight shards and 24,374,793,024 weight bytes.
- Generation command: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout --foreground --signal=TERM --kill-after=10s 21600s /venv/main/bin/python -u scripts/gen_target.py --target gemma12 --split common --output-dir results/naacl_experiments`. Log: `logs/naacl_experiments/20261011/G-G12-generation.log`.
- Generation started at about 19:13:38 UTC on the pinned revision, requested batch 48, common split (20,000 prompts), 128 new tokens, BF16/SDPA. All ten 2,000-prompt shards completed: shard sizes were 779,355; 781,473; 784,769; 782,823; 788,418; 781,881; 767,721; 777,939; 783,227; and 780,256 bytes. `GEN_COMPLETE` was logged at 21:25:51 UTC. The complete manifest records 20,000 prompts, `wall_seconds=7895.85` (2h11m36s), requested batch 48, actual batch sizes 1–48, BF16/SDPA, and `status=COMPLETE`. GPU memory use during generation was about 29 GiB.
- Target self-score and Kiwi alignment started at 21:25:57 UTC. Kiwi alignment completed all 10 shards by 21:33:48 UTC: 20,000 prompts and 2,549,337 aligned rows; the manifest reports `status=COMPLETE`. The continuation runner validates both `status=COMPLETE` manifests before evicting G1 and staging M1.
- The target scorer loaded the pinned Gemma model and passed `TF_HEAD_CHECK_OK scale=1 max_abs_error=0`. All 10 target-score shards completed successfully by 21:55:42 UTC: 20,000 prompts and 2,549,337 token positions, with `head_scale=1.0` and `head_check_max_abs_error=0.0` on every shard. The manifest reports `status=COMPLETE`; summed shard scoring time was 1,765.07 seconds, and the command log records `COMMAND_EXIT=0`.
- G1 eviction completed at 21:55:45 UTC. Gemma-12B's five weight shards were removed; Gemma-4B weights were already absent. Tokenizer/config files were retained. Eviction log: `logs/naacl_experiments/20261011/G1-eviction.log`.
- Queued target self-score command: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout --foreground --signal=TERM --kill-after=10s 14400s /venv/main/bin/python -u scripts/tf_score.py --model gemma12 --target gemma12 --split common --role target --output-dir results/naacl_experiments`. Log: `logs/naacl_experiments/20261011/G-G12-target-TF.log`.
- Queued Kiwi alignment command: `timeout --foreground --signal=TERM --kill-after=10s 3600s /venv/main/bin/python -u scripts/align_morphology.py --target gemma12 --split common --output-dir results/naacl_experiments`. Log: `logs/naacl_experiments/20261011/G-G12-alignment.log`. The continuation runner waits for the generation completion line and validates its manifest before launching these two jobs; it validates both scoring and alignment manifests before evicting G1.

### G-M8 run record

- Following G1 eviction, `download_models.py --plan --pair M1` reported 30.92 GiB free and 23.78 GiB requested (Ministral-3-3B 7.17 GiB plus Ministral-3-8B 16.61 GiB). The full M1 pair download ran from 21:57:20 to 22:00:51 UTC and completed successfully. Logs: `logs/naacl_experiments/20261011/G-M8-download-plan.log` and `logs/naacl_experiments/20261011/G-M8-download.log`.
- Generation started at 22:00:51 UTC with `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout --foreground --signal=TERM --kill-after=10s 10800s /venv/main/bin/python -u scripts/gen_target.py --target ministral8 --split common --batch-size 96 --output-dir results/naacl_experiments`. It is processing the common split (20,000 prompts), 128 new tokens, BF16/SDPA, requested batch 96. OOM backoff records any smaller actual batch sizes. Log: `logs/naacl_experiments/20261011/G-M8-generation.log`.
- All ten generation shards completed by 22:50:50 UTC. In shard order, their 2,000-prompt Parquet sizes were 877,148; 876,622; 880,848; 873,949; 883,578; 880,078; 872,634; 875,539; 875,280; and 878,064 bytes. The manifest records 20,000 prompts, revision `d4883f9b36aa2e5d775730d3fdba3d30de51a8ef`, requested batch 96, actual batch sizes `[1, 2, 3, 4, 5, 6, 7, 11, 13, 14, 85, 90, 91, 96]`, and `wall_seconds=2939.60` (48m59.6s), with `status=COMPLETE`. No OOM or generation error occurred; GPU memory stayed near 21 GiB.
- Transformers emitted nonfatal warnings that the checkpoint contains separate embedding and language-model head weights despite the config's tied-weight mapping, and that `max_new_tokens=128` takes precedence over the model's `max_length=262144`. Both warnings are preserved in the generation log; generation loaded and proceeded.
- Target self-score command: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout --foreground --signal=TERM --kill-after=10s 14400s /venv/main/bin/python -u scripts/tf_score.py --model ministral8 --target ministral8 --split common --role target --output-dir results/naacl_experiments`. The first attempt at 22:50:36 failed before scoring because `_body_and_head` looked for `model.language_model`; Transformers 5.19 nests the text transformer at `model.model.language_model` with the LM head at `model.lm_head`. No TF shard or metadata was written. The failed attempt is preserved as `logs/naacl_experiments/20261011/G-M8-target-TF-loader-failure.log`. `scripts/tf_score.py` was corrected to support the actual Mistral3 wrapper, and the retry is logged to `logs/naacl_experiments/20261011/G-M8-target-TF.log`; it passed `TF_HEAD_CHECK_OK scale=1 max_abs_error=0` and is scoring shard 0. Keep this unit open until the retry manifest is complete.
- Kiwi alignment command: `timeout --foreground --signal=TERM --kill-after=10s 3600s /venv/main/bin/python -u scripts/align_morphology.py --target ministral8 --split common --output-dir results/naacl_experiments`. Log: `logs/naacl_experiments/20261011/G-M8-alignment.log`. After the corrected TF retry and alignment manifests both report `status=COMPLETE`, verify them and evict M1 with `/venv/main/bin/python -u scripts/download_models.py --evict-pair M1`; log: `logs/naacl_experiments/20261011/M1-eviction.log`.
- The corrected TF retry passed `TF_HEAD_CHECK_OK scale=1 max_abs_error=0` and completed at 23:13:28 UTC. All 10 shards scored 20,000 prompts and 2,498,617 token positions; every shard recorded `head_scale=1.0` and `head_check_max_abs_error=0.0`. The manifest reports `status=COMPLETE`; summed shard scoring time was 1,221.75 seconds.
- Kiwi alignment completed at 22:58:06 UTC. Its manifest reports `status=COMPLETE`, 20,000 prompts, 10 shards, and 2,498,617 aligned rows; the command exited 0.
- The first M8 pipeline attempt ended after the initial loader failure, with target scoring exit 1 and alignment exit 0 at 22:58:06 UTC. After correcting `_body_and_head`, the scoring retry completed and both manifests were validated. M1 eviction ran from 23:13:45 to 23:13:48 UTC: two Ministral-3-3B and four Ministral-3-8B weight shards were removed; tokenizer/config files were kept. Log: `logs/naacl_experiments/20261011/M1-eviction.log`. `/dev/shm` returned to about 31 GiB free.

## 10. Follow-on execution log (2026-10-12)

The follow-on queue resumes from commit `51e857356477a48d9c8525da5087b928ca982947`.
The RTX A6000 is available, `/dev/shm` has 31 GiB, and the overlay has 24 GiB
free. The five target-generation units and their target self-scores/alignment
are already complete. The next planned work is cross-model TF-Qwen, TF-G, TF-M,
the full V0 sample, then H and W. The accepted 98.11% G0 pilot remains distinct
from the still-required full V0 sample (1,000 Q2 and 500 G1 prompts).

### Qwen grid preflight

- Downloaded pinned Qwen3-0.6B, 1.7B, 4B and 8B snapshots to `/dev/shm/naacl_models` and smoke-loaded each on the A6000. The combined checkpoint weights use about 27.1 GiB of the 31 GiB tmpfs. Log: `logs/naacl_experiments/20261012/TF-Qwen-download-smoke.log`.
- Tokenizer audits passed for Q1–Q6. Logs: `logs/naacl_experiments/20261012/TF-Qwen-tokenizer-audits.log`; reports: `results/naacl_experiments/provenance/tokenizer_audits/`.

### Q1 cross-model scoring and simulation

- Draft scoring command: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout --foreground --signal=TERM --kill-after=10s 14400s /venv/main/bin/python -u scripts/tf_score.py --model qwen06 --target qwen17 --split common --role draft --batch-size 64 --chunk-positions 32 --output-dir results/naacl_experiments`. Log: `logs/naacl_experiments/20261012/TF-Q1-draft.log`.
- Completed 20,000 prompts in 10 shards from 07:32:04 to 07:38:31 UTC (6m27s), producing 2,485,747 scored token positions. BF16/SDPA on the RTX A6000; the LM-head calibration check passed with `scale=1` and `max_abs_error=0`; manifest status is `COMPLETE`.
- Simulator command: `/venv/main/bin/python -u scripts/simulate_sd.py --pair Q1 --split common --output-dir results/naacl_experiments`. Log: `logs/naacl_experiments/20261012/TF-Q1-simulate.log`. Completed all planned block sizes from 07:38:37 to 07:42:20 UTC; the manifest reports 20,000 prompts, 2,485,747 positions and `status=COMPLETE`.

| Block size | Eligible proposals | Rejections | Rejection rate |
| ---: | ---: | ---: | ---: |
| 1 | 1,339,733 | 184,068 | 0.137392 |
| 2 | 1,774,649 | 238,454 | 0.134367 |
| 4 | 2,111,560 | 279,347 | 0.132294 |
| 8 | 2,315,386 | 299,674 | 0.129427 |
| 16 | 2,415,979 | 306,802 | 0.126989 |

The target TF argmax disagreed with the saved target token at 30,497 positions;
these near-tie-sensitive disagreements are retained in the simulation summary
and must be interpreted alongside the live-decoder V0 validation. Q1 evidence
is under `results/naacl_experiments/teacher_forced/generated/qwen17/qwen06/common/`
and `results/naacl_experiments/simulated/Q1/common/`.

**Queue status at 07:42 UTC:** Q1 complete; Q2, Q5, Q4, Q6 and Q3 cross-model
TF/simulation remain. Then run full V0 for Q2 and G1, TF-G, TF-M, H and W. The
W corpus presently has only the 1,000-document cached seed; the plan permits
shrinking W from 5,000 to 2,000 prompts before dropping it, so prepare the
additional 1,000 documents from the pinned Wikipedia revision and record the
resulting scope before running W.

### Q2 cross-model scoring, simulation, and full V0 validation

- Q2 draft TF command: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 21600s /venv/main/bin/python -u scripts/tf_score.py --model qwen06 --target qwen4 --split all --role draft --batch-size 64 --chunk-positions 32 --output-dir results/naacl_experiments`. Log: `logs/naacl_experiments/20261012/TF-Q2-draft.log`.
- Completed 40,000 prompts in 20 shards from 07:43:38 to 07:54:07 UTC. All shards passed the LM-head check (`scale=1`, `max_abs_error=0`); 4,945,226 token positions were scored. Manifest status: `COMPLETE`.
- Offline simulation was run on both `all` (40,000 prompts) and `common` (20,000 prompts). Logs: `TF-Q2-simulate.log` and `TF-Q2-simulate-common.log` under the same log directory. The V0 runner requires the `common`-split artifact; an initial preflight against the `all`-split output exited before decoding and is preserved in `V0-Q2-full.log` before the successful retry.

| Split | k | Eligible proposals | Rejections | Rejection rate |
| --- | ---: | ---: | ---: | ---: |
| common | 1 | 1,361,129 | 237,582 | 0.174548 |
| common | 2 | 1,798,526 | 307,393 | 0.170914 |
| common | 4 | 2,132,560 | 359,038 | 0.168360 |
| common | 8 | 2,326,917 | 383,194 | 0.164679 |
| common | 16 | 2,417,032 | 390,863 | 0.161712 |

- Full Q2 V0 command: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout 14400s /venv/main/bin/python -u scripts/validate_live_sd.py --pair Q2 --prompts 1000 --output-dir results/naacl_experiments`. It completed 1,000 prompts by 08:12 UTC; artifacts are under `results/naacl_experiments/validation/v0/Q2/`.
- The report's selected shared-context rejection-label agreement is **98.0577%** (95% prompt-cluster bootstrap CI 97.9196%–98.1808%) on 58,385 joined proposal positions, above the amended 95% threshold. However, **the full V0 gate is FAIL**: it also requires zero unmatched positions within the common-prefix contexts, while 949 live and 947 simulated positions were unmatched. The outer-joined all-position agreement is 88.6536% across 100,252 rows. The live target sequence exactly matched the cached continuation on 405/1,000 prompts; common-prefix token fraction was 54.5771%.
- Interpretation: the CPU simulator deterministically replays teacher-forced argmaxes and is exact for the saved greedy continuation under its stated assumptions. This Q2 live comparison supports high agreement on comparable shared-prefix proposals, but does **not** clear the plan's strict full V0 gate or validate unmatched/diverged contexts. Keep simulator-derived rates labeled as offline estimates until the divergence/unmatched-position issue is resolved and Q1/G1 validation is reviewed. Do not describe the full V0 as passing based only on the 98.06% conditional agreement.

**Queue status at 08:12 UTC:** Q1 and Q2 draft TF plus simulator outputs are complete; Q2 full V0 ran but failed its strict gate for unmatched shared-context positions. Continue with the remaining Qwen grid pairs, then validate G1 and investigate whether numerical divergence between cached target generation and the live decoder accounts for the unmatched contexts before treating simulator estimates as live-validated.

### Q5 cross-model scoring and simulation

- Draft scoring command: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout --foreground --signal=TERM --kill-after=10s 14400s /venv/main/bin/python -u scripts/tf_score.py --model qwen06 --target qwen8 --split common --role draft --batch-size 64 --chunk-positions 32 --output-dir results/naacl_experiments`. Log: `logs/naacl_experiments/20261012/TF-Q5-draft.log`.
- Scoring completed 20,000 prompts in 10 shards, 2,465,034 token positions. Every shard passed the LM-head check (`scale=1`, `max_abs_error=0`); summed shard time was 376.93 seconds. Manifest status: `COMPLETE`.
- Simulation command: `timeout --foreground --signal=TERM --kill-after=10s 21600s /venv/main/bin/python -u scripts/simulate_sd.py --pair Q5 --split common --output-dir results/naacl_experiments`. Log: `logs/naacl_experiments/20261012/TF-Q5-simulate.log`; all five block sizes completed.

| Block size | Eligible proposals | Rejections | Rejection rate |
| ---: | ---: | ---: | ---: |
| 1 | 1,373,031 | 271,425 | 0.197683 |
| 2 | 1,810,852 | 350,190 | 0.193384 |
| 4 | 2,142,767 | 408,582 | 0.190680 |
| 8 | 2,330,261 | 434,826 | 0.186600 |
| 16 | 2,413,811 | 442,668 | 0.183390 |

Q5 outputs are under `results/naacl_experiments/teacher_forced/generated/qwen8/qwen06/common/` and `results/naacl_experiments/simulated/Q5/common/`. As with Q1/Q2, these are offline replay estimates and inherit the Q2 full-V0 caveat until live validation is resolved.

**Queue status at 08:25 UTC:** Q1, Q2, and Q5 TF/simulation are complete. Q2 full V0 remains below the strict gate due to unmatched shared-context positions. Remaining Qwen pairs: Q4, Q6, Q3; then TF-G, TF-M, full G1 V0, H, and W.

### Q4 cross-model scoring and simulation

- Draft scoring command: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout --foreground --signal=TERM --kill-after=10s 14400s /venv/main/bin/python -u scripts/tf_score.py --model qwen17 --target qwen4 --split common --role draft --batch-size 64 --chunk-positions 32 --output-dir results/naacl_experiments`. Log: `logs/naacl_experiments/20261012/TF-Q4-draft.log`.
- Scoring completed all 20,000 common prompts in 10 shards, 2,475,047 positions. Every shard passed the LM-head check (`scale=1`, `max_abs_error=0`); summed shard wall time was 419.95 seconds. Manifest status: `COMPLETE`.
- Simulation command: `timeout --foreground --signal=TERM --kill-after=10s 21600s /venv/main/bin/python -u scripts/simulate_sd.py --pair Q4 --split common --output-dir results/naacl_experiments`. Log: `logs/naacl_experiments/20261012/TF-Q4-simulate.log`; all block sizes completed.

| Block size | Eligible proposals | Rejections | Rejection rate |
| ---: | ---: | ---: | ---: |
| 1 | 1,332,525 | 180,343 | 0.135339 |
| 2 | 1,766,628 | 234,860 | 0.132943 |
| 4 | 2,105,535 | 277,164 | 0.131636 |
| 8 | 2,311,336 | 299,216 | 0.129456 |
| 16 | 2,411,680 | 307,460 | 0.127488 |

Q4 outputs are under `results/naacl_experiments/teacher_forced/generated/qwen4/qwen17/common/` and `results/naacl_experiments/simulated/Q4/common/`. Treat these simulation rates as offline estimates pending resolution of the Q2 V0 unmatched-position gate.

**Queue status at 08:37 UTC:** Q1, Q2, Q4, and Q5 TF/simulation are complete. The next pair is Q6 (Qwen 1.7B→8B), then Q3 (Qwen 4B→8B). Q2 full V0 still has strict-gate status `FAIL` due to unmatched shared-context positions.

### Q6 cross-model scoring and simulation

- Draft scoring command: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout --foreground --signal=TERM --kill-after=10s 14400s /venv/main/bin/python -u scripts/tf_score.py --model qwen17 --target qwen8 --split common --role draft --batch-size 64 --chunk-positions 32 --output-dir results/naacl_experiments`. Log: `logs/naacl_experiments/20261012/TF-Q6-draft.log`.
- Scoring completed 20,000 prompts in 10 shards, 2,465,034 positions. Every shard passed the LM-head check (`scale=1`, `max_abs_error=0`); summed shard wall time was 420.01 seconds. Manifest status: `COMPLETE`.
- Simulation command: `timeout --foreground --signal=TERM --kill-after=10s 21600s /venv/main/bin/python -u scripts/simulate_sd.py --pair Q6 --split common --output-dir results/naacl_experiments`. Log: `logs/naacl_experiments/20261012/TF-Q6-simulate.log`; all five block sizes completed.

| Block size | Eligible proposals | Rejections | Rejection rate |
| ---: | ---: | ---: | ---: |
| 1 | 1,339,912 | 205,215 | 0.153156 |
| 2 | 1,774,572 | 266,839 | 0.150368 |
| 4 | 2,112,493 | 315,125 | 0.149172 |
| 8 | 2,313,978 | 339,756 | 0.146828 |
| 16 | 2,408,471 | 348,325 | 0.144625 |

Q6 outputs are under `results/naacl_experiments/teacher_forced/generated/qwen8/qwen17/common/` and `results/naacl_experiments/simulated/Q6/common/`. These remain offline estimates pending the V0 gate follow-up.

**Queue status at 08:49 UTC:** Q1, Q2, Q4, Q5, and Q6 TF/simulation are complete. Q3 (Qwen 4B→8B) is the last Qwen grid pair. Q2 full V0 remains flagged `FAIL` because of unmatched shared-context positions.

### Q3 cross-model scoring and simulation

- Draft scoring command: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout --foreground --signal=TERM --kill-after=10s 14400s /venv/main/bin/python -u scripts/tf_score.py --model qwen4 --target qwen8 --split common --role draft --batch-size 64 --chunk-positions 32 --output-dir results/naacl_experiments`. Log: `logs/naacl_experiments/20261012/TF-Q3-draft.log`.
- Scoring completed all 20,000 common prompts in 10 shards, totaling 2,465,034 positions. All shards passed the LM-head check (`scale=1`, `max_abs_error=0`); summed shard wall time was 768.12 seconds. Manifest status: `COMPLETE`.
- Simulation command: `timeout --foreground --signal=TERM --kill-after=10s 21600s /venv/main/bin/python -u scripts/simulate_sd.py --pair Q3 --split common --output-dir results/naacl_experiments`. Log: `logs/naacl_experiments/20261012/TF-Q3-simulate.log`; all five block sizes completed.

| Block size | Eligible proposals | Rejections | Rejection rate |
| ---: | ---: | ---: | ---: |
| 1 | 1,315,672 | 156,545 | 0.118985 |
| 2 | 1,746,689 | 204,382 | 0.117011 |
| 4 | 2,085,610 | 242,496 | 0.116271 |
| 8 | 2,295,967 | 263,792 | 0.114894 |
| 16 | 2,401,277 | 272,482 | 0.113474 |

Q3 outputs are under `results/naacl_experiments/teacher_forced/generated/qwen8/qwen4/common/` and `results/naacl_experiments/simulated/Q3/common/`. These rates are offline replay estimates; Q2's failed strict full-V0 gate remains a caveat for live-decoder generalization.

**Queue status at 09:07 UTC:** the full six-pair Qwen draft×target grid is now teacher-forced scored and simulated, including the Q1 baseline. Next: Gemma and Ministral cross-model scoring, full G1 live validation, then human-written text and the permitted 2k Wikipedia core-lite job. Resolve the Q2 V0 unmatched-position gate before treating simulator estimates as fully live-validated.
