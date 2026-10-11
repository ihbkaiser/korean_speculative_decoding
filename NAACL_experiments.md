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
| Sun 20:30–21:30 | Timing pilot: 200 prompts per target gen and TF pass. V0 on the first 100 Q2 prompts | Run the simulator on the 100 prompts; compare with V0 labels | **G0:** agreement ≥ 99% and every job's ETA known; fix tier |
| Sun 21:30–Mon 04:00 | G-Q4 (40k), G-Q1.7, G-Q8, then TF-Qwen | Kiwi alignment of each continuation set as it lands | — |
| Mon 04:00–09:00 | G-G12, TF-G | RQ1: Tab. 2 and Fig. 2 on Q2. Draw the 300-item human-validation sample | **G1 (09:00):** RQ1 result checked against Section 2's last column; Qwen grid complete |
| Mon 09:00–13:00 | G-M8, TF-M, rest of V0 (1,000 Q2 + 500 G1) | RQ3 diagnostics and particle confusion matrix; send the validation sheet to the annotator | — |
| Mon 13:00–16:00 | H (human-written text), then W (Wikipedia-ko) | RQ2 heatmap, grid and stability; RQ5 cost decomposition | **G2 (16:00):** all core jobs done, else apply cut order |
| Mon 16:00–Tue 09:00 | Optional tier in order O3, O1, O2, O4; reruns of any failed shard first | RQ4 falsification battery and second analyzer; update paper tables and figures | — |
| Tue 09:00–11:30 | Buffer only; no new long jobs after 09:00 | Checksum and copy all outputs (continuations, TF logs, V0, timings) off the machine | **G3 (11:30):** artifacts safe; GPU released at 12:00 |

All analysis after Tue 12:00 is CPU-only and can continue until the submission deadline.

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
