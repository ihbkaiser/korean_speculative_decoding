# KORSpec A6000 experiment plan

Oct 11, 2026 · @ihb

The paper becomes a pure analysis paper built on five research questions, with no new method; future directions go in the discussion. The A6000 48GB runs until 12:00 on Oct 13, 2026 (about 41 GPU-hours). To fit, live speculative decoding is replaced by target greedy generation plus teacher-forced scoring, which reproduces greedy SD rejections exactly for any block size. The saved compute goes to generality and validity: all five frozen pairs, a Qwen draft x target grid, a second domain and human-written text.

**Revision (18:50, Oct 11).** The method (BADS) and everything that only served it are removed: the step-cost benchmark, policy replay and the live wall-clock check. Added in their place: a quantified cost of fusion for SD (RQ5), a particle confusion analysis, a human-written-text replication, and human validation of the morphology labels.

## 1. The key change: replay greedy SD decisions from saved model scores

Under greedy SD, a draft proposal at position t is only verified if every earlier proposal in its block was accepted. Each accepted proposal equals the target's token, so the draft predicts position t from the target's greedy prefix. Teacher-forced draft scores on that continuation therefore reproduce the proposal decisions when the same deterministic argmax decisions are used. This is an exact replay conditional on the saved continuation and matching runtime decisions; cache, precision, or near-tie differences can make a live decoder diverge.

Two cheap passes give everything the live decoder gave:

1. **Target pass:** batched greedy generation (HF `generate`, 128 new tokens), then one teacher-forced forward to log the target's entropy, top-5 tokens and margin at every position.
2. **Draft pass:** one teacher-forced forward over prompt + target continuation, logging the draft argmax, its log-probability, entropy and top-2 margin at every position.

A CPU replay then walks each continuation with block size k: it accepts while draft argmax equals the target token, marks the first mismatch as a rejection, and drops the rest of the block. This gives eligible proposals and rejection labels for k = 4 and other k values, without another GPU pass. It does not measure live decoding latency.

**Validation scope.** The discovery study found SD and teacher-forced labels agree at 99.88% of positions. The dedicated Q2 V0 run reports 98.0577% agreement on joined shared-context proposal positions, above the user-selected 95% gate, while the strict zero-unmatched check failed; outer-joined agreement was 88.6536%, and the common-prefix token fraction was 54.5771%. These figures support the planned shared-context token analyses with explicit scope limits. They do not validate diverged positions or establish live latency; see Section 4 for unmatched counts and the G1 status.

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

### CPU result: Q2 RQ1 Table 2a

The first full-sample RQ1 fit is complete from the Q2 40k `all` artifacts. The analysis joined the k=4 eligible replay rows to target-path Kiwi alignment, Q2 draft TF, and target self-TF; token IDs and argmax IDs matched across the saved sources. It retained visible, `ALIGNED` proposals whose eojeol had complete single-eojeol token coverage and exact tokenizer roundtrip, giving **804,045 proposals from 34,211 documents**. The full control set is eojeol token-count fixed effects, k=4 block-slot fixed effects, a 5-df cubic spline of generated position, eojeol character length, draft/target entropy, and draft/target top-two margins. Standard errors are clustered by document; AME intervals use the delta method over the cluster sandwich covariance.

| Predictor (higher = more boundary error) | AME | Clustered SE | 95% CI | p |
| --- | ---: | ---: | ---: | ---: |
| 1 − boundary F1 | −0.00985 | 0.00115 | [−0.01212, −0.00759] | 1.45e−17 |
| 1 − boundary recall (fusion) | −0.00167 | 0.00113 | [−0.00389, 0.00055] | 0.141 |
| 1 − boundary precision (splitting) | −0.01032 | 0.00113 | [−0.01254, −0.00810] | 7.86e−20 |

All three logistic fits converged. These conditional associations are negative for F1 error and splitting error, while fusion error is not distinguishable from zero; they do not support the plan's anticipated positive scalar-error story. This is an observational association, not a causal effect. The operational boundary-set definition and empty-set convention are recorded in the metadata and should accompany the table. Boundary error is 1−score; both empty boundary sets give precision/recall/F1=1, while an empty predicted or gold set scores 0 on the corresponding precision or recall. Reproduction command: `/venv/main/bin/python -u scripts/analyze_naacl_cpu.py --config configs/naacl_experiments.yaml`. CSV, Markdown, and fit metadata are under `results/naacl_experiments/analysis/rq1_table2a_ame.*`; execution log: `logs/naacl_experiments/20261012/CPU-Q2-RQ1.log`.

### CPU result: Table 1, Table 2b, and Figure 2

All eight core pairs joined without unmatched event-to-alignment rows; target token IDs and target/draft TF argmax IDs matched the saved k=4 replay. Q2 uses the 40k `all` split; the other pairs use the frozen common 20k split. Pair-level counts and rejection rates are in `table1_pair_summary.csv`; class counts, shares, and class rejection rates are in `table1_alignment_classes.csv`.

Table 2b reports adjusted discrete AMEs for CROSS versus WITHIN proposals from pair-specific logistic models with eojeol-token-count fixed effects, k=4 slot fixed effects, a 5-df cubic position spline, log eojeol length, draft/target entropy, and draft/target top-two margins. Standard errors cluster by document.

| Pair | N proposals | Raw CROSS rejection | Raw WITHIN rejection | Adjusted CROSS−WITHIN AME | 95% CI | p |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Q1 0.6B→1.7B | 968,737 | 18.44% | 11.01% | +0.09 pp | [−0.11, +0.28] | 0.393 |
| Q2 0.6B→4B | 1,974,984 | 23.44% | 14.41% | +0.04 pp | [−0.11, +0.19] | 0.600 |
| Q3 4B→8B | 973,884 | 14.43% | 10.29% | −1.42 pp | [−1.61, −1.24] | 2.20e−50 |
| Q4 1.7B→4B | 976,040 | 17.31% | 11.28% | −0.69 pp | [−0.89, −0.49] | 4.83e−12 |
| Q5 0.6B→8B | 999,387 | 26.43% | 16.69% | +0.37 pp | [+0.14, +0.59] | 0.00128 |
| Q6 1.7B→8B | 985,750 | 19.41% | 13.08% | −0.79 pp | [−0.99, −0.58] | 4.63e−14 |
| G1 4B→12B | 822,145 | 1.36% | 1.55% | −0.39 pp | [−0.46, −0.32] | 5.08e−30 |
| M1 3B→8B | 1,010,655 | 15.95% | 10.29% | −0.01 pp | [−0.18, +0.16] | 0.920 |

All eight models converged. The controlled effects vary in sign across the Qwen grid; Q1, Q2, and M1 are not distinguishable from zero. This is an observational model result, not a causal effect. Full counts, raw rates, and estimates are in `rq1_table2b_cross_within.csv` and `rq1_table2b_metadata.json`.

Figure 2 uses each CROSS token's observed fine Kiwi boundary and assigns each WITHIN token to its nearest fine boundary. **Thirty-eight** boundaries met the planned minimum of 200 proposals in each class. **Thirty-six** boundary-specific adjusted contrasts were estimable; two supported fits are retained as diagnostics but excluded from the plot and Benjamini–Hochberg tests: `NP→JKO` had zero clustered variance, and `VA-I→EC` had a rank-deficient design. Nine of the 36 estimable contrasts have BH q<0.05. The largest positive estimates are `NNB→JKS` (+33.94 pp, q=1.51e−235), `NNG→JC` (+31.78 pp, q=2.49e−28), and `VCP→EC` (+15.00 pp, q=1.03e−8); negative estimates include `EP→EF` (−9.80 pp, q=9.97e−4) and `XSV→EC` (−6.42 pp, q=7.30e−4). Full estimates and diagnostics are in `rq1_fig2_boundary_landscape.csv` and `rq1_fig2_metadata.json`.

Reproduction: `/venv/main/bin/python -u scripts/analyze_naacl_cpu.py --config configs/naacl_experiments.yaml --analysis rq1b-table1-fig2`, followed by the final boundary pass `/venv/main/bin/python -u scripts/analyze_naacl_cpu.py --config configs/naacl_experiments.yaml --analysis rq1-fig2`, then `figures/gen_fig_naacl_rq1_boundary_landscape.py`. Figure exports are `figure2_boundary_landscape.pdf` and `figure2_boundary_landscape.png`. Logs: `CPU-RQ1-table1-fig2.log`, `CPU-Q2-RQ1-fig2-final.log`, and `CPU-RQ1-summary.log` under `logs/naacl_experiments/20261012/`.

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
- The report's selected shared-context rejection-label agreement is **98.0577%** (95% prompt-cluster bootstrap CI 97.9196%–98.1808%) on 58,385 joined proposal positions, above the user-selected 95% operational threshold. The original runner also imposed a zero-unmatched-position condition and therefore records its strict status as `FAIL`: 949 live and 947 simulated positions were unmatched. The outer-joined all-position agreement is 88.6536% across 100,252 rows. The live target sequence exactly matched the cached continuation on 405/1,000 prompts; common-prefix token fraction was 54.5771%.
- Interpretation: the CPU simulator deterministically replays teacher-forced argmaxes and reproduces the saved greedy continuation's proposal decisions under the stated assumptions. The user accepted V0 under the amended ≥95% shared-prefix agreement gate, so use simulator estimates for the planned token-level analyses. Keep the unmatched counts, 54.58% common-prefix fraction, and 88.65% outer-joined agreement visible as scope diagnostics. These are not live latency measurements, and the validation does not directly cover unmatched or diverged positions.

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

### G1 Gemma scoring, replay, and full live validation

- Qwen checkpoint eviction (after finishing the grid and Q2 validation) used `scripts/download_models.py --evict-pair Q1` and `--evict-pair Q3`; logs: `Qwen-evict-Q1.log` and `Qwen-evict-Q3.log`. This freed `/dev/shm` for Gemma-12B.
- Gemma-12B was downloaded to `/dev/shm/naacl_models`; Gemma-4B was downloaded to `/workspace/naacl_gemma_models` because both checkpoints do not fit together in `/dev/shm`. Download logs: `G-G12-download.log` and `G-G4-download.log`. Both snapshots passed an A6000 BF16/SDPA one-token smoke forward. Gemma-4B smoke log: `G-G4-smoke-load.log`. Gemma-12B's first smoke-load retry was rejected by the downloader's 1-GiB free-space preflight even though its completed checkpoint was present; the same repository smoke helper was run directly on that local snapshot and passed. Both attempts are preserved in `G-G12-smoke-load.log`.
- Tokenizer audit: `/venv/main/bin/python -u scripts/audit_tokenizers.py --pair G1 --output-dir results/naacl_experiments`; `TOKENIZER_AUDIT G1 PASS`. Log: `TF-G1-tokenizer-audit.log`; report: `results/naacl_experiments/provenance/tokenizer_audits/G1.json`.
- Gemma-4B draft TF command: `NAACL_MODEL_DIR_GEMMA4=/workspace/naacl_gemma_models NAACL_MODEL_DIR_GEMMA12=/dev/shm/naacl_models PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout --foreground --signal=TERM --kill-after=10s 14400s /venv/main/bin/python -u scripts/tf_score.py --model gemma4 --target gemma12 --split common --role draft --batch-size 32 --chunk-positions 32 --output-dir results/naacl_experiments`. Log: `TF-G1-draft.log`.
- TF completed all 20,000 prompts in 10 shards and 2,549,337 positions. All shards passed the LM-head check (`scale=1`, `max_abs_error=0`); summed shard time was 732.38 seconds. Manifest status: `COMPLETE`.
- Simulator command: `timeout --foreground --signal=TERM --kill-after=10s 21600s /venv/main/bin/python -u scripts/simulate_sd.py --pair G1 --split common --output-dir results/naacl_experiments`; log: `TF-G1-simulate.log`.

| Block size | Eligible proposals | Rejections | Rejection rate |
| ---: | ---: | ---: | ---: |
| 1 | 1,294,085 | 29,408 | 0.022725 |
| 2 | 1,721,239 | 35,288 | 0.020502 |
| 4 | 2,059,836 | 38,629 | 0.018753 |
| 8 | 2,283,008 | 40,155 | 0.017589 |
| 16 | 2,414,797 | 40,717 | 0.016861 |

- Full G1 live validation command: `NAACL_MODEL_DIR_GEMMA4=/workspace/naacl_gemma_models NAACL_MODEL_DIR_GEMMA12=/dev/shm/naacl_models PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout --foreground --signal=TERM --kill-after=10s 14400s /venv/main/bin/python -u scripts/validate_live_sd.py --pair G1 --prompts 500 --output-dir results/naacl_experiments`. Log: `V0-G1-full.log`; artifacts: `results/naacl_experiments/validation/v0/G1/`.
- G1 full V0 measured 99.8213% shared-context rejection-label agreement (95% prompt-cluster bootstrap CI 99.7786%–99.8601%) and 99.6120% all-position joined-label agreement. This exceeds the user-selected 95% operational threshold. The original runner reports strict status `FAIL` because it additionally requires zero unmatched positions; it found 714 live and 716 simulated unmatched rows. Target sequences matched exactly on 442/500 prompts; common-prefix token fraction was 90.1743%.
- Together with Q2 (98.0577% shared-context agreement, 88.6536% all-position agreement, 949/947 unmatched rows), V0 meets the user's amended shared-prefix agreement gate for the planned simulator analyses. Preserve the runner's strict status and unmatched counts as diagnostics. Neither V0 run measures latency or directly validates unmatched/diverged positions.

G1 TF and replay outputs are under `results/naacl_experiments/teacher_forced/generated/gemma12/gemma4/common/` and `results/naacl_experiments/simulated/G1/common/`. G1 live validation is under `results/naacl_experiments/validation/v0/G1/`.

### M1 cross-model draft scoring and replay

- Ministral pair tokenizer audit: `TOKENIZER_AUDIT M1 PASS`; both pinned models have identical serialized tokenizers, vocabularies, special tokens, and probe IDs/offsets. Report: `results/naacl_experiments/provenance/tokenizer_audits/M1.json`; log: `logs/naacl_experiments/20261012/TF-M1-tokenizer-audit.log`.
- Draft TF command: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout --foreground --signal=TERM --kill-after=10s 14400s /venv/main/bin/python -u scripts/tf_score.py --model ministral3 --target ministral8 --split common --role draft --batch-size 64 --chunk-positions 32 --output-dir results/naacl_experiments`. Log: `logs/naacl_experiments/20261012/TF-M1-draft.log`.
- TF completed all 20,000 common prompts in 10 shards, totaling 2,498,617 positions. Every shard passed the LM-head check (`scale=1`, `max_abs_error=0`); summed shard wall time was 565.54 seconds. Manifest status: `COMPLETE`. Model revisions: Ministral-3-3B `6f9c4b12a95b139af68670a6713616b757923735`; Ministral-3-8B `d4883f9b36aa2e5d775730d3fdba3d30de51a8ef`.
- Simulation command: `timeout --foreground --signal=TERM --kill-after=10s 21600s /venv/main/bin/python -u scripts/simulate_sd.py --pair M1 --split common --output-dir results/naacl_experiments`. Log: `logs/naacl_experiments/20261012/TF-M1-simulate.log`; all five block sizes completed.

| Block size | Eligible proposals | Rejections | Rejection rate |
| ---: | ---: | ---: | ---: |
| 1 | 1,335,792 | 163,248 | 0.122211 |
| 2 | 1,774,690 | 216,690 | 0.122100 |
| 4 | 2,115,278 | 254,827 | 0.120470 |
| 8 | 2,324,623 | 275,176 | 0.118374 |
| 16 | 2,428,883 | 282,755 | 0.116414 |

M1 outputs are under `results/naacl_experiments/teacher_forced/generated/ministral8/ministral3/common/` and `results/naacl_experiments/simulated/M1/common/`. These are offline replay estimates; M1 has not had full live validation.

### H human-text scoring: Qwen cells

- Command template: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout --foreground --signal=TERM --kill-after=10s 14400s /venv/main/bin/python -u scripts/tf_score.py --model MODEL --target TARGET --split common --role ROLE --continuation human --max-prompts 10000 --batch-size 32 --chunk-positions 32 --output-dir results/naacl_experiments`. The complete human-text Qwen matrix has nine cells: three target self-scores and all six planned cross-size draft/target cells. Each scored 9,722 prompts / 1,171,450 positions; all five shards per cell passed (`head_scale=1`, `head_check_max_abs_error=0`).
- The human branch uses each model's tokenizer to take the document's real next 128 tokens after a 128-token prompt. It considered the first 10,000 common-split documents and skipped the same 278 documents without a full prompt and continuation, leaving 9,722 prompts / 1,171,450 continuation positions per cell.

| Cell | Model / target | Prompts | Positions | Summed shard time | Result |
| --- | --- | ---: | ---: | ---: | --- |
| Target qwen17 | qwen17 / qwen17 | 9,722 | 1,171,450 | 256.41 s | Complete; all five head checks pass |
| Target qwen4 | qwen4 / qwen4 | 9,722 | 1,171,450 | 404.28 s | Complete; all five head checks pass |
| Target qwen8 | qwen8 / qwen8 | 9,722 | 1,171,450 | 573.44 s | Complete; all five head checks pass |
| Draft qwen06 | qwen06 / qwen4 | 9,722 | 1,171,450 | 178.43 s | Complete; all five head checks pass |
| Draft qwen06 | qwen06 / qwen17 | 9,722 | 1,171,450 | 148.68 s | Complete; all five head checks pass |
| Draft qwen17 | qwen17 / qwen4 | 9,722 | 1,171,450 | 204.01 s | Complete; all five head checks pass |
| Draft qwen06 | qwen06 / qwen8 | 9,722 | 1,171,450 | 148.77 s | Complete; all five head checks pass |
| Draft qwen17 | qwen17 / qwen8 | 9,722 | 1,171,450 | 235.98 s | Complete; all five head checks pass |
| Draft qwen4 | qwen4 / qwen8 | 9,722 | 1,171,450 | 342.34 s | Complete; all five head checks pass |

The three missing cells were added after an audit showed the original six-cell H run did not cover all Qwen pairs. Their logs are `H-qwen06-draft-qwen17.log`, `H-qwen17-draft-qwen4.log`, and `H-qwen06-draft-qwen8.log`; download/smoke and pair-scoped eviction logs are `H-Qwen06-missing-download-smoke.log`, `H-Qwen17-missing-download-smoke.log`, `H-Qwen06-Q5-download-smoke.log`, and `H-Q{1,4,5}-draft-eviction.log`. Existing cell logs remain `H-{qwen17-target,qwen4-target,qwen8-target,qwen06-draft-qwen4,qwen17-draft-qwen8,qwen4-draft-qwen8}.log`.

Human-text alignment used the same pinned target tokenizer and validated each reconstructed token sequence against the completed target self-TF rows before applying Kiwi. Command template: `scripts/align_morphology.py --config configs/naacl_experiments.yaml --target TARGET --split common --continuation human --max-prompts 10000`. All five targets completed: Qwen17 9,722 prompts / 1,171,450 rows; Qwen4 9,722 / 1,171,450; Qwen8 9,722 / 1,171,450; Gemma12 9,604 / 1,127,951; Ministral8 9,651 / 1,133,929. Outputs are under `results/naacl_experiments/aligned/kiwi/{target}/human/common/`. Logs: `H-{qwen17,qwen8,gemma12,ministral8}-human-alignment.log`; Qwen4's first adapter attempt failed before writing a shard on a field-name mismatch, and the corrected complete retry is `H-qwen4-human-alignment-retry.log` (the failed attempt is retained as `H-qwen4-human-alignment.log`).

### H human-text scoring: Gemma cells

- Command template: `NAACL_MODEL_DIR_GEMMA4=/workspace/naacl_gemma_models NAACL_MODEL_DIR_GEMMA12=/dev/shm/naacl_models PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout --foreground --signal=TERM --kill-after=10s 14400s /venv/main/bin/python -u scripts/tf_score.py --model MODEL --target gemma12 --split common --role ROLE --continuation human --max-prompts 10000 --batch-size 32 --chunk-positions 32 --output-dir results/naacl_experiments`.
- Gemma tokenization skipped 396 of the first 10,000 common documents because they were too short for a 128-token prompt plus continuation; both cells scored the remaining 9,604 documents / 1,127,951 continuation positions across five shards. Every shard passed (`head_scale=1`, `head_check_max_abs_error=0`).

| Cell | Model / target | Prompts | Positions | Summed shard time | Result |
| --- | --- | ---: | ---: | ---: | --- |
| Target gemma12 | gemma12 / gemma12 | 9,604 | 1,127,951 | 844.90 s | Complete; all five head checks pass |
| Draft gemma4 | gemma4 / gemma12 | 9,604 | 1,127,951 | 348.11 s | Complete; all five head checks pass |

Gemma outputs are under `results/naacl_experiments/teacher_forced/human/gemma12/{gemma12,gemma4}/common/`. Logs: `H-gemma12-target.log`, `H-gemma4-draft-gemma12.log`, and `H-gemma12-download-smoke.log`. Gemma12 weights were removed from `/dev/shm` after scoring; Gemma4 remains in `/workspace/naacl_gemma_models` for W. The first attempted eviction mistakenly used the download `--model` flag and stopped at the storage preflight without changing files; the subsequent `--evict-pair G1 --model-root /dev/shm/naacl_models` removed all Gemma12 weight shards. Both command logs are preserved as `H-Gemma12-eviction-misflagged-download-attempt.log` and `H-G1-eviction.log`.

### H human-text scoring: Ministral cells

- Command template: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout --foreground --signal=TERM --kill-after=10s 14400s /venv/main/bin/python -u scripts/tf_score.py --model MODEL --target ministral8 --split common --role ROLE --continuation human --max-prompts 10000 --batch-size 32 --chunk-positions 32 --output-dir results/naacl_experiments`.
- Ministral tokenization skipped 349 of the first 10,000 common documents as too short for a 128-token prompt plus continuation; both cells scored 9,651 documents / 1,133,929 continuation positions across five shards. Every shard passed (`head_scale=1`, `head_check_max_abs_error=0`).

| Cell | Model / target | Prompts | Positions | Summed shard time | Result |
| --- | --- | ---: | ---: | ---: | --- |
| Target ministral8 | ministral8 / ministral8 | 9,651 | 1,133,929 | 545.18 s | Complete; all five head checks pass |
| Draft ministral3 | ministral3 / ministral8 | 9,651 | 1,133,929 | 274.12 s | Complete; all five head checks pass |

Ministral outputs are under `results/naacl_experiments/teacher_forced/human/ministral8/{ministral8,ministral3}/common/`. Logs: `H-ministral8-target.log`, `H-ministral3-draft-ministral8.log`, `H-M1-download-smoke.log`, and `H-M1-eviction.log`. The 3B and 8B weight shards were evicted from `/dev/shm` after both cells.

Across the complete H study, tokenizer-specific filtering left 9,722 Qwen, 9,604 Gemma, and 9,651 Ministral documents. All 13 planned target/draft cells completed with passing per-shard head checks, and all five target tokenizers have complete human-text morphology alignment. Do not treat the family-specific document counts as identical sample sizes. Human-text next-token mismatches are teacher-forced observational labels; they are not live speculative-decoding outcomes.

### W Wikipedia-ko data preparation

- Applied the plan's storage cut from 5,000 to 2,000 prompts: retain the original 1,000-document local seed and append 1,000 additional unique articles.
- Preparation command: `/venv/main/bin/python -u scripts/prepare_naacl_wikipedia.py`. Log: `logs/naacl_experiments/20261012/W-data-prep.log`.
- The new script streams `wikimedia/wikipedia`, config `20231101.ko`, split `train`, pinned at `b04c8d1ceb2f5cd4588862100d08de323dccfbaa`. Before writing outputs, it verified the cached seed's source indices, titles, and text against that exact revision. It produced 2,000 ordered prompts; no duplicate extension texts were skipped.
- Frozen data: `data/naacl_wikipedia_2k.parquet` (SHA-256 `a96dc3086f73c587373036b287cf53708f43a5a69d0d0d8464c2f3ea0785944b`) and `data/naacl_wikipedia_2k_ids.txt`. Provenance, seed hash, ordered ID/text hashes: `results/naacl_experiments/provenance/wikipedia_2k_manifest.json`.
- Dedicated run config: `configs/naacl_wikipedia.yaml`; it points to the frozen 2k pool and isolates W outputs under `results/naacl_experiments/wikipedia/`. W inference remains pending: generate Qwen4 and Gemma12 continuations; score/alignment for each target; teacher-force the Qwen06 and Gemma4 drafts; then run Q2 and G1 replay summaries.

**Queue status at 11:25 UTC:** Qwen grid, G1 cross-model TF/replay/full V0, M1 draft TF/replay, and all 10 H cells are complete and preserved. W data is frozen and ready for inference at the plan-approved 2,000-document size; W target generation, target TF/alignment, draft TF, and replay remain. Q2 and G1 V0 exceed the 95% shared-context agreement level but fail the additional zero-unmatched check. Interpret that condition before claiming full live validation.

### W Wikipedia-ko Q2 run

- Pair: Qwen3-0.6B draft → Qwen3-4B target, on the frozen 2,000-document Wikipedia-ko pool. Generation used greedy decoding with 128 prompt and 128 new-token limits, bfloat16, SDPA, and the pinned Qwen revisions in the run metadata. The target produced 249,908 continuation tokens across four 500-document shards; generation wall time was 307.86 s.
- The target self-score and draft score each covered all 2,000 prompts / 249,908 tokens in four shards. Every shard passed its head check (`head_scale=1`, maximum absolute error 0); summed shard times were 91.69 s for the target and 31.43 s for the draft. Kiwi alignment completed all four shards with 249,908 aligned rows.
- Offline replay completed for k ∈ {1, 2, 4, 8, 16}:

| Block size k | Eligible proposals | Rejections | Rejection rate |
| ---: | ---: | ---: | ---: |
| 1 | 138,123 | 25,406 | 18.3938% |
| 2 | 181,734 | 31,978 | 17.5960% |
| 4 | 215,490 | 37,629 | 17.4621% |
| 8 | 234,647 | 39,570 | 16.8636% |
| 16 | 243,885 | 40,362 | 16.5496% |

- These are deterministic offline replay rates over the saved target continuation and teacher-forced logits; they are not live decoding latency measurements. The Q2 full-V0 unmatched-context caveat above also applies, so describe these as offline estimates pending resolution of that live-validation issue.
- Artifacts are under `results/naacl_experiments/wikipedia/`: generated continuations, target and draft TF shards, Kiwi alignment shards, and per-k event tables plus summaries. Run metadata records pinned model revisions, prompt/data hashes, runtime environment, shard counts, and head checks. Logs: `W-Q2-download-smoke.log`, `W-Q2-generation.log`, `W-Q2-target-TF.log`, `W-Q2-alignment.log`, `W-Q2-draft-TF.log`, `W-Q2-simulate.log`, and `W-Q2-eviction.log` under `logs/naacl_experiments/20261012/`. Both Qwen model weight sets were evicted after the pair; tokenizer/config files were retained.

**Queue status after W-Q2:** Q2's 2k Wikipedia generation, target TF, alignment, draft TF, and offline replay are complete and preserved. Next run G1 (Gemma4→Gemma12) on the same frozen pool. Continue to report replay as offline estimates; the Q2/G1 full-V0 unmatched-position failures remain unresolved.

### W Wikipedia-ko G1 run

- Pair: Gemma-3-4B-pt draft → Gemma-3-12B-pt target on the same frozen 2,000-document pool. Target generation used greedy decoding, 128-token prompt/continuation limits, bfloat16, SDPA, and pinned Gemma revisions. All four 500-document shards completed; generation wall time was 923.21 s.
- Target and draft TF each scored 2,000 prompts / 255,179 continuation tokens across four shards. Every shard passed its head check (`head_scale=1`, maximum absolute error 0). Summed shard times: 176.93 s target, 74.15 s draft. Kiwi alignment completed all four shards with 255,179 aligned rows.
- Offline replay completed for k ∈ {1, 2, 4, 8, 16}:

| Block size k | Eligible proposals | Rejections | Rejection rate |
| ---: | ---: | ---: | ---: |
| 1 | 129,587 | 3,025 | 2.3343% |
| 2 | 172,358 | 3,601 | 2.0893% |
| 4 | 206,325 | 4,094 | 1.9843% |
| 8 | 228,711 | 4,254 | 1.8600% |
| 16 | 241,794 | 4,324 | 1.7883% |

- Target TF label disagreement is 204 positions for each block size. As for W-Q2, these are deterministic offline saved-logit replay estimates, not live decoding or latency measurements. G1 live V0 supports agreement on shared-prefix contexts; unmatched/diverged positions remain a limit on claims about full live behavior.
- Artifacts are under `results/naacl_experiments/wikipedia/continuations/gemma12/`, `teacher_forced/generated/gemma12/`, `aligned/kiwi/gemma12/`, and `simulated/G1/`. Logs under `logs/naacl_experiments/20261012/`: `W-G1-download-smoke.log`, `W-G1-generation.log`, `W-G1-target-TF.log`, `W-G1-alignment.log`, `W-G1-draft-TF.log`, `W-G1-simulate.log`, and `W-G1-eviction.log`. Gemma12 weights were evicted from `/dev/shm` after scoring; Gemma4 remains in `/workspace/naacl_gemma_models`.

**Queue status after W-G1 (2026-10-12 12:03 UTC):** W-Q2 and W-G1 are complete on the frozen 2k Wikipedia sample, and the planned H and W GPU cells are complete. CPU RQ1 and RQ2 analyses have since been completed. Next: RQ3–RQ5 analyses, remaining morphology/frequency falsification and validation artifacts, then update the paper draft. V0 is accepted under the user's 95% shared-prefix agreement gate; report unmatched-position counts as a scope diagnostic and do not present offline replay as a live latency result.

### RQ2 CPU analysis: boundary stability across pairs and text settings

- Analysis command: `/venv/main/bin/python -u scripts/analyze_naacl_rq2.py`; final log: `logs/naacl_experiments/20261012/CPU-RQ2-final.log`. Boundary summaries cover eight core pairs on the common 20k prompt pool, Q2 and G1 on the frozen 2k Wikipedia pool, and all eight human-text pairs. Each human target/draft TF join matched at every scored position; the human outcome is **draft-versus-target argmax disagreement at the same human-written prefix**, not a live speculative-decoding rejection.
- **Figure 3a** uses coarse Kiwi boundary families. A cell is shown only with at least 200 CROSS and 200 WITHIN k=4 replay proposals. All 40 core pair-by-family cells meet this threshold. The global rate differences vary by tokenizer family: Qwen pairs show a positive N→P difference from +6.7 to +15.5 pp, while G1 is −0.3 pp and M1 is +6.2 pp. The fine-boundary source table retains counts and rates for every supported label.
- **Figure 3b** reports adjusted CROSS−WITHIN AMEs with document-clustered 95% CIs. Qwen global and N→P estimates (percentage points) are:

| Pair | Global AME [95% CI] | N→P AME [95% CI] |
| --- | ---: | ---: |
| Q1 0.6B→1.7B | +0.09 [−0.11, +0.28] | −0.01 [−0.62, +0.61] |
| Q2 0.6B→4B | +0.02 [−0.20, +0.23] | +1.44 [+0.79, +2.09] |
| Q3 4B→8B | −1.42 [−1.61, −1.24] | −1.95 [−2.49, −1.40] |
| Q4 1.7B→4B | −0.69 [−0.89, −0.49] | −0.99 [−1.57, −0.41] |
| Q5 0.6B→8B | +0.37 [+0.14, +0.59] | +3.00 [+2.34, +3.66] |
| Q6 1.7B→8B | −0.79 [−0.99, −0.58] | −0.34 [−0.96, +0.27] |

- **Table 3** compares all 18 settings pairwise on fine Kiwi POS-transition boundaries. Spearman rank and sign agreement use only boundaries with at least 100 CROSS and 100 WITHIN rows in both settings. The top-five set is the five largest positive CROSS−WITHIN effects among each setting's supported fine boundaries. Of 153 setting pairs, 136 are estimable and 17 are marked not estimable because fewer than three boundaries meet shared support; every non-estimable comparison involves Wikipedia Q2, which has only one supported fine boundary. Among core generated pairs, mean rank correlation is 0.618, mean sign agreement 0.796, and mean top-five Jaccard is 0.306 (28 comparisons). Human-versus-human values are 0.668, 0.794, and 0.413 (28 comparisons). Same-pair generated-versus-human comparisons average 0.367 rank correlation, 0.792 sign agreement, and 0.186 top-five Jaccard (8 comparisons). These summarize stability across distinct outcomes; human disagreement is not treated as an SD rejection.
- Replay outcomes remain scoped by V0: the accepted ≥95% threshold applies to shared-context Q2 proposal positions. The 54.58% common-prefix fraction and 88.65% outer-joined agreement remain visible validation diagnostics. These RQ2 replay associations are not live-latency results, and their estimates do not cover diverged contexts. Wikipedia Q2 stability is underpowered at the fine-boundary level.
- Artifacts: `results/naacl_experiments/analysis/rq2_boundary_effects.csv` (coarse and fine boundary rates/support), `rq2_qwen_grid.csv` (fit status and estimate provenance), `rq2_table3_setting_stability.csv`, and `rq2_metadata.json`. Figures: `figure3a_rq2_boundary_pair_heatmap.{pdf,png}` and `figure3b_rq2_qwen_grid.{pdf,png}`. Figure source: `figures/gen_fig_naacl_rq2.py`; analysis source: `scripts/analyze_naacl_rq2.py`.

**Queue status after RQ2 (2026-10-12):** RQ1 and RQ2 CPU exhibits are complete. Next per plan: RQ3 token-level mechanism analyses, RQ4 falsification/robustness analyses, and RQ5 rejection-cost decomposition; then finalize remaining morphology validation and paper-facing summaries. Continue to label replay estimates with the V0 shared-context scope and keep the human-text disagreement outcome separate.

### RQ3 CPU analysis: token geometry and rejected-candidate diagnostics

- Analysis command: `/venv/main/bin/python -u scripts/analyze_naacl_rq3.py`; final analysis and figure logs: `logs/naacl_experiments/20261012/CPU-RQ3.log` and `CPU-RQ3-figure.log`. These diagnostics use the Q2 40k k=4 saved replay and target/draft teacher-forced scores. Of 717,681 rejection events, 677,871 have the saved target token equal to the target TF argmax; the other 39,810 remain in the CSVs as a separate target-inconsistent group. Rank and particle panels use the target-consistent subset.
- **Boundary locus.** Among 168,861 visible, aligned, single-boundary CROSS tokens, the N→P internal boundary falls in the 25–50% position bin for 21,743/25,299 (85.9%) and the 50–75% bin for the remaining 3,556 (14.1%). This is the character position of the morpheme boundary within the fused target token, not a causal affix effect.
- **Target rank and draft entropy.** Among target-consistent CROSS rejections, the rejected Qwen 0.6B argmax is ranked 2–5 in the Qwen4 target's saved top-20 for 27,994/33,653 positions, ranks 6–10 for 3,467, ranks 11–20 for 1,315, and below rank 20 for 877 (right-censored). Target-consistent aligned CROSS rejections have median draft entropy 2.565 nats; WITHIN rejections have median 3.861 nats. Rank is limited to the saved target top-20 list.
- **Particle confusion.** There are 5,842 target-consistent, single-boundary N→P CROSS rejection events. Kiwi finds no particle in the decoded draft candidate token for 83.1%; topic/subject category swaps total 372 events. Thus this run does not support the prespecified prediction that topic/subject swaps dominate. Draft-side particle labels come from Kiwi on the nominal plus the decoded candidate surface; treat them as a surface diagnostic, particularly for subword fragments.
- **Draft-size trend.** Descriptive global and N→P slopes (pp per additional draft billion parameters) are −0.642 and −2.206 for the 4B target, and −0.489 and −1.338 for the 8B target. The 1.7B target has only one draft size and no estimable slope. These are descriptive trends over two or three grid points per target; no inferential CI is reported.
- Outputs: `rq3_affix_locus_by_boundary.csv`, `rq3_rejected_draft_target_rank.csv`, `rq3_rejection_entropy.csv`, `rq3_rejection_entropy_overall.csv`, `rq3_particle_confusion.csv`, `rq3_qwen_grid_draft_size_trends.csv`, and `rq3_metadata.json` under `results/naacl_experiments/analysis/`. Figure 5 is `figure5_rq3_token_diagnostics.{pdf,png}`; source is `figures/gen_fig_naacl_rq3.py` and analysis source is `scripts/analyze_naacl_rq3.py`.
- All simulator-derived values remain scoped to the saved greedy continuation and V0's shared-context agreement gate; they are not live latency results and do not validate diverged contexts.

### RQ5 CPU analysis: rejection cost and counterfactual accepted length

- Analysis command: `/venv/main/bin/python -u scripts/analyze_naacl_rq5.py`; log: `logs/naacl_experiments/20261012/CPU-RQ5.log`. The analysis joins the Q2 40k target continuation, draft argmaxes, Kiwi alignment, and saved replay events for k ∈ {1, 2, 4, 8, 16}. It charges one rejected proposal plus every invalidated suffix proposal in that block to the rejected position's alignment family.
- At k=4, the replay has 717,681 rejection events and 2,128,451 units of wasted draft work under that accounting. The five coarse single-boundary CROSS families account for 251,794 rejections (35.08% of all) and 751,148 wasted draft tokens (35.29%). N→P alone accounts for 90,193 rejections (12.57%) and 273,076 wasted draft tokens (12.83%). The `OTHER_OR_UNALIGNED` bucket accounts for 64.77% of rejections; class attribution is therefore limited by the visible aligned morphology subset.
- **Counterfactual.** At each k, replace each CROSS position's observed mismatch with an independent Bernoulli outcome whose rejection probability equals the eligible WITHIN rate for that coarse boundary; retain other positions' observed mismatch labels. Dynamic programming then computes expected target calls with greedy correction and bonus-token transitions. At k=4, observed replay yields 3.452 target tokens per target call; the counterfactual is 3.477 (document-cluster 95% CI 3.471–3.483), a point difference of +0.025. Across k=1, 2, 4, 8, 16, the differences are +0.003, +0.009, +0.025, +0.054, and +0.084 tokens per call. This is a modest model-based counterfactual, not a measured live-decoder gain or a causal estimate.
- The 500-resample document-cluster intervals hold estimated WITHIN rates fixed, so they do not include uncertainty from estimating those rates. Counterfactual positions are treated as independent conditional on boundary family; correlations among mismatch events are not modeled. Both the cost attribution and counterfactual remain conditional on saved continuations and V0's shared-context validation scope; they do not validate diverged contexts or measure wall-clock latency.
- Outputs: `rq5_cost_decomposition.csv`, `rq5_counterfactual_accepted_length.csv`, and `rq5_metadata.json` under `results/naacl_experiments/analysis/`. Figure 7 is `figure7_rq5_cost_and_accepted_length.{pdf,png}`; source: `figures/gen_fig_naacl_rq5.py`; analysis source: `scripts/analyze_naacl_rq5.py`.

**Queue status after RQ5 (2026-10-12):** RQ1–RQ3 and RQ5 CPU exhibits are complete. Next: RQ4 token-frequency/type-removal and pseudo-boundary falsification, then the alignment/analyzer robustness rows and human validation sheet. Human annotation and MeCab-ko remain outstanding; report those as pending unless completed, and do not substitute an automatic result for an annotator's judgment without labeling it.

### RQ4 CPU analysis: token types, random cuts, frequency, and analyzer robustness

- The Q2 RQ4 models use the same k=4 replay outcome and controls as Table 2b. The completed GLM run is logged in `CPU-RQ4-final.log`; final table, pseudo-boundary draws, metadata, and annotation sheet were assembled by `/venv/main/bin/python -u scripts/finalize_naacl_rq4.py --pseudo-reps 100` (`CPU-RQ4-finalize.log`). The post-projection rerun completed on Oct 12 with exit code 0; log: `CPU-RQ4-finalize-after-full40k.log`. Earlier failed/interrupted attempts are retained as `CPU-RQ4.log` and `CPU-RQ4-retry.log` for provenance.
- **Baseline and sample checks.** The full Q2 estimate is +0.041 pp (95% CI −0.111 to +0.192; p=.600). The strict exact-roundtrip and complete-eojeol filter retains all 1,974,984 primary CROSS/WITHIN proposals, so its estimate is identical by construction. Excluding multi-boundary CROSS tokens leaves 1,971,165 rows (3,819 fewer) and gives +0.020 pp (95% CI −0.133 to +0.172; p=.800). The common-20k estimate is +0.016 pp (95% CI −0.198 to +0.231; p=.880).
- **Token-type removal.** Remove the top N CROSS target-token IDs within each fine boundary from both CROSS and same-boundary WITHIN rows, then refit. The AMEs are +3.78 pp (N=10; 95% CI +3.09 to +4.47), +3.53 pp (25; +2.18 to +4.88), +3.42 pp (50; +1.63 to +5.22), and +4.54 pp (100; +1.86 to +7.22). These fits retain only 5,045, 1,384, 712, and 307 CROSS rows, respectively, from 149,457 at baseline. The estimate rises after removal, but CROSS support collapses; treat this as a sparse-support stress test, not evidence that a large population-level effect survives token removal.
- **Pseudo-boundaries.** For each strict-aligned eojeol with at least one Kiwi boundary, sample the same number of distinct internal character cuts uniformly without replacement. Among the 1,526,566 eligible CROSS/WITHIN proposals in that population, the observed raw rejection-rate gap is +8.64 pp. Across 100 random-cut draws, the median pseudo gap is +7.39 pp (95% draw interval +7.30 to +7.50); none reaches the observed gap (empirical two-sided tail probability 0.0099, the resolution limit with 100 draws). Arbitrary cuts therefore reproduce most of the raw gap, although the observed value is above this pseudo distribution. This result weakens a purely morphology-specific reading and is not a randomized causal test.
- **Frequency controls.** Adding a spline of external log token count from the old 1,000-article Korean Wikipedia cache gives −1.86 pp (95% CI −2.01 to −1.71); replacing the spline with frequency deciles gives −1.91 pp (−2.06 to −1.76). The Wikipedia counts remain an out-of-domain proxy for Q2 and share the W domain. The planned disjoint FineWeb2 cache is now complete: 40,000 normalized, unique `kor_Hang` train documents from pinned revision `af9c13333eb981300149d5ca60a8e9d659b276b9`, scanned in source range [250,000, 500,000), with collected rows spanning source indices 250,000–298,110. Text hashes were deduplicated against all 40,000 frozen Q2 prompts; zero overlap was found. The pinned Qwen3-4B tokenizer (revision `906bfd4b4dc7f14ee4320094d8b41684abff8539`) counted 43,497,725 tokens. The spline fit is −1.111 pp (95% CI −1.262 to −0.959), and the decile fit is −1.205 pp (−1.355 to −1.056), both converged on 1,974,984 proposals from 35,960 documents. Both frequency adjustments move the estimate below zero while the unadjusted AME is near zero, so report the result as specification-sensitive, associational evidence. Two cache-builder attempts printed their completion marker but aborted during interpreter shutdown; the separately run artifact validation passed all uniqueness, range, zero-overlap, and token-total checks. See `CPU-RQ4-fineweb2-frequency*.log`, `CPU-RQ4-fineweb2-frequency-validation.log`, and `CPU-RQ4-finalize-with-frequency.log` under `logs/naacl_experiments/20261012/`.
- **MeCab-ko.** `mecab-ko` 1.0.2 with `mecab-ko-dic` 1.0.0 parsed 162,137 unique eojeol strings exactly. On 1,860,584 classified strict-alignment proposals, the MeCab CROSS−WITHIN AME is −0.037 pp (95% CI −0.192 to +0.118; p=.639). This automatic second-analyzer comparison is not human validation. MeCab labels use cumulative character offsets from parser surfaces; exact endpoint matches are excluded from the contrast.
- **Human review sheet.** `rq4_human_validation_300.csv` contains 300 rows: 50 each for EXACT, WITHIN_SPLIT, and CROSS crossed with accepted/rejected outcome. For each CROSS stratum, 35 of 50 are N→P. The annotator fields are blank; human validation remains pending.
- **Proposal-side labels.** The pinned Qwen06 tokenizer projected each eligible teacher-forced draft argmax on its saved target-prefix context for a seed-random sample of 1,000 common-pool prompts. There were 105,624 eligible proposals across 35,571 blocks, with no projection-block errors; 47,111 candidates had valid CROSS/WITHIN proposal labels. The adjusted proposal-side AME is +4.82 pp (95% CI +4.11 to +5.54; 976 documents). Proposal-side and target-side classes agree on 99.34% of 37,553 valid comparable CROSS/WITHIN rows. The completed full-pool projection covers all 40,000 Q2 prompts: 4,260,935 eligible proposal events across 1,432,443 blocks, with no projection-block errors. Of these, 1,914,148 had valid proposal-side CROSS/WITHIN labels across 39,311 documents; proposal-side labels match target-side classes on 99.42% of 1,528,036 comparable rows. The adjusted full-pool proposal-side AME is +4.773 pp (95% CI +4.657 to +4.889), converged with no rank deficiency. This is a saved-prefix proposal-side refit; the 1k estimate is exploratory, and neither estimate is causal.
- Outputs: `rq4_table4_robustness.csv`, `rq4_top_type_removal.csv`, `rq4_pseudo_boundary_draws.csv`, `rq4_human_validation_300.csv`, `rq4_proposal_side_sample_rows.parquet`, `rq4_proposal_side_sample_summary.json`, `rq4_proposal_side_full40k_summary.json`, `rq4_proposal_side_full40k_rows.parquet.gz`, `fineweb2_disjoint_frequency_documents.parquet`, `fineweb2_disjoint_qwen4_token_frequency.parquet`, `fineweb2_disjoint_frequency_metadata.json`, and `rq4_metadata.json` under `results/naacl_experiments/analysis/`. The full-pool row file is archived as gzip level 1 because the raw Parquet is 109,163,759 bytes; restore it with `gzip -dc results/naacl_experiments/analysis/rq4_proposal_side_full40k_rows.parquet.gz > results/naacl_experiments/analysis/rq4_proposal_side_full40k_rows.parquet`. The archive's decompressed SHA-256 is recorded in the summary JSON. Figure 6 is `figure6_rq4_falsification.{pdf,png}`; source: `figures/gen_fig_naacl_rq4.py`; analysis, proposal projection, frequency-cache builder, and finalization sources: `scripts/analyze_naacl_rq4.py`, `scripts/analyze_naacl_rq4_proposal_side_sample.py`, `scripts/build_naacl_fineweb2_disjoint_frequency.py`, and `scripts/finalize_naacl_rq4.py`.
- As with RQ1–RQ5, these are saved-continuation replay analyses within V0's shared-context agreement scope. They do not measure live decoding latency or validate unmatched/diverged contexts.

**Queue status after RQ4 update (2026-10-12, 21:14 UTC):** The RQ1–RQ3 and RQ5 analyses and all machine-run RQ4 robustness analyses are complete. The full-pool proposal-side fit and the 100-draw pseudo-boundary analysis completed; MeCab-ko remains an automatic analyzer comparison. The 300-row human sheet is still unannotated because no annotator is available. O1 target generation is live at 3/10 shards; O2's Japanese pool is frozen and its two model runs are queued behind O1; O4 tokenizer compatibility passed and its model scoring is queued after O2. Saved-token analyses remain within the user's 95% V0 shared-prefix agreement gate and do not measure live decoding latency.

### Optional O3: FLORES-200 domain extension

- O3 used all **1,012** official FLORES-200 English `devtest` prompts with the frozen template “Translate the following sentence into Korean. English: … Korean:”. The prompt pool and config are reproducible with `scripts/prepare_naacl_flores_o3.py`; the script verifies the pinned source checksums and copies only English source text into `data/naacl_flores_1012.parquet`. Korean references remain checksum-verified provenance and were not passed to generation, scoring, alignment, or analysis.
- All five planned target models completed greedy generation, target self-TF, and Kiwi alignment: Qwen3-1.7B, 4B, and 8B; Gemma-3-12B; and Ministral-3-8B. Draft TF and k=4 replay completed for Q1–Q6, G1, and M1. Every target and draft TF log has a passing head-calibration record (`scale=1`, `max_abs_error=0`). Pair replay logs report 45,586 / 43,086 / 40,317 / 41,647 / 43,041 / 41,461 / 104,419 / 110,163 eligible proposals, respectively; observed k=4 rejection rates were 32.31%, 31.74%, 15.35%, 22.39%, 32.70%, 21.99%, 1.05%, and 12.65%.
- Adjusted global CROSS−WITHIN and nominal→particle (N→P) estimates use the same covariates and document-clustered uncertainty as the core RQ2 fit. Values are percentage points:

| Pair | Global AME (95% CI) | N→P AME (95% CI) |
| --- | ---: | ---: |
| Q1, 0.6B→1.7B | −2.01 [−3.86, −0.17] | +6.66 [+1.70, +11.62] |
| Q2, 0.6B→4B | +0.17 [−1.54, +1.88] | +6.72 [+0.35, +13.10] |
| Q3, 4B→8B | −1.20 [−2.51, +0.11] | +1.07 [−2.34, +4.48] |
| Q4, 1.7B→4B | −0.50 [−2.06, +1.07] | +2.94 [−2.17, +8.05] |
| Q5, 0.6B→8B | −0.93 [−2.77, +0.91] | +13.90 [+8.60, +19.20] |
| Q6, 1.7B→8B | −1.74 [−3.20, −0.28] | +1.94 [−2.12, +5.99] |
| G1, Gemma-4B→12B | −0.18 [−0.64, +0.29] | Not estimable: N→P lacked both outcome levels |
| M1, Ministral-3B→8B | +0.18 [−0.75, +1.10] | +4.44 [+1.62, +7.25] |

- Same-pair fine-boundary rank stability against the Korean FineWeb2 core was estimable only for M1 (7 shared supported boundaries; Spearman ρ=0.86; sign agreement 0.57). Q1–Q6 and G1 had fewer than three shared supported fine boundaries. This 1,012-prompt study therefore does not establish stable boundary-level generalization across domains. Across the pair grid, N→P estimates are positive for some settings and near zero or too uncertain for others.
- O3 is an exploratory saved-continuation replay on translated prompts. It does not validate live decoding on FLORES, measure latency, or establish a causal morphology effect. Outputs are under `results/naacl_experiments/flores_optional/`; analysis source is `scripts/analyze_naacl_flores_o3.py`; run logs are `O3-*.log` plus `CPU-RQ2-O3-analysis.log` under `logs/naacl_experiments/20261012/`.

### Optional O1: Qwen3-14B scale extension

- The public Qwen3-14B-Base checkpoint at pinned revision `0b0bd3732e2c374d483664439ea334928b65f304` finished downloading to `/dev/shm/naacl_models/Qwen--Qwen3-14B-Base/0b0bd3732e2c374d483664439ea334928b65f304`; the exact download record is `logs/naacl_experiments/20261012/O1-qwen14-download.log`.
- Target greedy generation for all 20,000 common-pool prompts started at approximately 20:21 UTC with requested batch size 48, BF16, SDPA, and 128 new tokens. Command: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True timeout --foreground --signal=TERM --kill-after=10s 21600s /venv/main/bin/python -u scripts/gen_target.py --config configs/naacl_o1_qwen14.yaml --target qwen14 --split common --batch-size 48`. Log: `logs/naacl_experiments/20261012/O1-qwen14-generation.log`. At 20:59 UTC on Oct 12, the process was live and two of ten 2,000-prompt shards were complete; this stage is not complete until all ten shards and the `COMPLETE` manifest validate.
- Planned follow-up after generation: target self-TF and Kiwi alignment, then draft TF and k=4 replay for O1_Q06, O1_Q17, and O1_Q4, evicting each pair after its run. Run the three-pair RQ2 scale analysis and record its estimates before marking O1 complete.
- Analysis implementation: `scripts/analyze_naacl_o1_qwen14.py` will report global CROSS−WITHIN and nominal→particle CROSS−nearest-WITHIN fits for each pair, with the same global covariates and document-clustered uncertainty used in the core RQ2 analysis.

**Queue status after O3 (2026-10-12, 21:16 UTC):** O3 is complete with all eight planned k=4 cells and its domain analysis. O1 common-split target generation is live with 4/10 shards complete. O2's 5,000 Japanese prompts are frozen and artifact-validated; its model runs are queued behind O1. The full-pool RQ4 proposal-side fit and remaining robustness rows are complete and pushed; human annotation remains pending. O4's Gemma tokenizer audit has passed and its one-pair analysis driver is prepared; its model run remains queued behind O2. O3 remains an offline saved-continuation replay, not a live-decoding result.

### Optional O2: Japanese control preparation

- Preparation command: `/venv/main/bin/python -u scripts/prepare_naacl_o2_japanese.py`; log: `logs/naacl_experiments/20261012/O2-japanese-prompt-prep.log`. It streamed the pinned FineWeb2 `jpn_Jpan` train revision `af9c13333eb981300149d5ca60a8e9d659b276b9`, source indices [0, 250,000), and completed with exit code 0. Of 250,000 scanned documents, 249,968 passed the frozen language/length/deduplication filters; 32 failed the 0.35 Japanese-script ratio. The seed-42 selection is frozen at 5,000 unique documents.
- Artifact validation found 5,000 unique IDs and normalized text hashes, minimum 95 characters, and minimum Japanese-script ratio 0.3651. Prompt parquet SHA-256 is `379208751a7cfcc3cef32b19f7389b0da0c3d7c303d73c4c7ba9162d88215504`; ordered ID SHA-256 is `540d04a066ba303982784186d0908c21a1308e645773bfa6f6f66c1ac8789032`. Files: `data/naacl_fineweb2_jpn_5000.parquet`, `data/naacl_fineweb2_jpn_5000_ids.txt`, `configs/naacl_o2_japanese.yaml`, and `results/naacl_experiments/provenance/o2_japanese_manifest.json`.
- `fugashi` 1.5.2 and full UniDic 3.1.0 are installed; installation log: `logs/naacl_experiments/20261012/O2-fugashi-unidic-install.log`. The config defines O2_Q06 (Qwen3 0.6B→4B) and O2_G4_12 (Gemma-3 4B→12B), with Fugashi alignment retaining nominal→topic は and nominal→case が roles. The Japanese runs and analysis remain pending until O1 releases the GPU/model storage.

### Optional O4: Gemma-3-1B tokenizer preflight

- O4 adds the pinned `google/gemma-3-1b-pt` revision `fcf18a2a879aab110ca39f8bffbccd5d49d8eb29` as a draft against the existing Gemma-3-12B common-pool continuation (`O4_G1_12`). Its weight files total 1,999,811,208 bytes (1.86 GiB). Gated-repository access was verified using the local `hf_token`; the credential itself was not logged.
- The tokenizer-only snapshot was downloaded to `/tmp/naacl_o4_gemma1_tokenizer` and compared with the pinned Gemma-3-12B tokenizer using `/venv/main/bin/python -u scripts/audit_tokenizers.py --config configs/naacl_o4_gemma1.yaml --pair O4_G1_12 --output-dir results/naacl_experiments`. Audit passed: identical serialized tokenizer and vocabulary, 262,145 tokens on both sides, matching added/special tokens, and supported offsets; all four Korean/mixed-text probes matched IDs and offsets. Report: `results/naacl_experiments/provenance/tokenizer_audits/O4_G1_12.json`; log: `logs/naacl_experiments/20261012/O4-tokenizer-audit.log`.
- The committed run config is `configs/naacl_o4_gemma1.yaml`; analysis driver: `scripts/analyze_naacl_o4_gemma1.py`. The O4 weight download, draft teacher-forced scoring on the existing Gemma-12B continuation, k=4 replay, and RQ2 analysis are still pending; they remain after O2 in the optional-run order.
