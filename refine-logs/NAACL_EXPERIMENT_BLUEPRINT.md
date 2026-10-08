# NAACL Experiment Blueprint v2 — concrete cards, model scales, data, main tables

Date: 2026-10-06. This is a proposed experiment design, not completed results. Vietnamese explanation; English section titles for the paper. No GPU jobs are launched by preparing this blueprint. `configs/paper_experiment_matrix_v2.yaml` is a design manifest, not a configuration understood by the current runners.

## 1. Paper thesis and section sketch

Primary claim C1: Direction and local environment of token–morpheme mismatch identify reproducible differences in actual draft–target greedy agreement in Korean.

Supporting claim C2: Morphological features provide incremental, held-out risk information beyond strong draft-confidence predictors; a small cost-aware scheduling prototype tests operational value. The predictive and runtime parts remain hypotheses.

The experiment section should follow the reader's questions rather than H1/H2/H3 experiment history:

```text
4 Experiments
  4.1 Experimental Protocol
      Models, tokenizer compatibility and scales
      Data, discovery/confirmation/test splits
      Actual proposal labels and morphological measurement
      Metrics and statistical inference
  4.2 Analysis I: Localizing Morphological Disagreement
      4.2.1 Aggregate alignment versus directional mismatch
      4.2.2 Boundary-specific signature across natural workloads
      4.2.3 Data scale, model scale and family replication
  4.3 Analysis II: Information Beyond Confidence and Lexical Identity
      4.3.1 Measurement validity and comparable contexts
      4.3.2 Confidence-conditioned and held-out predictive utility
      4.3.3 What linguistic content do the models disagree about?
  4.4 Operational Implication: Morphology-Aware Draft Scheduling
      Cost/headroom, simple controller, tuned baselines and actual latency
```

About 0.8 page for protocol, 1.5 for Analysis I, 1.8 for Analysis II and 0.7 for operational implication; adjust to actual venue limits. Introduction/related work/method definitions/discussion need the remaining space. Three core main tables and three or four figures; exhaustive grids belong in the appendix. There is no finite experiment package that makes a paper immune to review or guarantees Oral.

## 2. Concrete models

### Primary Qwen3-Base scale matrix

| ID | Draft model ID | Target model ID | Why run |
|---|---|---|---|
| Q1 | `Qwen/Qwen3-0.6B-Base` | `Qwen/Qwen3-1.7B-Base` | Reproduce small-target signature |
| Q2 | `Qwen/Qwen3-1.7B-Base` | `Qwen/Qwen3-4B-Base` | Canonical analysis and predictor development |
| Q3 | `Qwen/Qwen3-0.6B-Base` | `Qwen/Qwen3-4B-Base` | Same target as Q2, weaker draft; fresh pinned replacement for historical P3 |
| Q4 | `Qwen/Qwen3-0.6B-Base` | `Qwen/Qwen3-8B-Base` | Larger target with small draft; operationally promising size ratio |
| Q5 | `Qwen/Qwen3-1.7B-Base` | `Qwen/Qwen3-8B-Base` | Bridge for fixed-draft target-scale comparison |
| Q6 | `Qwen/Qwen3-4B-Base` | `Qwen/Qwen3-8B-Base` | Stronger draft at the same 8B target |

Fixed target 8B gives a draft-capacity comparison 0.6/1.7/4B. Fixed draft 0.6B gives targets 1.7/4/8B; fixed draft 1.7B gives targets 4/8B. The same prompts and canonical target-generated sequence can be shared within target checkpoint × dtype × backend. Do not pool different targets as if their linguistic contexts were identical. To hold prefixes strictly fixed across target sizes, add the same-prefix diagnostic described in E04.

The 8B-Base checkpoint exists and is pretrained: [official model card](https://huggingface.co/Qwen/Qwen3-8B-Base). No unverified `Qwen3-14B-Base` checkpoint is required.

### Cross-family and instruction robustness

| ID | Draft | Target | Role |
|---|---|---|---|
| G1 | `google/gemma-3-4b-pt` | `google/gemma-3-12b-pt` | Second family and tokenizer, text-only input |
| QI1 | `Qwen/Qwen3-1.7B` | `Qwen/Qwen3-4B` | Instruction robustness, both non-thinking |
| GI1 | `google/gemma-3-4b-it` | `google/gemma-3-12b-it` | Instruction + family robustness |

Gemma 3 uses multilingual training and accepts text-only input: [4B PT card](https://huggingface.co/google/gemma-3-4b-pt), [12B PT card](https://huggingface.co/google/gemma-3-12b-pt), [implementation documentation](https://huggingface.co/docs/transformers/model_doc/gemma3). Load its documented model/processor class; do not assume the existing Qwen runner supports it. Use no images and no artificial image tokens. Verify shared vocabulary, token IDs and special-token semantics within each pair; compatibility is planned, not established by model names.

I prefer Gemma 3 over Gemma 2 for the main external-family check because the [Gemma 2 card](https://huggingface.co/google/gemma-2-2b) describes primarily English content. Gemma 2 2B→9B can be a cheaper optional pilot but cannot replace a convincing Korean-output replication if generation drifts to English. Gemma access may require the account's existing model license entitlement.

QI1 uses each checkpoint's chat template with `enable_thinking=False`, as documented in the [Qwen3 card](https://huggingface.co/Qwen/Qwen3-4B). PT/base and IT/chat results must be separate panels. Changes in alignment or language quality are reported, not silently filtered after outcome inspection.

### Revisions

Existing small-Base revisions can be reused: 0.6B `da87bfb608c14b7cf20ba1ce41287e8de496c0cd`; 1.7B `ea980cb0a6c2ae4b936e82123acc929f1cec04c1`; 4B `906bfd4b4dc7f14ee4320094d8b41684abff8539`. For every new checkpoint resolve and save immutable commit SHA at preflight. New revision placeholders in the design manifest deliberately prevent treating it as execution-ready. Historical FP16 P3 is supplementary; do not relabel it as the newly pinned BF16 Q3.

## 3. Data design and whether 1,000 prompts are enough

The old sample has about 200 N→P CROSS observations per pair, often from fewer than 100 contributing prompts. That is the issue for fine particle categories and adjusted overlap comparisons. A large number of WITHIN_SPLIT control tokens cannot create support for rare CROSS observations.

### Proposed data inventory

| ID | Source/task | Development/training | Final evaluation | Use |
|---|---|---|---|---|
| W | `wikimedia/wikipedia`, `20231101.ko`, continuation | 4,000 source articles train, 1,000 dev | 5,000 source articles test | Main natural distribution and statistical support |
| F | Original FLORES-200 `dev`, English→Korean | Do not tune on it | All 997 rows from the archive already used by the expansion plan | Fresh confirmation + workload transfer |
| N | `csebuetnlp/xlsum`, `korean`, news continuation | Official validation 550 for diagnostics/calibration; train corpus for frequency/optional news predictor | All official test 550 | Natural news domain |
| I | `beomi/KoAlpaca-v1.1a`, Korean instruction responses | 300 prompts for development | 1,000 prompts test | Real Korean instruction generation |
| S | XL-Sum Korean summarization | Official validation 550; controller development subset | Official test 550 | IT-only generation task |

Wiki source: [dataset card](https://huggingface.co/datasets/wikimedia/wikipedia). Exclude the old 1,000 and already-declared 1,000 expansion article IDs from W before sampling. Hash-sort eligible article IDs with seed 20261006, sample 10,000 uniformly without replacement, group near-duplicates before partitioning, one prompt per source document. Pin dataset revision and save source and prompt hashes. This random fresh set replaces claims based solely on the first articles in local dataset order; preserve the old set as discovery.

FLORES `devtest` 1,012 rows have already been explored and are discovery. Complete old expansion for archival confirmation if needed, but do not call inspected rows an untouched final test. Before testing on F, freeze feature definitions, thresholds and analysis; if any F results have already informed choices, label that portion as discovery and hold out fresh W/N/I instead. Keep source pairs grouped if the same sentence appears in more than one language experiment. [Official release](https://github.com/facebookresearch/flores/blob/main/flores200/README.md).

XL-Sum Korean has 4,407 train /550 dev /550 test articles: [official repository](https://github.com/csebuetnlp/xl-sum). 550 is a valid whole external split, not grounds to pretend it contains 1,000 independent examples. If N has inadequate boundary support, add a prespecified random 2,000 train-article supplemental continuation set; report separately from official test and do not reuse articles used to train an in-domain predictor. N and S share articles: keep their dependence visible and never add them to an independent-document sample count.

KoAlpaca has about 21k Korean Q&A records and instruction/output fields: [dataset card](https://huggingface.co/datasets/beomi/KoAlpaca-v1.1a). Deduplicate normalized instructions and group identical source URLs; use only question/instruction as input, never dataset answers. Hash-sample 1,300 groups and divide 300/1,000. This is an inference workload, not a claim that model pretraining was free of these questions.

### Coverage rather than arbitrary N

Engineering support targets on W test: N→P CROSS >=500 and >=200 contributing source documents per primary pair; particle-category breakdown preferably >=100 CROSS and >=50 contributing documents before interpreting a category contrast. These are analysis support goals, not venue rules or guarantees of power. Main RD half-CI target <=5 percentage points on the large Wiki cells, evaluated with prompt clustering and common support. Use discovery clusters for a prespecified power simulation of a conservative +5 pp effect before sampling the final test; do not substitute iid-token formulas.

If a category is sparse, choose to mark it non-estimable or increase data by a preregistered support-count rule. Do not extend sample size until a favorable p-value appears. Boundary-enriched natural/synthetic data go into a diagnostic set, never the natural-prevalence or overall-runtime denominator.

## 4. Shared decoding and logging configurations

### New main setting

BF16; SDPA; one resident pair per process on a single GPU; no quantization or CPU offload in primary latency results; `eval`, `inference_mode`, `use_cache`; greedy (`do_sample=False`, `num_beams=1`); `K=4`; up to 256 new tokens; normal EOS stopping; no repetition penalty, forced minimum length or n-gram blocking. Seed 20261006 fixes data selection, not independent greedy generation. Log torch/transformers/tokenizer/Kiwi/library versions and exact GPU.

For W/N, construct a common raw-text prefix of at most 600 Unicode characters, cut at a complete whitespace boundary. Shorten at shared whitespace boundaries if any selected tokenizer exceeds 512 input tokens; retain exactly the same raw prefix for all families in that comparison. Include standard model-specific BOS as required. Base models receive raw continuation text, not chat wrappers. For F keep the existing saved English→Korean template exactly, with input cap 256 tokens (audit truncation rather than silently changing the source). I uses a Korean-only response instruction plus the checkpoint chat template; input cap 512, exclude overlength source prompts by a rule frozen before generation. S uses a Korean summary instruction and up to 2,048 source-input tokens; analyze source truncation rates.

The old 128/128 FP16 setting remains a replication/sensitivity condition; new effects are not interchangeable with old coefficients. BF16 is a proposed numerical choice, not a promise that all kernel paths have identical argmaxes.

Minimum event log: source/prompt ID and hashes; pair/revisions; round and slot; committed prefix; draft candidate token ID; target argmax; accepted/first-rejected/invalidated; proposal block; draft confidence/entropy at the actual prefix; target metrics if collected; morphology contexts and validity; output IDs; EOS and output length; runtime statistics. Store full vocab distributions only for a selected diagnostic prefix set, not for every token.

Actual outcomes include accepted-prefix proposals and first rejection. Suffix proposals invalidated by an earlier mismatch are not rejection observations. Label accepted and rejected candidates under the same projection context. Retrospective full-block measurements and streaming prefix-only features have separate columns. Use no target continuation attributes in deployment-feature tables.

### Inference statistics

Primary report: raw rates plus adjusted RD with 95% intervals; CROSS contributing source documents/unique token IDs; per-cell estimation validity. Prompt-clustered sandwich inference and 2,000 prompt-cluster bootstrap replicates (seed 20261006). Shared prompts across pairs/workloads remain grouped in pooled comparisons. Holm correction over five boundary-environment contrasts within pair/workload, with N→P designated as the confirmation contrast before test. Report interactions, convergence and overlap diagnostics; sparse-category non-estimability is a result.

## 5. Experiment cards

Cards are grouped into five blocks: measurement (E00–E01), signature/generalization (E02–E06), explanation/prediction (E07–E10), decoder robustness (E11–E12), operational prototype (E13–E16). Not every card generates new traces; most reuse a carefully logged baseline.

### E00 — Tokenizer, numerical parity and decoder validity

- **Priority/claim:** MUST; all claims.
- **Models/data:** Each new pair, 64 W +32 F +32 N development prompts, K=4. Record exact source IDs.
- **Compared paths:** Cached target AR, greedy speculative path, true batched target verifier. BF16 primary; FP16 on Q2/Q5 sensitivity. Compare tokenizer vocabulary and backend/offset probes within pair.
- **Metrics:** Exact output/stopping parity; first divergence; proposal-prefix argmax parity; top-1/top-2 margins at mismatches; cache/call counts.
- **Gate:** No silent mismatch exclusions. Resolve failures by documented backend/precision/cache corrections and rerun pilot; only a path that meets the declared output contract can support an exactness claim. A changed canonical numerical path must be applied to every competing method. If it remains nonidentical, do not claim lossless greedy speedup.
- **Artifact:** `A1_decoder_validity`; table of parity and alignment coverage by pair. A forensic FP32 replay of up to 128 near-tie prefixes is diagnostic, not an undeclared benchmark fallback.
- **Reuse/implementation:** Existing tokenizer tests help; Gemma3 and generalized GPU selection require runner support. Existing sequential decoder remains valid for observational logging but is not the final acceleration baseline.

### E01 — Human validation and exclusion sensitivity

- **Priority/claim:** MUST; linguistic validity C1.
- **Sample:** 600 blinded cases: 360 from W/F/N (60 accepted +60 rejected per workload, balance CROSS/SPLIT and oversample N→P); 120 ambiguous/context-sensitive cases; 120 Gemma cases. Avoid repeatedly sampling one document; disclose quotas and prevalence weights for population estimates.
- **Labels:** Candidate char span, CROSS/SPLIT/EXACT/ambiguous, boundary type, particle category where justified. Two Korean annotators independently; adjudication. Hide outcome, model identity and automated segmentation from annotation view.
- **Compared measures:** Full-block, strict-prefix, and streaming-local parser contexts. Existing rejected-only sheet is insufficient for outcome-dependent error assessment.
- **Metrics:** Class/environment precision-recall, inter-annotator agreement, confidence, errors by outcome/context/family, proportion excluded; sensitivity on expert/high-confidence labels.
- **Gate:** Predeclare engineering precision targets (e.g. >=90% class precision); correct systematic errors and freeze labeling before confirmation. If not met, narrow the linguistic claim.
- **Artifact:** Figure 1 examples and `A2_annotation/exclusions`.

### E02 — Primary natural-distribution signature (MAIN)

- **Claim/question:** C1; does local boundary structure recur on actual proposals?
- **Models:** Q1/Q2/Q3.
- **Data/config:** W test5,000 + F997 + N550 per pair, new main configuration; all five environments. Q2 also generates W train4,000/dev1,000 for predictor development. Keep discovery artifacts separate.
- **Comparisons:** EXACT/SPLIT/CROSS; coarse score; single-boundary CROSS vs nearest-adjacent SPLIT. Multi-boundary cases as sensitivity.
- **Models for interpretation:** (a) raw; (b) adjusted only for prefix/candidate structural features + nonlinear draft confidence/frequency; (c) richer retrospective candidate context; (d) target-side context as explicitly labeled diagnostic. Main claim should not rely only on (d). Do not use future target lengths as deployable covariates.
- **Metrics:** Raw rejection rates; RD/CIs; CROSS support/unique candidate IDs; environment interactions; prevalence-weighted error contributions. Matching/overlap in E07 determines robustness.
- **Gate:** Recurrence and adequate estimation support on frozen test; publish heterogeneous/negative environments too. No assertion that every non-N→P effect is zero.
- **Artifact:** MAIN Table 1 and Figure 2 boundary profile.

### E03 — Data-scale and precision of the conclusion

- **Priority/claim:** MUST CPU reuse; C1 stability.
- **Data:** Nested deterministic subsets of W test: 500/1,000/2,000/5,000 source docs. Do not call W train/dev an independent 10k test. A 10k test extension is optional if frozen support-based sampling requires it.
- **Compared estimates:** Same raw/adjusted/common-support estimator at each N, main Q1/Q2/Q3; 20 seeded subsamples at 1,000 and 2,000 for estimator variability, not 20 independent generation seeds.
- **Metrics:** RD, CI width, N→P CROSS count/contributing docs, particle cells estimable, overlap coverage.
- **Gate:** Estimates stabilize with growing N; broad CIs or shifting effects must remain visible. This is a precision diagnostic, not repeated hypothesis testing.
- **Artifact:** Figure 3a RD/CI versus dataset size; appendix full category support.

### E04 — Controlled model-scale grid

- **Priority/claim:** MUST for capacity discussion.
- **Models/data:** Q1–Q6 on the same prespecified 2,000 subset of W test. Q1–Q3 reuse E02; only Q4–Q6 new generation. Optional Q5 F997 for larger-target workload transfer.
- **Configuration:** Same raw prefixes, K4, BF16, max_new256; within-target reference sharing.
- **Comparison A:** Target8B fixed, draft0.6/1.7/4B. **B:** Draft0.6B fixed, target1.7/4/8B. **C:** Draft1.7B fixed, target4/8B.
- **Metrics:** Overall acceptance; N→P raw/adjusted/overlap RD; confidence-conditioned risk; draft-target distributions on same prefixes. Capacity×boundary interaction with source clustering.
- **Same-prefix diagnostic:** 1,000 committed prefixes sampled from Q5 test trajectories, not chosen by rejection; score every compatible Qwen checkpoint at identical text/token prefix. This separates realized-path changes from conditional argmax/distribution changes, but is a diagnostic outside each model's natural SD trajectory.
- **Gate:** Do not infer a capacity-scaling law from a monotonic-looking plot or parameter ratio alone; checkpoint training differences remain. Report effect size and interactions.
- **Artifact:** Figure 3b fixed-target draft scale, Figure 3c fixed-draft target scale; appendix `A3_scale_grid`.

### E05 — Family/tokenizer replication

- **Priority/claim:** MUST for broader-than-Qwen claim.
- **Pair:** G1 Gemma3-4b-pt→12b-pt, W shared2,000 + F997 + N550, pure text.
- **Pilot:** E00 +100 prompts to inspect Korean-output rate and N→P support. Do not discard English outputs without reporting them. A frozen Hangul-share>=0.8 slice is a secondary scope sensitivity, not replacement of the all-output population.
- **Metrics:** Same primary contrast and online-utility metrics; alignment/exclusion rates, unique CROSS identities, language/repetition audit. Tokens/sec is not comparable across vocabularies; also report visible chars/sec for cross-family descriptive plots.
- **Gate:** Positive recurring signal with trustworthy Korean morphology supports broader scope; failure supports a narrower Qwen/tokenizer-family finding. The family comparison changes architecture, training and tokenizer together and cannot identify a tokenizer-only cause.
- **Artifact:** G1 panel in MAIN Table 1; appendix all environments.

### E06 — Instruction state and task generalization

- **Priority/claim:** STRONG/Oral package; deployment relevance.
- **Pairs/data:** QI1 and GI1; I1,000 Korean Q&A prompts; S550 official news summarization prompts. IT models only, proper chat templates; Qwen thinking disabled; up to256 output tokens, S inputcap2,048.
- **Comparisons:** Same actual proposal morphology contrasts; versus Base findings qualitatively with separate task/context accounting. Optional same raw user content rendered to Base as a separate interface control, not a clean SFT-only intervention.
- **Metrics:** Signature, predictor transfer, Korean-output rate, output length, repetition; summary ROUGE only to characterize target workload, not to claim SD quality superiority when outputs are identical.
- **Gate:** Signature/utility that survives instruction generation increases applicability; mixed results narrow the claim. Training-stage contrasts also include template differences.
- **Artifact:** Appendix `A4_instruction`; promote a compact panel into Analysis I if it changes the thesis.

### E07 — Common support and structural explanation

- **Priority/claim:** MUST; robustness of C1.
- **Data:** E02 actual proposal rows, N→P environment; Q1/Q2/Q3 separately and same-source dependence preserved.
- **Design:** Primary overlap weighting from a propensity model for CROSS membership using measured structural, confidence, lexical-frequency and boundary-context features; require propensity in [0.05,0.95]. Weights CROSS=1-e and SPLIT=e; define the overlap-population estimand explicitly. Check weighted covariate SMD and effective sample sizes.
- **Sensitivity:** Match candidate char length +proposal slot +particle category exactly when enough support, with a 0.2 SD logit-propensity caliper. Source fixed effects/matching are additional sensitivity, not an impossible universal requirement that removes all cases.
- **Metrics/gate:** RD/CIs, max abs SMD<=0.10 goal, treated support retained, contributing docs, ESS. If balance/common support fails, state that geometry-independent inference is not established. Do not present low-coverage weighted estimates as effects on all Korean proposals. Retrospective features here are not claimed as deployable predictors.
- **Artifact:** Balance/overlap plot and `A5_overlap`; one main result sentence.

### E08 — Frequency, token identity and morphology granularity

- **Priority/claim:** MUST CPU reuse; alternative explanation.
- **Controls:** Separate token, nominal, particle and eojeol counts from a frequency corpus that excludes final W/N test documents, with exact tokenizer/normalization version. No claim of reproducing full pretraining frequencies.
- **Comparisons:** Nonlinear frequencies; leave-top-10 nominal/particle/token patterns out (rank by count, not rejection); leave-one-supported-particle-group out; held-out nominal surface and token identity prediction. Show random source split and identity-held-out split separately, not as the same task.
- **Predictive token baseline:** Smoothed token-ID rejection lookup fitted on train: (reject_count+10*train_rejection_rate)/(token_count+10), unseen-ID fallback to train global rate. Compare confidence+lookup versus confidence+lookup+morphology.
- **Caveat:** A morphology class can be nearly constant for a token identity; token fixed effects may destroy identifiability. Report this instead of forcing a morphology coefficient with token FE. Frequency and top-pattern controls cannot prove morphology-only causality.
- **Artifact:** `A6_lexical`; held-out metrics in MAIN Table 2; supported H4 particle results in appendix.

### E09 — Incremental predictive utility (MAIN)

- **Claim:** C2; information beyond confidence.
- **Training:** Q2 W4,000; grouped CV internal to train; W1,000 dev for calibration/tuning; W5,000 test. F997/N550 transfer without retuning. Train Q1/Q3 equivalents on a smaller1,000 train-source subset only if pair-specific calibration is needed; record their narrower training scope.
- **Feature sequence:** P0 draft entropy/top1 probability/top1-top2 margin/proposal slot/generation position/candidate char length/prefix-local structure/nonlinear log token count; P1=P0+coarse alignment when computable from available context; P2=P0+directional class; P3=P2+environment interaction; P4=P0+token lookup; P5=P4+direction/environment.
- **Timing of features:** Strictly streaming candidate prefix (zero future draft tokens) primary; one-token-lookahead and full-block variants secondary with explicit costs. A full-block variant is observable after drafting, but cannot claim savings of that completed draft work. Missing/ambiguous morphology gets an explicit flag and confidence-only fallback.
- **Learners:** Regularized logistic regression, standardized continuous features, C=[0.01,0.1,1,10,100], class weights none, max_iter2,000, select dev log-loss. Strong nonlinear check: HistGradientBoostingClassifier loss=log_loss, learning_rate=.05, max_iter200, max_leaf_nodes15, min_samples_leaf50, l2_regularization1, early_stopping=False; source-grouped tuning rather than row-random internal split. Every variant uses the same learner/tuning budget. Model specification from [official sklearn docs](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.HistGradientBoostingClassifier.html).
- **Metrics:** Test log-loss/Brier primary; AUPRC, AUROC secondary; 10 equal-count-bin calibration; risk–coverage at top1/5/10/20% predicted-risk flags; fraction of first-rejections and wasted suffix proposals captured. Prompt-bootstrap paired differences; report all proposals and N→P slice separately.
- **Gate:** Reliable held-out improvement beyond both confidence and token lookup; quantify practical size. Development-significant regression alone fails the gate. No threshold choice on F/N test.
- **Artifact:** MAIN Table 2 and Figure 4 predictive risk/coverage.

### E10 — Linguistic content and controlled boundary challenge

- **Priority/claim:** STRONG; explanation rather than causal proof.
- **Natural diagnostic:** 400 expert-audited contexts (100 each accepted N→P CROSS, rejected N→P CROSS, matched SPLIT and other-boundary cases). Score actual draft/target distributions at the same committed prefix, full vocab on the selected contexts only. Log target rank of draft candidate, candidate log-prob gap, entropy, top-5 alternatives; annotate nominal lexical choice/particle choice/token-surface segmentation/other.
- **Controlled set:** 200 Korean expert-validated sentence frames ×3 nominal frequency bands ×2 particle conditions =1,200 contexts, with natural subject/object/topic/adverbial constructions and consistent allomorphs. Use lexical fillers chosen independently of rejection; separate frame/filler held-out split. Tokenizer boundaries are measured, not assumed to obey desired CROSS/SPLIT quotas.
- **Systems:** Q2/Q5/G1 on fixed-prefix scoring. Optional teacher-force up to16 following reference tokens for surface-continuation diagnostics; label this as scoring, not actual greedy SD generation.
- **Metrics:** Candidate rank/agreement/distribution overlap; particle-vs-lexical error shares; effect within validated frame strata; compare confident and uncertain positions.
- **Gate:** A specific account (e.g. particle-choice disagreement) needs annotation and recurrence, not illustrative examples. Altering nominal/particle also changes token identities and linguistic likelihood; this is a controlled diagnostic, not proof that crossing alone causes rejection.
- **Artifact:** Short main qualitative panel; full challenge in `A7_content_challenge`.

### E11 — K, context, length and precision robustness

- **Priority/claim:** MUST focused grid; decoder-specificity.
- **K sweep:** Q2 and Q5, W500 +F500 dev/test-reserved robustness subset, K=[1,2,4,8,16], main maxnew256. K4 reuse if exact config matches. Stronger selection/slot controls are required as K changes realized eligible proposals.
- **Context/output one-factor sweeps:** Q5 W300 common documents; context raw prefixes selected to target approximately [128,512,2,048] Qwen tokens, maxnew256; separately maxnew=[128,256,512] at common short prefix. Use sufficiently long articles, same documents and record different prompt endings. For fixed-prefix isolation add selected same-prefix scoring from E04; do not attribute context effects solely to one factor when prompts change.
- **Precision:** Q2/Q5 128 forensic prompts BF16 vsFP16, exactness/margins/effect direction. FP32 only selected prefixes when needed. Quantized run optional appendix and always compared with the same quantized target reference.
- **Metrics:** Boundary effect/support, confidence-conditioned risk, accepted prefix, invalidation, parity. Partition 1-token/all-EOS outputs and heavy-repeat outputs via prespecified audit, report full population first.
- **Artifact:** `A8_decoder_robustness`; one main sentence if stable.

### E12 — Beyond greedy: distribution overlap and optional exact sampling

- **Priority:** Distribution diagnostic STRONG; actual stochastic SD OPTIONAL unless claiming general SD acceptance.
- **Cheap distribution test:** Q2/Q5/G1, 2,000 predeclared same-prefix positions across W/F, compute alpha_T=sum_v min(p_T(v),q_T(v)) for T=[0.7,1.0], without top-p/top-k truncation. This is the expected one-step acceptance under standard speculative sampling at those prefixes, not empirical rejection on stochastic realized paths. Use full distributions, not an uncorrected top-k truncation.
- **Optional actual sampling:** Q2, 200 W +200 F prompts, seeds[17,29,43], T=.7 and top_p1/top_k disabled initially, K4, maxnew256. Implement proposal sampling, acceptance min(1,p/q), correct residual correction and bonus sampling, and audit finite-vocab toy distributions plus theory before model runs.
- **Metrics:** Stochastic acceptance/path effects and policy correctness. Exact sampled outputs need not match target-only with the same seed; numerical equality is not the stochastic distribution-preservation test.
- **Gate:** Without implementation, title/claims explicitly refer to greedy draft–target disagreement; overlap evidence alone does not license exact stochastic runtime claims.
- **Artifact:** `A9_sampling`; move a short result into Analysis II only if it materially changes scope.

### E13 — Cost and oracle headroom before method engineering

- **Priority:** MUST before speed claim; cheap gate.
- **Pairs/data:** Q4/Q5 on 200 W +200 F development prompts; optional G1 on100 W. True batched verification and efficient draft cache required after E00.
- **Profile:** Draft decode, target prefill/block verify/correction, cache maintenance, parser/controller, unavoidable lookahead. Compute baseline prevalence and reachable triggers; distinguish high rejection risk from amount of avoidable suffix work.
- **Oracle:** Offline policy selects among K1/2/4/8/16 under measured cost and known outcomes; call it cost-aware oracle for this restricted policy class, not universal SD optimum. Trace replay is an estimate until executed policy reproduces costs/round boundaries.
- **Gate:** Stop method expansion if detector/call overhead exceeds estimated headroom; report negative feasibility. Never turn a tiny population-level opportunity into an assumed double-digit speedup.
- **Artifact:** Headroom/overhead panel accompanying MAIN Table 3.

### E14 — Minimal morphology-aware controller and fair runtime (MAIN, gated)

- **Method:** Preserve natural candidate tokens. Score risk after each available candidate; if risk crosses calibrated threshold, close the drafted prefix and invoke standard verification/correction. Do not force a non-N→P alternative. Prefix parser last20 eojeols primary; compare full committed-prefix parsing on dev to validate the local approximation.
- **Pairs:** Q4/Q5 primary; G1 or GI1 only if pilot headroom exists. Development prompts200 per workload, drawn only from development partitions; fixed final500 W +500 F prompts/pair for runtime. F runtime prompts come from the frozen test, never its tuning pool; tune F policy only via Wiki transfer or previously designated FLORES discovery, not F997 test.
- **Baselines:** target AR; fixedK=[1,2,4,8,16] with best dev choice; SVIP entropy stopping; max-probability confidence stopping; confidence+morphology risk controller. All have the same target kernels, precision and efficient cache path. [SVIP](https://aclanthology.org/2025.emnlp-main.844/) and [HF confidence adaptation](https://huggingface.co/blog/dynamic_speculation_lookahead) motivate strong comparators; call reimplementations SVIP-style if not reproducing the paper algorithm.
- **Concrete tuning:** Kcap=[8,16]; entropy thresholds from dev-calibration quantiles[.5,.7,.8,.9,.95]; probability thresholds[.3,.5,.7,.9,.95]; predicted-risk thresholds[.1,.2,.3,.4,.5]. Select one setting per policy/pair using median dev decoding latency. Cost-model shortlist top3 settings before exact dev timing if full grid is expensive; freeze all choices before final tests. A max-probability threshold implementation that auto-updates during the test is a separate policy; do not silently compare it to frozen thresholds.
- **Measurements:** 3 warmed repeats minimum; random/counterbalanced policy order, fixed workload order within repeat; CUDA synchronize around timed regions; exclude model load consistently, include detector/controller/cache work; report decode-only and full prompt-to-output latency separately; bootstrap paired source-prompt speed ratios. No artificial min_new_tokens.
- **Metrics:** tokens/sec within tokenizer; visible chars/sec across families; total latency and 95%CI; target/draft calls per output token; wasted proposals; parser overhead; exact IDs/stopping; activation rate. Do not call accepted tokens divided by sequential token checks “accepted tokens per batched verify call”.
- **Gate:** Correct outputs and measured benefit versus the strongest tuned baseline on ordinary test data. If only a boundary-enriched subset improves, report a specialized finding. Runtime negative stays negative even if selected-proposal rejection rate drops.
- **Artifact:** MAIN Table 3 with same backend and paired CIs.

### E15 — Controller specificity and simplicity

- **Priority:** MUST if E14 claims morphology benefit.
- **Data/pair:** Q4 W500 runtime subset, same3 repeats; extend only if result needs explanation.
- **Variants:** Drop morphology from same risk learner; all-boundary signal; N→P-only signal; random cut opportunities matched using development-derived activation/slot rates; strict-prefix vs1-draft-token-lookahead vsfull-block detector. Never match random policy activation using test rejection labels.
- **Metrics:** Latency/overhead/coverage, output parity. Classifier accuracy or equal activation alone does not establish causal morphology value.
- **Gate:** Morphology gain survives deleting unrelated components and is not reproduced by arbitrary shortening. If a confidence-only rule is equally good, prefer it and limit the morphology claim.
- **Artifact:** Small ablation panel in Table3 or `A10_controller`.

### E16 — Reuse the negative intervention and publish limitations

- **Priority:** MUST reuse, no additional GPU by default.
- **Existing evidence:** N→P candidate suppression worsens immediate acceptance; matching is inconclusive; batched prototype numerical parity failed; original P3 revisions unresolved.
- **Role:** Explicitly distinguish risk prediction, proposal substitution and cost-aware scheduling. Negative substitution does not refute every controller; association does not imply substitution will help.
- **Artifact:** Analysis II short paragraph; appendix counterfactual/matching/provenance tables with actual numbers and limitations.

Optional E17: cross-lingual Japanese/English controls, only after Korean thesis is stable. English has no directly equivalent N→P taxonomy, so compare conditional predictive value of general directional features, not raw Korean environment coefficients. Not required for a paper scoped to Korean. Optional E18: morphology-aware draft distillation with equal-budget random-token loss/distillation baseline; this is a larger method project, not needed to complete the two-analysis paper.

## 6. Main tables and figure templates

### MAIN Table 1 — Natural proposal signature

Caption should say actual greedy draft candidates, primary N→P CROSS-minus-SPLIT association, observed supports and prompt-clustered95% CIs. Discovery and frozen test in different tables/panels.

| Workload | Pair | N→P CROSS n / contributing docs | Raw reject CROSS / SPLIT | Adjusted RD [95% CI] | Overlap RD [95% CI] / retained support |
|---|---|---|---|---|---|
| Wiki test5k | Q1 | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE |
| Wiki test5k | Q2 | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE |
| Wiki test5k | Q3 | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE |
| FLORES dev997 frozen | Q1/Q2/Q3, separate rows | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE |
| News test550 | Q1/Q2/Q3, separate rows | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE |
| Wiki2k/F997/N550 | G1, separate rows | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE |

If too wide, omit overlap column from main and reference A5; never omit CROSS contributing-document counts. Full population/exclusion/class counts go into A0. Overall CROSS-vs-SPLIT and all five environments appear in Figure2 rather than expanding Table1 to dozens of rows.

### MAIN Table 2 — Predictive information beyond confidence

| Predictor | Wiki test log-loss ↓ | Wiki Brier ↓ | Wiki AUPRC ↑ | FLORES transfer log-loss ↓ | News transfer log-loss ↓ | Reject/wasted-work coverage at10% flags ↑ |
|---|---|---|---|---|---|---|
| P0 confidence+structure+frequency | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE |
| P1 +coarse alignment | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE |
| P2 +direction | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE |
| P3 +environment | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE |
| P4 confidence+token lookup | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE |
| P5 lookup+morphology | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE |

Report paired improvement CIs for P3-P0 and P5-P4; absolute metrics alone conceal small gains. Main table is canonical Q2; other pairs, nonlinear learner, identity-held-out splits and feature timing are appendix. Show calibration/risk–coverage and confidence-conditioned error in Figure4.

### MAIN Table 3 — Cost-aware prototype, if E13/E14 pass

| Pair/workload | Policy | Decode latency ↓ | Speed ratio vs best tuned baseline [95%CI] | Draft proposals/output ↓ | Detector time share ↓ | Exact outputs |
|---|---|---|---|---|---|---|
| Q4 or Q5 /Wiki or F, separate panels | Target AR | TO MEASURE | TO MEASURE | — | 0 | TO MEASURE |
| Same | Best fixedK | TO MEASURE | TO MEASURE | TO MEASURE | 0 | TO MEASURE |
| Same | Entropy/SVIP-style | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE |
| Same | Confidence-adaptive | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE |
| Same | Confidence+morphology | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE | TO MEASURE |

Put offline oracle estimates in a visibly separate panel, never as measured speed. If prototype fails, replace Table3 with a compact feasibility table: detector overhead, trigger prevalence, oracle headroom, actual negative pilot. An honest negative operational implication preserves an analysis paper's scope.

### Figures

- F1: Human-validated token/morpheme diagrams for candidate vs target, one accepted and one rejected example.
- F2: Environment forest plot across natural datasets/pairs; display support and unknown/sparse cells.
- F3: Dataset-size stability +fixed-target draft-scale +fixed-draft target-scale, same-source subsets.
- F4: Confidence-conditioned risk/risk–coverage; cost/headroom inset if it fits. Omit a main figure if it repeats tables without new information.

### Appendix inventory

A0 populations/language/length/repetition/exclusions; A1 numerical correctness and pinned environment; A2 annotations/context sensitivity; A3 scale grid; A4 IT/task transfer; A5 overlap/balance; A6 frequency/identity/H4; A7 content/challenge; A8 K/context/precision; A9 sampling; A10 controller ablations; A11 historical/negative/provenance audit.

## 7. Budget and launch order

Baseline evidence trajectories in the recommended large package, excluding diagnostic/timing reruns:

| Component | Planned trajectories |
|---|---:|
| Q1/Q2/Q3 × (W5,000 +F997 +N550) | 19,641 |
| Q2 extra W train/dev5,000 | 5,000 |
| Q4/Q5/Q6 ×W shared2,000 | 6,000 |
| G1 ×(W shared2,000 +F997 +N550) | 3,547 |
| QI1/GI1 ×(I1,000 +S550) | 3,100 |
| Total planned baseline trajectories | 37,288 |

At maxnew256 this is a ceiling of9,545,728 target output tokens across model trajectories; real EOS lengths vary. Target-only references are additional work but reusable by exact target/revision/prefix/backend/dtype. Labels/regressions/predictors/subsampling should reuse traces. Runtime repetitions, development grids and optional stochastic runs add substantial compute and are gated separately.

No credible total GPU-hour figure exists before pilots on the actual hardware. Estimate sum over pair/workload of `N × measured_sec_per_prompt`, add reference/scoring/parsing passes and peak-memory measurements. BF16 Gemma3 4B+12B is of order32GB nominal parameter storage before buffers/cache/vision module; a single80GB GPU is a sensible proposed environment, not a verified fit/performance guarantee. 24GB may handle small Qwen pairs/short sequences; do not introduce offload or quantization into only one method's timing.

Recommended order:

1. E00/E01 and data manifests; start annotation while CPU data preparation proceeds.
2. E02 Q2 W dev/train first; test definitions freeze; E07/E09 cheap checks before all expensive family runs.
3. E02 full frozen W/F/N; E03 CPU stability; E04 six-pair shared2k scale; E05 G1.
4. E06 IT/task and E08/E10 explanation; E11 focused robustness and optional E12.
5. E13 cost gate; E14 numerical-valid efficient baseline and controller pilot; E15 only if morphology improves latency. Reuse E16 negative results.

Lean publishable-scope package: Q2 W10k +Q1/Q3 W2k test +three pairs F997 +G1 W1k/F997; E00/E01/E07/E09 +focusedE11, and a small cost audit. It sacrifices the full scale/news/instruction breadth and must use narrower claims. The large package is a planned upper scope, not a requirement to run every Cartesian product.

## 8. What this package can and cannot support

- Repeated actual-proposal effects with validated labels and overlap support: robust linguistic association.
- Held-out incremental information: useful risk feature beyond confidence, even if runtime is nonpositive.
- Positive runtime vs tuned adaptive baselines with equality: a limited morphology-aware scheduling method in measured regimes.
- Multiple sizes/families/tasks: defined broader scope, not universal Korean behavior or a tokenizer-only causal law.
- More data alone does not repair post-outcome features, incorrect morphology, lack of overlap, model-revision ambiguity or a sequential acceleration baseline.
- All results and main cells must be filled from actual measured artifacts; every blank template here is intentionally unfilled.
