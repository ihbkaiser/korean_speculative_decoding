# Experiment Plan: hai phần phân tích và một đề xuất SD ngắn

Ngày: 2026-10-06. Ngôn ngữ báo cáo: tiếng Việt; section titles gợi ý bằng tiếng Anh. Đây là kế hoạch, không phải kết quả đã chạy.

**Problem:** Nhận diện phần linguistic structure liên quan draft–target disagreement tiếng Hàn, và kiểm tra liệu tín hiệu đó có giá trị ngoài confidence cho speculative scheduling.

**Dominant contribution:** Đo và giải thích một local compatibility signature trên actual draft proposals.

**Supporting contribution:** Kiểm tra signal đó có dùng được cho online risk prediction và một controller nhỏ hay không; runtime-positive chưa được xác nhận.

## Claim map

| Claim | Minimum evidence | Anti-claim cần loại |
|---|---|---|
| C1. Direction × local environment của mismatch mang association tái lập với greedy disagreement | Actual-proposal labels, expert audit, fresh confirmation, common-support estimates và pair/workload heterogeneity | Target-side substitution artifact; Kiwi lỗi; chỉ frequency/geometry/extrapolation hoặc một lexical slice |
| C2. Morphology cung cấp incremental online risk information, có thể hỗ trợ scheduling ở settings phù hợp | Held-out confidence-vs-confidence+morphology comparison; practical cost/headroom; controller vs tuned entropy nếu feasible | Significant regression bị diễn giải thành predictive gain; target/future leakage; giảm rejection do bỏ các proposal khó |

Không claim morphology causally causes rejection, không claim universal Korean behavior, không claim stochastic exactness từ greedy parity.

## Bố cục main paper

### 3. Experimental Protocol

Khoảng 0.7–1 trang main text: data/task/splits; pinned P1/P2 và supplementary P3; tokenizer sharing; greedy K; actual proposal outcome; invalidated suffix exclusions; annotation and alignment; clustered inference; distinction between explanatory and online features. Đưa full audit tables/formulas vào appendix.

Định nghĩa primary observation: proposal tại prefix thực tế trước khi xảy ra rejection đầu tiên hoặc chính first rejection. Feature hình thái lấy từ candidate text với context khai báo, applied identically to accepted/rejected. SD rejection ở greedy là candidate ID khác target argmax; không đồng nghĩa lỗi ngữ pháp hay task-quality error.

### 4. Analysis I: Which Token–Morpheme Mismatches Predict Disagreement?

Khoảng 1.2–1.5 trang. Narrative: F1 đơn scalar không phân biệt split/fusion → tách hướng mismatch → localize theo boundary type → hỏi cái gì tái lập và cái gì thay đổi theo workload.

#### 4.1 Aggregate alignment obscures directional differences

Đặt EXACT, WITHIN_SPLIT, CROSS cạnh nhau, fragmentation cụ thể và uncertainty. Coarse result là motivation ngắn, không một section độc lập dài. Nếu wording nói “không giải thích”, báo interval và held-out incremental metrics; nonsignificant beta không chứng minh null.

Figure 1: ví dụ Korean do annotator xác nhận, có token bars, morpheme bars, candidate draft và target. Figure phải phân biệt target token và rejected candidate.

#### 4.2 Boundary-specific effects across workloads and model pairs

Primary: actual-proposal all-environment contrasts; raw probabilities và adjusted risk differences cùng support. Dùng forest plot environment × workload × pair. Nominal→particle là recurring large contrast, không nói mọi other boundary zero. Replication gồm Wiki continuation và English→Korean generation; target output language/quality được audit riêng. Không coi FLORES reference morphology là morphology của generated output.

Table 1: workload, pair, all valid proposals, primary CROSS/SPLIT, N→P CROSS, contributing prompts, raw/adjusted differences, 95% CIs. Pooled estimates secondary, bootstrap theo source prompt để giữ dependent model-pair observations cùng cluster.

Kết section: “Where the token crosses is more informative than an aggregate disagreement score, and the boundary profile varies with workload.” Đây là thesis cần fresh confirmation, không đóng băng “only N→P”.

### 5. Analysis II: Is Morphology More Than a Proxy for Uncertainty and Token Identity?

Khoảng 1.5–2 trang. Narrative: effect lớn có thể là confidence, identity, geometry hoặc đo nhãn → kiểm tra alternative explanations → trực tiếp đo information ngoài mẫu và error coverage.

#### 5.1 Measurement stability and comparable contexts

Expert validation và strict-prefix/full-block sensitivity; future-target controls được tách riêng. Common-support matching/weighting với balance plots, giữ estimand và coverage rõ. Nếu population overlap rất thấp thì báo giới hạn identification, không tiếp tục thêm controls để tìm significance.

#### 5.2 Confidence-conditioned and lexical generalization

Train cùng predictor family với features (a) draft-only confidence/structure; (b) thêm coarse F1; (c) thêm directional class; (d) thêm environment interaction. Các variants cùng split, capacity và tuning budget. Log-loss/Brier/calibration primary cho information; AUPRC và risk–coverage cho rare-event utility. AUROC secondary. Bootstrap difference theo prompt; đánh giá transfer workload và held-out token/nominal surfaces nếu đủ support.

Mọi deployable feature phải có trước target verification và không đọc target continuation. Draft entropy/top-1 probability/top-1–top-2 margin/frequency từ prefix và candidate; parser chỉ dùng context mà policy thực sự đã generate. Nếu full-block cần K proposals để ổn định nhãn, chi phí này phải tính và không claim tiết kiệm những draft calls đã tiêu.

#### 5.3 What do the rejected candidates disagree about?

Blind audit mẫu balanced accepted/rejected × CROSS/SPLIT: nominal lexical continuation, particle choice, surface segmentation alternative, khác biệt ngoài môi trường N→P, hoặc ambiguous. Với subset đã validate, score draft/target tại cùng prefix để kiểm tra low-entropy confident disagreement, target rank của draft candidate và probability mass trên alternative classes. Đây là diagnostic của distributions/candidate content, không morphology-only causal experiment.

Không cần exhaustive H4 taxonomy trong main. H4 particle categories/geometry/frequency đặt appendix trừ khi có replicated mechanism với adequate support. Negative N→P suppression là một paragraph/table nhỏ ở đây: signal không đủ để chọn candidate thay thế tốt hơn.

Figure 2: confidence-conditioned rejection / risk–coverage + balance or held-out improvement. Table 2: nested predictors, in-domain held-out và cross-workload transfer. One short qualitative panel annotated thật, không chọn chỉ failed CROSS.

### 6. Operational Implication: Morphology-Aware Draft Scheduling

Khoảng 0.5–0.8 trang, đặt câu hỏi feasibility. Không đổi tokenizer, không blacklist N→P, không làm yếu acceptance rule.

Candidate controller: giữ natural proposals; sau khi candidate-risk cao được phát hiện, verify available prefix sớm và tránh generate thêm suffix chỉ khi online detector quyết định kịp. Quyết định từ calibrated risk và measured cost, không từ morphology flag alone. Option rẻ: entropy policy + morphology correction term; train predictor/control on development data và freeze trên test.

MVP comparisons: target-only; best validated fixed-K; tuned draft-entropy/SVIP-style adaptive baseline; cùng baseline + morphology. Random activation/slot-matched và all-boundary guard là specificity controls, không thay strong adaptive baseline. SVIP đã dùng draft entropy để điều khiển draft length: https://aclanthology.org/2025.emnlp-main.844/.

Table 3: latency/tokens per second, ratio vs best baseline, paired 95% CI, output parity, parser/controller overhead và proposals wasted. Oracle column được ghi rõ offline upper-headroom diagnostic, không deployable result. Nếu negative, đặt feasibility limitation; không quảng bá method.

## Năm experimental blocks và protocol

### B0 — Measurement and provenance audit (MUST)

- Claim: C1.
- Data: discovery Wiki/FLORES P1/P2; stratify accepted/rejected, primary classes, workload, context-sensitive và ambiguous cases. Existing 200-row sheet chỉ rejected nên bổ sung accepted và unclassifiable strata.
- Systems: exact candidate projection; strict-prefix/full-block contexts; two Korean annotators independently, blinded to rejection/model/detector labels where context permits, then adjudication.
- Metrics: class/environment precision-recall against expert labels, inter-annotator agreement, coverage, exclusion rate by outcome/workload, class-change rate by available context.
- Setup: sample size khoảng 300–500 để phủ strata, chốt trước; uncertainty intervals theo sampling design. Không cần GPU. Nếu annotator thấy marked candidate span không đúng, ghi alignment failure riêng.
- Success: precision đủ cao và không có systematic accepted/rejected mislabeling; threshold validation thực tế đặt trước (ví dụ precision >=90% là engineering gate, không venue standard). Rerun analysis trên adjudicated/high-confidence subset để kiểm tra sensitivity.
- Failure: sửa taxonomy/projection; thu hẹp hoặc bỏ linguistic claim trước benchmark lớn.
- Target: setup + Figure 1; appendix audit.

### B1 — Fresh proposal-level signature and overlap confirmation (MUST)

- Claim: C1; scope of generalization.
- Data: discovery existing runs; confirmation additional Wiki and FLORES sets đã khai báo trong expansion. Wiki/P1 đã đủ 1,000; other cells partial nên hoàn tất cần thiết. Additional consecutive Wikipedia là fresh slice, không gọi random corpus sample. Nếu muốn population claim, lấy seed-random document sample khác.
- Systems: pinned P1/P2 primary; same greedy K=4/128 configuration trước, actual candidate labels applied identically. Primary all-environment contrast; target-side counterpart diagnostic.
- Metrics: raw/adjusted rejection RD and CIs; support by CROSS prompts/identities; common-support coverage and SMD; interactions across environment/workload.
- Setup: Freeze taxonomy/model specification/confirmatory contrasts trước analysis mới. Discovery và confirmation báo riêng trước bất kỳ combined estimate. Matching không dùng outcome, prior overlap gate; nếu exact prompt/slot/length không có support, chuyển sang prespecified overlap-restricted estimand với calipers/weighting và báo coverage.
- Success: positive recurring N→P contrast on confirmation, adequate overlap for relevant interpretation, no major measurement artifact. Confidence intervals phải phản ánh limited contributing prompts; không xem p-value của 100k control rows là support cho sparse CROSS.
- Failure: không tuyên bố đã loại structure confounding; báo failed overlap/unstable profile. More observations chỉ hữu ích nếu tạo thêm comparable cases.
- Target: Table 1 and boundary forest.

### B2 — Incremental online predictive utility (MUST, highest information gain)

- Claim: C2.
- Split: by source document/prompt, giữ cùng source của các model pairs trong một partition; development/calibration và final test tách rõ. Existing explored devtest là discovery, không gọi untouched test. Fresh expansion có thể làm confirmation/test nếu freeze choices trước khi đọc labels.
- Variants: identical regularized logistic/GAM family baseline; +F1; +direction; +environment. Có baseline token identity/frequency phù hợp để kiểm tra memorization; held-out token/nominal identity test supplementary với support counts.
- Features: draft entropy/confidence/margin/slot/available context; candidate frequencies; online parser morphology. Exclude target entropy, target-generated eojeol/fragmentation/first-last positions from deployable variants. Keep explanatory oracle model separately.
- Metrics: paired change held-out log-loss/Brier/calibration; AUPRC and risk–coverage/wasted-draft coverage; entropy-conditioned RD. CIs by prompt; cross-workload transfer both directions where support allows.
- Seeds: greedy trajectories deterministic; 3 source-grouped split seeds for stability nếu budget cho phép. Khi training/sampling stochastic, 3 seeds. Bootstrap seeds không được đếm là independent inference runs.
- Success: improvement persists on final test/transfer, beyond tuned confidence baseline; define minimum practical improvement from cost model rather than arbitrary significance.
- Failure: retain descriptive C1; do not claim useful controller from adjusted regression.
- Target: Table 2 and Figure 2.

### B3 — Distribution/content diagnosis and external validity (MUST for Oral ambition; scope limited)

- Claim: explain scope of C1/C2, not causal morphology.
- Data: Korean-expert-adjudicated natural proposal subset; normal and boundary-rich slices reported separately. One additional model family whose draft and target share tokenizer internally, plus optionally a natural Korean instruction/story workload.
- Comparison: draft/target distributions at identical committed prefix; behavior across particle types and token identities; natural vs enriched observations. N→P suppression result already negative: reuse, không chạy lại chỉ để tìm positive result.
- Metrics: distribution overlap or candidate target rank; low-confidence vs confident disagreement; annotated content categories; same primary boundary RD/predictor utility on new family.
- Setup: choose model pair after tokenizer compatibility and Korean-capability pilot; no specific model recommendation without budget/context. Stop expansion if no N→P support. Target-side full-sequence entropy is diagnostic only.
- Success: recurring signature/predictive benefit beyond Qwen or clear boundary of validity; specific distribution/content account confirmed across settings.
- Failure: narrower Qwen/shared-tokenizer-family claim remains possible. Hypotheses about grammatical choice should not be asserted if audit finds chiefly lexical substitutions.
- Target: short subsection in Analysis II; appendix full H4.

### B4 — Cost-aware scheduling feasibility (CONDITIONAL MUST if proposing a method)

- Claim: operational part of C2.
- Stage 1: profile real draft, target batched verification, cache update/resynchronization, parser/controller costs. Count prevalence, reachable trigger positions, avoidable suffix calls and extra rounds. Offline oracle chooses policy under measured costs, not just perfect rejection prediction. No analytical throughput claim from saved trace counts alone.
- Stage 2: make fixed-K baseline use real block target verification and efficient cache reuse; choose stable precision/backend and define identical reference path for all methods. Investigate near-tie logits in failed batched parity before broad timing. Do not relax equality silently.
- Stage 3: incremental detector or cheap learned lookup/controller with deployment-time inputs. If detection needs full block, account for no draft-call saving and consider it a verification policy only.
- Data/splits: development thresholds/cost calibration frozen; Wiki and FLORES final timing distinct; typical and boundary-rich slices separately. K=2/4/8 development sweep; compare best-fixed-K on frozen test. At least short vs longer contexts where economically relevant.
- Variants: target-only, tuned fixed-K, tuned entropy adaptation, entropy+morphology; random/all-boundary controls on representative settings, all sharing backend/cache optimization.
- Metrics: synchronized paired latency, throughput, first-token vs generation timing scope, exact greedy IDs/stopping, accepted tokens per real verify forward, proposals wasted, target/draft calls, detector overhead, trigger frequency. Model loading excluded consistently; actual controller work included.
- Repetitions: >=3 warmed repeats, interleave/counterbalance policy order; >=200 pilot prompts/workload, final all test prompts only if headroom and correctness pass. Prompt-cluster paired bootstrap; report repeat variability.
- Success: positive latency CI vs strongest tuned baseline on ordinary held-out data, correct outputs, no artifact from weaker cache/backend or favorable trigger-only slice. Runtime gain may be small and meaningful only in certain regimes; quantify those.
- Failure: if oracle cost budget cannot exceed detector overhead, stop method engineering and report practical limitation. If runtime nonpositive, do not call reduction in rejection a speedup. Greedy output equality gives same target behavior; quality metrics are needed only for approximate outputs or changed target/task settings.
- Target: Table 3, short implication section.

## Must-run versus appendix

Main: B0/B1/B2; concise B3 for explanatory depth/external validity; B4 only when method claim is retained. Appendix: exhaustive frequency specifications, all H4 categories, sparse geometry contrasts, full alignment audits, old target-side numbers, model revisions and numerical parity details.

Nice-to-have: Japanese comparison after Korean thesis is stable; stochastic SD only if claim extends beyond greedy; larger K/context grid, multiple GPU regimes and batching after batch=1 prototype passes. Morphology-aware distillation is another research project unless scheduling economics warrants it.

## Run order and stop/go

1. M0: B0 + audit target-vs-proposal features, P3 provenance, current partial artifacts. Human time is critical; CPU audits in parallel with annotation are independent. No launch authorized by this planning document alone.
2. M1: Reuse completed Wiki/P1 expansion, complete remaining P1/P2 cells needed for B1, freeze confirmation analysis, calculate overlap. Do not merge discovery/confirmation first.
3. M2: B2 on existing/fresh valid rows using online feature definitions. If no incremental signal, reduce emphasis on method before spending on more GPU benchmark.
4. M3: Focused B3: identical-prefix scores on adjudicated subset, then new-family pilot/full replication if support exists.
5. M4: B4 cost profile and oracle gate; repair standard batched baseline; only then controller pilot and final timings.

## Compute/data budget

CPU: B0 audits, B1 projection/matching/regression, B2 predictors; may require tens of minutes to hours depending parser/caching, no measured estimate is available for this machine.

Human: two Korean annotators for 300–500 cases plus adjudication, validate environment as well as class.

Existing expansion card estimated ~20 GPU hours for all 1,000 Wiki +997 FLORES prompts per pair including its parity-pilot accounting on RTX A4000; this is historical planning evidence, not a current quote. Wiki/P1 completed in reported 15,951.8 seconds (~4.43h) for that run. Remaining cells need fresh throughput estimate on the same hardware/settings; compute collection and independent reference audit separately. New-family and method timing budgets cannot be reliably quantified before pilot.

Scaling more GPU generation is lower priority than measuring overlap and predictor gain. If limited budget: B0 → use completed Wiki/P1 + existing P1/P2 for B1/B2 → finalize confirmation in remaining cells → oracle profile → targeted external family.

## Checklist

- [ ] Main numbers are actual proposal measurements with declared label context.
- [ ] Expert validation covers accepted and rejected, plus ambiguity/exclusion.
- [ ] Discovery and fresh confirmation are reported separately.
- [ ] Common support and contributing CROSS prompts are visible.
- [ ] Online utility excludes target/future information.
- [ ] Baselines include tuned uncertainty policy, not just matched guards.
- [ ] Negative suppression and partial/failed experiments are represented honestly.
- [ ] Method claim is conditional on synchronized latency and correct batched decoding.
- [ ] Generality wording matches tested families/workloads and greedy scope.
