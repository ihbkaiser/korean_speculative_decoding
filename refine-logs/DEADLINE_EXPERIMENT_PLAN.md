# Kế hoạch đóng thực nghiệm trước 23:59 thứ Bảy 10/10/2026

> **Scope reset theo RQ (07/10/2026):** kế hoạch full-grid dưới đây không còn là queue để submit. Dùng [`RQ_FIRST_DIRECT_EXPERIMENTS_20261007.md`](RQ_FIRST_DIRECT_EXPERIMENTS_20261007.md) làm scope hiện hành: 4.2.1 đã xong; phân tích RQ2 tái sử dụng artifact hiện có; chỉ còn một random Wikipedia confirmation nhỏ cho RQ3 nếu cần bảo vệ claim tổng quát. Không chạy scale/family/task grid hoặc MADS runtime nếu paper chưa thêm RQ tương ứng.

Ngày lập: 07/10/2026. Mọi mốc dùng Asia/Bangkok, UTC+7. Phần cứng được người dùng xác nhận: RTX3090 và hai B200 cùng host, dùng dài hạn. Lịch làm việc chính bắt đầu 09:00 ngày07/10; thời gian trước09:00 dành cho chuẩn bị/download nếu có người vận hành. Không có SSH/host được truy cập, job được submit hay kết quả mới được tạo trong phiên lập kế hoạch này.

Đọc cùng `NAACL_EXPERIMENT_BLUEPRINT.md` (E00–E18) và `HARDWARE_EXECUTION_PLAN.md` (phân việc v3, L01–L06). v4 là kế hoạch có hạn chót và đặc tả phương pháp; không biến manifest thành CLI mà runner hiện đã hỗ trợ.

## 1. Mục tiêu và định nghĩa “làm hết”

Đến23:59 ngày10/10 có một bộ thực nghiệm đã đóng, đủ viết hai phần phân tích và một phần đề xuất phương pháp ngắn. Không bảo đảm kết quả dương, NAACL acceptance hay Oral.

- C1: đo morphology-associated disagreement trên ACTUAL draft candidates, kiểm tra scale/workload/family và validity; không suy diễn nhân quả từ association.
- C2: kiểm tra incremental online risk information ngoài confidence/frequency/identity; đánh giá một controller draft-length nhỏ bằng latency thực, kể cả kết quả âm.
- Deliverables: T1 signature/generalization, T2 held-out prediction, T3 runtime/cost/parity; figures data/model scale, risk coverage và overhead; appendix validity/overlap/identity/robustness/negative evidence; provenance và danh sách thiếu/skipped.

Mục tiêu đầy đủ: toàn bộ68,011 analysis SD trajectories được liệt kê trong v3, gồm các optional large pairs. Bộ ưu tiên47,929 trajectories phải đứng trước các card phụ. E11/E12/E10 và controller có ngân sách riêng bên dưới. Exact stochastic sampling, cross-lingual E17 và draft distillation E18 là nhánh mở rộng nghiên cứu đã được đánh dấu optional từ v2; không thuộc bộ đóng bốn ngày. Phương pháp trong lịch này là controller greedy SD, không phải training một drafter mới.

Hai B200 không tự bảo đảm lịch khả thi. Chỉ chốt scope đầy đủ sau pilot thời gian và memory ngày07/10. Đặt full grid là mục tiêu, kèm tiêu chí cắt phần phụ được ghi trước và giữ nguyên main/method evidence.

## 2. Nhân lực và đường phụ thuộc

Người dùng đã xác nhận: một người triển khai, chưa có annotator tiếng Hàn. Lịch dành cho người đó cùng trợ lý code, quản lý ba hàng đợi tự chạy qua đêm. Một người không làm đồng thời hai task code độc lập. Hạ tầng tải models/corpus và inference đã qua gate chạy nền; analysis CPU chạy tự động trên shard đã hoàn tất.

E01/E10 human validation không thể hoàn thành theo thiết kế600 items x2 Korean annotators với nhân lực hiện có. Thay phần triển khai trong sprint bằng audit tự động600 candidates có accepted/rejected, ambiguous spans và disagreement giữa full-context/strict-prefix/local20; kiểm tra exact offsets, coverage và sensitivity của kết quả theo labeler/context. Đây là agreement/robustness của pipeline, không phải gold accuracy. Human annotation giữ status DEFERRED_NO_ANNOTATOR, không thay bằng LLM labels rồi gọi là gold.400 natural examples chỉ có thể phục vụ error summaries/illustrations thăm dò;1,200 controlled frames có thể được tạo/chấm như một exploratory appendix, nhưng chưa có linguistic validation nên không dùng để kết luận particle-choice mechanism/causality. Claim chính phải ghi rõ KiWi-defined boundaries và giới hạn measurement. Nếu annotator xuất hiện sau này, dùng rubric/sampling đã lưu và bổ sung study riêng.

Đường phụ thuộc: data/revision freeze -> generic runner + actual proposal schema -> cache/batched verification correctness -> Q2 train/dev -> online feature audit + predictor -> dev cost/tuning -> freeze controller -> final runtime -> ablations/CI -> paper artifacts. Family/large-model replication chạy song song sau adapter/pair preflight.

Phần phương pháp bắt đầu từ ngày07/10, không đợi chạy xong mọi mô hình lớn.

## 3. Lịch theo ngày/giờ

| Mốc | Người triển khai và workstation3090 | B200-A | B200-B | Artifact / điều kiện kết thúc |
|---|---|---|---|---|
| 07/10 00:00–09:00, nếu vận hành được | Chuẩn bị env, dữ liệu/model snapshots, kiểm tra quyền Gemma | Prefetch/pin models, chưa chạy confirmation | Prefetch/pin family/large models | Cache và manifest sơ bộ; không coi đây là giờ code được bảo đảm |
| 07/10 09:00–12:00 | Freeze source documents/splits/exclusions; thống nhất candidate schema; generalized device/dtype runner | Kiểm tra UUID/VRAM/driver/backend; loader Qwen | Kiểm tra loader/tokenizer/Gemma/MoE/large; download nền | D00 frozen manifest; H00 hardware/revisions; annotate pilot sheet |
| 07/10 12:00–16:00 | Sửa draft cache; target block verification; stop callback trong draft loop; parity/toy rejection/EOS/cache tests | Pilot Q2/Q5 và same-prefix target reference | Adapter preflight G1/L01/L02 và memory72B; smoke test tùy model sẵn | Engine v1; không scale decoder còn lỗi |
| 07/10 16:00–18:00 | Đo throughput/storage; freeze engine/data/schema; tạo automatic-audit sheet và rubric để annotation sau này | H01 timed pilot; xác nhận Q2train/dev | H01 timed pilot các target lớn; chốt queue budget | GateG0 18:00: scope có ETA, engine hợp lệ |
| 07/10 18:00–08/10 09:00 | CPU projector/frequency streaming và automatic-audit; không cần ngồi canh cả đêm | Q2 Wtrain4k +Wdev1k trước; sau đó Q2test/F/N | G1 W2k/F/N trước; L01a/L02a sau | Q2 training/dev shards; family replication đầu tiên |
| 08/10 09:00–12:00 | Train P0/P1 predictors; leak tests; token-identity baseline; validation online last20 versus full-prefix | Tiếp tục Q2/Q1/Q3 | L01a/L02a; gom model theo target nếu phù hợp | GateG1 12:00: predictor/feature path dùng được; không cần effect dương để tiếp tục evaluation |
| 08/10 12:00–20:00 | E03/E07/E08; candidate label audit; triển khai morphology controller và strong baselines | Q1/Q3, rồi Q4/Q5/Q6; reserved slots cho E13 | L01b và IT nhỏ; reserved dev-timing slot, GPU-A idle khi cần | Predictor và các policy; preliminary cost/parity |
| 08/10 20:00–09/10 09:00 | Sensitivity audit; CPU regressions/bootstrap; tự động tạo table skeleton | Hoàn thiện core; Qwen2.5 target7/14B nếu queue cho phép | Dense32/72B, Gemma12->27B, IT32B và7->32B theo ưu tiên | Core corpus gần hoàn tất; large extras tiến triển |
| 09/10 09:00–12:00 | Review dev cost/risk; khóa threshold/cap/models; chốt sampling runtime; kiểm tra exploratory frames | E10/E11/E12 focused slots +remaining corpus | Dev timing/controller IT32 nếu đủ điều kiện +remaining corpus | GateG2 12:00: predictor/policy/config SHA frozen; technical parity pass |
| 09/10 12:00–16:00 | Hoàn thành AnalysisII và support checks; bảngT1/T2 nháp | Hoàn tất corpus/diagnostics đến cutoff | Hoàn tất large corpus đến cutoff | GateG3 16:00: không khởi chạy thêm card analysis dài |
| 09/10 16:00–18:00 | Kiểm tra merged shard coverage; benchmark warmup/dry run; sửa plumbing thôi | Dừng heavy company-host jobs; hoàn tất write flush | Load/warm runtime pairs; validate test references | 18:00 final runtime config + test manifest frozen |
| 09/10 18:00–10/10 09:00 | Có thể chạy3090 L05 riêng nếu config/hardware audit pass; không chạy CPU-heavy trong timing3090 | Idle trong benchmark host, hoặc chỉ can thiệp khi kiểm tra isolation cho phép | Final runtime Q4 rồiQ5; six policies; three repeats | T3 raw timing/parity; checkpoint mỗi workload/policy block |
| 10/10 09:00–15:00 | Paired CIs từ blocks hoàn tất; E16 negative limitations; draft narrative | Idle trong benchmark host | Hoàn tất primary runtime; E15 ablations trước optional IT32/L05-B200 | Primary method comparison +specificity/overhead |
| 10/10 15:00–18:00 | Kiểm tra parity, CI, coverage, completed/skipped; không tune theo test | Idle/rescue correctness run trong slot riêng nếu cần | Kết thúc remaining registered benchmarks; không thêm policy/model | GateG4 18:00: raw results freeze, cả kết quả âm |
| 10/10 18:00–21:00 | Merge/report/render tablesfigures, toàn bộ Appendix ledger; kiểm tra số liệu/claim | Không cần GPU; chỉ tính lại artifact deterministic nếu thiếu | Không thêm thí nghiệm mới | T1/T2/T3 +figures +reproducibility bundle |
| 10/10 21:00–23:59 | Audit claim/evidence; viết Experiments/Method sketch; compile khi tích hợp vào paper | Reserved cứu lỗi nhỏ, không full grid mới | Reserved cứu lỗi nhỏ | Bộ thực nghiệm và bản viết đóng trước23:59 |

Corpus window danh nghĩa18:00 ngày07/10 đến16:00 ngày09/10=46h/GPU. Dành ít nhất6h/GPU trong window cho pilot thêm, cost/tuning, scoring và robustness; không coi cả46h là main generation. Final runtime window18:00 ngày09/10 đến18:00 ngày10/10=24h, ưu tiên chạy một B200 với GPU còn lại idle. Slot dự kiến không phải thời gian chạy đã đo.

## 4. Queue phân công và counts

### Core47,929 trajectories — chạy trước

| Queue | Owner | Models | Cells / SD trajectories |
|---|---|---|---:|
| C01 | B200-A | Q2 Qwen3-1.7B-Base ->4B-Base | Wtrain4k/Wdev1k/Wtest5k/F997/N550 =11,547 |
| C02 | B200-A | Q1 .6B-Base ->1.7B-Base | Wtest5k/F997/N550 =6,547 |
| C03 | B200-A | Q3 .6B-Base ->4B-Base | Same =6,547 |
| C04/C05/C06 | B200-A | Q4 .6->8B; Q5 1.7->8B; Q6 4->8B, all Base | Wtest2k each =6,000 |
| C07 | B200-B | G1 Gemma3-4b-pt ->12b-pt | Wtest2k/F997/N550 =3,547 |
| C08/C09 | B200-B | Qwen3-1.7B ->4B; Gemma3-4b-it ->12b-it | I1k/S550 each =3,100 |
| C10/C11 | B200-B | Qwen3-Base1.7/4B ->30B-A3B-Base | Wtest2k/F997/N550 each =7,094 |
| C12 | B200-B | Gemma3-4b-pt ->27b-pt | Wtest2k/F997/N550 =3,547 |

### Full scope adds20,082 trajectories — scheduled, capacity-gated

| Queue | Owner v4 | Models | Cells / SD trajectories |
|---|---|---|---:|
| X01/X02 | B200-A after core | Qwen2.5-1.5B ->7B/14B, Base | Wtest2k/F997 each =5,994 |
| X03/X04 | B200-B after required large/family | Qwen2.5-1.5B ->32B/72B, Base | Same each =5,994 |
| X05 | B200-B | Gemma3-12b-pt ->27b-pt | Wtest2k/F997/N550 =3,547 |
| X06 | B200-B | Qwen2.5-7B ->32B, Base | Wtest2k/F997 =2,997 |
| X07 | B200-B | Qwen3-4B ->32B, nonthinking IT | I1k/S550 =1,550 |

Full SD corpus A=36,635; B=31,376; total68,011. v4 moves the two smaller Qwen2.5 anchors toA to relieve the slower large-target queue. Do not move unfinished shards silently between hardware/backends; one owner per shard and configuration hash.

References are a separate GPU cost. With exact overlap/prompt contracts, the full listed corpus has approximately47,373 unique target-reference trajectories: core33,835; four dense targets11,988; IT32 adds1,550. These are planning counts for fully matched reusable references, not evidence of cache compatibility. Run target-only AR once per unique target/revision/dtype/backend/prompt set; compare EVERY final output to its reference. Coverage changes or numerical revalidation can require more work. Full corpus plus references is already115,384 generation paths before diagnostics or timing.

Keep main BF16/SDPA/batch1/K4/output256/EOS protocol from v2. Pin all new revisions. No softening of EOS or shorter outputs to create apparent speedups. Base/PT versus IT results are separate. For newly adapted pairs, do not assume the tokenizer or cache types are compatible.

### Focused diagnostic work, mainly reuse

- E03 nested W500/1k/2k/5k: CPU reuse only.
- E04 shared-prefix:1,000 prefixes across Q1–Q6; extend corresponding L01/L02/L03 cells as scheduled. Count scoring positions separately from SD trajectories.
- E07 overlap weighting/strict matching; E08 train-only frequency/identity and leave-out token/nominal identities: CPU reuse; report balance/ESS/failures, not just coefficients.
- E09 six feature groups from v2, logistic and HistGB; use eligible candidates only. CPU feature processing streams parquet shards, no single giant dataframe assumed to fit workstation RAM. Start4 workers if available and profile RAM; reduce workers on memory pressure.
- E10:400 natural examples và1,200 exploratory frames scored for Q2/Q5/G1 =3,600 frame/pair observations nếu frame schema kiểm tra được. Không có annotator nên natural-case linguistic error categories và frame grammaticality chưa được human-validated; mark exploratory và không dựa vào chúng để kết luận cơ chế. One-step distributions/top alternatives suffice; không decode256tokens cho mỗi frame. Human-validated E10 remains DEFERRED_NO_ANNOTATOR.
- E11 K sweep:Q2/Q5, W500/F500, K1/2/4/8/16; reuse K4 only if exact engine/config matches, leaving8,000 new SD paths. Context/output one-factor grid on Q5 W300 has5 unique context-length combinations; about1,500 paths before reuse. Precision128 prompts xQ2/Q5 xBF16/FP16: reuse matching BF16 references/traces;256 new FP16 SD paths plus their references. Total additional SD work up to9,756 if no context reuse. References/parity still have extra cost.
- E12 distribution overlap:Q2/Q5/G1,2,000 registered positions, T.7/1; about6,000 prefix/pair evaluations, both temperatures from the same logits. Chunk/stream vocab probabilities. Exact stochastic SD implementation stays outside this sprint.
- E16 existing suppression/matching/provenance/failed parity artifacts: reuse and disclose limitations, no unnecessary GPU reruns.
- L06 support expansion20k additional SD paths is capacity-gated and must be registered from DEV precision/support before final test effects are inspected. No promise that it fits by10/10; if not, report CI/support limitations. Batch reference/scoring parity pilots1/4/8/16 belong in07/10 preflight, not realized SD batch changes.

## 5. Đề xuất phương pháp: Morphology-Aware Draft Stopping (MADS), tên tạm

Đây là một đề xuất cần kiểm chứng, không khẳng định novelty hoặc speedup đã đạt. Dynamic confidence stopping và learned lookahead đã có trước: [HF dynamic speculation](https://huggingface.co/blog/dynamic_speculation_lookahead), [DISCO/dynamic lookahead](https://arxiv.org/abs/2405.04304). SVIP dùng draft entropy để điều chỉnh độ dài: [EMNLP2025 paper](https://aclanthology.org/2025.emnlp-main.844/). Contribution cần kiểm tra là thông tin hình thái tiếng Hàn bổ sung và giá trị sau khi tính chi phí, không phải adaptive K tự nó.

### 5.1 Predictor và feature contract

Training:Q2 Wtrain4k; Wdev1k chia theo document thành Dmodel500 và Dpolicy500 bằng split manifest trước chạy. Chọn model/hyperparameters trênDmodel; tune scheduling trênDpolicy, dùng200 documents cố định cho full dev timing. Có thể dùng200 FLORES DISCOVERY examples đã đánh dấu để tune translation; nếu không có development pool hợp lệ thì transfer Wiki thresholds, không tune F997 test. Idev300 và XL-Sum validation dùng choIT32 tuning riêng; không lấy test feedback để đổi predictor.

- P0 confidence learner: draft log top1 probability, top1–top2 margin, entropy, current slot/generation position, online token length/whitespace markers, previous-round accepted-length information nếu trace hỗ trợ. Train-only frequency/token lookup được phân tích như strong controls; cùng information/tuning budget giữa predictor so sánh.
- P1 morphology learner: P0 + online CROSS/SPLIT/ALIGNED/UNKNOWN indicators, internal boundary count, N->P và predicate->ending flags trên accepted history + CURRENT candidate. Không target entropy, target-side fragmentation, completed eojeol future length hay future block context.
- Logistic C=[.01,.1,1,10,100], max_iter2,000, không class_weight; dùng sparse features khi cần. HistGB của v2 dùng cho AnalysisII, không mở một cuộc search mô hình lớn cho production. Chính sách mặc định dùng logistic nhanh; chỉ thay đổi learner trướcG2 theo development evidence và latency.
- Missing/uncertain spans: ghi detector_status/coverage và dùngP0 fallback; không ép nhãn N->P. Kiểm tra fallback/calibration theo workload.
- Train/evaluate labels: actual eligible proposal disagreement. Các proposal sau rejection đầu block là INVALIDATED, không phải rejection labels. P1 được chấm cả trên same held-out population lẫn registered identity-held-out controls.
- Predictive metrics: log loss/Brier primary; AUPRC/risk coverage secondary; paired bootstrap theo document. Report zero-shot transfer sang Q5/L01a/L02a/IT32 theo đúng tên, không lén refit bằng test labels.

### 5.2 Online stopping rule

Sau khi candidatej đã được sinh nhưng TRƯỚC forward draft cho candidate tiếp theo:

1. Tính confidence features. Detector xử lý accepted text +proposal prefix1..j; context local20 eojeols chính, so với full accepted-prefix trên development để kiểm tra approximation. Decode/token spans/incomplete UTF-8 phải được xử lý và ghi coverage.
2. Predictor cho r_j là estimated rejection risk nếu candidatej đến vị trí được target kiểm tra. Đặt S_j=product_i<=j(1-r_i) làm cumulative-survival proxy. Đây là proxy cần calibration/check thực nghiệm, không giả định r_i độc lập hoặc oracle acceptance.
3. Nếu S_j <=tau, EOS, đạtKcap hoặc max output còn lại: đóng block ởj; nếu không tiếp tục draft.
4. Target verify toàn bộ block đã thực sự sinh bằng một causal block forward. Accept prefix đến mismatch đầu; correction/bonus vẫn target argmax. Rollback/resync cả target và draft KV đúng trạng thái. Không thay candidate bằng token “an toàn”.

Rule giữ current candidate trong block: nó chỉ tiết kiệm các candidate CHƯA sinh. Không claim tiết kiệm chi phí current candidate. Placement callback trong vòng draft là bắt buộc; `guard.inspect` sau khi đã sinh hếtK hiện tại không đáp ứng rule này.

Tuning: Kcap=[8,16], tau=[.1,.2,.4,.6,.8]. P0 vàP1 dùng cùng rule/cap/grid, để morphology là khác biệt được kiểm tra. Chi phí draft/verify/parser đo trênDEV; replay chỉ shortlist top3 configurations, sau đó chọn bằng actual dev wall-clock. Không coi replay gain hoặc survival proxy là latency thật.

Bản4ngày không cần learned neural controller, target fine-tuning hay speculative sampling. Exactness claim chỉ là greedy contract với parity đã kiểm tra trên hardware/backend đã khai báo.

### 5.3 Triển khai bắt buộc

- Generic device UUID/dtype/model registry; hiệnE2 gắn GPU7/A100/FP16, guard gắn GPU3/A100, expansion gắnA4000.
- Draft cache giữ qua rounds; không re-prefill toàn bộ history mỗi round. Target batched verifier/crop phải xử lý accepted/rejected/EOS/end-length và cache position. Existing batched branch có stored parity failure, không được mặc định dùng cho final benchmark.
- Qwen runtime engine trước; Gemma adapters có cache khác cần kiểm tra riêng. Có thể đối chiếu HF assisted generation reference và học cách tái sử dụng verified engine, nhưng custom hook/interface phải pin version và test; không hứa private HF API đã ổn định.
- Main analysis logs actual candidate IDs/confidence/outcome rồi morphology offline. Runtime detector nằm trên critical path, toàn bộ decode/CPU projection/predictor/sync được tính. Offline annotation không được thay detector runtime trong bảngspeedup.
- Engine semantic implementation chung cho mọi runtime policy. Nếu một pair không thể bảo đảm declared parity đếnG2, dừng speed claim cho pair đó; report failed validity. Không loại riêng prompt gây divergence hay chạy lại đến có PASS mà che failure.

## 6. Baselines, benchmark budget và ablations

Six primary policies:

| ID | Policy | Development choices |
|---|---|---|
| M0 | Target-only AR | Same target/dtype/backend/EOS; no speculative overhead |
| M1 | Best fixedK SD | K1/2/4/8/16, explicitly disable implicit confidence stopping |
| M2 | Max-probability dynamic stopping | Kcap8/16, probability thresholds.3/.5/.7/.9/.95; freeze updates |
| M3 | SVIP entropy policy | Official equation/procedure if reproduced; label SVIP-style if only entropy threshold. Dev quantiles.5/.7/.8/.9/.95, cap8/16 |
| M4 | Learned confidence-only stopping | P0, survival rule above, same budget asM5 |
| M5 | Morphology-aware stopping | P1, same rule/cap/budget; includes real detector cost |

Baselines M2 vàM4 đều được đo: chỉ thắng một threshold yếu không đủ. Không gọi generic entropy threshold là officialSVIP. Phần entropy/adaptive computation cost chỉ được thêm vào policy có dùng nó; AR/fixedK không bị gánh logging artificial để controller nhìn có lợi.

E13 dev tuning: fixed5 +AR +shortlist3 cho mỗi adaptive family(M2–M5)=18 configs/pair. Stage1 actual timing trên50W +50F DEV (hoặc100W nếu không cóFdev):18x100x2pairs=3,600 runs. Validate one winner/policy trên200W+200F DEV:6x400x2=4,800 runs. Planning total8,400 runs, repeats/warmup bổ sung nếu noise cao. Không tự động mở grid lớn hơn. Cần slots benchmark kiểm soát tải host; có thể reserve trong ngày08/10 và09/10, tạm dừng main queue.

E14 primary test: Q4 vàQ5, mỗi pair W500 +F500 cố định, six policies x3 counterbalanced warm repeats. Gross36,000 runs; share AR ONLY if prompt/target/config/hardware/repeat contract thật sự identical ->33,000 unique timed runs. TargetAR estimates cùng repeat schedule và môi trường; không chia sẻ một run tình cờ bị ảnh hưởng tải khác. Nếu cần repeatAR theo mỗi block, giữ36,000 budget.

E15: Q4 W500,3repeats. Drop morphology chính làM4 có thể reuse. Five additional registered variants: P1 không environment flags (general boundary only), N->P-only morphology, random stopping matched DEV slot/activation,1-candidate-lookahead morphology, full-block morphology guard. 5x500x3=7,500 timed runs. Lookahead/full-block phải thực sự sinh và tính chi phí extra draft/context, không oracle annotations. Nếu thiếu thời gian, first-tier ablations general-boundary/N->P/random trước, lookahead/fullblock sau; không claim phần chưa chạy.

Optional L04 runtime: IT4->32B, I500/S300,6policies x3=14,400 runs. Tune onIT development. Optional L05:Q3 W500, AR/bestfixed/strongest nonmorph/morph x3 on EACH machine =6,000B200 +6,0003090 runs. Select strongest nonmorph onDEV before testing. Same-prefix device audit first. Compare each device's own AR/baseline; system-level comparison includes different CPU, not a GPU-only causal conclusion.

Primary runtime+all5ablations=40,500 unique B200 timed runs (43,500 if no ARreuse). Full method extras=60,900B200 timed runs +6,0003090 runs (assuming ARreuse). This is separate from68,011 analysis SD trajectories,47,373 target references,~9,756 robustness SD paths, scoring and8,400 dev-timing runs. Do not sum “prompts” as if it were the full compute budget.

Measurements: include prompt prefill, draft/target/cache/controller/parser; exclude model loading consistently. Also show decode-only breakdown. CUDA sync at prompt timing boundaries; do not synchronize every token solely for trace metrics. Disable heavy disk writes/parsing of diagnostic labels within timed regions, retaining lightweight counters; M5's real online parser cost stays in region. Three repeats are timing repeats, not3 independent documents. Aggregate per-source prompt across repeats and paired cluster bootstrap(2,000resamples; raise after completion if inexpensive). Primary speedup ratio=sum latency_AR /sum latency_policy on the SAME outputs; compare M5 with strongest tuned nonmorph. CI matters; a tiny positive point estimate with CI across1 is inconclusive. Output/EOS/stop divergence counts are always shown.

Không yêu cầu kết quả dương để được hoàn thành đề xuất: chạy và report M5 even if negative. Nếu dev costs không có headroom, vẫn làm registered negative feasibility comparison nếu correctness pass, nhưng không mở L04/L05/expensive tuning. Do not silently select positive boundary-enriched prompts for the main speed table.

## 7. Capacity gate và kế hoạch dự phòng có định lượng

H01: mỗi Base pair64Wdev/32Fdev/32Ndev parity; IT pair64Idev/32Sdev. Time8warmup+32dev prompts perworkload, record weighted seconds/output-token/prompt, EOS distribution, VRAM, storage. Measure corpus mode(with logging) separately from benchmark mode. Statistical pilot/annotation selection is separate from final test.

For each queue:

```text
ETA = sum_cells(N_SD * t_SD + N_unique_reference * t_AR)
    + scoring + diagnostics + load/download + projected failed-job recovery
```

Use workload-stratified pilot distributions/p95 plus a20% execution margin; record actual running ETA at every completed200-prompt shard. Model total/active parameters are not timing estimates.

- Main corpus budget: about40h usable/GPU in the46h window BEFORE counting all added diagnostics. For SD-only full allocation this would require mean <=3.93s(A) and <=4.59s(B). These are optimistic upper ceilings: references/overhead/margin make permissible SD means lower. Gate on the full ETA, not these scalar bounds.
- Final method slot: reserve2h of24h for loads/warmups/audit/rescue, leaving22h timed work. Core40,500runs requires weighted mean about<=1.96s/run; full60,900B200 runs requires<=1.30s/run. They are feasibility thresholds, NOT measured performance or promises. If core is slower, reduce optional scopes before sacrificing primary comparisons.
- Never use3090 timing to predictB200 latency or multiply theoretical hardwareFLOPS ratios.

AtG0 choose/register capacity tier BEFORE final test effects:

| Tier | Registered analysis scope | Method scope | Capacity action |
|---|---|---|---|
| F FULL |68,011 SD paths +references+focused diagnostics | Two8B pairs,sixpolicies,all5ablations; IT32/hardware if timing ETA fits | Target of this plan; all large rows run regardless of sign |
| C CORE |47,929 SD paths +references+focused diagnostics | Two8B pairs,sixpolicies; three specificity ablations first | Skip v3 optional dense/extra large/IT32 runtime; registered missing ledger |
| R RESCUE |29,282 SD paths:Q2 giữ toàn bộ train/dev/test/F/N;Q1/Q3 W2k/F997/N550;Q4–Q6 W1k each;G1/L01a/L02a W1k/F997/N550 each | Q4/Q5 cùngW200/F200,sixpolicies/3repeats;three specificity variants onQ4 W200 | Explicitly a reduced study; not “all full cards completed” |

Rescue reference count if reusable is23,735, separate from29,282SD. Rescue runtime gross14,400minus1,200sharedAR=13,200; three additional general-boundary/N->P/random ablations xW200x3=1,800, total15,000 timed runs. With22h usable slot this permits a weighted mean up to5.28s/run before additional safety margin. Use fixed nested prompt lists registered atG0, not a subset selected from helpful test cases. R omitsIT robustness andL01b/additional large grids, so resulting claims/scope must be narrowed.

R is a predefined contingency, not selected because of test signs. If even technical parity fails, provide association/prediction results and the method specification/negative validity report; no valid acceleration claim exists. Do not report requested experimental work as completed when the underlying job is missing.

Trim order for time: L06 extra20k -> exact-sampling/crosslingual/distillation already deferred -> L05/L04 runtime extras -> L03extra7draft32/L02b -> L03target72/32(and anchors if scale panel is dropped) -> lower-priority lookahead/fullblock ablations. Keep Gemma family replication, coreactual-candidate labeling, confidence/identity controls, primary runtime and random/drop-morph comparisons ahead of model-size breadth. If an already registered cell is partially run, report count/reason and complete fixed shards rather than choosing helpful results.

G1 andG2 are technical/measurement gates, not filters on a positive morphology effect. With no annotator, measurement validity here is AUTOMATED consistency/coverage/sensitivity, not semantic gold accuracy. If online morphology adds no predictive value, report that inT2 and the corresponding negativeT3 comparison. Don't retrain/tune after looking at test.

If start later than09:00 on07/10, subtract elapsed time from slot budgets and recompute; do not keep the same completion promise. Downloads/access unresolved by noon07/10 reduce scope explicitly. No-annotator limitation is already confirmed and accounted for above; the sprint can finish computational/method evidence but cannot claim human-validation completion.

## 8. Kết quả cần đóng trước hạn

T1: actual candidate N->P and general-boundary associations with raw/adjustedRD,95%documentCI, exposed docs/identities/coverage; Qwen capacity and family/large panels. Main text uses modest claims if overlap/support insufficient.

T2: held-out feature groups P0confidence/P1structure+direction+environment versus train-only token/identity lookup controls; logloss/Brier, pairedCI, riskcoverage, domain/size transfer. Offline explanatory features cannot be presented as deployable online predictors.

T3: six runtime policies; full latency/relativegain/acceptedtokens perverify/CPUoverhead/VRAM/parity. Negative gain remains in table. Ablations and hardware/IT32 are shown only if actually completed.

Figures: nested data-scale; fixed-draft/fixed-target capacity; predictive riskcoverage; overhead/headroom. Appendix: tokenizer/cache/labelvalidity, humanannotation rubric/agreement/accuracy only if actualgold, matchingSMD/ESS, lexicalheldout, K/context/precision/distributiondiagnostics, failedintervention and provenance.

Final status ledger at18:00 on10/10: planned/completed/failed/partial/skipped for every card, immutableconfig and prompt hashes, denominators and missing-cell reasons. Figures/tables must be generated from completed outputs; no template number is treated as a result. New `paper.tex` prose should distinguish discovery versus confirmation and replace original target-side wording/numbers where unsupported.

This plan uses the experiment-plan workflow. No automatic background scheduler, remote access or human message is created merely by writing these files.
