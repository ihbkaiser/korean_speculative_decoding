# Review repo và định hướng NAACL

Ngày: 2026-10-06. Bản thảo được đọc: `paper.tex`; không tìm thấy `paper.txt` trong repo.
Phạm vi: đọc bản thảo, source của pipeline, báo cáo và bảng kết quả lưu trong repo; không chạy lại generation, regression hay GPU benchmark.
Reviewer route: local self-review; review_independence: same-family/local-self-review; acceptance_status: provisional. Đây là đánh giá của cùng assistant, không phải phản biện độc lập hay quyết định của NAACL.

## Nhận định chính

Repo đã vượt giai đoạn quan sát một tương quan ban đầu: có tái lập theo cặp model và workload, đo trên actual draft proposals, matching, phân rã particle/geometry và một can thiệp proposal. Tuy nhiên bản thảo mới trình bày lớp kết quả target-side ban đầu. Evidence mạnh nhất hiện tại là **một linguistic risk marker trong greedy draft–target disagreement**, chưa phải nguyên nhân hình thái học hay một phương pháp tăng tốc.

Thesis nên kiểm tra tiếp: “Hướng và môi trường của token–morpheme mismatch mô tả những vùng draft–target không tương thích; đặc biệt nominal–particle fusion có tín hiệu lặp lại, nhưng giá trị dự đoán ngoài mẫu và giá trị điều khiển inference cần được kiểm chứng riêng.”

## Vòng 1: audit evidence

| Evidence | Đã thấy | Giới hạn |
|---|---|---|
| Coarse misalignment | Giả thuyết quan hệ dương không được hỗ trợ; adjusted beta -0.0785, p=.084 trên Wiki/P3 | Không phải bằng chứng tương đương với zero; không suy ra misalignment có lợi |
| Target-side CROSS vs SPLIT | Wiki M5 adjusted differences P1/P2/P3 +5.79/+6.62/+6.07 pp | Feature lấy từ continuation target, khác morphology của rejected draft token |
| Target-side nominal→particle | Wiki +20.9/+18.7/+24.7 pp; FLORES +25.5/+19.6/+30.8 pp | Target-side position descriptor; không phải proposal-level causal effect |
| Actual proposal CROSS vs SPLIT | Wiki OR P1/P2/P3 1.446/1.442/1.330; FLORES 1.406/3.046/1.905 | General validation ban đầu tái dùng target labels cho accepted tokens; boundary validation sau đó reproject mọi accepted/rejected proposal |
| Actual proposal nominal→particle | Wiki P1 +12.25 [7.35,17.16], P2 +10.35 [5.66,15.03] pp; FLORES P1 +16.43 [10.72,22.14], P2 +28.17 [21.58,34.75] pp | Structural controls vẫn có thành phần target-side; chưa phải online predictor |
| Stronger matching | Wiki +6.8 [-10.0,25.6], FLORES +10.3 [-12.0,34.6] pp | Chỉ 44/29 matched pairs; max abs SMD 2.81/2.02; criterion chưa đạt |
| N→P avoidance intervention | Pooled acceptance thay đổi Wiki -67.87 pp, FLORES -46.52 pp | Thay top-1 draft bằng candidate khác, nên morphology và identity/rank thay đổi đồng thời |
| Guard implementation | Có CPU opportunity audit và correctness checkpoints | Chưa có pilot/full throughput report; sequential target verification, full draft trước khi cắt |

Nguồn: `runs/20260926T184145Z_pilot1000/results_review.md`; `runs/e2_model_pair_replication/combined/e2_housekeeping_summary.md`; `runs/e2_model_pair_replication/h3/h3_summary.md`; `runs/flores200_en_ko_replication/flores_replication_summary.md`; `runs/proposal_side_validation/proposal_side_validation.md`; `runs/proposal_side_boundary_validation/nominal_to_particle_adjusted_results.csv`; `runs/proposal_boundary_matched_analysis/matched_analysis_report.md`; `runs/proposal_boundary_counterfactual/counterfactual_report.md`.

### Những điểm draft cần sửa

1. Abstract/setting nói phân tích individual draft proposals, trong khi H2/H3 gốc lấy morphology của token target tại vị trí đó. Trong `src/h2_analysis.py`, `generated_ids` đến từ `reference_token_ids`, còn outcome đến từ `event.rejected`. Đưa proposal-side analysis thành primary và target-side thành diagnostic; không thay số một cách cơ học vì estimand và population khác nhau.
2. “Concentrated at nominal–particle” cần giải thích là một contrast lớn, tái lập, không đồng nghĩa nguồn đóng góp lớn nhất vào rejection tổng. H3 pooled N→P chỉ chiếm 6.9% single-boundary CROSS; OTHER chiếm 55.6% và có contribution score lớn hơn.
3. Predicate→ending gần zero trên Wiki, nhưng target-side FLORES +14.3/+8.9/+11.1 pp. Không viết sự vắng mặt của hiệu ứng này như một quy luật chung của tiếng Hàn. Cần actual-proposal all-environment analysis để chốt narrative theo workload.
4. H4 gợi ý heterogeneity theo particle, chưa nhận diện mechanism: dưới 100 CROSS/category/pair; full-nominal/full-particle geometry chỉ 21/8/7 observations; particle-surface fixed effects thiếu support hoặc không estimable. Không gọi geometry hay một loại particle là explanation đã xác nhận.
5. Frequency robustness không tương đương loại trừ token identity hoặc lexical confounding. Không có một surface pattern chi phối không chứng minh token identities không đóng vai trò.
6. Không gọi các cặp Qwen là ba model-family replications độc lập. Pairs dùng chung models, tokenizer và prompts. Teacher forcing trùng rejection 99.88% là consistency check.

## Vòng 2: phản biện và diễn giải cạnh tranh

**Concern 1 — measurement depends on outcome/future text.** Target-side morphology có thể hợp lệ cho descriptive analysis của positions, nhưng không đại diện rejected candidate. Proposal-side sửa một phần; full-block segmentation vẫn nhạy với context, và covariates fragmentation/relative position/first-last trong script lấy từ target table. Phải tách retrospective explanatory model và prefix-available predictive model.

**Concern 2 — common support/extrapolation.** CROSS N→P và SPLIT khác mạnh ở geometry/last-token position. AME bằng cách đổi morph class trong regression có thể dự đoán trên cấu hình hiếm hoặc bất khả. Matching hiện tại không xác nhận được robustness; cũng không bác bỏ association. Cần overlap audit, estimand giới hạn common support và confirmation sample mới.

**Concern 3 — uncertain morphology.** General proposal projection có agreement prefix/full-block khoảng 74–78% ở subset cùng valid. Annotation sheet đã chuẩn bị nhưng chưa có expert labels, và chỉ chứa rejected proposals. Audit mới phải có accepted, rejected và ambiguous cases; hai annotators tiếng Hàn, adjudication và báo theo outcome/workload.

**Concern 4 — significance vs practical value.** Actual single-boundary N→P candidates chiếm khoảng 0.243/0.258% all valid proposals trên Wiki P1/P2, 0.712/0.815% trên FLORES. Số này bao gồm fragmentation=1 nên lớn hơn primary regression population. Một effect cục bộ lớn vẫn có thể cho scheduling gain tổng rất nhỏ. Đo error coverage, discarded draft work và wall time.

**Concern 5 — decoder is not yet an acceleration benchmark.** Official path xác minh target sequential và re-prefill draft mỗi vòng. Full draft block đã tính xong trước `guard_controller.inspect`. Cắt verification chưa tiết kiệm draft computation; đường target sequential cũng chưa hiện thực lợi ích parallel verification của SD chuẩn. Batched prototype fail parity 1/8 preflight, nên cần giải quyết numerical/backend issue với reference phù hợp trước benchmark method. Không suy ra causal/risk result vô hiệu chỉ vì đường này chậm.

**Concern 6 — reproducibility.** Historical P3 weights không xác định; pinned-P2 target continuations khác historical-P3 ở 160/1,000 prompts. Đây là cảnh báo attribution/provenance, không phải chứng minh cause. Dùng pinned P1/P2 primary; P3 historical sensitivity hoặc rerun pinned P3 nếu cần controlled capacity comparison.

## Kết luận reviewer

Điểm mạnh: câu hỏi gắn NLP với inference, taxonomy vượt qua metric tổng hợp, association tái lập ở hai workloads và actual proposals, repo có negative experiments hữu ích.

Điểm chặn hiện tại: manuscript mô tả nhầm measurement unit, morphology chưa human-validated, common support yếu, chưa có held-out incremental predictive utility, một model family, chưa có runtime result đáng tin. Ở trạng thái hiện tại tôi chưa đánh giá evidence đủ vững cho main-conference/Oral ambition; đây là một hướng analysis paper có triển vọng nếu xử lý các điểm này.

Không cần biến paper thành method-heavy paper để làm contribution rõ hơn. Narrative nên là: aggregate metric fails → local signature replicates → test competing explanations and predictive information → small, honest operational feasibility test. Oral là kết quả selection của venue; không có checklist thực nghiệm bảo đảm Oral.

## Claims theo outcome tương lai

| Kết quả | Claim được phép |
|---|---|
| Proposal effect + annotation + held-out confirmation đạt, predictor không cải thiện | Reproducible descriptive association; chưa có predictive utility ngoài confidence |
| Online morphology cải thiện predictor ngoài mẫu, runtime không cải thiện | Useful risk signal; detector overhead/rarity hoặc scheduling economics hạn chế practical gain |
| Oracle có gain nhưng deployable controller không có | Có headroom ở policy class đã xét; implementation/detection là bottleneck |
| Online controller cải thiện synchronized latency vs tuned entropy, giữ output đúng | Prototype morphology-aware scheduling hữu ích trong settings đã đo |
| Common-support estimate không ổn định/không đủ support | Evidence phụ thuộc regression/extrapolation; thu hẹp claim và không diễn giải causal |
| Cross-family không tái lập | Qwen-tokenizer-family-specific finding |
| N→P avoidance tiếp tục âm | Không dùng suppression làm proposed method; chưa kết luận mọi morphology-aware intervention vô ích |

## Trạng thái artifact tại thời điểm đọc

- Expansion traces: Wiki/P1 1,000 prompts; Wiki/P2 32; FLORES/P1 165; FLORES/P2 32. Có pilot projection P2/FLORES, chưa có combined confirmation report.
- Guard `greedy_equality.csv`: 1,700 rows, đều là Wikipedia, zero output/stop mismatch trong rows đã lưu. Đây là correctness evidence đã hoàn thành từng phần, không phải full method evaluation; chưa đủ FLORES và repetitions.
- Không tìm thấy `refs.bib` hoặc bibliography style/class assets trong file listing; manuscript có `\bibliography{refs}`. Hoàn thiện packaging trước submission.

Chi tiết experiment blocks, estimands, splits, metrics, success gates và thứ tự triển khai: `refine-logs/EXPERIMENT_PLAN.md`. Tracker: `refine-logs/EXPERIMENT_TRACKER.md`.
