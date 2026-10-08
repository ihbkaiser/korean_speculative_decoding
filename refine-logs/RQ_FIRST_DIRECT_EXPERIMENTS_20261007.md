# RQ-first scope: chỉ giữ thí nghiệm trả lời trực tiếp câu hỏi nghiên cứu

Ngày 2026-10-07. `paper.tex` hiện chưa ghi RQ thành câu hỏi đánh số. Các RQ dưới đây được rút ra từ claim và kết quả trong bản thảo. Tài liệu này thay thế danh sách chạy rộng ở `DEADLINE_EXPERIMENT_PLAN.md` và `NAACL_EXPERIMENT_SUBSECTION_CARDS_20261007.md` cho phạm vi bài hiện tại. Không submit một job chỉ vì card cũ đánh dấu MUST.

## RQ của bài hiện tại

### RQ1 — Điểm aggregate có giải thích draft rejection không, hay cần phân biệt hướng mismatch?

**Câu trả lời hiện có:** Đã chạy 4.2.1 trên Modal T4, tái phân tích parquet candidate-side P1/P2 ở Wikipedia và FLORES. So sánh cùng population giữa controls; controls + `1 − boundary F1`; controls + loại mismatch; controls + cả hai. Scalar không có đóng góp ổn định; CROSS so với WITHIN_SPLIT có adjusted rejection difference dương trong cả bốn cell. Đây là câu trả lời trực tiếp cho RQ1; không chạy lại và không cần inference mới.

**Nguồn:** `runs/proposal_side_boundary_validation/aggregate_vs_directional/report.md`, `model_comparison.csv`, `population_audit.csv`. P1/P2 ở đây là mã pair cũ, không tự đổi thành Q1/Q2 mới.

### RQ2 — Hiệu ứng có tập trung ở loại ranh giới hình thái cụ thể, nhất là nominal→particle, không?

**Câu trả lời hiện có:** Candidate-side analysis và environment report đã có trong repo cho P1/P2 trên Wikipedia/FLORES. Report cho thấy N→P CROSS–WITHIN_SPLIT khác biệt dương, còn nhiều environment khác có hỗ trợ ít hơn; matched analysis hiện có không đạt cân bằng covariate đủ tốt để diễn giải nhân quả. Dùng adjusted association làm kết luận chính; trình bày matching/suppression như giới hạn hoặc negative evidence.

**Việc cần làm trước khi viết, không cần GPU:** khóa một định nghĩa outcome/eligibility chung giữa report 4.2.1 và environment report; kiểm tra từng bảng chỉ dùng actual accepted proposals và first rejection, không tính suffix `INVALIDATED`; đồng thời phân biệt candidate-side projection với target-side segmentation. Nếu bảng và định nghĩa nhất quán, không chạy lại morphology analysis chỉ để đổi tiêu đề subsection.

**Nguồn:** `runs/proposal_side_boundary_validation/proposal_side_validation.md`, `runs/proposal_side_boundary_validation/metadata.json`, `runs/proposal_side_boundary_validation/pairs/`, `runs/proposal_side_boundary_validation/` environment outputs và `runs/proposal_boundary_matched_analysis/matched_analysis_report.md`. Dùng P1/P2 làm bằng chứng chính; P3 chỉ sensitivity cho tới khi provenance model revision được giải quyết.

### RQ3 — Kết quả có phụ thuộc vào việc lấy các ví dụ đầu theo thứ tự dataset không?

Đây là điểm còn thiếu trực tiếp nhất trong bản thảo: paper nói 1.000 Wikipedia đầu theo thứ tự; 1.000 mẫu mở rộng kế tiếp cũng không phải random sample. FLORES là workload khác nhưng không sửa được selection bias của Wikipedia.

**Chỉ một run mới được khuyến nghị cho RQ này:**

- **Máy:** B200-A. Phân tích outcome và bootstrap chạy trên Modal T4 theo chỉ dẫn trước; RTX 3090 không chạy phân tích.
- **Model:** một pair đã pin rõ revision, Qwen3-1.7B-Base → Qwen3-4B-Base (Q2). Không chạy grid nhiều scale/family cho câu hỏi này.
- **Dataset:** 2.500 bài Wikipedia Korean mới, hash-sampled uniform từ nguồn đã pin; loại toàn bộ ID trong 2.000 bài cũ và duplicate/near-duplicate đã phát hiện trước split. Không dùng FLORES devtest/dev để tune.
- **Decode:** cùng một cấu hình cố định cho cả prompt: BF16, greedy, K=4, tối đa 256 token mới; kiểm tra exact parity trước khi chạy hết.
- **Outcome/contrast đăng ký trước:** actual valid proposals; accepted + first rejection, bỏ hậu tố invalidated. Kiểm tra CROSS vs WITHIN_SPLIT và riêng N→P CROSS vs adjacent WITHIN_SPLIT; cùng model specification chính của RQ1/RQ2, báo adjusted risk difference, 95% document-cluster CI, coverage và số nguồn tài liệu đóng góp.
- **Ngưỡng coverage:** báo trước nếu N→P có dưới 500 candidate hoặc dưới 200 tài liệu đóng góp thì ước lượng subgroup chưa đủ support; chỉ được thêm một shard đã đăng ký trước dựa trên số lượng/coverage, không nhìn rejection effect hay p-value để quyết định dừng.
- **Nó trả lời gì:** association và N→P localization có lặp lại trên tài liệu được lấy ngẫu nhiên hay kết quả chỉ phản ánh đầu danh sách Wikipedia cũ?

**Không chạy trong card này:** Q1/Q3 trên cùng data mới; thêm FLORES/news; model scale/family; context/K/precision grid; classifier/predictor; annotation frames. Các phần đó không cần để trả lời RQ3 và sẽ không được gom vào như “confirmation”. Nếu không có thời gian cho run này, paper phải giới hạn claim vào các corpus/samples/model pairs thực sự đã quan sát và giữ sampling-order limitation nổi bật.

## Câu hỏi phương pháp là một RQ riêng, không phải mặc định của bài hiện tại

Trong `paper.tex`, bài nói rõ hiện chưa đề xuất inference method; MADS mới là ý tưởng ở phần Discussion. Vì vậy **không chạy E09/E13–E15 trong gói RQ1–RQ3**. Có thể mô tả MADS như hướng tiếp theo, nhưng không claim predictor utility hay speedup.

Chỉ nếu quyết định thêm RQ4 — “Morphology có cải thiện latency speculative decoding so với controller chỉ dùng confidence không?” — mới chạy hai kiểm tra trực tiếp: (1) held-out predictor trên cùng proposal population: confidence vs confidence+morphology, giữ token identity/frequency làm đối chứng; (2) một runtime test end-to-end trên một pair/target đã chọn, MADS vs tuned fixed-K và confidence/entropy controller, exact greedy output parity, tính parser/controller cost. Nếu bước (1) không tăng held-out utility hoặc pilot cost/headroom âm thì dừng, không mở runtime grid. Đây là scope riêng và cần sửa thesis của paper trước khi chạy.

## Các card cũ chuyển khỏi queue chạy hiện tại

Không chạy chỉ để làm bảng lớn hơn: 5.000 Wiki × Q1/Q2/Q3 + FLORES + news; Qwen 8B capacity grid; Gemma; IT/KoAlpaca/XL-Sum task grid; 30B/27B/72B; K/context/precision sweep; distributional stochastic SD; 400+1.200 linguistic frames khi chưa có annotator; full MADS benchmark. Chúng chỉ quay lại nếu được gắn với một RQ và nếu kết quả cần thiết để bảo vệ claim cụ thể.

## Bố cục paper theo RQ

1. **Experimental protocol:** mẫu/pair/version, eligibility/projection, covariates và clustered uncertainty.
2. **RQ1:** aggregate score so với directional mismatch — bảng chính, dùng kết quả 4.2.1 đã chạy.
3. **RQ2:** boundary environment, trọng tâm N→P và các environment có đủ support — dùng trace/report hiện có; gộp replication FLORES ở đây.
4. **RQ3:** random-sample replication — chỉ thêm nếu chạy được card 2.500 bài nêu trên; nếu chưa chạy thì chuyển sample-order bias thành limitation, không mô tả là confirmation.
5. **Discussion:** đề xuất MADS ngắn như implication/hypothesis, không claim method result nếu chưa thêm RQ4 và benchmark trực tiếp.

## Trạng thái thực thi trong phiên này

Không có job inference B200 mới được submit trong phiên sửa scope này. Workspace hiện không có kết nối/SSH tới host B200; các kết quả 4.2.1 đã chạy trước đó từ Modal T4. Card RQ3 là cấu hình tối thiểu để queue khi runner/host được kết nối, không phải kết quả đã chạy.
