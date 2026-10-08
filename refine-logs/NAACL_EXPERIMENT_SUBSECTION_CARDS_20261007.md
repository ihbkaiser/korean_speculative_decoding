# NAACL experiment cards by subsection

> **Lưu ý:** đây là danh mục thiết kế rộng trước khi người dùng yêu cầu thu hẹp theo RQ. Nó không còn là danh sách cần chạy. Scope hiện hành là [`RQ_FIRST_DIRECT_EXPERIMENTS_20261007.md`](RQ_FIRST_DIRECT_EXPERIMENTS_20261007.md).

Ngày: 2026-10-07. Đây là kế hoạch thực nghiệm theo bố cục trong `NAACL_EXPERIMENT_BLUEPRINT.md`, không phải báo cáo các job sẽ tự chạy. Tên model, quy mô dữ liệu và tiêu chí dưới đây là thiết kế đề xuất; chỉ kết quả có log và artifact mới được đưa vào paper.

## Phân công máy và quy ước chung

- **B200-A:** sinh trace/inference cho nhóm Qwen3-Base chính và grid scale Qwen 0.6B–8B.
- **B200-B:** Gemma, instruction-tuned, model lớn và prototype MADS. Khi đo latency cuối, chỉ để một B200 hoạt động; B200 còn lại idle và host không chạy job nặng khác, hoặc phải đo/ghi rõ ảnh hưởng nhiễu.
- **RTX 3090:** chỉ preflight inference cho model nhỏ hoặc thí nghiệm phụ so sánh hệ thống với B200. Theo chỉ dẫn của người dùng, không chạy hồi quy, bootstrap hay phân tích parquet trên máy này.
- **Modal T4:** mọi phân tích offline trên parquet: feature extraction, hồi quy, propensity/matching, predictor, bootstrap, bảng/figure thống kê. Đây là nơi đã chạy 4.2.1.

Thiết lập chính cho trace mới: BF16, greedy, K=4, tối đa 256 token mới, EOS chuẩn, batch 1, một cặp model resident trên một GPU, không quantization/offload. Ghi revision bất biến, tokenizer, backend, GPU, source/prompt hash và output IDs. Với nhãn SD, tính accepted proposals và proposal bị target từ chối đầu tiên; hậu tố bị vô hiệu sau lần từ chối đó là `INVALIDATED`, không gán thành thêm rejection.

Các tập chính: **W** = Wikipedia tiếng Hàn, 10.000 tài liệu mới đã chia theo nguồn thành train 4.000/dev 1.000/test 5.000; **F** = FLORES-200 English→Korean, 997 câu ở `dev` nếu phần này chưa bị dùng để chọn phân tích; 1.012 câu `devtest` cũ là discovery; **N** = XL-Sum Korean news, official test 550; **I** = KoAlpaca, test 1.000 nhóm câu hỏi, chỉ dùng câu hỏi làm prompt; **S** = XL-Sum Korean summarization, test 550. N và S dùng cùng nguồn bài báo nên không tính là hai mẫu độc lập.

## 4.1 Experimental Protocol

### EXP-4.1a — Decoder, tokenizer và numerical parity (E00; bắt buộc trước trace chính)

- **Máy:** B200-A cho Qwen; B200-B cho Gemma/IT. RTX 3090 chỉ chạy smoke test nếu cần; không dùng kết quả 3090 để trộn vào test chính.
- **Model:** từng cặp sẽ xuất hiện trong paper: Q1–Q6, G1, QI1, GI1 và các cặp lớn nếu được bật.
- **Dataset:** 64 prompt W + 32 F + 32 N cho mỗi cặp Base; 64 I + 32 S cho cặp IT, lấy từ development/discovery đã khai báo trước.
- **Làm gì / kiểm chứng:** trên cùng prompt, so output và điều kiện dừng giữa target-only AR, speculative path và batched verifier; kiểm tra vocab/token ID, special token, cache position, argmax và lỗi near-tie. Cổng bắt buộc: exact output/stop parity trước khi gọi kết quả là greedy speculative decoding lossless. Nếu không đạt, sửa backend/cache rồi chạy lại pilot; nếu vẫn lệch thì không claim speedup chính xác cho cặp đó.

### EXP-4.1b — Độ tin cậy phép đo morphology (E01; phần người cần làm đang hoãn)

- **Máy:** không cần GPU cho annotation; hồi quy/độ phủ sau annotation chạy Modal T4.
- **Model:** mẫu cân bằng từ Qwen và Gemma, accepted/rejected, CROSS/SPLIT/EXACT và trường hợp mơ hồ.
- **Dataset:** thiết kế 600 trường hợp: 360 W/F/N, 120 ambiguous/context-sensitive, 120 Gemma; hai annotator tiếng Hàn độc lập và adjudication.
- **Làm gì / kiểm chứng:** người gán span ký tự, loại quan hệ token–morpheme, loại boundary/particle khi xác định được; ẩn model, outcome và nhãn tự động. Tính precision/recall, agreement và tỷ lệ loại trừ. **Hiện chưa có annotator tiếng Hàn nên card này chưa chạy được**; không thay gold bằng nhãn LLM. Trong bản hiện tại chỉ báo cáo kiểm tra tự động, độ phủ, lỗi projection và sensitivity; không viết “human-validated”.

## 4.2 Analysis I: Localizing Morphological Disagreement

### 4.2.1 Aggregate alignment versus directional mismatch

### EXP-4.2.1 — So sánh điểm aggregate với hướng mismatch (ĐÃ CHẠY; tái phân tích offline)

- **Máy:** Modal T4; không inference mới, không dùng RTX 3090/B200.
- **Model:** các cặp legacy P1/P2 trong parquet candidate-side hiện có; phép chiếu morphology dùng tokenizer Qwen3-0.6B-Base đã pin và Kiwi 0.24.0. Không tự động đổi tên P1/P2 thành Q1/Q2 mới.
- **Dataset:** Wikipedia và FLORES từ parquet đã lưu. Bốn cell P1/P2 × Wikipedia/FLORES; cùng complete-case population giữa bốn model specification trong mỗi cell.
- **Làm gì / kiểm chứng:** tính `1 − boundary F1` theo candidate-eojeol, rồi fit (M0) controls, (M1) controls + scalar aggregate, (M2) controls + loại mismatch, (M3) controls + cả hai. Outcome là actual rejection; clustered SE theo prompt; loại proposal suffix đã invalidated.
- **Kết quả hiện có:** scalar aggregate không có đóng góp độc lập ổn định sau controls. CROSS so với WITHIN_SPLIT có adjusted rejection-risk difference dương trong cả bốn cell: Wikipedia P1 +3.97 pp, P2 +2.82 pp; FLORES P1 +5.91 pp, P2 +12.44 pp; các CI 95% đã loại 0 sau hiệu chỉnh Holm. Đây là association trên corpus cũ, không phải causal effect hay fresh confirmation.
- **Artifact:** [report.md](../runs/proposal_side_boundary_validation/aggregate_vs_directional/report.md), CSV model comparison, population audit và parquet cache cùng thư mục. Giữ 4.2.1 như kết quả đã hoàn tất; không chạy lại trừ khi phát hiện lỗi tái lập.

### EXP-4.2.2 — Boundary-specific signature trên workload tự nhiên (E02; kết quả chính)

- **Máy:** B200-A sinh trace; Modal T4 làm feature scoring và thống kê sau đó.
- **Model:** Q1 `Qwen/Qwen3-0.6B-Base → Qwen/Qwen3-1.7B-Base`; Q2 `1.7B-Base → 4B-Base`; Q3 `0.6B-Base → 4B-Base` với revision pin mới.
- **Dataset:** mỗi cặp W-test 5.000 + F 997 + N-test 550. Riêng Q2 có W-train 4.000/W-dev 1.000 để phục vụ phân tích dự báo 4.3.2. Nếu F-dev đã góp phần chọn giả thuyết/ngưỡng thì hạ F xuống discovery và dùng W/N/I còn giữ kín làm confirmation.
- **Làm gì / kiểm chứng:** trích nhãn từ proposal thật ở K=4; so EXACT/SPLIT/CROSS, coarse alignment, và tương phản xác nhận CROSS một boundary (ưu tiên N→P) với WITHIN_SPLIT lân cận. Báo raw rejection, adjusted risk difference/CI, số proposal CROSS, số tài liệu đóng góp và tương tác workload. Điều chỉnh cấu trúc proposal, confidence và tần suất; giữ biến nhìn thấy sau khi target chạy riêng thành retrospective diagnostic.
- **Kiểm chứng claim:** directional/local boundary signal có lặp lại trên output thực, workload tự nhiên, và không chỉ do một tập Wikipedia/FLORES hay do hậu nghiệm target-context. Hiệu ứng âm hoặc không ước lượng được cũng phải hiện trong bảng.
- **Artifact:** Main Table 1 và boundary-profile figure.

### EXP-4.2.3 — Quy mô dữ liệu, quy mô model và replication theo family (E03–E06; cộng large-model extension)

**EXP-4.2.3a: độ ổn định theo cỡ dữ liệu (E03)**

- **Máy:** Modal T4; chỉ tái phân tích trace W đã sinh, không inference mới.
- **Model:** Q1/Q2/Q3.
- **Dataset:** các tập con lồng nhau từ W-test: 500/1.000/2.000/5.000 tài liệu.
- **Làm gì / kiểm chứng:** tính lại cùng ước lượng raw/adjusted/common-support; 20 lần chọn document cluster ở N=1k và 2k để xem dao động ước lượng, không gọi là 20 seed generation. Theo dõi CI, số N→P CROSS và số nguồn độc lập. Kiểm chứng mẫu nhỏ có quá bất ổn hay kết luận dần hội tụ theo cỡ mẫu.

**EXP-4.2.3b: grid quy mô Qwen3-Base (E04)**

- **Máy:** B200-A sinh Q4–Q6; Q1–Q3 dùng lại trace E02. Phân tích trên Modal T4.
- **Model:** Q1–Q6: Q1 .6→1.7B, Q2 1.7→4B, Q3 .6→4B, Q4 .6→8B, Q5 1.7→8B, Q6 4→8B, tất cả Base.
- **Dataset:** cùng tập W-test 2.000 document cho cả sáu cặp; Q5 có thể thêm F 997 như transfer panel nếu vẫn là test chưa dùng để tune.
- **Làm gì / kiểm chứng:** so sánh target 8B với draft 0.6/1.7/4B; target 1.7/4/8B với draft cố định 0.6B; target 4/8B với draft cố định 1.7B. Đo acceptance và boundary risk contrasts. Thêm cùng-prefix diagnostic trên 1.000 prefix đã chọn trước từ Q5 để tách hiệu ứng prefix tự nhiên khỏi khác biệt tại cùng context.
- **Kiểm chứng claim:** hiệu ứng có bền khi thay tỷ lệ năng lực draft/target không. Đây không tự chứng minh quy luật scaling vì checkpoint cũng khác dữ liệu huấn luyện.

**EXP-4.2.3c: replication model family (E05)**

- **Máy:** B200-B sinh Gemma; Modal T4 phân tích.
- **Model:** G1 `google/gemma-3-4b-pt → google/gemma-3-12b-pt`, text-only.
- **Dataset:** W 2.000 + F 997 + N 550. Chạy pilot 100 prompt trước để kiểm tra Korean-output rate, tokenizer compatibility và N→P support.
- **Làm gì / kiểm chứng:** lặp lại contrast boundary và predictive features trên family/tokenizer khác; báo toàn bộ output gồm output không phải tiếng Hàn, thêm Hangul-share ≥0.8 chỉ là sensitivity đã định trước. Kiểm chứng kết quả có vượt khỏi Qwen không; không quy khác biệt riêng cho tokenizer vì architecture/training cũng đổi.

**EXP-4.2.3d: task và instruction robustness (E06; tăng sức thuyết phục)**

- **Máy:** B200-B; Modal T4 cho phân tích.
- **Model:** QI1 `Qwen/Qwen3-1.7B → Qwen/Qwen3-4B` với thinking tắt; GI1 `Gemma-3-4b-it → Gemma-3-12b-it`.
- **Dataset:** I-test 1.000 prompt từ KoAlpaca (chỉ câu hỏi), S-test 550 từ XL-Sum summarization; đúng chat template, S input tối đa 2.048 token.
- **Làm gì / kiểm chứng:** so sánh signature, predictor transfer, Korean-output rate, độ dài/repetition riêng cho instruction và summarization. Summary ROUGE chỉ mô tả workload; nếu output parity chính xác thì không dùng để tuyên bố chất lượng sinh tốt hơn.
- **Vai trò:** nên có để mở rộng kết luận khỏi raw continuation; tách Base và IT vì task/template/training thay đổi cùng lúc.

**EXP-4.2.3e: model lớn hơn trên B200 (L01/L02; ưu tiên sau core, capacity-gated)**

- **Máy:** B200-B, một cặp/lần; đo memory và tốc độ pilot trước full shards. Modal T4 cho phân tích.
- **Model:** L01a/b Qwen3-1.7B/4B-Base → Qwen3-30B-A3B-Base; L02a Gemma3-4B-PT → 27B-PT. L02b 12B→27B là panel phụ.
- **Dataset:** mỗi pair W 2.000 + F 997 + N 550.
- **Làm gì / kiểm chứng:** xem boundary signal ở target vận hành lớn hơn và thêm family/scale. Báo Qwen MoE như một operating point (30B tổng tham số, 3B active), không trộn với dense scaling law. Nếu thiếu thời gian, giữ E04 Qwen 8B + G1; L01/L02 không được làm chậm test chính.
- **Mở rộng xa hơn:** Qwen2.5-1.5B→7/14/32/72B là P2/optional; 72B chỉ sau memory gate, cần xem cách bố trí hai B200 và parity riêng. Không cần để hoàn tất luận điểm chính.

## 4.3 Analysis II: Information Beyond Confidence and Lexical Identity

### 4.3.1 Measurement validity and comparable contexts

### EXP-4.3.1a — Common support và so sánh context tương đương (E07)

- **Máy:** Modal T4, dùng trace E02.
- **Model:** Q1/Q2/Q3, phân tích riêng từng pair.
- **Dataset:** các actual proposal rows của W/F/N; contrast chính N→P CROSS với WITHIN_SPLIT.
- **Làm gì / kiểm chứng:** propensity overlap weighting dựa trên cấu trúc candidate, confidence, frequency và boundary-context; propensity [0.05, 0.95]. Kiểm tra SMD, effective sample size, retained documents; sensitivity exact match theo token length/slot/particle group và caliper 0.2 SD.
- **Kiểm chứng claim:** hiệu ứng có còn khi so các candidate nằm trên vùng covariate overlap hay chỉ do hai nhóm vốn khác nhau. Nếu overlap yếu, kết luận giới hạn ở association trên population quan sát được.

### EXP-4.3.1b — Decoder/feature robustness (E11; focused appendix)

- **Máy:** B200-A tạo trace; Modal T4 tóm tắt/ước lượng. RTX 3090 chỉ có thể chạy parity smoke test phụ.
- **Model:** Q2 và Q5 `Qwen3-1.7B-Base→4B-Base` và `1.7B-Base→8B-Base`.
- **Dataset:** K sweep: W 500 + F 500, K={1,2,4,8,16}; Q5 context sweep trên W 300 ở khoảng 128/512/2.048 token; output-length sweep 128/256/512; BF16-vs-FP16 forensic set 128 prompt mỗi pair.
- **Làm gì / kiểm chứng:** độ nhạy của boundary association/acceptance với K, context, output length, precision và near-tie argmax. K=4 tái sử dụng nếu config chính xác trùng. Giữ test diagnostics tách khỏi việc tune controller; không gom context/output sweep thành một factorial claim nếu prompt thay đổi.

### 4.3.2 Confidence-conditioned and held-out predictive utility

### EXP-4.3.2a — Frequency, lexical identity và morphology granularity (E08)

- **Máy:** Modal T4 cho mọi fit/ablation offline; trace E02 từ B200-A.
- **Model:** Q1/Q2/Q3 cho association; Q2 là predictor chuẩn.
- **Dataset:** frequency corpus/train-only W/N; không tính tần suất từ final W/N test. Predictor split Q2 W train 4.000/dev 1.000/test 5.000.
- **Làm gì / kiểm chứng:** nonlinear token/nominal/particle/eojeol frequencies; leave-top-10 surface patterns; leave-one-supported-particle-group-out; held-out nominal/token identity. So sánh confidence + token-ID lookup với confidence + lookup + morphology. Báo random-source split và identity-held-out split riêng.
- **Kiểm chứng claim:** morphology có mang thông tin khác tần suất/nhớ token cụ thể không. Nếu token identity gần như mã hóa morphology, báo giới hạn nhận dạng thay vì ép diễn giải nhân quả.

### EXP-4.3.2b — Giá trị dự báo ngoài confidence (E09; bảng chính 2)

- **Máy:** B200-A tạo trace; Modal T4 train/evaluate predictor, bootstrap và calibration. Không fit predictor trên RTX 3090.
- **Model:** Q2 làm chính; Q1/Q3 chỉ thêm predictor riêng nếu cần calibration theo pair. Transfer sang F/N không tune lại.
- **Dataset:** W-train 4.000/W-dev 1.000/W-test 5.000; FLORES 997 và XL-Sum news 550 để transfer. Các train/dev/test document phải theo split manifest.
- **Làm gì / kiểm chứng:** so sánh cùng learner/budget: P0 draft confidence (entropy, top-1, margin, slot, position, prefix features, nonlinear frequency); P1 coarse alignment; P2 directional morphology; P3 direction×environment; P4 confidence+train-only token lookup; P5 lookup+morphology. Logistic regression điều chuẩn C∈{.01,.1,1,10,100}, chọn trên dev; HistGradientBoosting là robustness. Primary feature chỉ được dùng candidate prefix đang có, không nhìn trước proposal tương lai.
- **Metrics / điều kiện:** test log-loss/Brier là chính; AUPRC/AUROC, calibration, risk-coverage và tỷ lệ bắt first rejection/wasted suffix là phụ; bootstrap theo document. Chỉ kết luận incremental utility nếu morphology cải thiện held-out so với cả confidence lẫn lookup; nếu không, 4.4 phải so method theo kết quả âm/không kết luận.

### 4.3.3 What linguistic content do the models disagree about?

### EXP-4.3.3 — Content/error analysis (E10; hiện chỉ làm bản mô tả)

- **Máy:** B200-A/B200-B lấy one-step distributions ở prefix cố định; Modal T4 tính rank/log-prob/entropy và bảng. Không cần full-vocab logging cho toàn corpus.
- **Model:** Q2, Q5 và G1.
- **Dataset:** một mẫu định trước từ W/F/N dev hoặc discovery, phân tầng accepted N→P, rejected N→P, SPLIT matched và boundary khác; nếu dùng F `devtest` cũ thì ghi rõ discovery.
- **Làm gì / kiểm chứng:** tại prefix giống nhau, đo rank draft token dưới target, log-prob gap, entropy, top-5 alternative và ví dụ lỗi bề mặt. Đây là diagnostic để gợi ý lexical-vs-particle disagreement, không chứng minh nguyên nhân.
- **Giới hạn:** thiết kế mạnh hơn cần 600 trường hợp được hai annotator kiểm tra và 200 câu frame tiếng Hàn chuyên gia xác thực (1.200 context controlled). Vì chưa có annotator, **không gọi ví dụ/nhãn tự động là human-validated hoặc causal mechanism**. Giữ subsection này ngắn, ghi hạn chế rõ; nếu sau này có annotator thì mới mở rộng card.

## 4.4 Operational Implication: Morphology-Aware Draft Scheduling

### EXP-4.4a — Đo cost/headroom trước khi code controller (E13; cổng dừng)

- **Máy:** B200-B, một GPU; engine đo thật. Modal T4 xử lý trace/headroom summaries.
- **Model:** Q4 `Qwen3-0.6B-Base→8B-Base` và Q5 `1.7B-Base→8B-Base`; G1 chỉ nếu còn headroom.
- **Dataset:** 200 W + 200 F development prompt/pair; không dùng final test để chọn rule.
- **Làm gì / kiểm chứng:** đo draft decode, target block verify/correction, cache, parser/controller và overhead. Dùng outcome đã biết trong trace để tính cost-aware oracle giới hạn trong K={1,2,4,8,16}, ước lượng phần draft suffix tối đa có thể tiết kiệm. Oracle là upper bound của policy class này, không phải optimum SD nói chung.
- **Cổng:** nếu overhead morphology lớn hơn headroom có thể đạt, không mở rộng method claim; ghi kết quả âm và chuyển trọng tâm sang phân tích/prediction.

### EXP-4.4b — Prototype MADS và runtime benchmark (E14; main Table 3, có điều kiện)

- **Máy:** B200-B trên một GPU; GPU B200 còn lại idle, không model reload giữa các policy. Modal T4 chỉ làm phân tích paired CI sau benchmark.
- **Model:** Q4 và Q5; G1 chỉ khi E13 pass và benchmark budget cho phép.
- **Dataset:** tune trên development 200 W+200 F/pair; final runtime 500 W+500 F/pair, không điều chỉnh threshold sau test.
- **Phương pháp:** giữ nguyên draft candidate; sau mỗi candidate, ước lượng rejection risk bằng prefix morphology + confidence; dừng draft block trước candidate kế tiếp nếu ngưỡng/cap đã đạt; target vẫn verify/correct bình thường. P0 confidence-only và P1 confidence+morphology dùng rule/cap/grid và ngân sách tune tương đương. Không thay token bằng “an toàn”.
- **Baselines:** target-only AR; fixed K∈{1,2,4,8,16}, chọn tốt nhất trên dev; max-probability stopping; entropy/SVIP-style stopping; learned confidence-only; confidence+morphology. Chỉ gọi SVIP nếu tái hiện đúng thuật toán; nếu không gọi SVIP-style.
- **Làm gì / kiểm chứng:** chạy ít nhất 3 repeat đã warm-up, policy order counterbalanced; exact output/EOS parity. Đo latency prompt-to-output và decode-only, speedup CI, tokens/sec trong cùng tokenizer, chars/sec cross-family, calls/token, wasted proposals, parser/controller overhead, VRAM. So MADS với baseline mạnh nhất đã tune trên dev.
- **Cổng kết luận:** chỉ nói morphology-aware scheduling hữu ích nếu MADS vừa đúng output vừa vượt baseline mạnh nhất trong test tự nhiên sau khi tính chi phí, với paired CI hỗ trợ. Nếu không, báo kết quả âm và trình bày paper là nghiên cứu về signal/risk, không tuyên bố tăng tốc.

### EXP-4.4c — Ablation chứng minh lợi ích đến từ morphology (E15; bắt buộc nếu claim method)

- **Máy:** cùng B200-B/exclusive runtime slot; phân tích trên Modal T4.
- **Model:** Q4 trên cùng workload/runtime subset với E14; M4 confidence-only có thể reuse nếu đúng trace/config.
- **Dataset:** W 500 prompt; 3 repeat; các activation/threshold được xác định từ dev.
- **Làm gì / kiểm chứng:** bỏ morphology; general-boundary vs chỉ N→P; random cut với activation/slot rate ghép từ dev; strict-prefix so với one-token lookahead và full-block parse nếu đủ thời gian. Đo lại latency thật gồm chi phí parser/lookahead.
- **Kiểm chứng claim:** MADS phải hơn confidence-only và random shortening; equal activation hay classifier AUC tự nó không chứng minh morphology tạo gain. Nếu confidence-only ngang bằng thì chọn baseline đơn giản hơn và thu hẹp claim.

### EXP-4.4d — Negative controls và các phương án không đạt (E16; tái dùng)

- **Máy:** Modal T4 nếu cần tổng hợp lại; không cần GPU mặc định.
- **Model/data:** các artifact intervention/matching/parity cũ trong repo.
- **Làm gì / kiểm chứng:** đưa vào appendix/limitations việc suppression N→P từng làm giảm immediate acceptance, matching cũ chưa kết luận, batched prototype từng fail parity và P3 cũ chưa xác định revision chắc chắn. Tách ba câu hỏi: morphology dự đoán rejection, đổi draft proposal có ích, và scheduling có giảm latency hay không. Đây là kết quả âm có sẵn, không trộn với Q1–Q6 mới.

### EXP-4.3/4.4 optional — Sampling distribution (E12)

- **Máy:** B200-A/B; Modal T4 phân tích distribution outputs.
- **Model:** Q2/Q5/G1.
- **Dataset:** 2.000 prefix đã đăng ký trước từ W/F để tính `sum_v min(p(v),q(v))` ở T=0.7 và 1.0. Chỉ làm stochastic SD thật nếu paper tuyên bố sampling decoding: Q2, 200 W+200 F, 3 seed, K=4, T=0.7, có rejection sampling/residual correction/bonus token và toy-distribution correctness audit.
- **Làm gì / kiểm chứng:** phân biệt one-step expected acceptance ở cùng prefix với empirical stochastic path outcome. Card này không cần cho paper giới hạn ở greedy; ưu tiên thấp hơn parity, E02, E07–E09 và runtime method.

## Trật tự ưu tiên và main-paper outputs

1. **Không thể bỏ:** 4.1a parity; 4.2.1 (đã xong); 4.2.2 E02; 4.2.3a/b và E05 ở mức core; 4.3.1a; 4.3.2a/b. Không có gold annotation thì ghi rõ giới hạn ở 4.3.1/4.3.3.
2. **Chỉ mở rộng method sau các cổng:** E13 trước, rồi E14, sau đó E15. Chạy controller không thay thế kết quả main về association/predictive utility.
3. **Tăng độ phủ nếu GPU-time đủ:** QI1/GI1; Qwen3-30B-A3B và Gemma3-27B trên B200-B. 72B và cross-hardware 3090/B200 để phụ lục/phase sau.

Main tables nên là: **Table 1** boundary contrasts trên W/F/N cùng Qwen/family; **Table 2** held-out predictive utility confidence → morphology → identity controls; **Table 3** AR/fixed/adaptive/MADS latency và ablations, chỉ khi E13–E15 pass. Các grid K/context/precision, overlap diagnostics, family/IT/large-model panels và negative interventions để phụ lục nếu không đổi kết luận chính.

Paper hiện tại trong `paper.tex` còn cấu trúc/kết quả cũ; khi cập nhật cần tổ chức lại theo blueprint này, phân biệt discovery với confirmation, và không gọi mọi card ở trên là đã chạy.
