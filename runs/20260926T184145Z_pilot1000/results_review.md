# Đánh giá kết quả full run

Run: `20260926T184145Z_pilot1000`  
Dữ liệu: 1.000 bài Wikipedia không rỗng đầu tiên theo thứ tự dataset `wikimedia/wikipedia`, config `20231101.ko`  
Mô hình: Qwen3-0.6B-Base draft, Qwen3-4B-Base target; FP16, greedy, batch 1, K=4

## CONFIRMED

- Full run hoàn tất và có marker `COMPLETE`: 173.046 dòng SD event, 127.245 token teacher-forced, 37.576 lần xuất hiện của eojeol.
- Hai tokenizer vượt qua kiểm tra compatibility. Target greedy và custom speculative decoding cho token ID giống hệt nhau trên cả 1.000 prompt.
- Giả thuyết theo chiều dương **không được ủng hộ**. Trong các fragmentation bin, không có slope dương có ý nghĩa thống kê. Bin `5+` cho slope âm có ý nghĩa ở cả hai outcome; các bin `2`, `3`, `4` không có slope đáng kể.

Các mô hình theo bin là `outcome ~ misalignment`, dùng sai số chuẩn cluster theo prompt. `n` là proposal SD đã được target xác minh (proposal bị vô hiệu hóa bị loại) và số token teacher-forced tương ứng. Slope là thay đổi log-odds khi misalignment tăng trọn dải 0→1.

| Bin | n | SD rejection | SD slope | p | Teacher disagreement | TF slope | p |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2 | 15.491 | 20,62% | +0,015 | 0,795 | 20,60% | +0,016 | 0,784 |
| 3 | 37.694 | 20,00% | −0,059 | 0,473 | 20,01% | −0,064 | 0,431 |
| 4 | 27.362 | 18,55% | −0,077 | 0,544 | 18,56% | −0,074 | 0,557 |
| 5+ | 34.306 | 16,17% | −0,676 | 1,54×10⁻⁵ | 16,16% | −0,676 | 1,46×10⁻⁵ |

Ở bin `5+`, slope thô tương đương odds ratio khoảng 0,51 cho mỗi mức tăng misalignment từ 0 đến 1 (95% CI xấp xỉ 0,37–0,69), tức chiều ngược với giả thuyết. p-value chưa hiệu chỉnh; kết quả `5+` vẫn thấp hơn ngưỡng Bonferroni 0,00625 cho 8 phép thử bin×outcome.

Logistic regression theo công thức đã yêu cầu có 114.853 quan sát đủ điều kiện:

- SD: `rejected ~ misalignment + fragmentation + draft_entropy + target_entropy + proposal_position`; β misalignment = −0,0785, 95% CI [−0,168; 0,011], p=0,084.
- Teacher forcing: `disagreement ~ misalignment + fragmentation + draft_entropy + target_entropy + output_token_position`; β misalignment = −0,0498, 95% CI [−0,142; 0,043], p=0,291.

Cả hai mô hình điều chỉnh đều không cho thấy quan hệ dương.

## EXPLORATORY

- Bin `5+` không đồng nhất: trong 6.082 eojeol, có 4.386 eojeol đúng 5 token, 1.114 có 6, 302 có 7; phần còn lại trải từ 8 đến 128 token. Có 3 eojeol 128 token và 1 eojeol 90 token.
- Phân tích độ nhạy hậu nghiệm cho bin `5+`: thêm nhóm số token (`5`, `6`, `7`, `8+`) vẫn cho slope âm (SD β=−0,622, p=9,9×10⁻⁵; teacher β=−0,621, p=9,6×10⁻⁵). Khi thêm entropy và vị trí token, hai slope tiến về 0 (SD β=−0,011, p=0,939; teacher β=−0,005, p=0,972). Điều này cho thấy liên hệ thô trong `5+` đi cùng entropy/vị trí của mô hình; không chứng minh tác động nhân quả của morphology.
- Hai nhãn SD rejection và teacher-forced disagreement trùng nhau tại 127.093/127.245 vị trí token hợp lệ (99,88%). Vì vậy đây gần như cùng một tín hiệu trên cùng continuation, không phải hai lần xác nhận độc lập. Teacher-forced target argmax khác token target đã sinh tại 167 vị trí (0,13%); full-sequence scoring và cached one-token decoding có thể tạo khác biệt số học FP16.
- Bảng độ nhạy theo bin đã điều chỉnh nằm ở `posthoc_adjusted_bin_sensitivity.csv`; mô hình lồng nhau cho `5+` nằm ở `five_plus_nested_sensitivity.csv`.

## FAILED / INCOMPLETE

- Một lần khởi chạy trước (`20260926T183817Z_pilot1000`) bị dừng sau prompt đầu tiên khi chuyển job vào tmux; không có parquet hoàn chỉnh và được đánh dấu trong `INTERRUPTED.txt`. Run cuối cùng ở trên đã hoàn tất đủ 1.000 prompt.

## HIGHEST VERIFIED RUNG

R7 (review kết quả) cho đánh giá 1.000 prompt đã khai báo. Artifact R6 là marker `COMPLETE`, các parquet và `run_metadata.json`; memo quyết định là file này. Kết quả không ủng hộ chiều dương của giả thuyết.

## EVIDENCE GAPS

- Prompt là 1.000 bài đầu tiên theo thứ tự dataset, không phải mẫu ngẫu nhiên có seed từ toàn bộ corpus.
- `5+` gộp nhiều số token. Slope âm thô biến mất khi điều chỉnh entropy/vị trí, nên chưa tách được hiệu ứng misalignment độc lập.
- Teacher-forced và rejection gần như trùng nhau. 167 target argmax khác continuation cần được kiểm tra bằng logit margin để xác định ảnh hưởng của đường tính FP16.
- Phân tích dựa trên Kiwi và boundary F1 chính xác theo character offset, chưa được người gán nhãn kiểm định. Đây là liên hệ trên continuation do model sinh, không phải kết luận nhân quả hay đại diện cho mọi văn bản tiếng Hàn.

## RECOMMENDED NEXT

Lặp lại R6 với 1.000 prompt Wikipedia lấy mẫu ngẫu nhiên bằng seed cố định; báo riêng fragmentation chính xác trong `5+`, định trước cả estimand thô lẫn estimand điều chỉnh entropy/vị trí, và tính teacher-forced theo cùng cached path với target greedy. Việc này sẽ kiểm tra khả năng khái quát ngoài lát cắt 1.000 bài đầu.

## Tài nguyên chạy

- Thời gian: 30.934,5 giây (khoảng 8 giờ 35 phút).
- VRAM đỉnh: 8,90 GiB allocated; 10,17 GiB reserved trên RTX 3090 24 GiB.
- Báo cáo mô hình: `logistic_regression.txt`, `fragmentation_bin_summary.csv`.
- Forest plot có khoảng tin cậy 95%: `figures/20260926T184145Z_pilot1000/binwise_misalignment_slopes.png`.
