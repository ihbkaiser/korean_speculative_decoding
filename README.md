# 한국어 형태소 경계와 Speculative Decoding

이 저장소는 한국어 띄어쓰기 단위인 어절(eojeol) 안에서 형태소 경계와 LLM 토큰 경계의 불일치가 speculative decoding의 draft-target 불일치 및 거부율과 관련되는지 조사합니다. 비교할 때 토크나이저 fragmentation을 통제합니다.

## 연구 가설

같은 fragmentation 수준의 어절끼리 비교했을 때, 형태소 경계와 토크나이저 경계의 불일치가 클수록 (1) 실제 speculative decoding에서 draft 제안 토큰의 rejection rate가 높고, (2) target continuation에 teacher-forcing했을 때 draft와 target의 greedy 예측이 더 자주 다를 것으로 가정합니다.

## 실험 설정

- Draft: `Qwen/Qwen3-0.6B-Base`
- Target: `Qwen/Qwen3-4B-Base`
- FP16, batch size 1, greedy decoding, speculative block size `K=4`
- Prompt 최대 길이 128 토큰, 생성 최대 길이 128 토큰
- 데이터: [`wikimedia/wikipedia`](https://huggingface.co/datasets/wikimedia/wikipedia), 설정 `20231101.ko`, `train` split
- 기본 데이터 규모: 1000개 비어 있지 않은 위키백과 문서 prompt
- 형태소 분석: `kiwipiepy`

모델을 로드하기 전에 두 tokenizer의 vocabulary, added tokens, backend 구성, special tokens, 입력 ID와 character offsets를 확인합니다. 호환성 검사가 실패하면 실험을 중단합니다. 실제 speculative decoding 결과는 같은 prompt의 target-only greedy 결과와 token ID 단위로 완전히 일치해야 합니다.

## 설치 및 실행

RTX 3090의 CUDA가 포함된 PyTorch 환경을 준비한 뒤:

```bash
source /venv/main/bin/activate
uv pip install -r requirements.txt
```

요청된 검증 순서:

```bash
python scripts/check_env.py
python -m pytest -q
python scripts/run_pilot.py --limit 5
python scripts/run_pilot.py --limit 20
```

위키백과 데이터를 미리 내려받으려면:

```bash
python scripts/prepare_data.py --limit 1000
```

20개 prompt 실행 및 분석:

```bash
python scripts/run_pilot.py --limit 20
python scripts/analyze.py runs/<run_id>
```

전체 1000개 prompt 실행은 명시적으로 요청할 때만 시작합니다:

```bash
python scripts/run_pilot.py --limit 1000
```

모델 가중치 및 데이터셋을 처음 실행할 때 다운로드하므로 Hugging Face Hub에 접근할 수 있어야 합니다. 모델이 실행될 때 run 설정, 환경 정보, tokenizer compatibility report, 참조 생성 결과, 출력 token ID, VRAM 측정값을 함께 저장합니다.

## 형태소와 tokenizer 경계 정의

각 생성 continuation을 whitespace-delimited 어절로 분리하고, Kiwi가 반환한 형태소 character spans와 tokenizer offset을 어절 내부 character 위치로 변환합니다. 양쪽 모두 마지막 어절 끝 경계는 내부 경계 집합에서 제외합니다.

- `fragmentation = # LLM tokens / eojeol` (어절마다 토큰 수)
- `misalignment = 1 - boundary_F1(morpheme_boundaries, tokenizer_boundaries)`
- Boundary 일치는 어절 안의 character offset이 정확히 같은 경우입니다. 두 집합이 모두 비면 F1=1, 하나만 비면 F1=0입니다.
- Primary fragmentation bins: `2`, `3`, `4`, `5+`. 토큰이 1개인 어절은 보관하지만 해당 primary-bin 비교에서 제외합니다.

생성 결과의 각 token position은 character-span overlap을 기준으로 어절에 연결됩니다. 따라서 morphology-conditioned 분석은 실제 continuation 내 어절 occurrence를 사용하며, surface form만으로 합치지 않습니다.

## Decoding 기록과 분석

Custom greedy speculative decoder는 각 draft proposal을 개별 기록합니다. 첫 불일치 지점에서는 target greedy token을 출력하고 해당 block의 나머지 proposal은 무효화합니다. 무효화된 proposal은 `rejected=null`로 기록되어 rejection 분모에 포함되지 않습니다. 모든 prompt에서 target-only greedy ID와 SD output ID가 같지 않으면 run은 실패 처리됩니다.

Teacher-forced 분석은 target-generated continuation의 모든 token 위치에서 draft와 target의 next-token greedy prediction, 실제 token log probability, entropy를 기록합니다. Primary analysis는 fragmentation bin별 rejection/disagreement 비율과 misalignment slope를 보고하고, prompt별 반복 관측을 고려해 logistic regression의 표준오차를 prompt 기준으로 cluster합니다.

### 중요 구현 설명 (English notes)

- **Exact greedy parity:** Target proposals are verified one token at a time through the target KV cache, using the same cached forward path as target-only greedy decoding. This avoids argmax changes that can arise from batched verification kernels near ties.
- **Invalidated proposals:** Proposals after the first mismatch are logged for audit, but their `rejected` value is null because the target never verified them on the valid decoding path.
- **Boundary score:** Morpheme/tokenizer boundaries are exact character offsets within one eojeol. `misalignment` is one minus their boundary F1; two empty boundary sets have F1 1.
- **Inference scope:** Regression standard errors are clustered by prompt. A small pilot is for pipeline validation and does not establish the hypothesis.

## Output

Mỗi lần chạy tạo `runs/<run_id>/` với các bảng chính:

- `sd_events.parquet`: một hàng mỗi draft proposal; chứa token ID, accepted/rejected, vị trí rejection đầu tiên, accepted-prefix length, proposal/output positions, log probabilities và entropy.
- `teacher_forced_tokens.parquet`: draft-target predictions, disagreement, log probabilities và entropy cho từng token của continuation tham chiếu.
- `eojeols.parquet`: số hình vị/token, boundary offsets, fragmentation bin, boundary F1 và misalignment theo từng occurrence.

Lệnh `scripts/analyze.py` ghi các summary CSV, logistic regression report vào thư mục run và hình vào `figures/<run_id>/`. Dataset đã chuẩn bị, log và kết quả chạy được lưu trong Git để tiện tái lập và xem lại; model cache (`.hf_home/`) và cache cục bộ vẫn được loại khỏi Git.

Thiết kế chi tiết, semantics của event log, quy tắc alignment và giới hạn thực nghiệm nằm trong [implementation.md](implementation.md).
