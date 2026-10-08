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

로컬 가상환경과 CUDA 12.8 PyTorch를 설치합니다:

```bash
uv venv --seed --python python3.13 .venv
uv pip install --python .venv/bin/python --index-url https://download.pytorch.org/whl/cu128 'torch==2.11.0+cu128'
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
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

Lệnh `scripts/analyze.py` ghi các summary CSV, logistic regression report vào thư mục run và hình vào `figures/<run_id>/`. Các phân tích H2 và E1 có thể chạy tiếp trên artifact hiện có, không cần sinh lại continuation hoặc chạy speculative decoding:

```bash
python scripts/analyze_h2.py runs/<run_id>
python scripts/analyze_e1.py runs/<run_id>
```

H2 tạo bảng token theo quan hệ tokenizer–morpheme. E1 nối bảng đó với tần suất token trong cache Wikipedia cục bộ, lưu cache tần suất để tái sử dụng và báo cáo mô hình điều chỉnh tần suất. Dataset đã chuẩn bị, log và kết quả chạy được lưu trong Git để tiện tái lập và xem lại; model cache (`.hf_home/`) và cache cục bộ vẫn được loại khỏi Git.

Thiết kế chi tiết, semantics của event log, quy tắc alignment và giới hạn thực nghiệm nằm trong [implementation.md](implementation.md).

## Table 1 pipeline (FineWeb2 Korean)

The NAACL Table 1 pipeline is implemented in `scripts/table1_pipeline.py` and
uses one frozen prompt pool for all five model pairs. The data stage records the
immutable FineWeb2 commit in `metadata/dataset_revision.json`; model auditing
records model SHAs and the tokenizer/parity gate in
`metadata/model_revisions.json` and `audit/tokenizer_compatibility.csv`.

```bash
python scripts/table1_pipeline.py prepare-data
python scripts/table1_pipeline.py audit-models --pair Q2 --device cuda
python scripts/table1_pipeline.py run-sd --pair Q2 --shard-index 0 --num-shards 8
python scripts/table1_pipeline.py align-morphology --pair Q2 --shard-index 0 --num-shards 8
python scripts/table1_pipeline.py build-table1
```

To supply a different local model snapshot for each pair, pass both model paths
to that pair's audit and SD commands. A custom override requires `--pair`, and
`run-sd` refuses a path that was not used by the corresponding audit, so an old
compatibility/smoke gate cannot accidentally be reused:

```bash
python scripts/table1_pipeline.py audit-models --pair Q1 --device cuda \
  --draft-model-path /models/Q1/draft --target-model-path /models/Q1/target
python scripts/table1_pipeline.py run-sd --pair Q1 --num-shards 8 --device cuda \
  --draft-model-path /models/Q1/draft --target-model-path /models/Q1/target
```

Repeat those two commands for `Q2`, `Q3`, `M1`, and `G1`, changing the two
paths each time. Then run `align-morphology` for every completed SD shard and
run `build-table1` once. The paths may also be Hugging Face model IDs; local
directories are loaded without a Hub revision.

For the direct no-pilot path, `run-table1` performs the full sequence for one
pair (full audit, every SD shard, and morphology alignment). Run it once per
pair, then build the combined table once. If the frozen prompt pool is missing,
the first invocation automatically runs the deterministic `prepare-data` stage:

```bash
python scripts/table1_pipeline.py run-table1 --pair Q1 --num-shards 1 \
  --draft-model-path /models/Q1/draft --target-model-path /models/Q1/target
python scripts/table1_pipeline.py run-table1 --pair Q2 --num-shards 1 \
  --draft-model-path /models/Q2/draft --target-model-path /models/Q2/target
python scripts/table1_pipeline.py run-table1 --pair Q3 --num-shards 1 \
  --draft-model-path /models/Q3/draft --target-model-path /models/Q3/target
python scripts/table1_pipeline.py run-table1 --pair M1 --num-shards 1 \
  --draft-model-path /models/M1/draft --target-model-path /models/M1/target
python scripts/table1_pipeline.py run-table1 --pair G1 --num-shards 1 \
  --draft-model-path /models/G1/draft --target-model-path /models/G1/target
python scripts/table1_pipeline.py build-table1
```

Each pair now persists smoke references/events/reports under `validation/`,
raw resumable shard logs under `runs/table1/<pair>/shards/`, consolidated
references/alignment/proposal joins under `results/`, tokenizer hashes, and
the runtime environment manifest. Q2 also gets a common-20k consolidated view
for cross-pair analysis. The final `build-table1` must report `COMPLETE`.

Each SD/alignment shard writes a `COMPLETE` marker only after exact token-ID
parity succeeds. Production reference generation uses same-length batch-64
target decoding for throughput. If a near-tied batch kernel disagrees with the
singleton cached verifier, only that prompt is regenerated with scalar target
greedy decoding and the fallback is recorded in the reference row and
`run_metadata.json`; a real SD-versus-scalar mismatch still aborts the shard.
Cached SD also verifies each `k=4` proposal block in one target forward and
rolls the target cache back after rejection. Re-running a shard resumes from
`progress.jsonl` and does not resample prompts. Incomplete pairs remain
`Status=INCOMPLETE` in Table 1; placeholder values are never generated.

If the full compatibility audit and 200-prompt smoke gate have already been
handled externally, use `run-table1-main` for the production path only. It
uses explicit model references from the CLI or `config.model_paths`, runs the complete frozen split, and keeps
the per-prompt exact SD/target parity check; it does not call `audit-models` or
`run_smoke_test`:

```bash
python scripts/table1_pipeline.py run-table1-main --pair Q1 --num-shards 1 \
  --draft-model-path /models/Q1/draft --target-model-path /models/Q1/target
```

For the company B200 machine, the immutable local snapshot paths are already
embedded in `configs/table1_pipeline.yaml`. The convenience launcher therefore
needs only a pair name, or `all` for sequential execution of all five pairs:

```bash
bash scripts/run_company_table1.sh Q1
bash scripts/run_company_table1.sh all
```

The `all` mode overlaps CPU morphology alignment with the next pair's GPU
inference: each pair runs SD first, then its alignment is started in the
background while the next pair starts. The final table build waits for every
alignment process and aborts if any one fails. Per-pair logs are split into
`logs/table1_<PAIR>_sd.log` and `logs/table1_<PAIR>_align.log`; the build log
is `logs/table1_build.log`.

SD progress checkpoints are buffered by default (`64` JSONL records per
flush), and terminal progress is printed every `100` prompts. This reduces
filesystem and console overhead while keeping resumable progress. Override
these values when needed:

```bash
PROGRESS_FLUSH_EVERY=64 PROGRESS_LOG_EVERY=100 \
  bash scripts/run_company_table1.sh all
```

If a run is interrupted between checkpoint flushes, at most the buffered
suffix is recomputed; prompt sampling and shard assignment remain unchanged.

### Updating the company checkout from GitHub Download ZIP

The ZIP updater handles the top-level `*-main/` directory created by GitHub.
By default it updates code/config/docs while preserving `data/`, `metadata/`,
`runs/`, logs, model/cache directories, `.venv/`, and credentials. Before
writing, it creates a recoverable backup outside the repository. It does not
need network access:

```bash
ZIP_PATH=/workspace/storage-shared/nlp/tungdd11/korean_speculative_decoding/korean_speculative_decoding-main.zip
FOLDER_PATH=/workspace/storage-shared/nlp/tungdd11/korean_speculative_decoding/repo

ZIP_PATH="$ZIP_PATH" FOLDER_PATH="$FOLDER_PATH" \
  bash /workspace/storage-shared/nlp/tungdd11/korean_speculative_decoding/repo/scripts/update_from_github_zip.sh --dry-run
ZIP_PATH="$ZIP_PATH" FOLDER_PATH="$FOLDER_PATH" \
  bash /workspace/storage-shared/nlp/tungdd11/korean_speculative_decoding/repo/scripts/update_from_github_zip.sh
```

The updater logs timestamped phases to the terminal. To keep a persistent log
and show every copied file, set `LOG_FILE` and `VERBOSE=1`:

```bash
LOG_FILE=/workspace/storage-shared/nlp/tungdd11/korean_speculative_decoding/update.log \
VERBOSE=1 ZIP_PATH="$ZIP_PATH" FOLDER_PATH="$FOLDER_PATH" \
  bash /workspace/storage-shared/nlp/tungdd11/korean_speculative_decoding/repo/scripts/update_from_github_zip.sh
```

It validates the whole ZIP, extracts only mutable code/config files into a
temporary directory, applies the overlay, and removes that directory in a
`finally` cleanup block even when the update fails. `data/`, `models/`, caches, logs,
artifacts, and credentials are not extracted or overwritten by default.
If a ZIP has both a GitHub wrapper and a redundant checkout wrapper such as
`korean_speculative_decoding-main/repo/...`, the updater removes both layers
when `FOLDER_PATH` ends in `repo`.

The default mode is an overlay, so local files absent from the ZIP are kept.
To mirror deletions from GitHub for the mutable part of the checkout, opt in
explicitly with `--delete-missing`:

```bash
ZIP_PATH="$ZIP_PATH" FOLDER_PATH="$FOLDER_PATH" \
  bash /workspace/storage-shared/nlp/tungdd11/korean_speculative_decoding/repo/scripts/update_from_github_zip.sh --delete-missing
```

Backups are stored in `../.repo_update_backups/`; each update prints a JSON
report listing added, updated, deleted, and preserved files.

The Modal entrypoint persists the HF cache and experiment artifacts in named
Volumes and supports the same stages:

```bash
modal run scripts/modal_table1.py --stage all
modal run scripts/modal_table1.py --stage run-sd --pair Q2 --num-shards 8
modal run scripts/modal_table1.py --stage benchmark --pair Q2 --benchmark-prompts 5 --benchmark-max-new-tokens 32
modal run scripts/modal_table1.py --stage benchmark-all --benchmark-prompts 5 --benchmark-max-new-tokens 32
modal run scripts/modal_table1.py --stage benchmark-batch-all --benchmark-prompts 16 --benchmark-max-new-tokens 32
modal run scripts/modal_table1.py --stage package
modal run scripts/modal_table1.py --stage upload --hf-repo-id <namespace>/korean-speculative-decoding-table1-bundle
```

All five pairs can run on one B200 each. In `stage=all`, audits remain
sequential because they update shared metadata, while the five SD jobs and the
five morphology-alignment jobs are submitted concurrently. The benchmark-all
stage also uses one B200 per pair. The upload stage reads `hf_token` only to
create an ephemeral Modal Secret; the token is ignored by git and excluded
from the zip.

The production Table 1 path selects the decoder and numeric inference path per
pair from the pinned YAML. All five pairs use FP32 + eager attention and
reference batch 64. Q1/Q3/M1/G1 use the persistent draft-cache
decoder after a B200 smoke parity check, while Q2 stays on the legacy sequential
decoder because its cached smoke run changed a valid proposal decision. The
global defaults remain `decoder: legacy`, FP16, and SDPA. In all cases, target
continuation IDs must remain exact within the configured numeric path.

The target-only reference pass uses same-length microbatches of 64 for all five
pairs. FP32 + eager is selected uniformly for this throughput-oriented run;
exact token parity against the former FP16 scalar baseline is not a required
gate for this configuration. Target proposal verification remains sequential.

On the pinned B200 runtime, FP32 SDPA was slightly slower than eager for Q1/Q2
batch 64, so eager is pinned uniformly. `flash_attention_2` is accepted by the
loader but is not part of the current Modal image; enabling it would require a
separate B200-compatible FlashAttention build.

For a fast preliminary check on the larger pairs, use `--quick-smoke`; this
checks 5 prompts with at most 32 generated tokens and records `QUICK_ONLY`.
Such an artifact is deliberately blocked from `run-sd`, which requires the
full 200-prompt/128-token smoke gate.
