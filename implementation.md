# Implementation plan

## Research question

Test whether, among Korean whitespace-delimited eojeols with the same LLM-token
fragmentation, greater disagreement between morpheme and tokenizer boundaries
is associated with (1) higher speculative-decoding rejection and (2) higher
draft/target greedy disagreement. The pilot uses Korean Wikipedia (`wikimedia/wikipedia`,
`20231101.ko`) and the Qwen3 base models on one CUDA GPU.

## Fixed experiment settings

- Draft: `Qwen/Qwen3-0.6B-Base`; target: `Qwen/Qwen3-4B-Base`.
- Both models run in evaluation mode with FP16 weights and activations, batch
  size 1, deterministic greedy decisions, and speculative block size K=4.
- Prompt token cap: 128; continuation cap: 128. The 20-prompt pilot is the
  default; `--limit 5`, `--limit 20`, or `--limit 1000` selects the row count.
- Dataset order and prompt extraction are deterministic. Seeds and resolved
  configuration are saved in each run directory.
- The target-only greedy continuation is the reference. The custom speculative
  decoder must produce the exact same token IDs, including a generated EOS when
  present, or stop with an error before writing a completed run.

## Repository layout and entry points

- `configs/pilot.yaml`: model IDs, dataset/config, decoding parameters, seed,
  output paths, and expected hardware/precision.
- `src/data.py`: load and deterministically select non-empty Wikipedia article
  prompts, save a local JSONL cache, and prepare capped token IDs.
- `src/models.py`: load models/tokenizers, compare tokenizer identity and
  behavior, and provide entropy/log-probability helpers.
- `src/speculative_decoding.py`: target-only greedy reference, custom K-token
  draft proposal/target verification, event records, and exact-output check.
- `src/morphology.py`: split generated text into eojeols and obtain Kiwi
  morphemes with character spans.
- `src/alignment.py`: tokenizer offsets, morpheme/tokenizer boundary F1,
  token-to-character and token-to-eojeol mappings.
- `src/analysis.py`: per-fragmentation-bin summaries, logistic regression,
  teacher-forced disagreement analysis, and figures.
- `scripts/check_env.py`: report Python/package/GPU/storage availability and
  check that the two requested tokenizers are compatible.
- `scripts/prepare_data.py`: materialize the deterministic prompt cache.
- `scripts/run_pilot.py`: run target reference, speculative decoding,
  teacher-forced scoring, morphology/alignment extraction, and parquet output.
- `scripts/analyze.py`: analyze one run and write CSV summaries and figures.
- `tests/`: offline unit tests using small deterministic fake tokenizers/models.

## Tokenizer compatibility gate

Load both model tokenizers as fast tokenizers. Compatibility requires equal
tokenizer implementation class, exact token-to-ID vocabulary, exact added-token
vocabulary, byte-for-byte identical fast-tokenizer backend serialization
(model, normalizer, and pre-tokenizer), the same special-token strings and IDs,
and identical `input_ids` and `offset_mapping` for a deterministic multilingual
probe set. Save the compatibility report in the run directory. Any mismatch or
unavailable offset mapping is a hard failure before model generation. Use the
target tokenizer for both models after the gate passes.

## Data and prompt preparation

Read the configured dataset split in its published order. For each non-empty
article, use its `text` field as one prompt, strip outer whitespace, and retain
the first requested number of prompts. Tokenize without added special tokens,
truncating on the right to 128 tokens. Save the source row index and prompt text
in JSONL so a run can be reconstructed without re-downloading the dataset.

## Custom speculative decoding and event semantics

For each prompt, first run target-only greedy decoding to at most 128 new tokens.
Then run custom greedy speculative decoding. Each round has at most four draft
proposals. For proposal j, record the draft token, draft-token log probability,
draft entropy, target probability assigned to that token, target entropy, the
target greedy token at that position, proposal position within the round, and
zero-based candidate output position. Check proposals sequentially through the
target KV cache, using the same one-token cached forward path as target-only
greedy decoding; this keeps argmax decisions identical even for near-tied
logits. Accept the longest prefix for which each draft token equals the target
argmax. At the first mismatch, mark that proposal as the first rejection, emit
the target argmax, and discard the remaining proposals in the round. Discarded later proposals are logged with
`invalidated_after_first_rejection=true` and a null `rejected` value because
they are not verified on the valid target path; their target statistics are
explicitly conditional on the hypothetical draft prefix. Every round stores accepted
prefix length and first-rejection position. EOS terminates generation when it
is emitted by the target path. These semantics make all proposal decisions
auditable while preserving greedy target output exactly.

The run aborts on any mismatch between target-only and speculative token IDs.
Rows in `sd_events.parquet` are proposal-level rows; run/prompt/round identifiers
and the exact IDs are retained. Fields include `accepted`, `is_first_rejection`,
`invalidated_after_first_rejection`, `accepted_prefix_length`,
`first_rejection_output_position`, `draft_logprob`, `target_logprob`,
`draft_entropy`, `target_entropy`, `proposal_position`, and
`output_token_position`.

## Teacher forcing

Score the target-generated reference continuation under both models in teacher
forcing. For every generated token position, compute draft and target next-token
argmax, log probability of the actual target token, and entropy. Define
`disagreement` as draft argmax != target argmax. This analysis is separate from
speculative proposal-path events and covers all reference continuation tokens.

## Korean morphology and alignment

For each target-generated continuation, decode the full prompt plus reference
IDs and use fast-tokenizer offsets. Check whether encoding the decoded text
round-trips to the non-special input IDs. If generation stops inside a UTF-8
byte sequence and the round trip fails, use incremental prefix decoding to
associate the changed suffix with the completing token, and record the fallback
prompt IDs in run metadata. Split the continuation on Unicode whitespace; each
non-whitespace span is one eojeol. Analyze each eojeol with Kiwi, retaining
morpheme character spans. Tokenize the eojeol independently without special
tokens and retain token character spans. Exclude the terminal eojeol boundary
from both internal-boundary sets.

- `morpheme_count`: Kiwi morpheme count.
- `llm_token_count`: tokenizer pieces for the eojeol.
- `fragmentation`: `llm_token_count / 1`.
- `morpheme_boundaries` and `tokenizer_boundaries`: sorted internal character
  offsets within the eojeol.
- `misalignment`: `1 - boundary_F1(morpheme_boundaries,
  tokenizer_boundaries)`. Empty/empty boundary sets have F1=1; only one empty
  set has F1=0. Boundary matching is exact character-offset matching.
- `fragmentation_bin`: token count 2, 3, 4, or 5+; token counts below 2 remain
  in the eojeol table but are omitted from the specified primary bins.

Map each reference output-token position to its eojeol by character-span
overlap. Proposal events join to the reference eojeol at their candidate output
position; positions beyond EOS/no eojeol remain null and are excluded from
morphology-conditioned models. Store one row per eojeol occurrence, not just
unique surface forms, in `eojeols.parquet`.

## Primary analysis

Report counts, mean rejection rate, and mean greedy-disagreement rate within
fragmentation bins 2, 3, 4, and 5+. Report the misalignment slope for rejection
and disagreement within each bin (with sample counts and p-values when
estimable). Cluster standard errors by prompt to account for repeated token
observations. Fit the proposal-level logistic model:

```text
rejected ~ misalignment + fragmentation + draft_entropy
           + target_entropy + proposal_position
```

Also fit/report teacher-forced logistic disagreement against the same
morphology/fragmentation predictors where data allow. Save model summaries and
analysis inputs as CSV/text alongside plots. Small or single-class bins are
reported as not estimable instead of silently producing a misleading fit.

## Outputs and failure handling

Each run writes `runs/<run_id>/config.yaml`, compatibility and environment
metadata, prompt/reference metadata, `sd_events.parquet`,
`teacher_forced_tokens.parquet`, and `eojeols.parquet`. Analysis writes figures
under `figures/<run_id>/` and machine-readable summaries under the run folder.
Writes use a temporary file followed by rename where practical. A run is marked
complete only after every prompt passes exact-output verification and all
parquet files are readable.

## Required tests and execution boundary

Unit tests cover tokenizer compatibility pass/fail, target-greedy versus custom
speculative output, first-rejection and accepted-prefix logging, boundary F1,
and token-offset character round trips. The requested validation sequence is:

1. environment check;
2. unit tests;
3. 5-prompt smoke test;
4. 20-prompt pilot.

The 1000-prompt run is documented as a command only and is not run during this
implementation task.
