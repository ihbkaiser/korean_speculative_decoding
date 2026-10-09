# Full Table 1 on the company B200

Selected workflow: `configs/table1_fast_b200.yaml`, batch 256, BF16/SDPA,
greedy K=4, max prompt/output 128/128. No scalar reference pass, scalar retry,
automatic batch reduction or OOM fallback. B256 passed all five bounded B200
pair smokes. B2048 only passed Q1 with a special allocator, for a small measured
throughput gain, and is not the full-run default.

## Start all five pairs

Use the existing company Python environment with B200-compatible PyTorch,
the frozen dataset and pinned model snapshots. `model_paths` in
`configs/table1_pipeline.yaml` must point to directories containing model
weights/configs on this machine. Do not resample the prompt pool.

```bash
cd /workspace/storage-shared/nlp/tungdd11/korean_speculative_decoding
git pull --ff-only origin main
# Only if dependencies are missing; retain your working CUDA/PyTorch installation:
python3 -m pip install -r requirements.txt
tmux new -s table1_b200
```

Inside tmux:

```bash
mkdir -p logs
set -o pipefail
PYTHONUNBUFFERED=1 NUM_SHARDS=1 bash run_table1.sh all 2>&1 | tee -a logs/table1_fast_b200_full.log
```

Detach with Ctrl-b, then d. Reattach using `tmux attach -t table1_b200`.
Do not start a second writer against the same checkpoint directory. If
`git pull` reports local conflicting edits, preserve/reconcile them; do not
reset or delete experiment artifacts to force an update.
If the company checkout has an older **untracked** `run_table1.sh`, Git may
refuse to overwrite it. Rename that file to a local backup, then repeat the
pull; the repository now supplies the selected launcher itself.

`all` runs Q1/Q2/Q3/M1/G1 sequentially on the GPU, starts each pair's CPU
morphology after its SD stage, waits for alignments, then builds Table 1.
Prompt counts are **20k/40k/20k/20k/20k**, or **120k pair-prompts**, not 20k total.
Cross-model comparisons use the common 20k; Q2 additionally gets a common-20k
analysis bundle. TQDM updates after checkpointed batches and shows resumed
counts. Both stdout and stderr are appended to disk; Python is unbuffered.

CPU alignment overlaps GPU work and can run concurrently for multiple pairs.
Full-size CPU RAM and disk requirements were not established by the bounded
smokes. Allow substantial headroom: JSONL retains every proposal, and final
alignment/join materialization uses host RAM. For a memory-constrained host,
run pairs with alignment sequentially instead of the overlap launcher:

```bash
set -euo pipefail
for pair in Q1 Q2 Q3 M1 G1; do
  bash run_table1.sh "$pair"
done
python3 scripts/table1_pipeline.py --config configs/table1_fast_b200.yaml --root . build-table1 2>&1 | tee -a logs/table1_fast_b200_build.log
```

## Resume

Rerun the exact same launch command with the same config, models, batch size
and `NUM_SHARDS`. Completed SD/alignments are skipped; unfinished SD resumes
from saved prompts. Every completed microbatch flushes its full records to
`progress.jsonl`; an interrupted in-flight batch can be regenerated. A truncated
final JSONL record is preserved separately as `*.incomplete-tail-*`; corruption
before the final record is fatal. Changing inference settings cannot silently
reuse incompatible fast checkpoints.

Fast artifacts live separately from historical strict `runs/table1` results.
Do not copy scalar checkpoints into the fast directory. Set `NUM_SHARDS` only
before the first launch and keep it fixed; with multiple shards every shard
is now included in the deferred CPU alignment loop.

## What is dumped, without sampling away events

| Location | Contents |
|---|---|
| `logs/table1_fast_b200_full.log` | Optional master console capture from the launch command above, append-only |
| `logs/table1_fast_b200_<PAIR>_sd.log` | SD stdout/stderr, loading, TQDM, throughput, ETA, errors (`all`) |
| `logs/table1_fast_b200_<PAIR>_align.log` | CPU alignment stdout/stderr for every shard (`all`) |
| `logs/table1_fast_b200_<PAIR>.log` | Combined pair-stage stdout/stderr when launching a single pair |
| `logs/table1_fast_b200_build.log` | Final table aggregation stdout/stderr |
| `runs/table1_fast_b200/<PAIR>/launches/*.json` | Per-launch history: resolved config, effective overrides, local model paths, runtime versions/CUDA/GPU/git commit; only allowlisted environment options |
| `metadata/table1_main_<PAIR>.json` | Latest per-pair launch manifest; history remains in `launches/` |
| `runs/table1_fast_b200/<PAIR>/shards/shard-*/progress.jsonl` | Every completed prompt's reference, all proposal events including invalidated proposals, and resume identity |
| Same shard: `references.parquet` | Prompt IDs/text, generated continuation IDs/text, split identity and verification provenance |
| Same shard: `sd_events.parquet` | All proposal-level events/statistics, not just accepted tokens or summary averages |
| Same shard: `tokenizer_compatibility.json` | Tokenizer classes, backend SHA256, vocabulary/special-token/offset compatibility and basic probes; not a substitute claim for the separate 1,000-string audit |
| Same shard: `aligned_tokens.parquet` | Target-path visible/eligible/excluded token rows, eojeol/morpheme spans, boundary classes/geometry/local WITHIN assignments and exclusion reasons |
| Same shard: `run_metadata.json`, `alignment_metadata.json`, completion markers | Frozen split/model/config identity, stage accounting and completion state |
| `results/table1_fast_b200/table1_<PAIR>_*.parquet` | Consolidated references, aligned tokens and proposal-analysis joins |
| `results/table1_fast_b200/table1_Q2_common20k_*.parquet` | Q2 common-set views for fair cross-pair comparison |
| `results/table1_fast_b200/table1_models_data_alignment.{csv,md,tex}` | Final Table 1 |
| `results/table1_fast_b200/table1_exclusion_breakdown.csv` | Explicit alignment exclusions and denominators |
| `results/table1_fast_b200/boundary_support_by_pair.csv` | Full fine-boundary counts/support, including unsupported transitions |
| `results/table1_fast_b200/table1_build_metadata.json` | Reconciled totals, missing shards and verification source |

Proposal fields include doc/pair/model identity, prompt length,
`generated_token_position` (the document's `generation_position`), block/slot,
token ID/text, accepted/rejected/invalidated flags, draft and target entropy,
proposed-token logprobs, target top1 ID/logprob, proposed-token target rank and
margin. Invalidated proposals are preserved but must not be counted as
independent evaluated rejections.

The historical stored `target_margin` is **top1 minus proposed** (nonnegative).
The signed margin in NAACL section 14 is the negative of this field; equivalently
compute `target_logprob_of_proposed_token - target_top1_logprob`. Rank bins and
`target_minus_draft_logprob` are derivable offline from the saved raw values;
no new model inference is needed. `later_proposals_invalidated` is a per-proposal
flag; counts of wasted proposals can be reconstructed by doc/block/slot.

## Validation contract and completion

The user-selected no-fallback mode does **not** claim independent scalar parity.
Reference rows identify `block_target_verified_no_scalar_reference`,
`scalar_parity_validated=false`, `independent_target_reference=false`.
The legacy `exact_sd_target=true` column denotes block-target verification in
this mode, not the original strict oracle. Keep this distinction in manuscripts.

Table 1 is usable only when all five pair rows and `table1_build_metadata.json`
show `COMPLETE`, expected prompt sets reconcile, and alignment exclusions/support
are reproducible from raw rows. Reaching 100% in the SD bar is not equivalent
to CPU alignment/table completion. An SD/CPU error must be investigated; logs
and progress remain on disk, never silently replaced with a fallback result.

Keep logs/artifacts backed up on experiment storage. Full raw runs, account
billing snapshots, transfer ZIPs and credentials are not auto-pushed to GitHub;
code, runbooks and selected small bounded benchmark/smoke evidence are versioned.
