# Morphology-aware boundary guard: frozen implementation plan

## Question and scope

Test whether a training-free proposal-scheduling rule can reduce end-to-end greedy speculative-decoding time by ending verification blocks before a projected target-side token that crosses one `NOMINAL_TO_PARTICLE` boundary. The run is isolated under `runs/morphology_aware_boundary_guard/`; H2, E2, H3, H4, and FLORES inputs and outputs are read-only inputs.

No model weights or tokenizer files will be modified. All model inference will be launched only with `CUDA_VISIBLE_DEVICES=3`; the process must see one device, `cuda:0`, which is physical GPU 3. `nvidia-smi -L` will be captured immediately before the first GPU launch and into `method_config.json`.

## Existing decoder and alignment path

The E2 decoder actually used for the saved model-pair runs is `src.e2_model_pairs.speculative_greedy_fast` (not a Transformers assisted-generation wrapper). Each prompt starts with a target prefill and a target KV cache. At each cycle, the draft is prefetched on the complete committed prefix, then extends a draft KV cache one greedy token at a time for up to `K=4`. The target checks draft proposals sequentially against its cached greedy next-token logits, stopping at the first mismatch. A mismatch appends the accepted draft prefix plus the target argmax correction and advances the target cache with that correction. A fully accepted block appends all proposals; the next draft cycle is re-prefilled from the committed output. There is no speculative-sampling/residual-distribution implementation in this repository; existing E2 and FLORES generation is greedy.

Online projection will reuse `src.alignment.token_ids_to_character_offsets`, `src.h2_analysis._span_tokenizer_ids`, H2's whitespace-delimited eojeol extraction and exact span eligibility, `src.h2_analysis._classify_token`, and H3's `boundary_type`/coarse POS taxonomy. A candidate is guardable only when full-prefix tokenization round-trips exactly, the target-side token ID maps exactly to its character span, the token is not special or prompt-crossing, Kiwi spans cover it without ambiguity, and exactly one internal morpheme boundary is crossed. Since all three pinned model tokenizers were previously verified identical, the proposal IDs are shared; text and target-token spans are nevertheless explicitly reprojected and exactness-checked rather than assumed to align.

## Frozen models, workloads, prompts, and decoding

Model pairs and pinned revisions come from `runs/e2_model_pair_replication/config.json`:

| Pair | Draft | Target |
|---|---|---|
| P1 | Qwen/Qwen3-0.6B-Base @ `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | Qwen/Qwen3-1.7B-Base @ `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` |
| P2 | Qwen/Qwen3-1.7B-Base @ `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` | Qwen/Qwen3-4B-Base @ `906bfd4b4dc7f14ee4320094d8b41684abff8539` |
| P3 | Qwen/Qwen3-0.6B-Base @ `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` | Qwen/Qwen3-4B-Base @ `906bfd4b4dc7f14ee4320094d8b41684abff8539` |

P3's historical Wikipedia draft revision is unresolved in the original run. This experiment uses the pinned E2/FLORES 0.6B revision above for its new P3 runs and will preserve that limitation in the report.

* **Wikipedia:** exact pretokenized E2 prompt IDs and hashes from `runs/e2_model_pair_replication/prompt_ids.jsonl` and `prompt_hashes.json`; 1,000 prompts in source order.
* **FLORES:** all 1,012 included prompt IDs and rendered prompt hashes from `runs/flores200_en_ko_replication/data_manifest.csv`; reuse the exact English→Korean prompt template and tokenizer path recorded in `flores_config.json`. Korean references remain provenance-only.
* Both workloads keep their saved `max_new_tokens=128`, greedy/no-sampling mode, `K=4`, EOS `151643`, `torch.float16`, SDPA, `use_cache=True`, and seed `3090`. The tokenizer files are not modified. No chat template is used in either existing path.

## Policies and fixed controls

1. `TARGET_ONLY`: cached target greedy decoding, timed as a speed reference.
2. `FIXED_K_SD`: unmodified E2 cached-path greedy algorithm, `K=4`.
3. `NP_BOUNDARY_GUARD`: produce the ordinary draft block, project it and parse committed generated text plus candidates. Find the earliest exact single-boundary `NOMINAL_TO_PARTICLE` target proposal at 1-based internal slot `j`, `1 <= j < min(K, remaining)`. Verify proposals `1..j-1` through the unchanged baseline path. If that prefix rejects, perform the normal correction path. If fully accepted, take the target greedy next token from the target's current next-token distribution, append it, advance the target cache, and rebuild the draft state from committed output at the next cycle.
4. `ALL_MORPH_BOUNDARY_GUARD`: same schedule, guarding the earliest exact single-boundary CROSS token of any H3 coarse boundary type.
5. `RANDOM_MATCHED_GUARD`: use the same morphology detector/eligible blocks as NP; choose an internal truncation slot deterministically from the Wikipedia-derived NP slot distribution with seed `20260930`, conditional on that block's valid internal slots. Thus the active opportunity blocks match NP within a decoding pass while the cut location is randomized.
6. `ENTROPY_MATCHED_GUARD`: guard the earliest internal draft proposal slot whose online draft entropy is at or above a per-pair threshold. Select the cutoff from a fixed 201-quantile grid of existing Wikipedia traces only. Because scalar entropy cutoffs can have discontinuous activation rates, calibrate and freeze deterministic slot-specific thinning probabilities on Wikipedia to match the NP activation count and slot distribution as closely as the threshold-eligible slots permit. Apply the same thresholds/probabilities on FLORES without tuning.

The initial implementation constructs the full normal proposal block and only then truncates. No streaming proposal optimization is planned before the baseline implementation passes correctness and performance gates.

The repository has no stochastic speculative decoder or residual-sampling formulas to alter. `correctness/sampling_logic_audit.md` will document this source audit: sampling behavior is not evaluated or claimed; only the frozen greedy path is in scope, and its acceptance/correction code is reused unchanged for every unguarded prefix and every prefix rejection.

## Frozen offline analysis, prompt subsets, and stages

**Stage 1:** CPU-only audit of existing per-round draft proposals and H3/FLORES morph spans. Join on prompt and output position, reconstruct each draft proposal block, and run the same online text projection/classifier. Report detector-valid NP and any-boundary candidates, their internal slot, positions saved from full-block verification, and whether the baseline rejects before, at, or after the cut. Distinguish detected candidates from actually reachable opportunities (the safe prefix must be fully accepted).

**Correctness suite:** 100 prompts per workload, fixed as the first 100 IDs of the precomputed pilot manifest, for all three pairs. Reuse target-only greedy reference IDs already saved for the same pinned target revision; verify those artifacts' prompt IDs/revisions before use. Run FIXED_K, NP guard, and ALL_MORPH guard; check exact token IDs and stop positions. `guard_decision_audit.csv` records the projected span, candidate token ID/text, eojeol, Kiwi surface/POS/span sequence, crossed POS transition, H3 boundary class, source exactness flags, chosen slot, and reason for every guard decision or fallback.

**Pilot:** choose 200 prompts from each workload with NumPy seed `3090`, sample without replacement, then retain source order; save the complete manifest before model inference. The first 100 form the correctness subset. Benchmark target-only plus every feasible policy for every pair/workload, three repeats each, identical prompt order. Warm each resident model/GPU before timing. Synchronize CUDA before and after each timed repeat; include draft computation, target verification, target re-synchronization, token projection, Kiwi/controller time, and per-prompt Python bookkeeping in the timed interval. Collect prompt-level elapsed time for paired prompt bootstrap intervals. Stage 4 is allowed only when correctness passes and NP is not more than 2% slower than FIXED_K in at least one pair/workload cell; otherwise record the prespecified negative pilot and stop.

**Full benchmark:** if the pilot gate passes, use every existing prompt in both workloads with the same variants, order, and at least three repeats. Do not pool Wikipedia and FLORES speed results. No model or morphology tuning follows inspection of FLORES timing.

## Measurements and decision rule

Each workload × pair × variant × repetition row will include synchronized wall time, prompts and generated target tokens, target tokens/s, seconds/target token, speed ratios to target-only and FIXED_K, accepted tokens/verify call, rejection rate, target calls/output token, draft proposals/output token, verified positions/output token, guard activation/position rates, invalid/ambiguous morphology fraction, detector time, re-synchronization time, and output equality. Report repeat-level mean/median plus prompt-cluster bootstrap 95% CIs; preserve raw per-prompt timings and guard events.

Classify `POSITIVE` only if greedy equality passes, measured wall time includes morphology overhead, NP has a positive bootstrap lower bound over FIXED_K in at least one workload and at least two pairs, RANDOM_MATCHED does not match/exceed that gain, and ALL_MORPH does not provide an equally convincing explanation. Reduced rejection without end-to-end speed is `NEGATIVE`. If the experiment cannot meet the gate or the intervals remain inconclusive, report `INCONCLUSIVE` with the actual completed stage and evidence.

## Planned artifacts

The run will contain `method_config.json`, GPU/environment and source hashes, this plan, stage-1 opportunity/validation tables, greedy/sampling/guard correctness audits, a fixed prompt manifest, pilot and (if gated) full benchmark tables, per-prompt metrics, guard events, throughput/overhead figures, diagnostics, and `method_summary.md` under the directory structure specified in the user request. GPU runs will be resumable/checkpointed, use a dedicated tmux session, and never change the source H2/E2/H3/H4/FLORES data.
