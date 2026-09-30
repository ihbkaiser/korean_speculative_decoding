# FLORES-200 English→Korean External Replication — Frozen Plan

## Study status and decision criteria

This is a confirmation replication of the saved H2/E2/H3/H4 analysis on FLORES-200 English→Korean `devtest`. It is a new prompt set and requires fresh target-greedy continuations and speculative-decoding traces. No H2/H3/H4 definition will be changed after examining any FLORES output. The analysis scripts will run only after all generation, exact-output parity checks, token alignment, and token-table construction have completed.

The focal H3 result will be classified as `REPLICATED` only when (1) the nominal→particle adjusted CROSS−SPLIT AME is positive in P1, P2, and P3; (2) the pooled FLORES AME is positive and its 95% prompt-clustered CI excludes zero; (3) no pair reverses the original positive direction; and (4) the environment pattern remains directionally consistent, operationalized as a positive nominal→particle minus lexical→lexical AME contrast in each pair where both contrasts are estimable. Every pair-specific p-value need not be below .05. If criteria are not met, the report will use `PARTIALLY_REPLICATED`, `NOT_REPLICATED`, or `INCONCLUSIVE` according to the observed direction, interval support, and estimability, without changing specifications.

## Frozen data, prompts, and hashes

- Source: original official FLORES-200 public release archive linked by the [Meta FLORES-200 README](https://github.com/facebookresearch/flores/blob/main/flores200/README.md), `https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz` (also linked there through `https://tinyurl.com/flores200dataset`). The archive response identifies S3 object version `KJZoZnGvyduN3osrx.67C_UQ7C9oNDic` and last-modified date `2022-07-14`.
- Split/languages: `devtest`, `eng_Latn` source, `kor_Hang` reference. The paired files each contain 1,012 lines; all 1,012 non-empty aligned rows are included in source order. Stable row IDs are 1–1,012 (one-based FLORES line order). The Korean reference is stored only in the provenance manifest and is never tokenized, generated against, aligned, used in filtering, or used for any statistical label.
- Archive SHA-256: `b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6`.
- English devtest SHA-256: `612e9fbe87997617c0fa8fa8929654a4f49b728d96738112c2b86ef6a1d78d88`.
- Korean devtest SHA-256: `540972696230a56e888f1fbd04b7000135c436351f5e467f99ce67fdccd8df0f`.
- Devtest metadata SHA-256: `8edac47f861fcf6b54dad2068c10ba8f741da8b8676861c6bc27fcd9f5483b36`.
- Per-row logical prompt: `Translate the following sentence into Korean.\nEnglish: <English FLORES sentence>\nKorean:`. It uses the same base-completion path as E2: no chat-template wrapper and `add_special_tokens=False`; tokenizer truncation is capped at 128 tokens. The exact rendered prompt SHA-256, tokenized prompt hash, and duplicate audit are saved in the manifest/audit. Duplicate prompts are reported, never dropped.

## Models, revisions, and generation

The three P1/P2/P3 pairs and pinned model revisions are copied from `runs/e2_model_pair_replication/config.json`:

| Size | Model ID | Immutable revision |
|---|---|---|
| 0.6B | `Qwen/Qwen3-0.6B-Base` | `da87bfb608c14b7cf20ba1ce41287e8de496c0cd` |
| 1.7B | `Qwen/Qwen3-1.7B-Base` | `ea980cb0a6c2ae4b936e82123acc929f1cec04c1` |
| 4B | `Qwen/Qwen3-4B-Base` | `906bfd4b4dc7f14ee4320094d8b41684abff8539` |

P1 is 0.6B→1.7B, P2 is 1.7B→4B, and P3 is 0.6B→4B. The historical P3 0.6B draft revision was not recorded in the original run and remains unresolved; the new FLORES P3 therefore uses the immutable 0.6B commit used by the E2 P1/P2 configuration, and the report will state this limitation. No prior Korean Wikipedia continuation is applicable to these prompts. Generate target-greedy continuations on this FLORES set for 1.7B and 4B; share the newly generated 4B target IDs between P2 and P3, while running independent P2 and P3 speculative passes.

Frozen generation: FP16, `model.eval()`, `torch.inference_mode()`, `use_cache=True`, SDPA attention, TF32 enabled, deterministic greedy decoding with no sampling, `K=4`, max prompt tokens 128, max new tokens 128, EOS ID 151643, and the E2 singleton batch size 1 for target generation and teacher-forced scoring. Singleton execution is required by the saved E2 parity benchmark: larger batches did not preserve exact token/teacher predictions. Speculative decoding remains sequential per prompt with the unchanged E2 cached algorithm. Every speculative output must equal its corresponding target-greedy token IDs and stopping position exactly; any mismatch stops the run for diagnosis rather than excluding that row.

## Tokenizer, GPU allocation, and reproducibility

Before loading model weights, run the E2 tokenizer identity/compatibility checks for all pinned revisions: vocabulary size and full token→ID map; special token IDs; BOS/EOS/PAD configuration; tokenizer class; backend serialization, normalizer, and pre-tokenizer; and exact token/offset parity on probes. Reuse one tokenizer only if all identities are compatible and the E1 token/eojeol frequency caches match that tokenizer. Otherwise stop before model loading and report the incompatibility.

Every GPU operation, including model loading/validation, target generation, speculative decoding, and teacher-forced scoring, is restricted to **physical GPU 3** by launching under `CUDA_VISIBLE_DEVICES=3`; inside the process the sole device must be `cuda:0`. Assert the visible device count is one and its reported device is the A100 80GB at physical index 3. Record `nvidia-smi -L`, GPU name, driver/CUDA, PyTorch/Transformers/Kiwi versions, attention backend, dtype, allocated/reserved peak memory, and process device in `flores_config.json` and runtime logs. No new GPU workload will be scheduled on GPUs 0, 1, 2, 4, 5, 6, or 7.

Seed Python, NumPy, and PyTorch with 3090 (the E2 seed), use deterministic greedy generation, and record generation settings. Checkpoint traces and teacher-forced tables atomically by prompt chunks; rerunning the CLI with resume skips only fully committed prompts.

## Frozen morphology, populations, and analyses

Use E2/H2 `src.e2_model_pairs.run_morphology` and `src.h2_analysis` span classification unchanged, with local `kiwipiepy` version 0.24.0. Retain H2 classes `EXACT`, `WITHIN_SPLIT`, `CROSS_MORPHEME`, `CROSS_EOJEOL`, and `KIWI_COMPLEX`; retain the original whitespace-delimited eojeol rule, exact-span/retokenization checks, and exclusions. Primary H2/E2 eligibility is `sd_valid=True`, fragmentation bins exactly `2,3,4,5,6,7,8+`, and `morph_class` in `WITHIN_SPLIT`/`CROSS_MORPHEME`.

The contextual fragmentation table reproduces the H2 prompt-cluster bootstrap by morphology × exact fragmentation bin (2,000 prompt resamples, seed 3090). H2 pair models use cluster-robust SE by `prompt_id`, two-sided tests, `WITHIN_SPLIT` and fragmentation bin `2` references:

- M1: rejection ~ morphology + fragmentation fixed effects.
- M2: M1 + `relative_position + first_token + last_token + proposal_slot + generation_position + token_char_length + eojeol_char_length`.
- M3: M2 + `draft_entropy + target_entropy`.

The H3 primary environment model keeps only `WITHIN_SPLIT` rows and `CROSS_MORPHEME` rows with `num_boundaries_crossed == 1`. CROSS environment is its single crossed boundary. SPLIT environment is assigned by the frozen H3 local-adjacent rule: inspect the immediately adjacent valid eojeol morpheme boundaries around the containing morpheme; use the nearer boundary by character distance, ties to following/right; if none maps, `OTHER`. Keep exactly `NOMINAL_TO_PARTICLE`, `PREDICATE_TO_ENDING`, `ENDING_TO_ENDING`, `LEXICAL_TO_LEXICAL`, `OTHER`; do not relabel or merge sparse categories.

H3 M5 formula per pair is `sd_rejected ~ C(morph_class, Treatment(reference='WITHIN_SPLIT')) * C(morph_environment, Treatment(reference='LEXICAL_TO_LEXICAL')) + C(fragmentation_bin, Treatment(reference='2')) + relative_position + first_token + last_token + proposal_slot + generation_position + token_char_length + eojeol_char_length + draft_entropy + target_entropy + bs(log_token_count, df=4, degree=3, include_intercept=False)`. Use prompt-clustered covariance, delta-method adjusted probabilities and AMEs, 95% normal intervals, two-sided p-values, and Holm correction over the five planned environment contrasts within each pair. The focal contrast is adjusted `P(reject|CROSS) − P(reject|SPLIT)` for `NOMINAL_TO_PARTICLE`, changing only morphology class over observed covariates.

For the pooled result, fit the same H3 pooled model-pair fixed effects, morphology class × environment and model-pair × morphology terms (no three-way interaction), standard M5 controls, and prompt-clustered covariance. Then rerun on the exact three-way intersection of prompt IDs that contribute eligible rows for all P1/P2/P3. Pair-specific analyses remain primary; pooled estimates are secondary.

Targeted H4 checks run only after the primary environment result is produced:

1. H4 L3-style N→P frequency control: H3 M5 restricted to the N→P environment, plus H4's prespecified particle category, nominal POS, nominal/particle character lengths, and cubic four-df splines for log nominal, particle, and eojeol frequencies. Reuse E1/H4 frequency caches only after hash/identity validation against the pinned 4B tokenizer and the same local Wikipedia corpus; missing cache entries are assigned zero only under the H4 documented rule.
2. Token-identity concentration: rank exact `(nominal_surface, particle_surface, token_surface)` patterns among N→P CROSS rows by counts only (never by rejection), and report top-1/5/10 shares of CROSS tokens and rejected CROSS tokens plus raw rejection. Preserve all patterns in output; apply H4's descriptive support note of at least 20 CROSS tokens and 10 prompts only to display labels.
3. Exact common-prompt sensitivity: restrict the pooled environment model and pooled N→P AME to the exact intersection of eligible prompt IDs across all three pair tables; do not substitute a union or a nominal source-row intersection.

No H4 particle-type or geometry decomposition will be fit unless the prespecified H3 N→P result reproduces. Failure will be reported without expanding the analysis.

## Audits and outputs

Include all 1,012 non-empty aligned FLORES source rows. Report duplicate raw/tokenized prompts, language audit (`Hangul letters / all Unicode letters >= 50%` marks generated continuation “predominantly Korean” solely for descriptive QA, never exclusion), SD validity/failure, exact target–SD parity, alignment/exclusions, morphology class and environment support, exact common-prompt intersection, model fit convergence/rank/sparse-cell/separation/extreme-coefficient/cluster diagnostics, frequency-cache identity, and GPU 3 runtime evidence. No output-quality or reference-based filter is allowed.

Write the frozen config and hashes, manifest, prompt audit, resumable traces, H2-compatible token tables, population/fragmentation/environment audits, pair and pooled result tables, H3 forest plots, original-versus-FLORES AME plot, diagnostics, and final interpretation under `runs/flores200_en_ko_replication/` using the paths requested in the task. The main report will keep the original E2/H3 and FLORES samples separate and use association language only; the task shift from free-form Korean Wikipedia continuation to English→Korean translation will be stated as a limitation.
