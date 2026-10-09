# NAACL Experiment Execution Order & Fill Guide
## Large-Scale Morphological Boundary Risk for Speculative Decoding in Korean

> **Purpose.** This is the execution-order rewrite of the previous experimental draft. It is organized by **what to run first, what each run produces, which table/figure it fills, and when a stage is considered complete**.
>
> **Important.** The paper-facing section order and the execution order are not identical. This file prioritizes execution dependencies so that every artifact can be filled incrementally without waiting for all experiments to finish.
>
> **Frozen scope for the deadline.** One large Korean web dataset, five draft→target pairs across three model families, one high-support Qwen reference configuration for the deepest analyses, and mostly offline robustness after the main inference runs.

---

> **Selected B200 execution mode (2026-10-10).** The user selected the separate
> BF16/SDPA microbatched workflow, batch 256, K=4, with target-block verification
> and **no independent scalar reference generation or scalar fallback**.
> Run `bash run_table1.sh all` on the company B200; see
> [the full-run and artifact runbook](docs/TABLE1_B200_RUNBOOK.md).
> Frozen splits remain Q2=40k, Q1/Q3/M1/G1=20k common each. All proposal,
> continuation, morphology, exclusion and support artifacts below remain required.
> The scalar-parity requirements and fallback description in the original sections
> below describe the **historical strict protocol**, not a validation claim for
> this selected mode. Fast metadata must retain `scalar_parity_validated=false`
> and `independent_target_reference=false`; do not present its completion markers
> as evidence of independent target-only/scalar equality. Five-pair B256 bounded
> smoke passed; full-data results are not yet established.

> **Current execution decision (2026-10-08).** The 5kâ€“10k Q2 pilot is
> intentionally skipped. The final Table 1 path runs each of the five frozen
> pairs directly on its complete prompt split. This is a declared scope
> decision, not evidence that the pilot was completed. Use the repository's
> `run-table1` command once per pair, keep `--num-shards` fixed for a pair, and
> run `build-table1` only after all five pairs have completed and aligned.

> **Artifact contract.** In addition to raw shard files, the implementation
> persists full-smoke event/reference parquet files, resumable progress and
> metadata, tokenizer hashes, environment metadata, consolidated
> eventâ€“alignment proposal tables, and a Q2 common-20k view. The target-only
> continuation is an SD correctness oracle, not linguistic gold; morphology
> remains automatic Kiwi alignment and scientific claims are observational
> unless separately validated.

> **Production-only rerun.** When audit/smoke has already been completed or is
> intentionally bypassed, use `run-table1-main` with explicit draft/target
> model paths. It runs the frozen production splits and per-prompt exact
> parity, but does not rerun the audit or the 200-prompt smoke gate.

# 0. Frozen Experimental Scope

## 0.1 Dataset

Use one large Korean web corpus:

```text
Dataset: FineWeb2 Korean / kor_Hang
Total frozen prompt pool: 40,000 raw Korean documents/prompts
Common set: 20,000 prompts shared by all five pairs
Extra set: 20,000 prompts used only by Q2
```

Freeze the exact prompt IDs before model inference.

```text
P_common = first 20,000 frozen prompt IDs
P_extra  = remaining 20,000 frozen prompt IDs
```

The same raw-text prompt IDs in `P_common` must be used for every model family. Token truncation is allowed to differ across tokenizers, but prompt **selection** must not differ.

### Required data files

```text
data/prompts_40k.parquet
data/common_20k_ids.txt
data/extra_20k_ids.txt
metadata/dataset_revision.json
metadata/data_sampling_config.yaml
```

Recommended frozen fields:

```text
doc_id
raw_text
normalized_text
text_hash
sample_order
is_common_20k
is_extra_20k
```

---

## 0.2 Model pairs

Run exactly these five core configurations, subject to tokenizer-compatibility validation:

| ID | Family | Draft → Target | Prompts | Role |
|---|---|---|---:|---|
| **Q1** | Qwen3 Base | 0.6B → 1.7B | 20k common | small-gap Qwen comparison |
| **Q2** | Qwen3 Base | 0.6B → 4B | 40k total | high-support reference configuration |
| **Q3** | Qwen3 Base | 4B → 8B | 20k common | larger-capacity Qwen comparison |
| **M1** | Ministral Base | 3B → 8B | 20k common | independent model family |
| **G1** | Gemma pretrained | 4B → 12B | 20k common | second independent model family/tokenizer |

`Q2` receives 40k prompts only to obtain enough support for fine-grained boundary analysis. For cross-model comparisons, use only the same `P_common` 20k prompts for all five pairs.

---

## 0.3 Main decoding configuration

```text
decoding: greedy speculative decoding
speculative block size: k = 4
max prompt length: 128 model tokens
max continuation: 128 target tokens
sampling: disabled
```

For every completed prompt:

```text
SD final token IDs == target-only greedy token IDs
```

must hold exactly.

If this invariant fails, downstream rejection analysis is invalid and the run must be debugged before proceeding.

The production implementation keeps this invariant without a global
batch-size-1 fallback. Target references are generated in same-length
microbatches of 64. Cached speculative decoding verifies each `k=4` proposal
block in one target forward on the B200 and rolls back the target cache after
the first rejection. When a batch kernel produces a rare near-tie disagreement,
that prompt alone is regenerated with singleton target greedy decoding; the
fallback token IDs, first-difference position, and source are persisted in the
reference row and shard metadata. A mismatch between singleton target greedy
and the speculative output remains fatal.

---

## 0.4 Core morphology definitions

For every visible generated token:

- **CROSS:** overlaps material from at least two morphemes.
- **WITHIN_SPLIT:** lies inside one morpheme but covers only part of that morpheme.
- **EXACT:** coincides with one complete morpheme.
- **AMBIGUOUS/OTHER:** exact surface alignment cannot be established.

The primary boundary-specific analysis uses only CROSS tokens crossing **exactly one** morpheme boundary.

A fine-grained boundary is:

\[
b = POS_L \rightarrow POS_R.
\]

Examples:

```text
NNG → JKS
NNG → JKO
NNP → JX
VV  → EC
VV  → EF
VA  → EF
EP  → EF
EP  → EC
```

Do not pre-select nominal→particle as the primary boundary class.

---

# 1. Master Artifact Checklist

This is the order in which artifacts should become complete.

| Order | Artifact | What it answers | Requires new LLM inference? | Completion status |
|---:|---|---|---|---|
| 1 | **Table 1** — Models/Data/Alignment | Is the pipeline/data usable? | Yes | [ ] |
| 2 | **Figure 1** — Boundary Support Matrix | Which boundaries have enough data? | No beyond Q2 main run | [ ] |
| 3 | **Table 2A** — Coarse Misalignment | Does scalar alignment explain rejection? | No | [ ] |
| 4 | **Table 2B** — Global CROSS vs WITHIN | Does mismatch direction matter globally? | Main five-pair runs | [ ] |
| 5 | **Figure 2** — Full Boundary-Risk Landscape | Which boundaries are actually risky? | No beyond Q2 main run | [ ] |
| 6 | **Table 3** — Boundary-Specific Effects | Exact numbers behind Figure 2 | No | [ ] |
| 7 | **Figure 3** — Boundary × Model-Pair Heatmap | Does boundary risk transfer across models/families? | Main five-pair runs | [ ] |
| 8 | **Table 4** — Cross-Model Stability | How stable is the ranking/sign? | No | [ ] |
| 9 | **Figure 4** — Exposure vs Conditional Risk | Which boundaries are common *and* risky? | No | [ ] |
| 10 | **Figure 5** — Token-ID Removal Ablation | Is morphology just a few fused tokens? | No | [ ] |
| 11 | **Table 5** — Robustness Summary | Is the landscape stable to analysis choices? | Mostly no | [ ] |
| 12 | **Figure 6** — Pseudo-Boundary Falsification | Is the signal linguistic rather than arbitrary geometry? | No | [ ] |
| 13 | **Figure 7** — Target Rank / Margin | What distributional disagreement accompanies risk? | No if logged | [ ] |
| 14 | **Figure 8** — Block Size / Speculative Waste | Does risk matter for SD cost? | Yes, small extra run | [ ] |

Artifacts 1–12 are the core analysis. Figures 7–8 are lower priority if the deadline becomes tight.

---

# 2. Stage 0 — Freeze Data, Configs, and Reproducibility Metadata

## Goal

Before generating a single long model run, make the experiment exactly reproducible.

## Run

Prepare the 40k FineWeb2 Korean prompt pool using deterministic preprocessing:

1. normalize Unicode consistently, preferably NFC;
2. strip leading/trailing whitespace;
3. reject empty/near-empty documents;
4. require sufficient Korean/Hangul content;
5. remove exact duplicates by normalized-text hash;
6. use a fixed random seed;
7. freeze the first 40k accepted samples and never re-sample per model.

Do not aggressively clean text unless the existing codebase already has a justified cleaner. The evaluation corpus should remain representative of Korean web text.

## Save

```text
configs/data.yaml
configs/models.yaml
configs/decoding.yaml
configs/morphology.yaml
configs/analysis.yaml

metadata/dataset_revision.json
metadata/model_revisions.json
metadata/tokenizer_hashes.json
metadata/environment.txt
```

Record:

```text
random seed
dataset revision/hash
model revisions
Transformers version
PyTorch version
CUDA version
GPU type
Kiwi version
code git commit
```

## Validation

- [ ] exactly 40k unique frozen documents;
- [ ] exactly 20k common IDs;
- [ ] exactly 20k extra IDs;
- [ ] common and extra sets do not overlap;
- [ ] text hashes do not duplicate across sets;
- [ ] manually inspect ~100 randomly sampled documents.

## Artifact filled

Nothing final yet. This stage provides the denominators and reproducibility metadata used by **Table 1**.

## Done when

The frozen data files can be consumed by all five model pairs without further sampling decisions.

---

# 3. Stage 1 — Tokenizer Compatibility Audit and SD Smoke Tests

## Goal

Determine whether all five proposed draft→target pairs can support exact token-level speculative decoding.

## Experiment 1A — Tokenizer compatibility

For every pair, compare:

```text
vocabulary size
full token-ID -> token-string mapping
special token IDs
tokenizer config/hash
```

Then sample at least 1,000 Korean strings and verify:

```text
encode_draft(s) == encode_target(s)
decode_draft(ids) == decode_target(ids)
```

### Output

```text
audit/tokenizer_compatibility.csv
```

Suggested columns:

```text
pair
vocab_size_match
mapping_match
special_token_match
random_encode_match_rate
status
```

Any pair failing exact token-ID compatibility is not valid for the standard SD implementation used here.

---

## Experiment 1B — SD correctness smoke test

Run each surviving pair on **200 common prompts** with `k=4`.

Verify:

1. speculative output equals target-only greedy output exactly by token ID;
2. EOS behavior is identical;
3. rejection accounting is correct;
4. proposal slot and block ID are logged;
5. later proposals invalidated by an earlier rejection are not counted as independent rejection events;
6. entropy/logprob values are finite;
7. visible token strings reconstruct generated text correctly.

### Required proposal-level fields

```text
doc_id
pair_id
draft_model
target_model
prompt_token_count
generation_position
proposal_slot
block_id
token_id
token_text
accepted
rejected
invalidated_by_earlier_rejection
draft_entropy
target_entropy
draft_logprob_of_proposed_token
target_logprob_of_proposed_token
target_top1_token_id
target_top1_logprob
target_rank_of_proposed_token
target_margin
later_proposals_invalidated
```

### Output

```text
validation/<pair>_smoke.parquet
validation/<pair>_report.json
```

## Stop rule

Do not start a long run for any pair until SD/target greedy equality is 100% on the smoke test.

## Done when

You have a green/red compatibility + correctness status for each of Q1/Q2/Q3/M1/G1.

---

# 4. Stage 2 — Q2 Early Run (5k–10k) and Provisional Table 1 / Figure 1

> **Current deadline status: SKIPPED by decision.** Do not execute the
> 5k–10k Q2 pilot or use a provisional Table 1 as evidence. The production
> path goes directly from the frozen prompt pool and full compatibility/smoke
> gate to the complete Q2 40k run through `run-table1 --pair Q2`. This section
> remains as the original contingency procedure and its checks are retained as
> optional diagnostics only.

## Goal

Before spending full compute, verify that the larger dataset actually gives broad morphological boundary coverage.

## Run

Run:

```text
Q2: Qwen3 0.6B → 4B
first 5k–10k frozen prompts
k = 4
max prompt = 128 tokens
max continuation = 128 tokens
```

Store all proposal-level metadata and final target continuation token IDs/text.

---

## Morphological alignment

Run Kiwi on the generated continuation and align visible token spans to morpheme surface spans within each eojeol.

For every visible target-path token, store:

```text
eojeol_text
eojeol_char_len
eojeol_token_count
token_index_in_eojeol
relative_position_in_eojeol
is_first_token
is_last_token
morpheme_surfaces
morpheme_pos_tags
morpheme_char_spans
alignment_class
crossed_boundary_count
alignment_status
alignment_exclusion_reason
```

For single-boundary CROSS tokens additionally store:

```text
left_pos
right_pos
fine_boundary
coarse_boundary
left_chars_in_cross_token
right_chars_in_cross_token
boundary_position_ratio
right_morpheme_fully_absorbed
```

For WITHIN_SPLIT tokens assign the nearest internal morpheme boundary:

```text
nearest_boundary_index
nearest_left_pos
nearest_right_pos
nearest_fine_boundary
nearest_boundary_distance_chars
nearest_boundary_tie
```

A tie/ambiguous local-boundary assignment remains part of the **global WITHIN count** but is excluded from boundary-specific support/effect analysis.

---

## Provisional Table 1

At 5k–10k prompts, produce a provisional Q2 row containing:

```text
# completed prompts
# generated target-path tokens
# visible candidate tokens
# eligible aligned tokens
aligned %
# CROSS
CROSS %
# WITHIN_SPLIT
WITHIN %
# single-boundary CROSS
alignment exclusion %
# observed fine boundaries
# supported boundaries at provisional N
```

This row is not the final paper number; its purpose is pipeline validation.

---

## Provisional Figure 1 — Boundary Support Matrix

Create POS-left × POS-right heatmap.

Cell value:

\[
\log(1 + N_{CROSS,b}).
\]

Also compute corresponding WITHIN counts per boundary.

### What to inspect

- Are there several well-supported non-nominal→particle transitions?
- Are ending→ending / predicate→ending / other classes now represented?
- Is the fine POS taxonomy too sparse?
- Are a few token IDs dominating the CROSS counts?

## Decision

If fine POS cells are extremely sparse even at 10k, define an additional **coarse POS-family landscape** now, but keep fine POS counts for appendix. Do not wait until all five models are finished to discover this problem.

## Done when

- [ ] Q2 pipeline produces sensible morphology counts;
- [ ] provisional Table 1 reconciles exactly;
- [ ] Figure 1 contains enough coverage to justify continuing the 40k run;
- [ ] all exclusion reasons are auditable.

---

# 5. Stage 3 — Complete Q2 40k Reference Run

> In the current no-pilot execution, this stage is not a separate continuation
> job: `run-table1 --pair Q2` performs the full Q2 40k inference and alignment
> after the full smoke gate.

## Goal

Generate the high-support dataset that will fill the core boundary-landscape artifacts.

## Run

Continue Q2 to all 40k frozen prompts.

```text
Q2: Qwen3 0.6B → 4B
P_common + P_extra = 40k
k = 4
```

Use sharding/resume rather than one monolithic job.

### Outputs

```text
runs/Q2_40k_sd.parquet
aligned/Q2_40k_aligned.parquet
results/Q2_exclusion_breakdown.csv
results/Q2_boundary_support.csv
```

---

## Final Table 1 Q2 row

Definitions must be fixed:

### Generated tokens

Total number of target greedy continuation tokens on successfully completed prompts.

Do **not** count all draft proposals.

### Visible candidate tokens

Visible generated target-path tokens before morphology filtering.

### Eligible tokens

Visible generated tokens remaining after removing:

- ambiguous morphology/span alignment;
- multi-eojeol token spans;
- exact-retokenization failures;
- special/non-visible tokens;
- other explicitly logged alignment failures.

### Alignment exclusion rate

\[
100\times
\frac{N_{visible}-N_{eligible}}
{N_{visible}}.
\]

### CROSS % / WITHIN %

Use `Eligible tokens` as denominator.

### Single-boundary CROSS

```text
alignment_class == CROSS
AND crossed_boundary_count == 1
```

### Supported boundaries for Q2

Use fine POS transitions with:

\[
N_{CROSS,b}\ge 200,
\qquad
N_{WITHIN,b}\ge 200.
\]

Support is defined only from counts, never from effect size.

---

## Final Figure 1 — Morphological Boundary Support Matrix

Finalize on Q2 40k.

Report:

```text
# fine boundaries observed
# boundaries satisfying 200/200 support
# supported non-nominal→particle boundaries
largest-support transitions
remaining sparse transition families
```

### Optional Appendix Figure A1

Ranked horizontal bars showing CROSS and WITHIN counts for each supported transition.

## Artifacts completed

- [x when done] **Table 1 — Q2 row**
- [x when done] **Figure 1 — final Q2 support matrix**

---

# 6. Stage 4 — Build Token-Frequency Cache

## Goal

Create the frequency covariate needed by the primary regressions without leaking evaluation documents.

## Data

Use a Korean corpus slice disjoint from the 40k evaluation prompts.

Compute frequency separately for each tokenizer family used in the final model set.

### Output fields

```text
token_id
token_count
token_frequency
log_token_frequency
frequency_decile
```

### Outputs

```text
frequency/qwen3_korean_token_frequency.parquet
frequency/ministral_korean_token_frequency.parquet
frequency/gemma_korean_token_frequency.parquet
```

Primary regression uses a flexible function of log token frequency.

## Done when

Every eligible analysis token can be joined to a tokenizer-specific frequency estimate or is explicitly marked missing.

---

# 7. Stage 5 — Q2 Core Statistical Analyses

Run these **before** waiting for all other model families. They establish the main phenomenon and main boundary landscape.

---

## Experiment 5A — Coarse Boundary Misalignment

### Question

Does an aggregate token–morpheme boundary agreement score explain rejection?

For each eojeol, define:

\[
misalignment = 1-F_1(B_T,B_M),
\]

where `B_T` and `B_M` are tokenizer and morphology internal boundaries.

### Population

Q2 40k eligible analysis observations.

### Model

```text
rejected ~ misalignment
         + fragmentation_FE
         + relative_position
         + is_first
         + is_last
         + proposal_slot
         + generation_position
         + token_char_len
         + eojeol_char_len
         + draft_entropy
         + target_entropy
         + flexible_log_frequency
```

Cluster standard errors by prompt/doc ID.

### Output

```text
results/Q2_coarse_misalignment.csv
```

### Fill

**Table 2A — Coarse Misalignment**

Suggested columns:

| Fragmentation | N | Rejection rate | Adjusted slope | 95% CI | p |
|---|---:|---:|---:|---:|---:|

### Interpretation

If the score is weak/null, retain the narrative that scalar misalignment mixes distinct tokenization events. If it becomes positive at scale, report that result honestly and use the boundary-specific analysis to determine where the effect comes from.

---

## Experiment 5B — Global CROSS vs WITHIN_SPLIT

### Question

Ignoring boundary identity, are CROSS proposals more failure-prone than WITHIN_SPLIT proposals?

### Population

```text
alignment_class ∈ {CROSS, WITHIN_SPLIT}
```

### Model

\[
logit\,P(R_i=1)
=
\alpha + \beta CROSS_i + \gamma^T X_i.
\]

Use the same standard controls as Experiment 5A.

### Report

```text
N CROSS
N WITHIN
adjusted P(reject | WITHIN)
adjusted P(reject | CROSS)
AME CROSS-WITHIN in percentage points
95% CI
odds ratio (secondary)
```

### Output

```text
results/Q2_global_cross.csv
```

### Fill

- **Table 2B — Q2 row**

### Decision

A null global effect does **not** end the study. Boundary-specific heterogeneity may still exist. It only means the later method should not treat every CROSS token as uniformly risky.

---

## Experiment 5C — Full Boundary-Risk Landscape

### Question

Which specific morphological boundaries are associated with higher rejection when crossed?

### Population

- Q2 40k;
- `CROSS` crosses exactly one boundary;
- boundary satisfies Q2 support threshold 200/200;
- `WITHIN_SPLIT` has an unambiguous nearest-boundary label;
- same supported boundary inventory.

### Model

```text
rejected ~ token_class * fine_boundary
         + fragmentation_FE
         + relative_position
         + is_first
         + is_last
         + proposal_slot
         + generation_position
         + token_char_len
         + eojeol_char_len
         + draft_entropy
         + target_entropy
         + flexible_log_frequency
```

Cluster SE by prompt.

For every boundary extract:

```text
n_cross
n_within
raw_cross_rejection
raw_within_rejection
adjusted_cross_probability
adjusted_within_probability
ame_pp
ci_low
ci_high
p_value
BH q_value
n_unique_cross_token_ids
```

### Output

```text
results/Q2_boundary_landscape.csv
```

### Fill Figure 2 — Full Boundary-Risk Landscape

Ranked forest plot:

- y-axis: fine POS boundary;
- x-axis: adjusted `CROSS − WITHIN` rejection difference (pp);
- 95% CI;
- sorted by AME;
- coarse family only as annotation.

Do **not** visually privilege nominal→particle.

### Fill Table 3 — Boundary-Specific Effects

Main-paper columns:

| Boundary | Coarse family | N CROSS | N WITHIN | Raw rej. CROSS | Raw rej. WITHIN | Adj. Δ pp | 95% CI | q | Unique CROSS token IDs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|

Main paper can display ~10–15 high-support rows; appendix must contain the complete supported inventory.

### Done when

You can answer:

- How many supported boundaries exist?
- How many are positive/near zero/negative?
- Which families are high risk?
- Is nominal→particle still special once other cells are adequately sampled?
- Is risk heterogeneous even within the same coarse family?

## Artifacts completed after Stage 5

- [ ] Table 2A
- [ ] Table 2B — Q2 row
- [ ] Figure 2
- [ ] Table 3

---

# 8. Stage 6 — Run the Four Remaining Pairs on the Common 20k

## Goal

Test whether the boundary-risk structure transfers across model scale and independently trained model families.

## Runs

```text
Q1: Qwen3 0.6B → 1.7B  on P_common 20k
Q3: Qwen3 4B   → 8B    on P_common 20k
M1: Ministral 3B → 8B  on P_common 20k
G1: Gemma 4B → 12B     on P_common 20k
```

Q2 common-set results are obtained by filtering the existing Q2 40k run to `P_common`; do not rerun Q2.

As each run finishes:

1. verify exact SD/target greedy equality;
2. run the same Kiwi alignment pipeline;
3. fill that pair’s Table 1 row immediately;
4. compute global CROSS result;
5. compute boundary support;
6. compute boundary-specific effects where support permits.

### Pair-specific support for cross-model analysis

For pair `m`, mark boundary `b` as supported when:

\[
N_{CROSS,b,m}\ge100,
\qquad
N_{WITHIN,b,m}\ge100.
\]

Do not assume that a boundary supported by Qwen is automatically supported by Ministral or Gemma.

### Outputs

```text
runs/Q1_20k_sd.parquet
runs/Q3_20k_sd.parquet
runs/M1_20k_sd.parquet
runs/G1_20k_sd.parquet

aligned/Q1_20k_aligned.parquet
aligned/Q3_20k_aligned.parquet
aligned/M1_20k_aligned.parquet
aligned/G1_20k_aligned.parquet

results/<pair>_global_cross.csv
results/<pair>_boundary_support.csv
results/<pair>_boundary_effects.csv
```

---

## Finalize Table 1

Recommended final Table 1 columns:

| Pair | Family | Draft→Target | Prompts | Generated tokens | Eligible tokens | Aligned % | CROSS | CROSS % | WITHIN | WITHIN % | Single-boundary CROSS | Exclusion % | Supported boundaries |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|

For the Q2 row, show the 40k analysis size. In the caption, state that cross-model comparisons use the common 20k subset.

### Consistency assertions

```text
single_boundary_cross <= CROSS <= eligible
WITHIN_SPLIT <= eligible
CROSS + WITHIN_SPLIT + EXACT + other eligible classes reconcile
exclusion reasons reconcile with visible - eligible
Q1/Q3/M1/G1 prompt IDs == common_20k exactly
Q2 contains common_20k + extra_20k exactly
```

---

## Finalize Table 2B

Run the global CROSS regression for all five pairs and fill:

| Pair | N CROSS | N WITHIN | Adj. P(reject|WITHIN) | Adj. P(reject|CROSS) | Δ pp | 95% CI |
|---|---:|---:|---:|---:|---:|---:|

## Artifacts completed after Stage 6

- [ ] **Table 1 final**
- [ ] **Table 2B final**

---

# 9. Stage 7 — Cross-Model / Cross-Family Boundary Generalization

## Experiment 7A — Boundary × Model-Pair Heatmap

### Question

Does the boundary-risk landscape transfer across Qwen scale and across Qwen/Ministral/Gemma tokenizers?

### Input

Boundary-specific AMEs for the five pairs on the common 20k prompt set.

### Comparison policy

For visualization, use the union of anchor-supported/relevant boundaries, but mark unsupported cells explicitly gray.

For statistical pairwise comparisons, use the **intersection of supported boundaries for the two model pairs being compared**.

Never impute a missing/unsupported boundary effect as zero.

### Fill Figure 3 — Boundary × Model-Pair Risk Heatmap

Columns:

```text
Qwen 0.6→1.7
Qwen 0.6→4
Qwen 4→8
Ministral 3→8
Gemma 4→12
```

Rows: supported fine boundary types.

Cell value:

\[
\Delta_{b,m}.
\]

Rows may be ordered by Q2 40k risk ranking or by common-set Q2 ranking; state the choice in the caption.

---

## Experiment 7B — Rank and Sign Stability

For each pair of model configurations, calculate over their shared-supported boundaries:

```text
# shared boundaries
Spearman rho
sign agreement
top-5 overlap
top-10 overlap
```

For every boundary also compute:

```text
mean AME across supported pairs
min/max AME
# pairs with positive AME
rank range
```

### Output

```text
results/cross_model_boundary_matrix.csv
results/cross_model_stability_pairwise.csv
results/cross_model_stability_by_boundary.csv
```

### Fill Table 4 — Cross-Model Stability

| Pair comparison | # shared boundaries | Spearman ρ | Sign agreement | Top-5 overlap | Top-10 overlap |
|---|---:|---:|---:|---:|---:|

### Interpretation

- high rank/sign stability → shared morphological risk structure is plausible;
- same signs but different magnitudes → shared structure + pair-specific calibration;
- low cross-family stability → risk should be treated as tokenizer/model-pair-specific.

## Artifacts completed

- [ ] Figure 3
- [ ] Table 4

---

# 10. Stage 8 — Tokenizer Exposure vs Conditional Risk

## Question

A boundary may be highly risky when crossed but almost never crossed by a tokenizer. Which boundaries are both common enough and consequential enough to matter?

For each model pair and boundary calculate:

### Boundary occurrence count

Number of actual local morphological instances of boundary `b` in the generated text.

### Exposure

\[
Exposure_{b,m}=P(CROSS\mid b,m).
\]

### Conditional risk

\[
Risk_{b,m}=\Delta_{b,m}.
\]

### Descriptive burden

\[
Burden_{b,m}=Exposure_{b,m}\times Risk_{b,m}.
\]

Treat burden as a descriptive prioritization statistic, not a causal quantity.

### Fill Figure 4 — Exposure vs Conditional Risk

Scatter plot:

- x-axis: Exposure;
- y-axis: adjusted boundary risk AME;
- point size: boundary occurrence count;
- panel/facet by tokenizer family or pair.

### Why this artifact matters

This figure tells the later SD method whether to prioritize:

- rare but catastrophic crossings;
- common moderate-risk crossings;
- or a mixture.

### Output

```text
results/boundary_exposure_risk.csv
```

## Artifact completed

- [ ] Figure 4

---

# 11. Stage 9 — Token-Identity Concentration and Removal Ablation

## Question

Is an apparent boundary effect actually driven by a small number of recurrent fused token IDs?

This is the most important alternative-explanation test after the main landscape.

## Input

Q2 40k aligned logs and `Q2_boundary_landscape.csv`.

## Experiment

For every supported boundary:

1. rank CROSS token IDs by count;
2. calculate share of observations explained by top 10/25/50/100 tokens;
3. refit the boundary effect after removing top 10;
4. repeat after top 25;
5. repeat after top 50;
6. top 100 may be appendix-only if sample size remains sufficient.

### Save

```text
boundary
top10_share
top25_share
top50_share
top100_share
top_n_removed
ame_pp
ci_low
ci_high
n_remaining
```

### Fill Figure 5 — Token-ID Removal Ablation

Main figure:

- choose ~5 high-support/high-risk boundaries;
- include ~2 neutral controls;
- x-axis = number of dominant CROSS token IDs removed;
- y-axis = adjusted AME.

### Associated table/appendix

Full concentration and removal results for all supported boundaries.

### Interpretation

If a boundary effect collapses after removing a handful of token IDs, do not present it as a general morphology-level effect. The later method may need token-conditioned risk.

### Outputs

```text
results/token_concentration.csv
results/token_removal_ablation.csv
```

## Artifact completed

- [ ] Figure 5

---

# 12. Stage 10 — Robustness Table

## Goal

Determine whether the *landscape*, not only the global CROSS coefficient, is stable to reasonable analysis choices.

Use Q2 40k as the main robustness dataset.

For each specification below recompute:

```text
global CROSS AME
boundary AME vector
Spearman rho vs main landscape
sign agreement vs main
top-10 overlap vs main
N retained
```

---

## Robustness 10A — Frequency specification

Run:

1. no frequency term;
2. linear log frequency;
3. flexible log frequency (main);
4. frequency-stratified comparison if implementation already exists.

---

## Robustness 10B — Multi-boundary tokens

Primary landscape uses exactly one crossed boundary.

Sensitivity:

```text
crossed_boundary_count >= 1
```

Either include count as a covariate or stratify:

```text
1
2
3+
```

Do not assign a unique fine boundary label to a token crossing several boundaries unless the analysis explicitly models the sequence.

---

## Robustness 10C — Alignment sensitivity

Create alternative populations:

```text
main standard alignment
strict exact surface alignment
exclude punctuation-heavy eojeol
exclude Latin/numeric-heavy eojeol
exclude retokenization anomalies
trim extreme token/eojeol lengths
```

---

## Robustness 10D — Dominant-token exclusion

Use the `remove top-50 CROSS token IDs` variant as one summary row.

---

## Fill Table 5 — Robustness Summary

| Specification | Global Δ pp | Landscape ρ vs main | Sign agreement | Top-10 overlap | N retained |
|---|---:|---:|---:|---:|---:|
| Main | ... | 1.00 | 1.00 | 10/10 | ... |
| No frequency | ... | ... | ... | ... | ... |
| Linear frequency | ... | ... | ... | ... | ... |
| Multi-boundary included | ... | ... | ... | ... | ... |
| Strict alignment | ... | ... | ... | ... | ... |
| Remove top-50 tokens | ... | ... | ... | ... | ... |

### Output

```text
results/robustness_summary.csv
```

## Artifact completed

- [ ] Table 5

---

# 13. Stage 11 — Pseudo-Boundary Falsification

## Question

Does the rejection penalty reflect genuine morpheme-boundary location, or would an arbitrary internal character cut produce a similar effect?

This experiment requires **no additional LLM inference**.

## Input

Q2 token spans, eojeol strings, true morphology spans, and rejection labels.

## Procedure

For each of 100 random seeds:

1. for every eligible eojeol, count the number of true internal morpheme boundaries;
2. sample the same number of valid internal character positions;
3. recompute pseudo-CROSS status under those pseudo-boundaries;
4. keep token geometry/eojeol fixed;
5. fit the analogous global pseudo-boundary crossing regression;
6. store the resulting pseudo-effect.

Optionally use a stricter null that approximately preserves segment-length distribution.

### Empirical p-value for a positive true effect

\[
p_{emp}=
\frac{1+\sum_s I[\Delta^{(s)}_{pseudo}\ge\Delta_{true}]}
{1+S}.
\]

### Fill Figure 6 — True vs Pseudo-Boundary Null

Plot:

- histogram/density of 100 pseudo-effects;
- vertical line = true morphology effect;
- empirical percentile/p-value in caption.

### Interpretation

- true effect clearly outside pseudo-null → supports linguistic localization;
- true effect comparable to null → morphology claim should be weakened toward generic token geometry.

### Output

```text
results/pseudo_boundary_null.csv
```

## Artifact completed

- [ ] Figure 6

---

# 14. Stage 12 — Distributional Disagreement Diagnostics (Lower-Cost Optional Core)

## Question

What target-distribution signature accompanies high-risk morphological crossings?

No new inference is required **if target rank/margin/logprob were logged during the main runs**.

For each draft proposal compute/store:

```text
target_rank_bin = {1, 2-5, 6-20, >20}
target_margin
target_minus_draft_logprob
draft_entropy
target_entropy
```

Recall:

\[
m_T = \log p_T(t_d|x)-\max_t\log p_T(t|x).
\]

Compare:

- high-risk CROSS boundaries;
- neutral CROSS boundaries;
- corresponding WITHIN_SPLIT controls.

Recommended transparent grouping:

```text
high-risk = top quartile of supported Q2 boundary AMEs
neutral = CI overlaps 0 and |AME| below a frozen threshold
```

Also keep continuous per-boundary results in the appendix so grouping does not hide structure.

### Fill Figure 7 — Target Rank / Margin Signature

Two panels:

A. target-rank categories;
B. target-margin distribution.

### Output

```text
results/disagreement_diagnostics.csv
```

## Interpretation

This is a diagnostic signature, **not** proof of a causal mechanism.

## Artifact completed

- [ ] Figure 7

If time is short, move this figure to appendix rather than delaying core falsification.

---

# 15. Stage 13 — Small Block-Size / Speculative-Waste Experiment (Last New Inference)

## Priority

This is the first experiment to cut if the deadline is tight.

## Question

Does morphological boundary risk persist across speculative block sizes, and do high-risk failures waste more downstream draft work?

## Prompt subset

Freeze 5,000 prompts from `P_common`.

## Runs

Use Q2 only:

```text
k = 2 : new 5k run
k = 4 : reuse main run
k = 8 : new 5k run
```

Total new inference: only 10k pair-prompt runs.

For every rejection save:

```text
block_size
block_id
rejection_position_in_block
later_proposals_invalidated
```

Define:

\[
Waste = \#\{\text{later proposals invalidated by the rejection}\}.
\]

Compute:

```text
boundary AME at k=2,4,8
rho(k2,k4)
rho(k4,k8)
mean invalidated proposals per rejection
mean invalidated proposals per boundary occurrence
```

### Fill Figure 8 — Block Size and Speculative Waste

Panel A:
- selected/global boundary AME vs k.

Panel B:
- downstream waste for high-risk vs neutral boundaries.

### Outputs

```text
results/block_size_sensitivity.csv
results/speculative_waste_by_boundary.csv
```

## Artifact completed

- [ ] Figure 8

---

# 16. Optional Experiments Only If Core Artifacts Are Already Finished

These retain useful ideas from the earlier experiment draft but should not block submission.

---

## Optional A — Boundary Geometry

For single-boundary CROSS tokens analyze:

```text
left_chars
right_chars
boundary_position_ratio
token_char_len
right_morpheme_fully_absorbed
```

Primary model:

```text
rejected ~ spline(boundary_position_ratio)
         + boundary
         + standard_controls
```

Optional interaction:

```text
rejected ~ boundary * spline(boundary_position_ratio)
         + standard_controls
```

Possible figure:
- adjusted rejection probability vs normalized boundary position;
- appendix 2D heatmap of left_chars × right_chars.

Output:

```text
results/boundary_geometry.csv
```

---

## Optional B — Generated Path vs Reference Path

Use the held-out next corpus segment under teacher forcing.

Measure:

```text
draft_top1
target_top1
top1_disagreement
draft_entropy
target_entropy
draft_top1_target_rank
target_margin
```

Do **not** call this SD rejection.

Compare generated-path boundary AMEs with reference-path disagreement effects.

Output:

```text
results/reference_path_boundary_effects.csv
```

Possible figure:
- x-axis = generated-path rejection effect;
- y-axis = reference-path disagreement effect.

---

## Optional C — Partial-Pooling / Hierarchical Boundary Model

If implementation is already available, fit boundary-specific CROSS slopes:

\[
\theta_b\sim\mathcal N(\mu_\theta,\sigma_\theta^2).
\]

Use only as a robustness analysis:

```text
hierarchical_ame
hierarchical_rank
rank_difference_vs_main
```

Do not build a new Bayesian infrastructure days before the deadline just for this analysis.

---

# 17. Exact Fill Order in the Manuscript

This is the order in which you should replace placeholders in the paper as results arrive.

## Fill 1 — Experimental Setup text

Can be completed before long runs:

- dataset identifier/revision;
- prompt sampling procedure;
- exact five model pairs that passed compatibility;
- decoding settings;
- morphology analyzer version;
- support thresholds;
- covariates/statistical model.

---

## Fill 2 — Table 1 provisional, then final

First fill Q2 after 5k–10k.

Then final Q2 40k.

Then add Q1/Q3/M1/G1 rows as runs finish.

Table 1 is final only when all model-pair rows reconcile with raw/aligned logs.

---

## Fill 3 — Figure 1

Finalize after Q2 40k morphology alignment.

This establishes whether the full landscape is statistically feasible.

---

## Fill 4 — Table 2A + Q2 row of Table 2B

Run Q2 core regressions as soon as Q2 40k is aligned.

Do not wait for other models.

---

## Fill 5 — Figure 2 + Table 3

These are the main Q2 boundary-landscape results.

At this point you should freeze the **actual scientific narrative**:

- concentrated risk;
- distributed risk;
- fine-grained heterogeneity;
- or little boundary structure.

---

## Fill 6 — Final Table 1 + Table 2B

Complete when all five main pair runs finish.

---

## Fill 7 — Figure 3 + Table 4

Complete cross-model/cross-family transfer analysis.

This determines whether the later method should use:

- shared universal boundary risk;
- shared boundary identity + pair-specific calibration;
- or fully pair/tokenizer-specific risk.

---

## Fill 8 — Figure 4

Exposure vs conditional risk.

This connects the statistical landscape to practical method prioritization.

---

## Fill 9 — Figure 5

Token-ID removal.

This must be complete before making a strong morphology-level claim.

---

## Fill 10 — Table 5

Consolidate frequency, alignment, multi-boundary, and token-exclusion robustness.

---

## Fill 11 — Figure 6

Pseudo-boundary falsification.

This should be complete before claiming that *linguistic* boundary location matters beyond arbitrary token geometry.

---

## Fill 12 — Figure 7 / Figure 8 if time permits

- target rank/margin diagnostic;
- block-size/speculative-waste bridge.

These strengthen the transition into the SD method but are not prerequisites for the core boundary-landscape claim.

---

# 18. Final Paper-Facing Experimental Section Order

The execution order above is optimized for running experiments. The manuscript should still read in a scientific narrative order:

## 4 Experimental Setup

### 4.1 Models and Data
### 4.2 Speculative Decoding Protocol
### 4.3 Morphological Alignment and Boundary Inventory
### 4.4 Covariates and Statistical Analysis

**Table 1** appears here.

---

## 5 Large-Scale Morphological Boundary Landscape

### 5.1 Boundary Support Census
- Figure 1

### 5.2 Aggregate Boundary Misalignment
- Table 2A (main or appendix)

### 5.3 Global CROSS vs WITHIN_SPLIT
- Table 2B

### 5.4 Full Boundary-Specific Rejection Landscape
- Figure 2
- Table 3

---

## 6 Cross-Model and Cross-Family Generalization

### 6.1 Boundary × Model-Pair Effects
- Figure 3

### 6.2 Rank/Sign Stability
- Table 4

### 6.3 Tokenizer Exposure vs Conditional Risk
- Figure 4

---

## 7 What Drives Boundary Risk?

### 7.1 Token-Identity Concentration
- Figure 5

### 7.2 Distributional Disagreement
- Figure 7 if retained in main paper

### 7.3 Boundary Geometry
- appendix unless especially strong

---

## 8 Robustness and Falsification

### 8.1 Frequency, Multi-Boundary, and Alignment Robustness
- Table 5

### 8.2 Pseudo-Boundary Falsification
- Figure 6

### 8.3 Block-Size and Speculative-Waste Sensitivity
- Figure 8 if completed

### 8.4 Reference-Path Analysis
- appendix / optional

---

# 19. Definition of “Complete” for Each Core Artifact

## Table 1 complete when

- [ ] all final model pairs are listed;
- [ ] prompt counts equal frozen run counts;
- [ ] generated/visible/eligible token counts reconcile;
- [ ] exclusion breakdown reconciles exactly;
- [ ] CROSS/WITHIN/single-boundary counts pass assertions;
- [ ] supported-boundary count reproducible from aligned parquet.

## Figure 1 complete when

- [ ] based on Q2 full 40k;
- [ ] support threshold clearly marked;
- [ ] counts independent of rejection outcome;
- [ ] fine/coarse taxonomy documented.

## Table 2A complete when

- [ ] coarse misalignment formula frozen;
- [ ] fragmentation-specific estimates present;
- [ ] adjusted overall estimate present;
- [ ] clustered SE used.

## Table 2B complete when

- [ ] all five pairs have global AMEs;
- [ ] same covariate policy used;
- [ ] model-family-specific frequency cache used;
- [ ] common 20k used for cross-pair comparison.

## Figure 2 complete when

- [ ] only supported boundaries included;
- [ ] AMEs and 95% CI plotted;
- [ ] multiple-testing-adjusted results available in Table 3;
- [ ] no pre-highlight of nominal→particle.

## Table 3 complete when

- [ ] raw counts/rates and adjusted AMEs both reported;
- [ ] unique token-ID support included;
- [ ] full inventory saved to appendix.

## Figure 3 complete when

- [ ] unsupported cells gray, not zero;
- [ ] pairwise comparison uses shared-supported sets;
- [ ] same common raw prompts used across all pairs.

## Table 4 complete when

- [ ] # shared boundaries stated for each comparison;
- [ ] Spearman/sign/top-K metrics computed consistently.

## Figure 4 complete when

- [ ] boundary occurrence denominator is clearly defined;
- [ ] exposure and risk are kept distinct;
- [ ] burden is explicitly described as descriptive.

## Figure 5 complete when

- [ ] removal is performed separately within boundary;
- [ ] sample size after each removal is reported;
- [ ] neutral controls included.

## Table 5 complete when

- [ ] every robustness row uses the same main boundary inventory or explicitly states otherwise;
- [ ] landscape correlation and sign agreement are reported, not only global AME.

## Figure 6 complete when

- [ ] at least 100 pseudo-boundary randomizations;
- [ ] boundary count per eojeol preserved;
- [ ] empirical p/percentile reported;
- [ ] no new LLM inference required.

---

# 20. Core Stop Rules for the Deadline

If time/compute becomes limited, cut experiments in this order:

1. block-size sensitivity / Figure 8;
2. target rank-margin Figure 7;
3. boundary geometry;
4. reference-path analysis;
5. hierarchical partial pooling.

Do **not** cut before submission if at all possible:

1. Table 1;
2. Figure 1 support matrix;
3. global CROSS analysis;
4. Figure 2 full boundary landscape;
5. at least one non-Qwen cross-family pair;
6. Figure 3 cross-model heatmap;
7. Figure 5 token-ID ablation;
8. Table 5 robustness;
9. Figure 6 pseudo-boundary falsification.

---

# 21. Final One-Line Workflow

```text
Freeze data/config
→ audit tokenizers + smoke test
→ Q2 5–10k checkpoint
→ provisional Table 1 + Figure 1
→ complete Q2 40k
→ final Q2 Table 1 + Figure 1
→ frequency cache
→ Table 2A + Q2 global CROSS
→ Figure 2 + Table 3 full Q2 landscape
→ run Q1/Q3/M1/G1 common 20k
→ final Table 1 + Table 2B
→ Figure 3 + Table 4 cross-model stability
→ Figure 4 exposure/risk
→ Figure 5 token-ID ablation
→ Table 5 robustness
→ Figure 6 pseudo-boundary falsification
→ optional Figure 7 disagreement diagnostics
→ optional Figure 8 block-size/waste
→ freeze empirical narrative
→ design SD method
```

---

# 22. Result Narrative Decision Tree

Do not force the original nominal→particle story. Use the observed results.

### If multiple boundary families are consistently risky across models

Claim:

> Morphological boundary identity provides a structured and partly transferable source of draft–target incompatibility.

Later method:
- shared boundary-risk prior, possibly calibrated per pair.

### If only a small set of fine boundaries is consistently risky

Claim:

> Boundary-crossing risk is selective rather than universal.

Later method:
- intervene only at high-risk boundary configurations.

### If Qwen landscape is strong but cross-family transfer is weak

Claim:

> Morphological risk is tokenizer/model-family dependent.

Later method:
- learn/calibrate risk per pair or tokenizer.

### If token-ID removal collapses most effects

Claim:

> The apparent morphological signal is substantially mediated by recurrent fused token identities.

Later method:
- token + morphology-conditioned risk.

### If pseudo-boundary null matches true morphology

Do not claim a uniquely morphological mechanism.

Claim instead:

> Token-internal boundary geometry, rather than linguistic morpheme identity alone, explains much of the disagreement structure.

Later method:
- geometry-aware rather than morphology-only SD.

