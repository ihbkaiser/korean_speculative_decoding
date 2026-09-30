# H3 implementation plan

## Scope and frozen inputs

This analysis reuses the completed E2 P1/P2 token tables and the P3 H2 token table. It does not run speculative decoding, target generation, or teacher-forced scoring. The H2 eligibility helper and exact exclusions are imported from `src/e2_model_pairs.py`; H2 `morph_class`, `sd_valid`, fragmentation, and rejection labels remain unchanged. P3 entropy is harmonized from its saved teacher-forced artifact using the existing E2 utility where the pooled E2 specification does so.

The pairs are P1 `Qwen/Qwen3-0.6B-Base@da87bfb608c14b7cf20ba1ce41287e8de496c0cd → Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1`, P2 `Qwen/Qwen3-1.7B-Base@ea980cb0a6c2ae4b936e82123acc929f1cec04c1 → Qwen/Qwen3-4B-Base@906bfd4b4dc7f14ee4320094d8b41684abff8539`, and existing P3 `Qwen/Qwen3-0.6B-Base → Qwen/Qwen3-4B-Base` from `runs/20260926T184145Z_pilot1000/`. The P3 historical model revision limitation/recovery metadata will be carried forward as recorded in the housekeeping artifacts; the current E2 config revisions will not be treated as proof of the historical P3 draft snapshot.

Input morphology fields are `overlapping_morpheme_ids`, surfaces, POS tags, spans, token character offsets, and the saved eojeol table. An initial integrity check found that not every eojeol morpheme appears in token-overlap arrays (in part because H2 excludes invalid/zero-width spans). Therefore, the analysis will tokenize only the already saved eojeol strings with the installed Kiwi version to recover the full ordered POS/span sequence; it will not alter any H2 token label. Those sequences will be cross-checked against saved H2 overlapping-morpheme records. Any discrepancies will be reported and token boundary assignment will use the H2 continuation-level overlap fields for exact crossed spans, with the eojeol-level parse used only to fill sequence context.

## Frozen morphology mapping and boundary taxonomy

The POS mapping will follow the installed `kiwipiepy` POS glossary and the user-prespecified groups: NOMINAL, PARTICLE, PREDICATE, ENDING, plus documented MODIFIER, DERIVATIONAL, OTHER_CONTENT, and OTHER_NONCONTENT groups for remaining observed and known Kiwi tags. Kiwi inflectional surface variants (for example `VV-I`, `VV-R`, `VA-I`, `VA-R`, and `XSA-I`) will map to their documented/base lexical group and be explicitly listed. Any observed tag absent from the local glossary will be listed and assigned explicitly rather than dropped.

The five frozen boundary classes are NOMINAL_TO_PARTICLE, PREDICATE_TO_ENDING, ENDING_TO_ENDING, LEXICAL_TO_LEXICAL, and OTHER. LEXICAL_TO_LEXICAL means both adjacent POS tags map to content-like coarse groups (NOMINAL, PREDICATE, MODIFIER, DERIVATIONAL, or OTHER_CONTENT), after excluding the first three directed grammatical classes. All other valid adjacent transitions map to OTHER.

For a CROSS token, a boundary is crossed when the token's exact H2 character span has positive overlap with both adjacent morphemes in the ordered eojeol sequence. `num_boundaries_crossed` is the number of adjacent morpheme transitions satisfying that condition; the ordered transition and coarse-class sequences are retained. Primary CROSS analyses require exactly one crossed transition. Multiple crossed transitions are retained for a separate exploratory count analysis.

For WITHIN_SPLIT, identify the containing morpheme from saved H2 overlap IDs. Candidate boundaries are the immediately preceding and following morpheme boundaries. Assign the sole candidate if one exists; if both exist, choose the smaller character distance from the token span to the boundary, breaking ties toward the following boundary. Assign OTHER if there is no adjacent boundary. Record `ADJACENT_BOUNDARY` as the source. CROSS single-boundary rows receive their crossed boundary type and `CROSSED_BOUNDARY` source.

The prespecified sensitivity labels each eojeol by its set of coarse boundary classes: a single class receives that class, no class receives OTHER, and multiple classes receive `MULTI_ENV`; `MULTI_ENV` rows are excluded only from the sensitivity fit. This rule is independent of rejection outcomes. If a class-by-environment cell is empty, all non-`MULTI_ENV` observations stay in the fit, the empty interaction column is omitted as non-identifiable, and its AME is reported as non-estimable.

## Population and frequency

Use the existing H2 eligible population (`sd_valid`, fragmentation categories 2, 3, 4, 5, 6, 7, 8+, and `CROSS_MORPHEME`/`WITHIN_SPLIT`). Keep CROSS rows with more than one crossed boundary out of primary H3a/H3b, but report them in audit and exploratory outputs. Attach the existing E1 token-frequency cache with the E2 helper and preserve the E1/E2 M5 cubic spline specification `bs(log_token_count, df=4, degree=3, include_intercept=False)`.

## Models and inference

H3a is fitted separately by pair among single-boundary CROSS rows: boundary category plus fragmentation fixed effects, the standard E2 structural controls, both saved entropy controls, and the E1/E2 frequency spline. Use `LEXICAL_TO_LEXICAL` as the reference only when it meets the fixed support rule (at least 100 CROSS tokens and 50 prompts); otherwise use the largest-support observed category as the computational reference and report standardized adjusted rejection probabilities for all categories over the same CROSS-only covariate distribution. This reference choice depends on support only, never rejection outcomes. Report clustered Wald omnibus tests over all estimable boundary terms.

H3b is fitted separately by pair on single-boundary CROSS plus all eligible WITHIN_SPLIT rows. Fit `morph_class * morph_environment` with the same M5 controls, reference `WITHIN_SPLIT`, cluster covariance by `prompt_id`, and a joint Wald test of the interaction block. For each of the five environments, calculate adjusted P(CROSS), adjusted P(SPLIT), and their difference on that environment's observed covariate distribution, changing only `morph_class`. Use the model's prompt-cluster covariance and the delta method for two-sided 95% intervals and p-values. Apply Holm correction across the five planned contrasts within each pair; non-estimable contrasts remain explicitly reported.

After pair-specific replication, fit the pooled model with pair fixed effects, `morph_class * morph_environment`, `model_pair * morph_class`, and the same M5 controls. Repeat on the exact 993-prompt three-way eligible-prompt intersection from the E2 overlap audit. Classify common-prompt change before looking at outcome estimates: unchanged if all environment AMEs shift by at most 0.5 percentage points; minor if the maximum shift is over 0.5 and at most 1.5 points; material if any sign reverses or any shift exceeds 1.5 points.

Exploratory analyses: fit a M5-adjusted boundary-count model among CROSS rows, and a fine-POS transition factor model restricted by the fixed support rule (at least 100 CROSS tokens and 50 prompts). No transition will be selected based on rejection rate or p-value. Compute descriptive contribution as the share of single-boundary CROSS tokens times the environment-specific adjusted excess rejection; report raw and normalized contribution scores.

All logistic models will record convergence, design-matrix rank, cluster count, sparse cells, and extreme coefficients. If a planned environment has inadequate support, retain the taxonomy and report imprecision/non-estimability rather than merging categories. Cluster-bootstrap resampling is not used; intervals use delta-method covariance clustered by prompt.

## Reproducibility, benchmark, and outputs

This is CPU-side analysis of local parquet/JSON artifacts and does not load model weights or query any GPU. The configuration will record input SHA-256 hashes, software versions, Kiwi mapping/version, formulas, references, thresholds, and deterministic analysis seed. Outputs include the saved sequence/enriched token tables, boundary/POS audits, pairwise and pooled estimates, common-prompt sensitivity, contribution analysis, figures (PNG/PDF), LaTeX table, and `h3_summary.md` under this directory. Figure 1 shows pair-specific adjusted CROSS-minus-SPLIT differences and 95% intervals; Figure 2 is a descriptive prevalence/effect plot.

No H3 definition or model specification will be modified after viewing rejection estimates.
