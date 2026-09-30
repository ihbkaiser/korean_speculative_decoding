# H4 implementation plan: Nominal→Particle decomposition

## Inputs and hard limits

- Analyze only saved artifacts; no speculative decoding, target generation, teacher-forced scoring, or model loading.
- P1/P2 token data: `runs/e2_model_pair_replication/p1_06b_to_17b/token_table.parquet` and `p2_17b_to_4b/token_table.parquet`.
- P3 token data: `runs/20260926T184145Z_pilot1000/h2_token_table.parquet`.
- H3 environment assignment and exact morpheme spans: `h3/h3_token_environment.parquet` and `h3/h3_eojeol_morphemes.parquet`.
- Reuse `runs/20260926T184145Z_pilot1000/e1_token_frequency_cache.parquet`, `e1_eojeol_frequency_cache.parquet`, and the exact source corpus `data/korean_wikipedia_20231101_ko.jsonl` (SHA-256 verified against E1 metadata). Token and eojeol frequencies are cached. Because morpheme-surface frequencies are not cached, count nominal/particle surfaces from the same corpus by applying the saved H3 Kiwi sequence representation to corpus eojeols; do not use rejection labels in construction.
- Pooling sensitivity uses the exact 993 prompt IDs from `combined/e2_prompt_overlap_audit.json`.

## H4 population

For each pair independently, filter explicitly to `sd_valid == True`, fragmentation category 2/3/4/5/6/7/8+, CROSS_MORPHEME/WITHIN_SPLIT, then restrict to the saved H3 `NOMINAL_TO_PARTICLE` environment. CROSS rows must additionally have exactly one crossed boundary. The saved H3 input was checked and all its rows satisfy `sd_valid` and the listed fragmentation bins. This uses the H3 primary local-adjacent assignment for WITHIN_SPLIT. Preserve all rows and document rows lacking a verifiable exact span as alignment failures rather than silently repairing them.

## Particle taxonomy (prespecified)

- `CASE_SUBJECT`: JKS
- `CASE_OBJECT`: JKO
- `CASE_ADVERBIAL`: JKB
- `CASE_GENITIVE`: JKG
- `OTHER_CASE`: JKC, JKV, JKQ
- `AUXILIARY_PARTICLE`: JX
- `CONJUNCTIVE_PARTICLE`: JC

No categories will be merged after inspecting outcomes. Unsupported levels remain in descriptive outputs and are not assigned fabricated estimates.

## Exact morphology and geometry

Reconstruct the ordered morphemes from saved H3 eojeol records. For each token, match saved H2 overlapping morpheme surface, fine POS, and global character span to the corresponding H3 records. For CROSS, the nominal and particle are the two morphemes bordering the unique crossed boundary. For SPLIT, reproduce H3's nearest-adjacent-boundary rule exactly: inspect the boundary immediately before/after the token-containing morpheme, choose minimum character distance, and prefer the following/right boundary on ties. Require the selected boundary to be NOMINAL→PARTICLE.

Span coverage is the integer character intersection of the exact token span with each full morpheme span; coverage fractions divide by each morpheme's character length. Geometry labels are mutually exclusive in this order: (1) both fully covered, `FULL_NOMINAL_PLUS_FULL_PARTICLE`; (2) partial nominal/full particle, `NOMINAL_SUFFIX_PLUS_FULL_PARTICLE`; (3) full nominal/partial particle, `FULL_NOMINAL_PLUS_PARTICLE`; (4) both partial, `PARTIAL_BOTH`; (5) otherwise, `OTHER_GEOMETRY`. This resolves the overlap between the supplied full-nominal-plus-particle and full/full descriptions without changing observed labels.

## Planned models and uncertainty

- H4a: pair-specific clustered logistic model with `morph_class * particle_category`, H3 M5 controls (fragmentation FE; relative position, first/last token, proposal slot, generation position, token and eojeol lengths; draft/target entropy; `bs(log_token_count, df=4, degree=3, include_intercept=False)`). Test the interaction jointly; calculate counterfactual adjusted P(reject) and CROSS−SPLIT AMEs using observed covariates and cluster-by-prompt delta-method covariance. Holm correction is within pair across prespecified particle-category contrasts.
- H4b: among CROSS only, fit geometry class and continuous coverage models with particle category, nominal POS, H3 M5 controls, and prompt-clustered covariance. Geometry-category adjusted probabilities are standardized over all CROSS rows in that pair, changing only geometry class. If sparse logistic fits are separated or extreme, use a clearly labeled prompt-clustered linear probability fallback for adjusted probabilities; retain sparse categories and flag non-estimability.
- H4c: L1 is H3 M5 on this population; L2 adds particle category, nominal POS and nominal/particle surface lengths; H3 `eojeol_char_length` is exactly equal to eojeol surface length in the saved rows, so retain the original H3 field instead of adding a duplicate column. L3 adds nonlinear log nominal-, particle-, and eojeol-frequency splines. Report AME changes without mediation language.
- Exact particle fixed-effect robustness uses surfaces meeting, before fitting, >=100 total rows, >=30 CROSS rows, and >=30 prompts. If interactions are rank-deficient or unstable, report that and retain the main model.
- Matched sensitivity is deterministic coarsened exact matching on pair, particle category, fragmentation category, token-length bin (1/2/3/4+ chars), nominal-surface-length bin (1/2/3/4+ chars), pooled eligible-sample decile bins for token frequency and draft entropy, and within-pair generation-position quartile. Bins use covariates only. Keep strata with both classes; report cross-weighted within-stratum raw differences and cluster-robust class contrasts with stratum fixed effects where estimable.
- Qualitative audit uses NumPy seed 20260929; sample 20 rejected CROSS, 20 accepted CROSS, and 20 matched WITHIN_SPLIT rows overall, stratified proportionally by pair where support permits. Sampling occurs only after the analysis set is frozen and is not used to define models.
- Recurrent CROSS token patterns are defined as `(nominal_surface, particle_surface, token_surface)` and reported by support/count before examining their rejection rate. Main repeated-pattern table requires >=20 CROSS rows and >=10 prompts; rare patterns remain in the overall counts but are not interpreted individually.
- A descriptive concentration table reports token/rejection shares among top-1/5/10 patterns ranked by pattern frequency only. Geometry character-count and coverage quantiles are written for CROSS rows by pair; they are not pooled with SPLIT rows.

## Audits, safeguards, and outputs

Check token/morpheme span matches and H3 environment reconstruction; verify only saved labels/entropy/frequency inputs are used; record convergence, design rank, cluster count, sparse cells, and extreme coefficients. Input hashes, formulas, mapping, bins, seed, and software versions go in `h4_config.json`.

Expected outputs under `runs/e2_model_pair_replication/h4/`:

- `implementation_h4.md`, `h4_config.json`, `h4_particle_mapping.json`
- `audits/h4_geometry_counts.csv`, `audits/h4_geometry_span_distributions.csv`, `audits/h4_geometry_audit.md`, `audits/h4_particle_audit.csv`, `audits/h4_alignment_audit.json`
- `pair_results/p1_h4_results.*`, `p2_h4_results.*`, `p3_h4_results.*`
- `pooled/h4_particle_results.csv`, `h4_geometry_results.csv`, `h4_lexical_robustness.csv`, `h4_particle_fixed_effects.csv`, `h4_matched_robustness.csv`, `h4_qualitative_audit.csv`, `h4_recurrent_pattern_concentration.csv`
- `h4_summary.md`

No H2/H3 morphology labels, eligibility semantics, or saved regression outputs will be changed.
