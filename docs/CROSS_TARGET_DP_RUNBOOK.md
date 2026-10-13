# Cross-target decision-point experiment (RQ6)

## Experiment card

- **Question:** Are low top-2-margin locations shared across target models on identical Korean text prefixes, or are they model/family specific?
- **Design:** Diagnostic, no directional success claim. Outcomes are interpreted as cross-target sharing, family dependence, or mostly target-specific locations.
- **Primary metric:** Pairwise Jaccard of each target's exact bottom-20% margin positions on the same common set, with a within-document permutation reference. The within-Qwen3 mean is compared with the cross-family mean.
- **Secondary metrics:** Margin Spearman correlation; exact bottom-10% and bottom-30% overlap; absolute margin `<1` prevalence; shared-target counts; matched Kiwi morphology comparison.
- **Data:** First 2,000 documents of the frozen FineWeb2 Korean common split. Select 20 evenly spaced eojeol onsets per eligible document, require at least 128 characters of preceding text, and use a 256-character context window snapped to an earlier eojeol onset. Every target receives the same source-text slice at each `(doc_id, char_offset)`.
- **Models and pinned revisions:** Qwen3-4B-Base `906bfd4b4dc7f14ee4320094d8b41684abff8539`; Qwen3-8B-Base `49e3418fbbbca6ecbdf9608b4d22e5a407081db4`; Qwen3-14B-Base `0b0bd3732e2c374d483664439ea334928b65f304`; Gemma-3-12B-pt `295efb63d01a7017928f273a94ebb86105c9526f`; Ministral-3-8B-Base-2512 `d4883f9b36aa2e5d775730d3fdba3d30de51a8ef`.
- **Normalization/alignment:** Use the existing NFC/strip FineWeb2 text and stable document IDs. Character offsets are Python Unicode codepoint offsets. Tokenize every shared prefix independently with each pinned tokenizer; do not compare token indices or raw margins across vocabularies.
- **Uncertainty:** 1,000 bootstrap replicates resampling documents and 1,000 independent within-document permutations that preserve each target's DP count per document. Fixed seed `20261013`.
- **Exploratory morphology set:** For two evenly spaced sampled eojeols per document, compare a reliable internal Kiwi morpheme boundary with the nearest character gap inside a morpheme, matched within the same eojeol. If Kiwi spans do not exactly cover the eojeol or no matched pair exists, exclude and report the reason.
- **Compute/storage:** One RTX A6000, BF16 and SDPA; load one model at a time and evict its weight shards after scoring. Save only reduced top-2/log-probability/entropy statistics, not vocabulary logits.

## Reproduction

From the repository root, with `/venv/main` active and the gated Gemma repository approved in Hugging Face:

```bash
/venv/main/bin/python scripts/prepare_cross_target_dp.py

for model in qwen4 qwen8 qwen14 gemma12 ministral8; do
  /venv/main/bin/python scripts/download_models.py --config configs/cross_target_dp.yaml --model "$model" --smoke-load
  /venv/main/bin/python scripts/run_cross_target_dp.py --config configs/cross_target_dp.yaml --model "$model" --batch-size 16 --evict-after
done

/venv/main/bin/python scripts/analyze_cross_target_dp.py --config configs/cross_target_dp.yaml
/venv/main/bin/python scripts/plot_cross_target_dp.py --config configs/cross_target_dp.yaml
```

For an interim two-model analysis once Qwen3-4B and Qwen3-8B are complete, select them explicitly:

```bash
/venv/main/bin/python scripts/analyze_cross_target_dp.py --config configs/cross_target_dp.yaml --models qwen4,qwen8
/venv/main/bin/python scripts/plot_cross_target_dp.py --config configs/cross_target_dp.yaml
```

The original 2K five-target run is complete. `results/cross_target_dp/` contains that full five-target analysis; `summary.json` records its model set and `cross_target_findings.md` contains the result tables and interpretation. A separate 10K Qwen3-4B/8B interim analysis is under `results/cross_target_dp_10k/`; its results only describe that two-target, within-family comparison. The earlier 2K interim output is preserved under `results/cross_target_dp/interim_qwen4_qwen8/`.

Running the two-model command above writes to the configured analysis output directory. Save or copy the existing five-target output before using it for a subset analysis; after the subset run, run the full command again to restore the five-target outputs.

If the Hugging Face token is stored locally in the ignored `hf_token` file, prefix the gated-model download command without printing the token:

```bash
HF_TOKEN="$(tr -d '\r\n' < hf_token)" /venv/main/bin/python scripts/download_models.py --config configs/cross_target_dp.yaml --model gemma12 --smoke-load
```

For a small GPU check before full scoring:

```bash
/venv/main/bin/python scripts/run_cross_target_dp.py --model qwen4 --position-limit 32 --batch-size 4 --run-root results/cross_target_dp/smoke
```

The run is resumable by model-position shard. If a shard already exists, its position IDs and pinned revision must match the frozen manifest. Use `--force` only to recompute that model's stored outputs.

## Outputs

`results/cross_target_dp/` contains `evaluation_positions.parquet`, `position_metadata.json`, per-target reduced-score shards and run manifests, `position_margins.parquet`, `pairwise_overlap.csv`, `pairwise_correlations.csv`, `bootstrap_ci.csv`, `permutation_results.csv`, `decision_point_prevalence.csv`, `shared_target_counts.csv`, `shared_target_buckets.csv` (with a 3+ group when at least three targets are selected), `family_consistency_bootstrap.csv`, `morphological_enrichment.csv`, `summary.json`, `cross_target_findings.md` with a populated result paragraph, the final analysis log, and publication PDF/PNG figures A–D. The model subset used for an analysis is recorded in `summary.json`; a subset analysis cannot support omitted pair or family comparisons. `NAACL_experiments.md` records the completed run and the pairwise results.

The primary source of cross-target alignment is `shared_prefix_sha256` plus `(doc_id, char_offset)`. The full prefix text is reconstructed from the frozen document and is not duplicated in the margins table.

## Interpretation limits

The absolute `<1`-nat threshold is secondary because the next-token vocabularies and margin scales differ. Bottom-percentile ranks and overlap on the common position set are primary. Eojeol-onset and Kiwi morphology findings are descriptive associations. They do not establish intrinsic linguistic ambiguity or causal effects. Existing RQ4 conclusions about draft failures remain conditional on a fixed target.


## 10K Qwen3-4B/8B extension (completed 2026-10-13)

This extension uses `configs/cross_target_dp_10k.yaml` and writes to `results/cross_target_dp_10k/`, so it does not overwrite the five-target 2K analysis. It scores only Qwen3-4B and Qwen3-8B. The 10,000-document candidate sample yields 9,595 eligible documents and 191,900 primary eojeol-onset positions; the morphology extension has 15,998 positions. Each model scores 207,898 positions including the morphology subset.

Reproduction commands for the scoring and subset analysis:

```bash
for model in qwen4 qwen8; do
  /venv/main/bin/python -I scripts/download_models.py --config configs/cross_target_dp_10k.yaml --model "$model" --smoke-load
  /venv/main/bin/python -I -u scripts/run_cross_target_dp.py --config configs/cross_target_dp_10k.yaml --model "$model" --batch-size 16 --evict-after
done

/venv/main/bin/python -I -u scripts/analyze_cross_target_dp.py --config configs/cross_target_dp_10k.yaml --models qwen4,qwen8
/venv/main/bin/python -I -u scripts/plot_cross_target_dp.py --config configs/cross_target_dp_10k.yaml
```

The pair analysis used 1,000 document-bootstrap replicates and 1,000 within-document permutations with seed `20261013`. It verified model revision, all per-shard SHA-256 values, the full frozen position order, and shared-prefix hashes before computing results. Qwen3-4B/8B bottom-20% Jaccard is 0.204 (95% document-bootstrap CI 0.201–0.207); Spearman margin correlation is 0.493 (95% CI 0.489–0.497). The within-document permutation mean Jaccard is 0.123 (95% interval 0.121–0.126), observed-minus-null is +0.080, and the unadjusted empirical upper-tail p-value is 0.001. This remains a within-Qwen comparison and does not support a cross-family conclusion.

The output directory contains the 104-shard runs and manifests for each model, `position_margins.parquet`, the metric CSVs, `summary.json`, `cross_target_findings.md`, `two_model_analysis.log`, and figures A–D in PDF and PNG. Model weights were evicted after scoring. To restore or compare the original five-target outputs, use the separate `results/cross_target_dp/` directory; do not run the subset analysis against that directory unless its five-target analysis outputs have been saved, because the analyzer writes to the configured output directory.
