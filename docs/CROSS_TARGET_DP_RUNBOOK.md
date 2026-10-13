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

`results/cross_target_dp/` contains `evaluation_positions.parquet`, `position_metadata.json`, per-target reduced-score shards and run manifests, `position_margins.parquet`, `pairwise_overlap.csv`, `pairwise_correlations.csv`, `bootstrap_ci.csv`, `permutation_results.csv`, `decision_point_prevalence.csv`, `shared_target_counts.csv`, `shared_target_buckets.csv` (including the required 3+ group), `family_consistency_bootstrap.csv`, `morphological_enrichment.csv`, `summary.json`, `cross_target_findings.md` with a populated result paragraph, and publication PDF/PNG figures A–D.

The primary source of cross-target alignment is `shared_prefix_sha256` plus `(doc_id, char_offset)`. The full prefix text is reconstructed from the frozen document and is not duplicated in the margins table.

## Interpretation limits

The absolute `<1`-nat threshold is secondary because the next-token vocabularies and margin scales differ. Bottom-percentile ranks and overlap on the common position set are primary. Eojeol-onset and Kiwi morphology findings are descriptive associations. They do not establish intrinsic linguistic ambiguity or causal effects. Existing RQ4 conclusions about draft failures remain conditional on a fixed target.
