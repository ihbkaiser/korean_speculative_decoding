# Frozen Table 1 inputs

These four files were downloaded unchanged on 2026-10-10 from the Modal volume
`korean-speculative-decoding-table1-artifacts`, under `/workspace/data/` and
`/workspace/metadata/`. This is the **full 40,000-document pool**, not the separate
256-prompt launcher-smoke fixture. No model weights or run checkpoints are included.

Source: [FineWeb2 by Hugging Face](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2),
Korean `kor_Hang`, `train`, immutable revision
`af9c13333eb981300149d5ca60a8e9d659b276b9`.
The pool was prepared by `src/table1_data.py` with seed 42, NFC/strip normalization,
minimum 64 characters, minimum Hangul ratio 0.35 and exact-text deduplication.
The first 20,000 rows are the common split; the remaining 20,000 are the extra split.
Q1/Q3/M1/G1 use the common split; Q2 uses all 40,000 rows.

Upstream attribution and terms: the [dataset card](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2/blob/af9c13333eb981300149d5ca60a8e9d659b276b9/README.md)
identifies the dataset license as [ODC-By 1.0](https://opendatacommons.org/licenses/by/1-0/)
and also refers to [Common Crawl's Terms of Use](https://commoncrawl.org/terms-of-use).
This repository's code license does not replace those data terms. These are
public-web research texts, not a guarantee that all content is free of personal,
offensive or otherwise sensitive material.

## Install or download

A normal Git checkout now includes the input files directly (no Git LFS).
The Parquet is 94,425,998 bytes. For an existing checkout, use `git pull --ff-only`
from the directory containing `scripts/table1_pipeline.py`. If local untracked
files conflict, preserve and compare them; do not delete or overwrite them blindly.

For manual downloads, run the following **inside that same repository directory**.
It refuses to overwrite an existing input file and pins the source to a commit.
Replace `DATA_COMMIT` with the data-publication commit, or a later commit containing
the same inputs. Do not download a GitHub HTML preview in place of the Parquet.

```bash
DATA_COMMIT=REPLACE_WITH_DATA_PUBLICATION_COMMIT
BASE_URL="https://raw.githubusercontent.com/ihbkaiser/korean_speculative_decoding/$DATA_COMMIT"
mkdir -p data metadata
for file in data/prompts_40k.parquet data/common_20k_ids.txt data/extra_20k_ids.txt metadata/dataset_revision.json data/table1_frozen.sha256; do
  if [ -e "$file" ]; then
    echo "Existing file: $file; preserve/compare it before downloading." >&2
    break
  fi
  curl --fail --location --retry 3 "$BASE_URL/$file" --output "$file" || break
done
sha256sum -c data/table1_frozen.sha256
```

Only launch inference after **all four checksum entries say OK**:

```bash
bash scripts/run_company_table1_fast.sh all --output-dir /path/to/new_fast_output
```

The output directory does not relocate these inputs. Do not mix old strict
checkpoints with fast-mode output or regenerate the pool to repair a path error.

## Integrity verified before publication

- 40,000 unique document IDs and unique normalized text hashes.
- Every stored text hash matches its UTF-8 text, and every pool index is in order.
- ID lists exactly match the first/last 20,000 Parquet rows and do not overlap.
- Dataset/config/split, text lengths and stored filtering ratios match metadata.
- Ordered ID SHA256 matches `metadata/dataset_revision.json`:
  `638d21a09e2f080c87e93b63a8f47009016074ddb7ed6fc2b8770991db681cd4`.
- Fast-config pair loading yields 20k/40k/20k/20k/20k records.

File-byte checksums are in `table1_frozen.sha256`; they also protect the original
line endings of the metadata and ID lists. A credential-pattern scan found two
reviewed false positives (a key-header reference in tutorial prose without a key
payload, and part of a news URL); no content was changed. This scan is not a
comprehensive audit of web-corpus content or proof of scalar decoding parity.
