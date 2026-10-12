# NAACL teacher-forced experiment runbook

This runbook follows [`NAACL_experiments.md`](../NAACL_experiments.md). It
reuses the frozen 40k FineWeb2 prompt pool, emits target continuations and
teacher-forced Parquet in 2,000-prompt shards, and keeps model weights in a
separate staging directory so they can be evicted after a family is finished.

## Environment and storage

The current Vast instance has an RTX A6000 48 GB, a 31 GiB `/dev/shm`, and a
32 GiB overlay filesystem. The model plan is about 82 GiB of *Transformers
weight shards* at the pinned revisions (the Ministral repositories also carry
duplicate `consolidated.safetensors` exports; the downloader skips those).
Keep outputs on persistent experiment storage when a volume is attached. On
this instance, `/workspace` is not persistent across recycle/destroy.

Use the existing PyTorch environment and install the pinned experiment stack:

```bash
source /venv/main/bin/activate
uv pip install --python /venv/main/bin/python -r requirements-naacl.txt
export NAACL_MODEL_DIR=/dev/shm/naacl_models
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Gemma-3 is gated on Hugging Face. Configure an authorized token as `HF_TOKEN`
in the host environment before downloading Gemma; the scripts read it from the
environment and do not write it to manifests or logs. The frozen pool is already
in `data/prompts_40k.parquet` and must not be resampled.

Check required model files and available space without downloading:

```bash
python scripts/download_models.py --plan
```

Download one checkpoint at a time with `--model KEY`. `--pair PAIR` downloads
both checkpoints when they fit together; `--evict-pair PAIR` removes their
weight files after their last job while retaining the small tokenizer/config
files used for alignment. Downloads resume from the Hugging Face cache in the
selected local directory.

## Qwen target continuations and scores

Qwen weights fit together in `/dev/shm`, so retain them through all six grid
cells and evict them after Qwen simulation and V0 validation. Run the target
generation and target self-score for each target exactly once:

```bash
python scripts/download_models.py --model qwen17 --smoke-load
python scripts/gen_target.py --target qwen17 --split common
python scripts/tf_score.py --model qwen17 --target qwen17 --split common --role target
python scripts/align_morphology.py --target qwen17 --split common

python scripts/download_models.py --model qwen4 --smoke-load
python scripts/gen_target.py --target qwen4 --split all
python scripts/tf_score.py --model qwen4 --target qwen4 --split all --role target
python scripts/align_morphology.py --target qwen4 --split all

python scripts/download_models.py --model qwen8 --smoke-load
python scripts/gen_target.py --target qwen8 --split common
python scripts/tf_score.py --model qwen8 --target qwen8 --split common --role target
python scripts/align_morphology.py --target qwen8 --split common
```

Then score all six draft/target cells. `qwen4/all` covers both Q2's 40k pool
and the common-20k grid cells; the scorer filters the latter by frozen
`pool_index` when simulating them.

```bash
python scripts/download_models.py --model qwen06 --smoke-load
python scripts/audit_tokenizers.py --pair Q1
python scripts/audit_tokenizers.py --pair Q2
python scripts/audit_tokenizers.py --pair Q5
python scripts/tf_score.py --model qwen06 --target qwen17 --split common --role draft
python scripts/tf_score.py --model qwen06 --target qwen4 --split all --role draft
python scripts/tf_score.py --model qwen06 --target qwen8 --split common --role draft

python scripts/audit_tokenizers.py --pair Q4
python scripts/audit_tokenizers.py --pair Q6
python scripts/tf_score.py --model qwen17 --target qwen4 --split common --role draft
python scripts/tf_score.py --model qwen17 --target qwen8 --split common --role draft

python scripts/audit_tokenizers.py --pair Q3
python scripts/tf_score.py --model qwen4 --target qwen8 --split common --role draft
```

Run the CPU simulator for each cell. It counts only verified proposals in the
rejection denominator, records invalidated suffix work without inventing its
token IDs, and skips target bonus positions as draft proposals:

```bash
for pair in Q1 Q2 Q3 Q4 Q5 Q6; do
  split=common
  [[ "$pair" == Q2 ]] && split=all
  python scripts/simulate_sd.py --pair "$pair" --split "$split"
done

```

## Live validation and remaining core family

After Qwen labels are simulated, compare them with the repository's live
microbatched cached decoder on the planned 1,000 Q2 prompts:

```bash
python scripts/validate_live_sd.py --pair Q2
```

After Q2 V0 is complete, release the Qwen weight files; tokenizer/config files
remain available for later audits and alignment.

```bash
python scripts/download_models.py --evict-pair Q1
python scripts/download_models.py --evict-pair Q2
python scripts/download_models.py --evict-pair Q3
```

For the G0 timing gate, use a separate output directory named
`results/naacl_pilot`. The pilot uses the first 200 common prompts for the
Qwen-4B continuation, target/draft TF scores, and alignment, then compares the
first 100 Q2 prompts with live SD. This keeps pilot shards separate from the
full 2k-prompt checkpoints:

```bash
pilot=results/naacl_pilot
python scripts/gen_target.py --target qwen4 --split common --max-prompts 200 --output-dir "$pilot"
python scripts/tf_score.py --model qwen4 --target qwen4 --split common --role target --max-prompts 200 --output-dir "$pilot"
python scripts/tf_score.py --model qwen06 --target qwen4 --split common --role draft --max-prompts 200 --output-dir "$pilot"
python scripts/align_morphology.py --target qwen4 --split common --max-prompts 200 --output-dir "$pilot"
python scripts/simulate_sd.py --pair Q2 --split common --max-prompts 100 --output-dir "$pilot"
python scripts/validate_live_sd.py --pair Q2 --prompts 100 --output-dir "$pilot"
```

Record measured per-job ETAs before starting the full split.

The validator requires at least 95% rejection-label agreement on eligible
proposals whose committed context is shared, with zero unmatched positions
in that context. It also writes `position_comparison.parquet`,
prompt-cluster bootstrap intervals, and agreement sensitivity at fixed
minimum-margin thresholds. Margin sensitivity is diagnostic and does not
change the gate. If simulation events predate the margin fields, regenerate
them with `--force` before validation:

```bash
python scripts/simulate_sd.py --pair Q2 --split common --max-prompts 100 \
  --output-dir "$pilot" --force
python scripts/validate_live_sd.py --pair Q2 --prompts 100 \
  --output-dir "$pilot" --force
```

The report's `g0_gate_status` covers label agreement only; the job-ETA check
remains a separate G0 requirement and must be recorded with the measured
schedule. The V0 report records that this ETA component was not assessed by
the validator.

The 95% threshold is a post-freeze amendment made after the 100-prompt Q2
pilot measured 98.11% agreement and was accepted. The original measurement is
retained; see `results/naacl_experiments/provenance/experiment_decisions.json`.

For G1, generate and score the Gemma-12B target on the common split, align its
continuations, and then load Gemma-4B as the draft. Since Gemma-4B plus 12B is
slightly larger than the 31 GiB tmpfs, use separate model roots for the V0 live
pair validation if needed:

```bash
export NAACL_MODEL_DIR_GEMMA12=/dev/shm/naacl_models
export NAACL_MODEL_DIR_GEMMA4=/workspace/naacl_gemma_models
python scripts/download_models.py --model gemma12 --smoke-load
python scripts/gen_target.py --target gemma12 --split common
python scripts/tf_score.py --model gemma12 --target gemma12 --split common --role target
python scripts/align_morphology.py --target gemma12 --split common
python scripts/download_models.py --model gemma4 --smoke-load
python scripts/audit_tokenizers.py --pair G1
python scripts/tf_score.py --model gemma4 --target gemma12 --split common --role draft
python scripts/simulate_sd.py --pair G1 --split common
python scripts/validate_live_sd.py --pair G1
```

For M1, download/score the Ministral-8B target and align it first, then load
Ministral-3B to score the draft. The pinned Rust tokenizer path enables
`fix_mistral_regex=True` for both models:

```bash
python scripts/download_models.py --pair M1 --smoke-load
python scripts/gen_target.py --target ministral8 --split common
python scripts/tf_score.py --model ministral8 --target ministral8 --split common --role target
python scripts/align_morphology.py --target ministral8 --split common
python scripts/audit_tokenizers.py --pair M1
python scripts/tf_score.py --model ministral3 --target ministral8 --split common --role draft
python scripts/simulate_sd.py --pair M1 --split common
```

The live validation entry point is restricted to Q2 and G1, as specified in
the plan. Its report includes both all-position agreement and agreement on
positions whose preceding committed tokens still match the generated target
continuation. The latter is the valid simulator comparison after a numerical
near-tie changes a target continuation; later contexts are no longer the same.
For the full validity check, run Q2 on 1,000 prompts and G1 on 500 prompts as
specified in the plan; pilot-sized or custom runs are labeled separately.
M1 does not need a V0 run.

## Table 1

Build the paper files after the required target, draft, alignment, and k=4
simulation shards have landed:

```bash
python scripts/build_table1.py
```

The outputs are `results/naacl_experiments/table1/table1.{csv,md,tex}` plus
`table1_build_metadata.json`. Missing jobs remain visibly `INCOMPLETE`; no
results are fabricated. A publishable Table 1 requires every planned pair row
to be `COMPLETE` and the V0 reports to be included in the paper's validity text.
