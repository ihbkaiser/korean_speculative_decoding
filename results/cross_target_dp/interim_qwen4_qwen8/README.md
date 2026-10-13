# Qwen3-4B / Qwen3-8B interim RQ6 snapshot

This directory preserves the two-target interim analysis before the planned five-target analysis replaces the top-level RQ6 aggregate outputs. It was generated with:

```bash
/venv/main/bin/python -u scripts/analyze_cross_target_dp.py --config configs/cross_target_dp.yaml --models qwen4,qwen8
```

The snapshot includes the exact interim analysis tables, summary, findings, figures, frozen evaluation-position manifest, and position metadata. Per-model raw score shards and run manifests remain under `../model_runs/qwen4/` and `../model_runs/qwen8/`.

The two-model results are descriptive for Qwen3-4B and Qwen3-8B only. They do not answer the planned cross-family comparison.
