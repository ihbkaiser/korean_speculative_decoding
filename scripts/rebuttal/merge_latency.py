#!/usr/bin/env python3
"""Merge per-family latency probes into the T1a headline CSVs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--latency-root", type=Path, default=ROOT / "results/rebuttal/latency")
    args = parser.parse_args()
    root = args.latency_root if args.latency_root.is_absolute() else ROOT / args.latency_root
    family_files = sorted(
        path for path in root.glob("*/latency.csv")
        if path.parent != root
    )
    if not family_files:
        raise FileNotFoundError(f"No family latency files found under {root}")
    latency = pd.concat([pd.read_csv(path) for path in family_files], ignore_index=True)
    latency = latency.drop_duplicates(["model", "op", "n_tokens", "context_len"], keep="last")
    latency.sort_values(["model", "op", "context_len", "n_tokens"], inplace=True)
    latency.to_csv(root / "latency.csv", index=False)

    cost_files = sorted(root.glob("*/cost_ratio.csv"))
    if cost_files:
        costs = pd.concat([pd.read_csv(path) for path in cost_files], ignore_index=True)
        costs = costs.drop_duplicates("pair", keep="last").sort_values("pair")
        costs.to_csv(root / "cost_ratio.csv", index=False)
    print(f"LATENCY_MERGED models={latency.model.nunique()} rows={len(latency)} output={root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
