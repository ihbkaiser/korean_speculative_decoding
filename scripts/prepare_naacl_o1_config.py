#!/usr/bin/env python3
"""Create the pinned configuration for optional O1 (Qwen3-14B target)."""

from __future__ import annotations

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "configs/naacl_experiments.yaml"
OUTPUT = ROOT / "configs/naacl_o1_qwen14.yaml"
QWEN14_REVISION = "0b0bd3732e2c374d483664439ea334928b65f304"


def main() -> int:
    config = yaml.safe_load(BASE.read_text(encoding="utf-8"))
    config["models"]["qwen14"] = {
        "id": "Qwen/Qwen3-14B-Base",
        "revision": QWEN14_REVISION,
        "family": "Qwen3",
        "size": "14B",
        "storage_gib": 27.50,
    }
    config["pairs"].update({
        "O1_Q06": {"family": "Qwen3", "draft": "qwen06", "target": "qwen14", "split": "common", "legacy_name": "0.6B→14B"},
        "O1_Q17": {"family": "Qwen3", "draft": "qwen17", "target": "qwen14", "split": "common", "legacy_name": "1.7B→14B"},
        "O1_Q4": {"family": "Qwen3", "draft": "qwen4", "target": "qwen14", "split": "common", "legacy_name": "4B→14B"},
    })
    config["paths"]["outputs"] = "results/naacl_experiments/o1_qwen14"
    OUTPUT.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(f"O1_CONFIG {OUTPUT.relative_to(ROOT)} qwen14_revision={QWEN14_REVISION}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
