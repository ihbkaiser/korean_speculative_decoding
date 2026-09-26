#!/usr/bin/env python3
"""Download the first deterministic Korean Wikipedia prompts into JSONL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data import prepare_prompt_cache


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/pilot.yaml"))
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    import yaml

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))["experiment"]
    limit = args.limit or int(config["pilot_prompts"])
    count = prepare_prompt_cache(
        config["data_cache"],
        dataset_id=config["dataset_id"],
        config_name=config["dataset_config"],
        split=config["dataset_split"],
        limit=limit,
    )
    print(f"Wrote {count} prompts to {config['data_cache']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
