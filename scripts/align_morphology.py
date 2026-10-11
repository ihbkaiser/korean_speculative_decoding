#!/usr/bin/env python3
"""Align frozen target continuations with Kiwi or optional MeCab-ko morphology."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kiwipiepy import Kiwi
from tqdm import tqdm

from src.naacl_experiment import atomic_json, atomic_parquet, load_config, load_tokenizer, output_root, read_parquet_records
from src.table1_morphology import _make_prompt_alignment_rows


class MecabKoAdapter:
    """Expose MeCab-ko nodes through the small Kiwi token interface we use."""

    def __init__(self):
        try:
            import MeCab
        except ImportError as exc:
            raise RuntimeError("MeCab-ko alignment requires the optional `python-mecab-ko` package") from exc
        self.tagger = MeCab.Tagger()

    def tokenize(self, text):
        node = self.tagger.parseToNode(text)
        offset = 0
        result = []
        while node is not None:
            surface = str(node.surface or "")
            if surface:
                feature = str(node.feature or "").split(",", 1)[0]
                tag = feature.split("+", 1)[0] or "UN"
                if text[offset : offset + len(surface)] != surface:
                    found = text.find(surface, offset)
                    if found < 0:
                        raise RuntimeError(f"MeCab surface is not in eojeol at offset {offset}: {surface!r}")
                    offset = found
                result.append(SimpleNamespace(form=surface, tag=tag, start=offset, len=len(surface)))
                offset += len(surface)
            node = node.next
        return result


def _references(directory: Path, selection):
    wanted = {str(row["doc_id"]): row for row in selection}
    found = {}
    for path in sorted(directory.glob("shard-*.parquet")):
        for row in read_parquet_records(path):
            doc_id = str(row["doc_id"])
            if doc_id in wanted:
                found[doc_id] = row
    missing = set(wanted) - set(found)
    if missing:
        raise RuntimeError(f"Missing {len(missing)} target continuation references under {directory}")
    return [found[str(row["doc_id"])] for row in selection]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--target", required=True, help="Target model key, e.g. qwen4")
    parser.add_argument("--split", choices=("common", "all", "extra"), required=True)
    parser.add_argument("--analyzer", choices=("kiwi", "mecab"), default="kiwi")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-path", type=Path, help="Tokenizer snapshot override; weights are not loaded")
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.target not in config["models"]:
        parser.error(f"unknown target model key {args.target!r}")
    selection = __import__("src.naacl_experiment", fromlist=["select_prompts"]).select_prompts(config, args.split)
    if args.max_prompts is not None:
        selection = selection[: args.max_prompts]
    out = output_root(config, args.output_dir)
    ref_dir = out / "continuations" / args.target / args.split
    references = _references(ref_dir, selection)
    tokenizer = load_tokenizer(config, args.target, args.model_path.expanduser().resolve() if args.model_path else None)
    if args.analyzer == "kiwi":
        analyzer = Kiwi()
    else:
        analyzer = MecabKoAdapter()
    destination = out / "aligned" / args.analyzer / args.target / args.split
    destination.mkdir(parents=True, exist_ok=True)
    shard_size = int(config["data"]["shard_prompts"])
    completed = []
    for shard_index, start in enumerate(range(0, len(selection), shard_size)):
        shard_selection = selection[start : start + shard_size]
        shard_ids = {str(row["doc_id"]) for row in shard_selection}
        shard_refs = [row for row in references if str(row["doc_id"]) in shard_ids]
        shard_path = destination / f"shard-{shard_index:05d}.parquet"
        if shard_path.exists() and not args.force:
            print(f"ALIGN_RESUME_SKIP {shard_path}", flush=True)
            completed.append({"shard": shard_index, "prompts": len(shard_selection), "resumed": True})
            continue
        aligned = []
        for reference in tqdm(shard_refs, desc=f"{args.target} {args.analyzer} {shard_index}", unit="prompt", dynamic_ncols=True, mininterval=1):
            adapter_reference = {
                "pair_id": args.target,
                "doc_id": str(reference["doc_id"]),
                "raw_text_hash": str(reference["raw_text_hash"]),
                "prompt_token_ids": reference["prompt_ids"],
                "target_continuation_token_ids": reference["continuation_ids"],
                "pool_index": int(reference["pool_index"]),
                "pool_split": str(reference["pool_split"]),
                "source_index": int(reference["source_index"]),
            }
            aligned.extend(_make_prompt_alignment_rows(adapter_reference, tokenizer=tokenizer, kiwi=analyzer))
        atomic_parquet(aligned, shard_path)
        completed.append({
            "shard": shard_index,
            "prompts": len(shard_selection),
            "aligned_rows": len(aligned),
            "resumed": False,
        })
        print(f"ALIGN_SHARD_COMPLETE target={args.target} analyzer={args.analyzer} shard={shard_index:05d} rows={len(aligned)}", flush=True)
    atomic_json(destination / "run_metadata.json", {
        "stage": "target_path_morphology_alignment",
        "target_key": args.target,
        "revision": config["models"][args.target]["revision"],
        "split": args.split,
        "analyzer": args.analyzer,
        "prompt_count": len(selection),
        "completed_shards": completed,
        "status": "COMPLETE",
    })
    print(f"ALIGN_COMPLETE target={args.target} split={args.split} analyzer={args.analyzer} output={destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
