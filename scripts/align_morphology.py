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

from src.naacl_experiment import (
    atomic_json,
    atomic_parquet,
    load_config,
    load_tokenizer,
    output_root,
    prompt_ids_sha256,
    read_parquet_records,
    select_prompts,
)
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


def _human_references(root: Path, config, target: str, split: str, max_prompts: int | None, tokenizer):
    """Rebuild the exact human next-token references used by TF scoring."""
    import pandas as pd

    selection = select_prompts(config, split)
    if max_prompts is not None:
        selection = selection[:max_prompts]
    max_prompt = int(config["data"]["max_prompt_tokens"])
    max_new = int(config["data"]["max_new_tokens"])
    candidates = []
    skipped = []
    for row in selection:
        ids = list(map(int, tokenizer(str(row["text"]), add_special_tokens=False)["input_ids"]))
        prompt_ids = ids[:max_prompt]
        continuation_ids = ids[len(prompt_ids) : len(prompt_ids) + max_new]
        if not prompt_ids or not continuation_ids:
            skipped.append(str(row["doc_id"]))
            continue
        candidates.append({
            **row,
            "prompt_ids": prompt_ids,
            "continuation_ids": continuation_ids,
        })

    target_dir = root / "teacher_forced" / "human" / target / target / split
    manifest_path = target_dir / "run_metadata.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Human target TF manifest is missing: {manifest_path}")
    manifest = __import__("json").loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("status") != "COMPLETE" or manifest.get("continuation") != "human":
        raise RuntimeError(f"Human target TF is not complete: {manifest_path}")

    tf_frames = []
    for path in sorted(target_dir.glob("shard-*.parquet")):
        tf_frames.append(pd.read_parquet(path, columns=["doc_id", "position", "target_token_id"]))
    if not tf_frames:
        raise FileNotFoundError(f"Human target TF shards are missing under {target_dir}")
    tf = pd.concat(tf_frames, ignore_index=True)
    if tf.duplicated(["doc_id", "position"]).any():
        raise AssertionError("Human target TF has duplicate token positions")
    tf["doc_id"] = tf["doc_id"].astype(str)
    observed_docs = set(tf.doc_id.unique())
    expected_docs = {str(row["doc_id"]) for row in candidates}
    if observed_docs != expected_docs:
        raise AssertionError(
            f"Human target TF doc coverage differs: observed={len(observed_docs)}, expected={len(expected_docs)}"
        )

    refs = []
    by_doc = {doc_id: group.sort_values("position") for doc_id, group in tf.groupby("doc_id", sort=False)}
    for row in candidates:
        doc_id = str(row["doc_id"])
        continuation_ids = row["continuation_ids"]
        saved = by_doc[doc_id]
        positions = saved.position.astype(int).tolist()
        token_ids = saved.target_token_id.astype(int).tolist()
        if positions != list(range(len(continuation_ids))) or token_ids != continuation_ids:
            raise AssertionError(f"Human target TF token IDs do not match source text for {doc_id}")
        refs.append({
            "pair_id": target,
            "doc_id": doc_id,
            "raw_text_hash": str(row["raw_text_hash"]),
            "prompt_ids": row["prompt_ids"],
            "continuation_ids": continuation_ids,
            "pool_index": int(row["pool_index"]),
            "pool_split": str(row["pool_split"]),
            "source_index": int(row["source_index"]),
        })

    expected_prompt_hash = prompt_ids_sha256(candidates)
    if manifest.get("prompt_count") != len(candidates) or manifest.get("prompt_ids_sha256") != expected_prompt_hash:
        raise AssertionError("Human alignment selection differs from the completed target TF run")
    return refs, len(selection), skipped


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--target", required=True, help="Target model key, e.g. qwen4")
    parser.add_argument("--split", choices=("common", "all", "extra"), required=True)
    parser.add_argument("--continuation", choices=("generated", "human"), default="generated")
    parser.add_argument("--analyzer", choices=("kiwi", "mecab"), default="kiwi")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--model-path", type=Path, help="Tokenizer snapshot override; weights are not loaded")
    parser.add_argument("--max-prompts", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.target not in config["models"]:
        parser.error(f"unknown target model key {args.target!r}")
    out = output_root(config, args.output_dir)
    tokenizer = load_tokenizer(config, args.target, args.model_path.expanduser().resolve() if args.model_path else None)
    if args.continuation == "generated":
        selection = select_prompts(config, args.split)
        if args.max_prompts is not None:
            selection = selection[: args.max_prompts]
        ref_dir = out / "continuations" / args.target / args.split
        references = _references(ref_dir, selection)
        skipped = []
        requested_prompts = len(selection)
    else:
        references, requested_prompts, skipped = _human_references(
            out, config, args.target, args.split, args.max_prompts, tokenizer
        )
    if args.analyzer == "kiwi":
        analyzer = Kiwi()
    else:
        analyzer = MecabKoAdapter()
    destination = out / "aligned" / args.analyzer / args.target
    if args.continuation == "human":
        destination = destination / "human"
    destination = destination / args.split
    destination.mkdir(parents=True, exist_ok=True)
    shard_size = int(config["data"]["shard_prompts"])
    completed = []
    for shard_index, start in enumerate(range(0, len(references), shard_size)):
        shard_refs = references[start : start + shard_size]
        shard_path = destination / f"shard-{shard_index:05d}.parquet"
        if shard_path.exists() and not args.force:
            print(f"ALIGN_RESUME_SKIP {shard_path}", flush=True)
            completed.append({"shard": shard_index, "prompts": len(shard_refs), "resumed": True})
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
            "prompts": len(shard_refs),
            "aligned_rows": len(aligned),
            "resumed": False,
        })
        print(f"ALIGN_SHARD_COMPLETE target={args.target} analyzer={args.analyzer} shard={shard_index:05d} rows={len(aligned)}", flush=True)
    atomic_json(destination / "run_metadata.json", {
        "stage": "target_path_morphology_alignment",
        "continuation": args.continuation,
        "target_key": args.target,
        "revision": config["models"][args.target]["revision"],
        "split": args.split,
        "analyzer": args.analyzer,
        "prompt_count": len(references),
        "requested_prompt_count": requested_prompts,
        "skipped_short_document_count": len(skipped),
        "completed_shards": completed,
        "status": "COMPLETE",
    })
    print(f"ALIGN_COMPLETE target={args.target} split={args.split} analyzer={args.analyzer} output={destination}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
