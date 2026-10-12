#!/usr/bin/env python3
"""Align frozen target continuations with Kiwi, MeCab-ko, or Japanese UniDic."""

from __future__ import annotations

import argparse
from importlib.metadata import version
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


class FugashiUniDicAdapter:
    """Expose UniDic morphemes through the repository's canonical POS groups."""

    def __init__(self):
        try:
            from fugashi import Tagger
            import unidic  # noqa: F401 - makes the selected dictionary explicit
        except ImportError as exc:
            raise RuntimeError("Japanese alignment requires `fugashi[unidic]` and its downloaded UniDic dictionary") from exc
        self.tagger = Tagger()

    @staticmethod
    def _canonical_pos(surface: str, feature) -> str:
        pos1 = str(getattr(feature, "pos1", ""))
        pos2 = str(getattr(feature, "pos2", ""))
        if pos1 == "名詞":
            return "NNP" if pos2 == "固有名詞" else "NNG"
        if pos1 == "助詞":
            if surface == "は" and pos2 == "係助詞":
                return "JX"
            if surface == "が" and pos2 == "格助詞":
                return "JKS"
            if pos2 == "格助詞":
                return "JKB"
            if pos2 == "接続助詞":
                return "JC"
            return "JX"
        if pos1 == "動詞":
            return "VV"
        if pos1 == "形容詞":
            return "VA"
        if pos1 == "助動詞":
            return "EP"
        if pos1 == "副詞":
            return "MAG"
        if pos1 == "連体詞":
            return "MM"
        if pos1 == "接続詞":
            return "JC"
        if pos1 == "接頭辞":
            return "XPN"
        if pos1 == "接尾辞":
            return "XSN"
        if pos1 == "感動詞":
            return "IC"
        if pos1 == "記号":
            return "SP"
        return "UN"

    def tokenize(self, text):
        offset = 0
        result = []
        for word in self.tagger(text):
            surface = str(word.surface or "")
            if not surface:
                continue
            if text[offset : offset + len(surface)] != surface:
                found = text.find(surface, offset)
                if found < 0:
                    raise RuntimeError(f"UniDic surface is not in eojeol at offset {offset}: {surface!r}")
                offset = found
            result.append(SimpleNamespace(
                form=surface,
                tag=self._canonical_pos(surface, word.feature),
                start=offset,
                len=len(surface),
            ))
            offset += len(surface)
        return result


def _japanese_particle_roles(row: dict) -> dict:
    """Identify nominal→は / nominal→が boundaries using aligned UniDic spans."""
    surfaces = row.get("morpheme_surfaces") or []
    tags = row.get("morpheme_pos_tags") or []
    spans = row.get("morpheme_char_spans") or []
    morphemes = []
    for surface, tag, span in zip(surfaces, tags, spans):
        if not isinstance(span, (list, tuple)) or len(span) != 2 or span[0] is None or span[1] is None:
            continue
        morphemes.append((int(span[0]), int(span[1]), str(surface), str(tag)))

    def role_at(position):
        left = next((item for item in morphemes if item[1] == position), None)
        right = next((item for item in morphemes if item[0] == position), None)
        if left is None or right is None or left[3] not in {"NNG", "NNP", "NNB", "NP", "NR"}:
            return None, None
        if right[2] == "は" and right[3] == "JX":
            return "TOPIC_HA", right[2]
        if right[2] == "が" and right[3] == "JKS":
            return "SUBJECT_GA", right[2]
        return None, right[2]

    roles_crossed = []
    surfaces_crossed = []
    if row.get("alignment_class") == "CROSS":
        for position in row.get("crossed_boundary_positions") or []:
            role, surface = role_at(int(position))
            if role:
                roles_crossed.append(role)
            if surface:
                surfaces_crossed.append(surface)

    nearest_role = None
    nearest_surface = None
    if row.get("alignment_class") == "WITHIN_SPLIT" and not row.get("nearest_boundary_ambiguous"):
        position = row.get("nearest_boundary_position")
        if position is not None:
            nearest_role, nearest_surface = role_at(int(position))

    return {
        "japanese_particle_roles_crossed": roles_crossed,
        "japanese_particle_surfaces_crossed": surfaces_crossed,
        "japanese_particle_role_nearest": nearest_role,
        "japanese_particle_surface_nearest": nearest_surface,
    }


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
    parser.add_argument("--analyzer", choices=("kiwi", "mecab", "fugashi"), default="kiwi")
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
    elif args.analyzer == "mecab":
        analyzer = MecabKoAdapter()
    else:
        analyzer = FugashiUniDicAdapter()
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
            prompt_rows = _make_prompt_alignment_rows(adapter_reference, tokenizer=tokenizer, kiwi=analyzer)
            if args.analyzer == "fugashi":
                prompt_rows = [
                    {**row, **_japanese_particle_roles(row)}
                    for row in prompt_rows
                ]
            aligned.extend(prompt_rows)
        atomic_parquet(aligned, shard_path)
        completed.append({
            "shard": shard_index,
            "prompts": len(shard_refs),
            "aligned_rows": len(aligned),
            "resumed": False,
        })
        print(f"ALIGN_SHARD_COMPLETE target={args.target} analyzer={args.analyzer} shard={shard_index:05d} rows={len(aligned)}", flush=True)
    analyzer_metadata = {}
    if args.analyzer == "fugashi":
        analyzer_metadata = {
            "fugashi_version": version("fugashi"),
            "unidic_package_version": version("unidic"),
            "unidic_dictionary_version": "3.1.0",
            "pos_mapping": "UniDic 名詞→NNG/NNP; は(係助詞)→JX; が(格助詞)→JKS; other particles map to canonical particle tags.",
        }
    atomic_json(destination / "run_metadata.json", {
        "stage": "target_path_morphology_alignment",
        "continuation": args.continuation,
        "target_key": args.target,
        "revision": config["models"][args.target]["revision"],
        "split": args.split,
        "analyzer": args.analyzer,
        **analyzer_metadata,
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
