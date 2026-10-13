#!/usr/bin/env python3
"""Freeze shared character-offset evaluation positions for cross-target RQ6."""

from __future__ import annotations

import argparse
from importlib.metadata import version
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cross_target_dp import add_matched_morphology_positions, sample_eojeol_positions
from src.naacl_experiment import atomic_json, atomic_parquet, load_config, select_prompts, sha256_file


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/cross_target_dp.yaml")
    parser.add_argument("--max-documents", type=int)
    parser.add_argument("--positions-per-document", type=int)
    parser.add_argument("--context-window-chars", type=int)
    parser.add_argument("--min-context-chars", type=int)
    parser.add_argument("--morphology-anchor-words-per-document", type=int)
    parser.add_argument("--skip-morphology", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    data_cfg = config["data"]
    n_docs = args.max_documents or int(data_cfg["max_documents"])
    per_doc = args.positions_per_document or int(data_cfg["positions_per_document"])
    window_chars = args.context_window_chars or int(data_cfg["context_window_chars"])
    min_context = args.min_context_chars or int(data_cfg["min_context_chars"])
    anchors = args.morphology_anchor_words_per_document
    if anchors is None:
        anchors = int(data_cfg["morphology_anchor_words_per_document"])

    source = ROOT / data_cfg["prompts"]
    documents = select_prompts(config, "common")[:n_docs]
    primary, metadata, included = sample_eojeol_positions(
        documents,
        max_documents=n_docs,
        positions_per_document=per_doc,
        context_window_chars=window_chars,
        min_context_chars=min_context,
    )
    morph_positions: list[dict] = []
    morph_metadata: dict = {"status": "NOT_RUN"}
    if not args.skip_morphology:
        from kiwipiepy import Kiwi

        morph_positions, morph_metadata = add_matched_morphology_positions(
            primary,
            included,
            Kiwi(),
            anchor_words_per_document=anchors,
            context_window_chars=window_chars,
        )

    output_dir = ROOT / config["paths"]["outputs"]
    output_dir.mkdir(parents=True, exist_ok=True)
    positions_path = output_dir / "evaluation_positions.parquet"
    metadata_path = output_dir / "position_metadata.json"
    if positions_path.exists() and not args.force:
        raise FileExistsError(f"{positions_path} exists; use --force to replace the frozen sample")

    all_positions = primary + morph_positions
    all_positions.sort(key=lambda row: (str(row["doc_id"]), int(row["char_offset"])))
    serializable = [
        {key: value for key, value in row.items() if key != "prefix_text"}
        for row in all_positions
    ]
    atomic_parquet(serializable, positions_path)
    position_ids = "\n".join(row["position_id"] for row in all_positions)
    import hashlib

    full_metadata = {
        "schema": config["schema"],
        "status": "POSITIONS_FROZEN",
        "data_path": str(source.relative_to(ROOT)),
        "data_sha256": sha256_file(source),
        "common_document_count_requested": n_docs,
        "position_count": len(all_positions),
        "position_ids_sha256": hashlib.sha256(position_ids.encode("utf-8")).hexdigest(),
        "positions_path": str(positions_path.relative_to(ROOT)),
        "positions_sha256": sha256_file(positions_path),
        "primary": metadata,
        "morphology": {
            **morph_metadata,
            "kiwipiepy_version": version("kiwipiepy") if not args.skip_morphology else None,
        },
        "configuration": {
            "positions_per_document": per_doc,
            "context_window_chars": window_chars,
            "min_context_chars": min_context,
            "morphology_anchor_words_per_document": anchors,
        },
        "prefix_reconstruction": "text[context_start:char_offset]; identical document text and Unicode offsets for every target, independently tokenized by each pinned tokenizer.",
    }
    atomic_json(metadata_path, full_metadata)
    print(
        f"POSITIONS_READY documents={metadata['documents_included']}/{n_docs} "
        f"primary={len(primary)} morphology={len(morph_positions)} total={len(all_positions)} "
        f"path={positions_path.relative_to(ROOT)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
