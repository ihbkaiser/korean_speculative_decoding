"""Kiwi-based generated-token alignment and Table 1 aggregation."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

from .alignment import token_ids_to_character_offsets
from .table1_alignment import (
    aggregate_alignment_rows,
    assert_alignment_invariants,
    build_aligned_token_row,
    support_counts_by_fine_boundary,
)
from .table1_data import load_prompt_pool


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _write_parquet(rows: list[dict[str, Any]], path: Path) -> None:
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    pd.DataFrame(rows).to_parquet(temporary, index=False, engine="pyarrow")
    temporary.replace(path)


def _markdown_table(frame: Any) -> str:
    """Render a compact GFM table without relying on pandas' tabulate extra."""
    columns = [str(column) for column in frame.columns]
    values = [["" if value is None else str(value) for value in row] for row in frame.itertuples(index=False, name=None)]
    widths = [len(column) for column in columns]
    for row in values:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], len(value))
    fmt = lambda row: "| " + " | ".join(value.ljust(widths[index]) for index, value in enumerate(row)) + " |"
    lines = [fmt(columns), fmt(["-" * width for width in widths])]
    lines.extend(fmt(row) for row in values)
    return "\n".join(lines) + "\n"


def _eojeol_rows(text: str) -> list[tuple[str, int, int]]:
    return [(match.group(0), match.start(), match.end()) for match in re.finditer(r"\S+", text, flags=re.UNICODE)]


def _continuation_window(full_text: str, prompt_char_end: int) -> tuple[int, str]:
    """Return a continuation start that never treats a prompt-fragment as an eojeol.

    Token truncation can end in the middle of a whitespace-delimited unit.  The
    partial unit contains both prompt and generated text, so assigning it to
    the generated continuation would leak an incomplete morphology context.
    We conservatively discard that entire unit and begin after its next
    whitespace boundary.
    """
    if prompt_char_end < 0 or prompt_char_end > len(full_text):
        raise ValueError("prompt_char_end must lie inside full_text")
    continuation_start = prompt_char_end
    if (
        0 < prompt_char_end < len(full_text)
        and not full_text[prompt_char_end - 1].isspace()
        and not full_text[prompt_char_end].isspace()
    ):
        boundary = re.search(r"\s", full_text[prompt_char_end:])
        continuation_start = len(full_text) if boundary is None else prompt_char_end + boundary.end()
    while continuation_start < len(full_text) and full_text[continuation_start].isspace():
        continuation_start += 1
    return continuation_start, full_text[continuation_start:]


def _kiwi_morphemes(eojeol: str, kiwi: Any) -> list[dict[str, Any]]:
    rows = []
    for token in kiwi.tokenize(eojeol):
        start = int(token.start)
        end = start + int(token.len)
        rows.append({
            "surface": str(token.form),
            "pos": str(token.tag),
            "start": start,
            "end": end,
        })
    return rows


def _token_core_text(text: str, start: int, end: int) -> str:
    """Return visible text, excluding whitespace marker material."""
    return text[start:end].strip()


def _make_prompt_alignment_rows(
    reference: dict[str, Any],
    *,
    tokenizer: Any,
    kiwi: Any,
) -> list[dict[str, Any]]:
    prompt_ids = [int(value) for value in reference["prompt_token_ids"]]
    generated_ids = [int(value) for value in reference["target_continuation_token_ids"]]
    all_ids = prompt_ids + generated_ids
    alignment = token_ids_to_character_offsets(tokenizer, all_ids)
    full_text = str(alignment["text"])
    prompt_offsets = [span for span in alignment["offsets"][:len(prompt_ids)] if span[1] > span[0]]
    prompt_char_end = max((int(span[1]) for span in prompt_offsets), default=0)
    continuation_start, continuation_text = _continuation_window(full_text, prompt_char_end)
    prompt_boundary_excluded = continuation_start > prompt_char_end
    eojeols = _eojeol_rows(continuation_text)
    special_ids = set(getattr(tokenizer, "all_special_ids", []) or [])

    # Count visible generated tokens per eojeol before classifying them.  A
    # token crossing an eojeol boundary is deliberately excluded later.
    token_targets: list[dict[str, Any]] = []
    for local_index, token_id in enumerate(generated_ids):
        absolute_index = len(prompt_ids) + local_index
        start, end = alignment["offsets"][absolute_index]
        raw_text = tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False)
        visible = token_id not in special_ids and end > start
        overlaps = [
            (idx, surface, continuation_start + char_start, continuation_start + char_end)
            for idx, (surface, char_start, char_end) in enumerate(eojeols)
            if min(end, continuation_start + char_end) > max(start, continuation_start + char_start)
        ]
        token_targets.append({
            "local_index": local_index,
            "token_id": token_id,
            "start": int(start),
            "end": int(end),
            "raw_text": raw_text,
            "visible": visible,
            "overlaps": overlaps,
        })

    per_eojeol = {}
    for target in token_targets:
        for idx, _, _, _ in target["overlaps"]:
            if target["visible"]:
                per_eojeol.setdefault(idx, []).append(target["local_index"])
    rows: list[dict[str, Any]] = []
    for target in token_targets:
        overlaps = target["overlaps"]
        visible = bool(target["visible"])
        if not visible:
            row = build_aligned_token_row(
                "",
                (0, 0),
                [],
                token_raw_text=target["raw_text"],
                token_core_text="",
                token_index_in_eojeol=0,
                eojeol_token_count=1,
                prompt_id=reference["doc_id"],
                visible_candidate=False,
                exclusion_reason="special_or_non_visible_token",
            )
            rows.append({
                **row,
                "pair_id": reference["pair_id"],
                "doc_id": reference["doc_id"],
                "raw_text_hash": reference["raw_text_hash"],
                "generated_token_position": target["local_index"],
                "token_id": target["token_id"],
                "token_raw_text": target["raw_text"],
                "token_core_text": "",
                "global_token_char_span": [target["start"], target["end"]],
                "prompt_boundary_excluded": prompt_boundary_excluded,
                "pool_index": reference.get("pool_index"),
                "pool_split": reference.get("pool_split"),
                "source_index": reference.get("source_index"),
            })
            continue

        exclusion_reason = None
        if not alignment["roundtrip_exact"]:
            exclusion_reason = "exact_retokenization_failure"
        elif prompt_boundary_excluded and target["end"] <= continuation_start:
            exclusion_reason = "prompt_boundary_partial_eojeol"
        elif not overlaps:
            exclusion_reason = "special_or_non_visible_token"
        elif len(overlaps) > 1:
            exclusion_reason = "multi_eojeol_span"

        if overlaps:
            eojeol_index, eojeol_text, eojeol_start, eojeol_end = overlaps[0]
            local_start = max(target["start"], eojeol_start) - eojeol_start
            local_end = min(target["end"], eojeol_end) - eojeol_start
            local_start = max(0, local_start)
            local_end = min(len(eojeol_text), local_end)
            core_text = _token_core_text(eojeol_text, local_start, local_end)
            # A whitespace-prefixed token's absolute span may begin before the
            # eojeol.  The core is the exact visible substring in the eojeol.
            if core_text:
                core_start = local_start + len(eojeol_text[local_start:local_end]) - len(eojeol_text[local_start:local_end].lstrip())
                core_end = core_start + len(core_text)
            else:
                core_start, core_end = local_start, local_end
            token_indices = sorted(per_eojeol.get(eojeol_index, []))
            token_index = token_indices.index(target["local_index"]) if target["local_index"] in token_indices else 0
            morphemes = _kiwi_morphemes(eojeol_text, kiwi)
            row = build_aligned_token_row(
                eojeol_text,
                (core_start, core_end),
                morphemes,
                token_raw_text=target["raw_text"],
                token_core_text=core_text,
                token_index_in_eojeol=token_index,
                eojeol_token_count=max(1, len(token_indices)),
                prompt_id=reference["doc_id"],
                visible_candidate=True,
                token_eojeol_count=len(overlaps),
                exclusion_reason=exclusion_reason,
            )
        else:
            row = build_aligned_token_row(
                "",
                (0, 0),
                [],
                token_raw_text=target["raw_text"],
                token_core_text="",
                token_index_in_eojeol=0,
                eojeol_token_count=1,
                prompt_id=reference["doc_id"],
                visible_candidate=True,
                exclusion_reason=exclusion_reason or "special_or_non_visible_token",
            )
        rows.append({
            **row,
            "pair_id": reference["pair_id"],
            "doc_id": reference["doc_id"],
            "raw_text_hash": reference["raw_text_hash"],
            "generated_token_position": target["local_index"],
            "token_id": target["token_id"],
            "token_raw_text": target["raw_text"],
            "token_core_text": row.get("token_core_text", ""),
            "global_token_char_span": [target["start"], target["end"]],
            "roundtrip_exact": bool(alignment["roundtrip_exact"]),
            "continuation_text": continuation_text,
            "prompt_boundary_excluded": prompt_boundary_excluded,
            "pool_index": reference.get("pool_index"),
            "pool_split": reference.get("pool_split"),
            "source_index": reference.get("source_index"),
        })
    return rows


def _shard_dir(root: Path, config: dict[str, Any], pair_id: str, shard_index: int, num_shards: int) -> Path:
    return root / config["paths"]["runs"] / pair_id / "shards" / f"shard-{shard_index:05d}-of-{num_shards:05d}"


def align_pair_shard(
    *,
    root: str | Path,
    config: dict[str, Any],
    pair_id: str,
    shard_index: int = 0,
    num_shards: int = 1,
    device: str = "cpu",
    target_model_path: str | Path | None = None,
    token: str | None = None,
) -> dict[str, Any]:
    import pandas as pd
    from kiwipiepy import Kiwi
    from transformers import AutoTokenizer

    root_path = Path(root)
    shard = _shard_dir(root_path, config, pair_id, shard_index, num_shards)
    if not (shard / "COMPLETE").exists():
        raise RuntimeError(f"Cannot align incomplete SD shard: {shard}")
    if target_model_path:
        candidate = Path(str(target_model_path)).expanduser()
        model_name = str(candidate.resolve()) if candidate.exists() else str(target_model_path)
        target_revision = None
    else:
        model_meta_path = root_path / config["paths"]["model_metadata"]
        model_meta = json.loads(model_meta_path.read_text(encoding="utf-8"))["pairs"][pair_id]
        target = model_meta["target"]
        target_path = target.get("local_path")
        model_name = target_path if target_path and Path(target_path).exists() else target["id"]
        target_revision = None if target_path and Path(target_path).exists() else target.get("resolved_revision")
    output = shard / "aligned_tokens.parquet"
    if output.exists() and (shard / "ALIGNMENT_COMPLETE").exists():
        run_metadata_path = shard / "run_metadata.json"
        if run_metadata_path.exists():
            run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
            if run_metadata.get("target_model") != model_name or run_metadata.get("target_revision") != target_revision:
                raise RuntimeError(f"Existing alignment belongs to a different target model: {shard}")
        return {"pair_id": pair_id, "status": "already_complete", "path": str(output)}
    references_path = shard / "references.parquet"
    if not references_path.exists():
        raise FileNotFoundError(references_path)
    refs = pd.read_parquet(references_path).to_dict(orient="records")
    if any(not row.get("exact_sd_target", False) for row in refs):
        raise AssertionError("morphology alignment cannot consume a parity-failed prompt")
    tokenizer = AutoTokenizer.from_pretrained(model_name, revision=target_revision, use_fast=True, token=token)
    kiwi = Kiwi()
    aligned_rows: list[dict[str, Any]] = []
    for reference in refs:
        aligned_rows.extend(_make_prompt_alignment_rows(reference, tokenizer=tokenizer, kiwi=kiwi))
    _write_parquet(aligned_rows, output)
    metadata = {
        "pair_id": pair_id,
        "shard_index": shard_index,
        "num_shards": num_shards,
        "prompt_count": len(refs),
        "visible_candidate_tokens": sum(bool(row.get("visible_candidate")) for row in aligned_rows),
        "aligned_rows": len(aligned_rows),
        "device": device,
        "target_model": model_name,
        "target_revision": target_revision,
    }
    _write_json(shard / "alignment_metadata.json", metadata)
    (shard / "ALIGNMENT_COMPLETE").write_text("All exact-parity prompt continuations were aligned.\n", encoding="utf-8")
    return {**metadata, "status": "complete", "path": str(output)}


def _read_shards(root: Path, config: dict[str, Any], pair_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], bool, list[str]]:
    import pandas as pd

    pair_dir = root / config["paths"]["runs"] / pair_id / "shards"
    refs: list[dict[str, Any]] = []
    aligned: list[dict[str, Any]] = []
    missing: list[str] = []
    if not pair_dir.exists():
        return refs, aligned, False, [str(pair_dir)]
    shard_dirs = sorted(path for path in pair_dir.iterdir() if path.is_dir() and path.name.startswith("shard-"))
    for shard in shard_dirs:
        if not (shard / "COMPLETE").exists() or not (shard / "ALIGNMENT_COMPLETE").exists():
            missing.append(str(shard))
            continue
        ref_path = shard / "references.parquet"
        aligned_path = shard / "aligned_tokens.parquet"
        if not ref_path.exists() or not aligned_path.exists():
            missing.append(str(shard))
            continue
        refs.extend(pd.read_parquet(ref_path).to_dict(orient="records"))
        aligned.extend(pd.read_parquet(aligned_path).to_dict(orient="records"))
    return refs, aligned, not missing, missing


def _read_event_shards(root: Path, config: dict[str, Any], pair_id: str) -> list[dict[str, Any]]:
    """Read raw proposal events for a pair without changing shard semantics."""
    import pandas as pd

    pair_dir = root / config["paths"]["runs"] / pair_id / "shards"
    events: list[dict[str, Any]] = []
    if not pair_dir.exists():
        return events
    for shard in sorted(path for path in pair_dir.iterdir() if path.is_dir() and path.name.startswith("shard-")):
        path = shard / "sd_events.parquet"
        if path.exists() and (shard / "COMPLETE").exists():
            events.extend(pd.read_parquet(path).to_dict(orient="records"))
    return events


def _write_pair_analysis_artifacts(
    root: Path,
    config: dict[str, Any],
    pair_id: str,
    references: list[dict[str, Any]],
    aligned: list[dict[str, Any]],
    events: list[dict[str, Any]],
) -> None:
    """Materialize consolidated joins for later NAACL analyses.

    The raw shard files remain authoritative.  These convenience files make
    rejection, morphology, prompt split, and target-distribution covariates
    available in one row-level table without rerunning inference.
    """
    import pandas as pd

    results_dir = root / config["paths"]["results"]
    results_dir.mkdir(parents=True, exist_ok=True)
    def write_bundle(suffix: str, bundle_refs: list[dict[str, Any]], bundle_aligned: list[dict[str, Any]], bundle_events: list[dict[str, Any]]) -> None:
        if bundle_refs:
            pd.DataFrame(bundle_refs).to_parquet(results_dir / f"table1_{pair_id}{suffix}_references.parquet", index=False)
        if bundle_aligned:
            pd.DataFrame(bundle_aligned).to_parquet(results_dir / f"table1_{pair_id}{suffix}_aligned_tokens.parquet", index=False)
        if not bundle_events or not bundle_aligned:
            return
        event_frame = pd.DataFrame(bundle_events)
        aligned_frame = pd.DataFrame(bundle_aligned)
        keys = ["pair_id", "doc_id", "generated_token_position"]
        missing_event = [key for key in keys if key not in event_frame.columns]
        missing_alignment = [key for key in keys if key not in aligned_frame.columns]
        if missing_event or missing_alignment:
            raise AssertionError(
                "cannot build proposal analysis artifact; missing join keys "
                f"events={missing_event}, alignment={missing_alignment}"
            )
        joined = event_frame.merge(
            aligned_frame,
            on=keys,
            how="left",
            suffixes=("", "_alignment"),
            validate="many_to_one",
        )
        joined.to_parquet(results_dir / f"table1_{pair_id}{suffix}_proposal_analysis.parquet", index=False)

    write_bundle("", references, aligned, events)
    if pair_id == "Q2":
        def is_common(row: dict[str, Any]) -> bool:
            pool_index = row.get("pool_index")
            return row.get("pool_split") == "common_20k" or (
                pool_index is not None and int(pool_index) < 20_000
            )
        common_refs = [row for row in references if is_common(row)]
        common_aligned = [row for row in aligned if is_common(row)]
        common_events = [row for row in events if is_common(row)]
        write_bundle("_common20k", common_refs, common_aligned, common_events)


def _table_row(pair_id: str, pair: dict[str, Any], refs: list[dict[str, Any]], aligned: list[dict[str, Any]], complete: bool, missing: list[str]) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    prompt_meta = [
        {
            "prompt_id": row["doc_id"],
            "completed": bool(row.get("completed", False) and row.get("exact_sd_target", False)),
            "generated_token_count": len(row.get("target_continuation_token_ids", []) or []),
        }
        for row in refs
    ]
    summary = aggregate_alignment_rows(aligned, completed_prompt_metadata=prompt_meta)
    assert_alignment_invariants(
        aligned,
        summary=summary,
        completed_prompt_metadata=prompt_meta,
    )
    support = support_counts_by_fine_boundary(
        aligned,
        min_cross=int(pair["support_threshold_cross"]),
        min_within=int(pair["support_threshold_within"]),
    )
    status = "COMPLETE" if complete else "INCOMPLETE"
    row = {
        "Pair": pair_id,
        "Family": pair["family"],
        "Draft→Target": f"{pair['draft'].split('/')[-1]}→{pair['target'].split('/')[-1]}",
        "# Prompts": summary["completed_prompt_count"],
        "Generated tokens": summary["generated_token_count"],
        "Visible candidate tokens": summary["visible_candidate_count"],
        "Eligible tokens": summary["eligible_count"],
        "Aligned %": summary["aligned_percent"],
        "CROSS": summary["cross_count"],
        "CROSS %": summary["cross_percent"],
        "WITHIN_SPLIT": summary["within_count"],
        "WITHIN %": summary["within_percent"],
        "Single-boundary CROSS": summary["single_boundary_cross_count"],
        "Alignment exclusion %": summary["exclusion_percent"],
        "# Supported boundaries": support["supported_boundary_count"],
        "Status": status,
    }
    if missing:
        row["Missing shards"] = ";".join(missing)
    exclusions = [
        {"Pair": pair_id, "Exclusion reason": reason, "Count": count, "Percent of visible": 100.0 * count / max(1, summary["visible_candidate_count"])}
        for reason, count in summary["exclusion_breakdown"].items()
    ]
    boundary_rows = []
    for boundary, counts in support["by_fine_boundary"].items():
        boundary_rows.append({
            "Pair": pair_id,
            "fine_boundary": boundary,
            "N_CROSS_single_boundary": counts["cross_count"],
            "N_WITHIN_LOCAL": counts["within_count"],
            "threshold_cross": support["cross_threshold"],
            "threshold_within": support["within_threshold"],
            "supported": counts["supported"],
        })
    return row, {"summary": summary, "support": support, "missing": missing}, exclusions + boundary_rows


def build_table1_outputs(*, root: str | Path, config: dict[str, Any]) -> dict[str, Any]:
    import pandas as pd

    root_path = Path(root)
    pool = load_prompt_pool(root_path / config["paths"]["prompts"])
    expected_common = {str(row["doc_id"]) for row in pool[:20_000]}
    expected_all = {str(row["doc_id"]) for row in pool[:40_000]}
    table_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    pair_summaries: dict[str, Any] = {}
    for pair_id, pair in config["pairs"].items():
        refs, aligned, complete, missing = _read_shards(root_path, config, pair_id)
        events = _read_event_shards(root_path, config, pair_id)
        observed_ids = [str(row["doc_id"]) for row in refs]
        if len(observed_ids) != len(set(observed_ids)):
            raise AssertionError(f"duplicate doc_id in pair {pair_id}")
        expected = expected_all if pair["prompt_split"] == "all_40k" else expected_common
        if not set(observed_ids).issubset(expected):
            raise AssertionError(f"pair {pair_id} contains prompts outside its frozen split")
        if complete and set(observed_ids) != expected:
            complete = False
            missing = missing + [f"prompt_set_missing:{len(expected - set(observed_ids))}"]
        row, details, extra_rows = _table_row(pair_id, pair, refs, aligned, complete, missing)
        _write_pair_analysis_artifacts(root_path, config, pair_id, refs, aligned, events)
        table_rows.append(row)
        pair_summaries[pair_id] = details
        for item in extra_rows:
            if "Exclusion reason" in item:
                exclusion_rows.append(item)
            else:
                boundary_rows.append(item)
        print(
            f"[{pair_id}] status={row['Status']} prompts={row['# Prompts']} "
            f"generated={row['Generated tokens']} eligible={row['Eligible tokens']} "
            f"aligned={row['Aligned %']:.2f}% CROSS={row['CROSS']} "
            f"WITHIN={row['WITHIN_SPLIT']} single_CROSS={row['Single-boundary CROSS']} "
            f"supported_boundaries={row['# Supported boundaries']}",
            flush=True,
        )
    results_dir = root_path / config["paths"]["results"]
    results_dir.mkdir(parents=True, exist_ok=True)
    table_frame = pd.DataFrame(table_rows)
    table_frame.to_csv(results_dir / "table1_models_data_alignment.csv", index=False)
    (results_dir / "table1_models_data_alignment.md").write_text(_markdown_table(table_frame), encoding="utf-8")
    (results_dir / "table1_models_data_alignment.tex").write_text(table_frame.to_latex(index=False, escape=True), encoding="utf-8")
    pd.DataFrame(exclusion_rows, columns=["Pair", "Exclusion reason", "Count", "Percent of visible"]).to_csv(results_dir / "table1_exclusion_breakdown.csv", index=False)
    pd.DataFrame(boundary_rows, columns=["Pair", "fine_boundary", "N_CROSS_single_boundary", "N_WITHIN_LOCAL", "threshold_cross", "threshold_within", "supported"]).to_csv(results_dir / "boundary_support_by_pair.csv", index=False)
    _write_json(results_dir / "table1_build_metadata.json", {
        "status": "COMPLETE" if all(row["Status"] == "COMPLETE" for row in table_rows) else "INCOMPLETE",
        "pairs": pair_summaries,
    })
    return {
        "status": "COMPLETE" if all(row["Status"] == "COMPLETE" for row in table_rows) else "INCOMPLETE",
        "rows": len(table_rows),
        "results_dir": str(results_dir),
    }
