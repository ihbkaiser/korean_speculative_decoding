#!/usr/bin/env python3
"""Assemble the plan's Table 1 from target, TF, simulation, and Kiwi shards."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.naacl_experiment import atomic_json, load_config, output_root, select_prompts


def _markdown(frame: pd.DataFrame) -> str:
    return frame.to_markdown(index=False) + "\n"


def _read_shards(directory: Path, pattern="shard-*.parquet"):
    paths = sorted(directory.glob(pattern))
    if not paths:
        return None
    return pd.concat((pd.read_parquet(path) for path in paths), ignore_index=True)


def _find_target_artifacts(out: Path, pair, split):
    target = pair["target"]
    references = out / "continuations" / target / split
    reference_split = split
    if not list(references.glob("shard-*.parquet")) and split == "common":
        references = out / "continuations" / target / "all"
        reference_split = "all"
    aligned = out / "aligned" / "kiwi" / target / split
    aligned_split = split
    if not list(aligned.glob("shard-*.parquet")) and split == "common":
        aligned = out / "aligned" / "kiwi" / target / "all"
        aligned_split = "all"
    target_tf = out / "teacher_forced" / "generated" / target / target / split
    target_tf_split = split
    if not list(target_tf.glob("shard-*.parquet")) and split == "common":
        target_tf = out / "teacher_forced" / "generated" / target / target / "all"
        target_tf_split = "all"
    return references, reference_split, aligned, aligned_split, target_tf, target_tf_split


def _count_ids(references: pd.DataFrame, split: str) -> int:
    frame = references
    if split == "common":
        frame = frame[frame["pool_index"].astype(int) < 20_000]
    elif split == "extra":
        frame = frame[frame["pool_index"].astype(int) >= 20_000]
    return frame["doc_id"].astype(str).nunique()


def _row_for_pair(out, pair_id, pair, prompts):
    expected = len(prompts)
    target = pair["target"]
    draft = pair["draft"]
    split = pair["split"]
    row = {
        "Pair": pair_id,
        "Family": pair["family"],
        "Draft → Target": pair["legacy_name"],
        "Prompt split": split,
        "Prompts": None,
        "Generated tokens": None,
        "Visible aligned tokens": None,
        "Alignment exclusions": None,
        "EXACT proposals": None,
        "WITHIN_SPLIT proposals": None,
        "CROSS proposals": None,
        "EXACT share": None,
        "WITHIN_SPLIT share": None,
        "CROSS share": None,
        "Eligible k=4 proposals": None,
        "k=4 rejections": None,
        "k=4 rejection rate": None,
        "Target TF label agreement": None,
        "Status": "INCOMPLETE",
    }
    missing = []
    refs_dir, ref_split, aligned_dir, aligned_split, target_tf_dir, target_tf_split = _find_target_artifacts(
        out, pair, split
    )
    refs = _read_shards(refs_dir)
    aligned = _read_shards(aligned_dir)
    target_tf = _read_shards(target_tf_dir)
    draft_tf_dir = out / "teacher_forced" / "generated" / target / draft / split
    if not list(draft_tf_dir.glob("shard-*.parquet")) and split == "common":
        draft_tf_dir = out / "teacher_forced" / "generated" / target / draft / "all"
    sim_path = out / "simulated" / pair_id / split / "events_k4.parquet"
    events = pd.read_parquet(sim_path) if sim_path.is_file() else None

    if refs is None:
        missing.append(f"continuations:{target}:{ref_split}")
    if aligned is None:
        missing.append(f"alignment:{target}:{aligned_split}")
    if target_tf is None:
        missing.append(f"target_tf:{target}:{target_tf_split}")
    if not list(draft_tf_dir.glob("shard-*.parquet")):
        missing.append(f"draft_tf:{draft}:{split}")
    if events is None:
        missing.append(f"simulation:{pair_id}:{split}:k4")
    if missing:
        return row, {"status": "INCOMPLETE", "missing": missing}

    refs["doc_id"] = refs["doc_id"].astype(str)
    aligned["doc_id"] = aligned["doc_id"].astype(str)
    target_tf["doc_id"] = target_tf["doc_id"].astype(str)
    events["doc_id"] = events["doc_id"].astype(str)
    if split == "common":
        refs = refs[refs["pool_index"].astype(int) < 20_000]
        aligned = aligned[aligned["pool_index"].astype(int) < 20_000]
        target_tf = target_tf[target_tf["pool_index"].astype(int) < 20_000]
    elif split == "extra":
        refs = refs[refs["pool_index"].astype(int) >= 20_000]
        aligned = aligned[aligned["pool_index"].astype(int) >= 20_000]
        target_tf = target_tf[target_tf["pool_index"].astype(int) >= 20_000]

    observed = refs["doc_id"].nunique()
    if observed != expected:
        missing.append(f"prompt_count:{observed}/{expected}")
    alignment_keys = aligned[["doc_id", "generated_token_position", "alignment_status", "alignment_class", "visible_candidate"]].copy()
    alignment_keys["position"] = alignment_keys["generated_token_position"].astype(int)
    proposals = events[events["eligible"].astype(bool)].merge(
        alignment_keys[["doc_id", "position", "alignment_status", "alignment_class", "visible_candidate"]],
        on=["doc_id", "position"],
        how="left",
        validate="many_to_one",
    )
    if proposals["alignment_class"].isna().any():
        missing.append("proposal_alignment_join")
    eligible = proposals
    aligned_proposals = eligible[
        eligible["visible_candidate"].fillna(False).astype(bool)
        & eligible["alignment_status"].eq("ALIGNED")
    ]
    class_counts = aligned_proposals["alignment_class"].value_counts().to_dict()
    total_eligible = len(eligible)
    aligned_count = len(aligned_proposals)
    for label, key in (("EXACT", "EXACT proposals"), ("WITHIN_SPLIT", "WITHIN_SPLIT proposals"), ("CROSS", "CROSS proposals")):
        count = int(class_counts.get(label, 0))
        row[key] = count
        share_key = {"EXACT": "EXACT share", "WITHIN_SPLIT": "WITHIN_SPLIT share", "CROSS": "CROSS share"}[label]
        row[share_key] = count / aligned_count if aligned_count else None
    row["Prompts"] = observed
    row["Generated tokens"] = int(refs["continuation_length"].astype(int).sum())
    row["Visible aligned tokens"] = aligned_count
    row["Alignment exclusions"] = total_eligible - aligned_count
    row["Eligible k=4 proposals"] = total_eligible
    row["k=4 rejections"] = int(eligible["rejected"].fillna(False).astype(bool).sum())
    row["k=4 rejection rate"] = row["k=4 rejections"] / total_eligible if total_eligible else None
    tf_cont = target_tf["target_token_id"].astype("int64")
    tf_argmax = target_tf["argmax_id"].astype("int64")
    row["Target TF label agreement"] = float((tf_cont == tf_argmax).mean()) if len(target_tf) else None
    if missing:
        row["Status"] = "INCOMPLETE"
    elif observed == expected:
        row["Status"] = "COMPLETE"
    return row, {
        "status": row["Status"],
        "missing": missing,
        "expected_prompts": expected,
        "observed_prompts": observed,
        "target_tf_positions": len(target_tf),
        "proposal_rows": len(eligible),
        "aligned_proposal_rows": aligned_count,
        "invalidated_suffix_rows": int(events["invalidated_after"].fillna(False).astype(bool).sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = load_config(args.config)
    out = output_root(config, args.output_dir)
    rows = []
    details = {}
    for pair_id, pair in config["pairs"].items():
        prompts = select_prompts(config, pair["split"])
        row, metadata = _row_for_pair(out, pair_id, pair, prompts)
        rows.append(row)
        details[pair_id] = metadata
        print(
            f"TABLE1 {pair_id} status={row['Status']} prompts={row['Prompts']} "
            f"eligible_k4={row['Eligible k=4 proposals']} reject={row['k=4 rejections']}",
            flush=True,
        )
        gc.collect()

    frame = pd.DataFrame(rows)
    results = out / "table1"
    results.mkdir(parents=True, exist_ok=True)
    frame.to_csv(results / "table1.csv", index=False)
    (results / "table1.md").write_text(_markdown(frame), encoding="utf-8")
    (results / "table1.tex").write_text(frame.to_latex(index=False, escape=True, na_rep=""), encoding="utf-8")
    validity = {}
    for pair_id in ("Q2", "G1"):
        report_path = out / "validation" / "v0" / pair_id / "agreement_report.json"
        if not report_path.is_file():
            validity[pair_id] = {"status": "PENDING", "path": str(report_path)}
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        agreement = report.get(
            "live_vs_simulation_eligible_label_agreement_shared_context",
            report.get("live_vs_simulation_eligible_label_agreement", {}),
        )
        validity[pair_id] = {
            "status": report.get("status", "UNKNOWN"),
            "g0_gate_status": report.get("g0_gate_status"),
            "sample_prompts": report.get("sample_prompts"),
            "eligible_label_agreement": agreement.get("rejection_label_agreement"),
            "cross_label_agreement": report.get(
                "live_vs_simulation_cross_label_agreement_shared_context",
                report.get("live_vs_simulation_cross_label_agreement", {}),
            ).get("rejection_label_agreement"),
            "path": str(report_path),
        }
    pair_artifacts_complete = all(row["Status"] == "COMPLETE" for row in rows)
    validity_complete = all(
        item["status"] == "COMPLETE" and item.get("g0_gate_status") == "PASS"
        for item in validity.values()
    )
    atomic_json(results / "table1_build_metadata.json", {
        "status": "COMPLETE" if pair_artifacts_complete and validity_complete else "INCOMPLETE",
        "pair_artifacts_status": "COMPLETE" if pair_artifacts_complete else "INCOMPLETE",
        "validity_status": "COMPLETE" if validity_complete else "PENDING",
        "plan_file": "NAACL_experiments.md",
        "plan_sha256": __import__("hashlib").sha256((ROOT / "NAACL_experiments.md").read_bytes()).hexdigest(),
        "pairs": details,
        "v0_validation": validity,
        "table_rows": len(rows),
    })
    status = "COMPLETE" if pair_artifacts_complete and validity_complete else "INCOMPLETE"
    print(f"TABLE1_BUILD status={status} pair_artifacts={pair_artifacts_complete} validity={validity_complete} path={results}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
