#!/usr/bin/env python3
"""Run FLORES-200 English→Korean SD traces using the frozen E2 pipeline."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from src import e2_model_pairs as e2
from src.data import encode_prompt

ARCHIVE_SHA256 = "b8b0b76783024b85797e5cc75064eb83fc5288b41e9654dabc7be6ae944011f6"
ENG_SHA256 = "612e9fbe87997617c0fa8fa8929654a4f49b728d96738112c2b86ef6a1d78d88"
KOR_SHA256 = "540972696230a56e888f1fbd04b7000135c436351f5e467f99ce67fdccd8df0f"
METADATA_SHA256 = "8edac47f861fcf6b54dad2068c10ba8f741da8b8676861c6bc27fcd9f5483b36"
S3_VERSION_ID = "KJZoZnGvyduN3osrx.67C_UQ7C9oNDic"
PAIR_ORDER = ["P1", "P2", "P3"]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(tmp, index=False)
    tmp.replace(path)


def load_flores_inputs(run_dir: Path, tokenizer: object, max_prompt_tokens: int) -> tuple[list[dict], pd.DataFrame, dict]:
    data_dir = run_dir / "data/flores200_dataset"
    eng_path = data_dir / "devtest/eng_Latn.devtest"
    kor_path = data_dir / "devtest/kor_Hang.devtest"
    metadata_path = data_dir / "metadata_devtest.tsv"
    for path, expected in [
        (run_dir / "data/flores200_dataset.tar.gz", ARCHIVE_SHA256),
        (eng_path, ENG_SHA256), (kor_path, KOR_SHA256), (metadata_path, METADATA_SHA256),
    ]:
        if not path.exists():
            raise FileNotFoundError(path)
        actual = e2.sha256_file(path)
        if actual != expected:
            raise RuntimeError(f"FLORES source hash mismatch for {path}: {actual}")

    en_sentences = eng_path.read_text(encoding="utf-8").splitlines()
    ko_sentences = kor_path.read_text(encoding="utf-8").splitlines()
    if len(en_sentences) != 1012 or len(ko_sentences) != 1012:
        raise RuntimeError(f"Expected 1,012 FLORES devtest rows; found EN={len(en_sentences)} KO={len(ko_sentences)}")

    prompts: list[dict] = []
    manifest: list[dict] = []
    for index, (english, korean_reference) in enumerate(zip(en_sentences, ko_sentences), start=1):
        if not english.strip() or not korean_reference.strip():
            raise RuntimeError(f"FLORES row {index} has an empty source or reference; input integrity failure")
        rendered = f"Translate the following sentence into Korean.\nEnglish: {english}\nKorean:"
        ids = encode_prompt(tokenizer, rendered, max_prompt_tokens)
        prompt_hash = sha256_bytes(rendered.encode("utf-8"))
        token_hash = sha256_bytes(json.dumps(ids, separators=(",", ":")).encode("ascii"))
        prompts.append({
            "prompt_id": index,
            "source_index": index,
            "title": "",
            "text": rendered,
            "rendered_prompt": rendered,
            "input_ids": [int(x) for x in ids],
            "prompt_hash": prompt_hash,
            "rendered_prompt_sha256": prompt_hash,
            "tokenized_prompt_sha256": token_hash,
            "english_source": english,
        })
        manifest.append({
            "flores_row_id": index,
            "prompt_id": index,
            "english_source": english,
            "korean_reference_provenance_only": korean_reference,
            "rendered_prompt": rendered,
            "rendered_prompt_sha256": prompt_hash,
            "tokenized_prompt_sha256": token_hash,
            "prompt_token_count": len(ids),
            "included": True,
            "exclusion_reason": "",
        })

    manifest_frame = pd.DataFrame(manifest)
    atomic_csv(manifest_frame, run_dir / "data_manifest.csv")
    manifest_sha = e2.sha256_file(run_dir / "data_manifest.csv")
    hash_lines = [
        f"{manifest_sha}  data_manifest.csv",
        f"{ARCHIVE_SHA256}  data/flores200_dataset.tar.gz",
        f"{ENG_SHA256}  data/flores200_dataset/devtest/eng_Latn.devtest",
        f"{KOR_SHA256}  data/flores200_dataset/devtest/kor_Hang.devtest",
        f"{METADATA_SHA256}  data/flores200_dataset/metadata_devtest.tsv",
    ]
    atomic_text(run_dir / "data_manifest.sha256", "\n".join(hash_lines) + "\n")

    rendered_dupes = manifest_frame.loc[manifest_frame.rendered_prompt_sha256.duplicated(keep=False), "flores_row_id"].astype(int).tolist()
    token_dupes = manifest_frame.loc[manifest_frame.tokenized_prompt_sha256.duplicated(keep=False), "flores_row_id"].astype(int).tolist()
    lengths = manifest_frame.prompt_token_count.astype(int)
    audit = {
        "source_row_count": len(manifest_frame),
        "included_row_count": int(manifest_frame.included.sum()),
        "empty_source_or_reference_rows": 0,
        "rendered_prompt_duplicate_rows": rendered_dupes,
        "rendered_prompt_duplicate_groups": int(manifest_frame.rendered_prompt_sha256.duplicated().sum()),
        "tokenized_prompt_duplicate_rows": token_dupes,
        "tokenized_prompt_duplicate_groups": int(manifest_frame.tokenized_prompt_sha256.duplicated().sum()),
        "token_count_min": int(lengths.min()),
        "token_count_median": float(lengths.median()),
        "token_count_max": int(lengths.max()),
        "prompts_truncated_at_128": int((lengths >= max_prompt_tokens).sum()),
        "rendered_prompt_hashes_ordered_sha256": sha256_bytes("\n".join(manifest_frame.rendered_prompt_sha256).encode()),
        "tokenized_prompt_hashes_ordered_sha256": sha256_bytes("\n".join(manifest_frame.tokenized_prompt_sha256).encode()),
    }
    lines = [
        "# FLORES prompt audit", "",
        f"Rows: **{audit['source_row_count']:,}**; included: **{audit['included_row_count']:,}**; empty input exclusions: **0**.",
        f"Duplicate rendered prompt groups: **{audit['rendered_prompt_duplicate_groups']}**; duplicate rendered rows: `{rendered_dupes}`.",
        f"Duplicate token-ID prompt groups: **{audit['tokenized_prompt_duplicate_groups']}**; duplicate tokenized rows: `{token_dupes}`.",
        f"Prompt token lengths: min {audit['token_count_min']}, median {audit['token_count_median']:.1f}, max {audit['token_count_max']}; reaching the 128-token cap: {audit['prompts_truncated_at_128']}.",
        f"Ordered rendered-prompt hash: `{audit['rendered_prompt_hashes_ordered_sha256']}`.",
        f"Ordered tokenized-prompt hash: `{audit['tokenized_prompt_hashes_ordered_sha256']}`.",
        "", "Korean references are provenance-only. They were not included in prompts, tokenized, generated against, or used in eligibility or statistical labels.", "",
    ]
    atomic_text(run_dir / "prompt_audit.md", "\n".join(lines))
    return prompts, manifest_frame, audit


def gpu3_preflight() -> tuple[object, dict]:
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "3":
        raise RuntimeError("Launch under CUDA_VISIBLE_DEVICES=3; refusing to use another physical GPU")
    import torch
    import transformers

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected exactly one visible CUDA device mapped to cuda:0; count={torch.cuda.device_count()}")
    torch.cuda.set_device(0)
    name = torch.cuda.get_device_name(0)
    props = torch.cuda.get_device_properties(0)
    if "A100" not in name or props.total_memory < 78 * 1024**3:
        raise RuntimeError(f"Expected physical GPU 3 A100 80GB mapped to cuda:0; found {name}, {props.total_memory}")
    if torch.cuda.current_device() != 0:
        raise RuntimeError("Framework-local CUDA device must be cuda:0")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(3090)
    torch.cuda.manual_seed_all(3090)
    random.seed(3090)
    np.random.seed(3090)
    gpu_list = subprocess.run(["nvidia-smi", "-L"], check=True, capture_output=True, text=True).stdout.strip()
    gpu_query = subprocess.run(
        ["nvidia-smi", "--id=3", "--query-gpu=name,driver_version,memory.total,memory.used,memory.free", "--format=csv,noheader"],
        check=True, capture_output=True, text=True,
    ).stdout.strip()
    env = {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "nvidia_smi_L_output": gpu_list,
        "nvidia_smi_physical_gpu3": gpu_query,
        "physical_gpu_index": 3,
        "GPU_name": name,
        "driver_version": gpu_query.split(",")[1].strip(),
        "nvidia_smi_cuda_version": "13.0",
        "torch_cuda_runtime_version": torch.version.cuda,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "attention_backend": "sdpa",
        "dtype": "torch.float16",
        "CUDA_VISIBLE_DEVICES": os.environ["CUDA_VISIBLE_DEVICES"],
        "framework_local_device": "cuda:0",
        "visible_device_count": torch.cuda.device_count(),
        "device_total_memory_bytes": props.total_memory,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM"),
        "pid": os.getpid(),
        "python": sys.version,
        "platform": platform.platform(),
    }
    return torch, env


def write_config(run_dir: Path, prompt_audit: dict, tokenizer_report: dict, gpu: dict, args: argparse.Namespace) -> dict:
    config = {
        "experiment": "External confirmation replication on FLORES-200 English to Korean",
        "source": {
            "official_release": "FLORES-200 original 2022 release",
            "archive_url": "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz",
            "README_url": "https://github.com/facebookresearch/flores/blob/main/flores200/README.md",
            "split": "devtest", "source_language": "eng_Latn", "reference_language": "kor_Hang",
            "row_count": 1012, "S3_version_id": S3_VERSION_ID,
            "archive_sha256": ARCHIVE_SHA256, "english_sha256": ENG_SHA256,
            "korean_reference_sha256": KOR_SHA256, "devtest_metadata_sha256": METADATA_SHA256,
            "prompt_hashes": prompt_audit,
        },
        "models": e2.MODELS,
        "historical_revision_note": "Historical P3 0.6B draft revision is unresolved; new FLORES P3 uses the pinned E2 0.6B revision da87bfb608c14b7cf20ba1ce41287e8de496c0cd.",
        "pairs": e2.PAIRS,
        "prompt_template": "Translate the following sentence into Korean.\\nEnglish: <English FLORES sentence>\\nKorean:",
        "prompt_wrapper": "E2 base-completion path; no chat template; AutoTokenizer add_special_tokens=False",
        "max_prompt_tokens": int(args.max_prompt_tokens),
        "max_new_tokens": int(args.max_new_tokens),
        "speculative_k": int(args.speculative_k),
        "dtype": "float16", "attention_backend": "sdpa", "eos_token_id": 151643,
        "generation": "greedy/no sampling; per-prompt deterministic cached decoding; use_cache=True",
        "generation_batch_size": 1, "teacher_forcing_batch_size": 1,
        "seed": 3090, "determinism": "greedy decoding; Python/NumPy/PyTorch/CUDA seed 3090",
        "GPU": gpu,
        "tokenizer_compatibility": tokenizer_report,
        "eligibility": "sd_valid=True; fragmentation bins exactly 2,3,4,5,6,7,8+; morph_class WITHIN_SPLIT or CROSS_MORPHEME; H2 CROSS_EOJEOL/KIWI_COMPLEX/exact-span exclusions unchanged",
        "H3_primary": "CROSS single boundary only; SPLIT nearest local adjacent boundary, ties right; taxonomy NOMINAL_TO_PARTICLE/PREDICATE_TO_ENDING/ENDING_TO_ENDING/LEXICAL_TO_LEXICAL/OTHER; H3 M5 controls and five planned Holm-adjusted contrasts per pair",
        "common_prompt_sensitivity": "Exact intersection of eligible prompt IDs across P1/P2/P3; pooled H3 model on this intersection",
        "frequency_cache": "E1 token and eojeol cache plus H4 morpheme cache, only if tokenizer and corpus hashes match",
        "prespecified_replication_criteria": {
            "REPLICATED": "NOMINAL_TO_PARTICLE adjusted CROSS-SPLIT AME > 0 in all three pairs; pooled AME > 0 with 95% cluster CI excluding zero; no direction reversal; positive N→P minus LEXICAL_TO_LEXICAL AME in every pair where both estimable.",
            "other_classes": ["PARTIALLY_REPLICATED", "NOT_REPLICATED", "INCONCLUSIVE"],
        },
        "output_only_descriptive_korean_audit": "Predominantly Korean means Hangul letters are at least 50% of all Unicode letters; never an exclusion.",
        "resume": bool(args.resume), "checkpoint_prompts": int(args.checkpoint_prompts),
        "run_script_arguments": vars(args),
    }
    e2.atomic_json(run_dir / "flores_config.json", config)
    atomic_text(run_dir / "environment.txt", "\n".join(f"{k}: {v}" for k, v in gpu.items()) + "\n")
    return config


def target_4b_references(torch: object, tokenizer: object, model: object, prompts: list[dict], out_dir: Path, args: argparse.Namespace) -> dict[int, dict]:
    progress_path = out_dir / "progress.json"
    progress = json.loads(progress_path.read_text()) if progress_path.exists() else {"completed_prompt_ids": [], "chunks": []}
    done = set(map(int, progress.get("completed_prompt_ids", [])))
    chunk_files = [out_dir / x for x in progress.get("chunks", [])]
    completed = pd.concat([pd.read_parquet(p) for p in chunk_files], ignore_index=True) if chunk_files else pd.DataFrame()
    by_id = {}
    if not completed.empty:
        for row in completed.to_dict("records"):
            by_id[int(row["prompt_id"])] = {"reference_token_ids": [int(x) for x in row["reference_token_ids"]], "reference_source": "FLORES 4B target greedy singleton"}

    pending = [p for p in prompts if int(p["prompt_id"]) not in done]
    torch.cuda.reset_peak_memory_stats(0)
    started = time.perf_counter()
    for start in range(0, len(pending), args.checkpoint_prompts):
        group = pending[start:start + args.checkpoint_prompts]
        records = []
        for i, prompt in enumerate(group, start=1):
            ids = e2.greedy_generate_single(model, prompt["input_ids"], args.max_new_tokens, int(tokenizer.eos_token_id))
            records.append({"prompt_id": int(prompt["prompt_id"]), "reference_token_ids": ids, "reference_source": "FLORES 4B target greedy singleton"})
            by_id[int(prompt["prompt_id"])] = {"reference_token_ids": ids, "reference_source": "FLORES 4B target greedy singleton"}
            if i % 8 == 0 or i == len(group):
                print(f"4B target-only {min(start+i,len(pending))}/{len(pending)} row={prompt['prompt_id']}", flush=True)
        chunk_index = len(progress.get("chunks", []))
        chunk_path = out_dir / f"references_{chunk_index:05d}.parquet"
        e2.atomic_parquet(pd.DataFrame(records), chunk_path)
        progress.setdefault("chunks", []).append(chunk_path.name)
        done.update(int(row["prompt_id"]) for row in group)
        progress["completed_prompt_ids"] = sorted(done)
        progress["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        e2.atomic_json(progress_path, progress)
    if done != {int(p["prompt_id"]) for p in prompts}:
        raise AssertionError("4B target reference checkpoint does not cover every FLORES row")
    if len(by_id) != len(prompts):
        raise AssertionError("Duplicate/missing 4B target reference rows")
    e2.atomic_parquet(pd.DataFrame([
        {"prompt_id": pid, **row} for pid, row in sorted(by_id.items())
    ]), out_dir / "target_references.parquet")
    e2.atomic_json(out_dir / "runtime_metadata.json", {
        "target_model": e2.MODELS["4B"],
        "prompt_count": len(prompts),
        "batch_size": 1,
        "elapsed_seconds_this_invocation": time.perf_counter() - started,
        "peak_allocated_vram_gb": round(torch.cuda.max_memory_allocated(0) / 1024**3, 3),
        "peak_reserved_vram_gb": round(torch.cuda.max_memory_reserved(0) / 1024**3, 3),
        "device": "cuda:0",
    })
    return by_id


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default="runs/flores200_en_ko_replication")
    parser.add_argument("--max-prompt-tokens", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--speculative-k", type=int, default=4)
    parser.add_argument("--checkpoint-prompts", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    for dirname in ["traces", "token_tables", "audits", "results", "figures"]:
        (run_dir / dirname).mkdir(parents=True, exist_ok=True)

    # The E2 preflight loads only the three pinned tokenizers and verifies that
    # offsets and the E1 cache identity match; model weights are not loaded here.
    tokenizer, tokenizer_report = e2.tokenizer_preflight(run_dir)
    prompts, manifest, prompt_audit = load_flores_inputs(run_dir, tokenizer, args.max_prompt_tokens)
    torch, gpu_env = gpu3_preflight()
    config = write_config(run_dir, prompt_audit, tokenizer_report, gpu_env, args)
    config["runtime"] = {"started_at_utc": datetime.now(timezone.utc).isoformat(), "pid": os.getpid()}
    e2.atomic_json(run_dir / "flores_config.json", config)
    print(f"GPU allocation verified: physical=3 local=cuda:0 name={gpu_env['GPU_name']}", flush=True)

    models = e2.load_models(torch, attention_backend="sdpa")
    model_memory = {
        "after_all_model_loads_allocated_gb": round(torch.cuda.memory_allocated(0) / 1024**3, 3),
        "after_all_model_loads_reserved_gb": round(torch.cuda.memory_reserved(0) / 1024**3, 3),
    }
    config["runtime"]["models_resident_memory"] = model_memory
    e2.atomic_json(run_dir / "flores_config.json", config)

    args_ns = SimpleNamespace(max_new_tokens=args.max_new_tokens, speculative_k=args.speculative_k, checkpoint_prompts=args.checkpoint_prompts)
    refs4b = target_4b_references(
        torch, tokenizer, models["4B"], prompts,
        run_dir / "traces/target_references_4b", args,
    )
    pair_summaries = {}
    for pair in PAIR_ORDER:
        pair_started = time.perf_counter()
        refs = {} if pair == "P1" else refs4b
        events, continuations, sd_meta = e2.run_sd_pair(
            torch, tokenizer, models, run_dir / "traces", pair, prompts, refs, args_ns, selected_batch=1,
        )
        if int(sd_meta.get("fallback_target_regenerations", 0)) != 0:
            raise AssertionError(f"{pair}: unexpected singleton target fallback; inspect target continuation parity")
        # Teacher-forced scoring uses only generated target continuations, never FLORES references.
        teacher, teacher_meta = e2.run_teacher_forcing(
            torch, tokenizer, models, run_dir / "traces", pair, prompts, continuations,
            args_ns, batch_size=1,
        )
        pair_table, eojeols, alignment_audit = e2.run_morphology(
            tokenizer, run_dir / "traces", pair, prompts, events, continuations, teacher, args_ns,
        )
        table_dir = run_dir / "token_tables" / e2.PAIRS[pair]["slug"]
        table_dir.mkdir(parents=True, exist_ok=True)
        for name in ["token_table.parquet", "eojeols.parquet", "alignment_audit.csv", "alignment_summary.json", "continuations.parquet", "sd_events.parquet", "teacher_forced_tokens.parquet"]:
            shutil.copy2(e2._pair_dir(run_dir / "traces", pair) / name, table_dir / name)
        pair_summaries[pair] = {
            "sd": sd_meta, "teacher_forced": teacher_meta,
            "generated_tokens": len(continuations), "teacher_rows": len(teacher),
            "morphology_rows": len(pair_table), "alignment_prompt_rows": len(alignment_audit),
            "exact_output_parity": bool(continuations.exact_match.astype(bool).all()),
            "wall_clock_seconds": time.perf_counter() - pair_started,
        }
        config["runtime"]["pair_summaries"] = pair_summaries
        config["runtime"]["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        e2.atomic_json(run_dir / "flores_config.json", config)
        print(f"{pair} complete; rows={len(pair_table)} exact parity={pair_summaries[pair]['exact_output_parity']}", flush=True)

    config["runtime"]["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    stage_allocated = [model_memory["after_all_model_loads_allocated_gb"]]
    stage_reserved = [model_memory["after_all_model_loads_reserved_gb"]]
    target_meta_path = run_dir / "traces/target_references_4b/runtime_metadata.json"
    if target_meta_path.exists():
        target_meta = json.loads(target_meta_path.read_text())
        stage_allocated.append(float(target_meta.get("peak_allocated_vram_gb", 0)))
        stage_reserved.append(float(target_meta.get("peak_reserved_vram_gb", 0)))
    for summary in pair_summaries.values():
        stage_allocated.extend([float(summary["sd"].get("peak_allocated_vram_gb", 0)), float(summary["teacher_forced"].get("peak_allocated_vram_gb", 0))])
        stage_reserved.extend([float(summary["sd"].get("peak_reserved_vram_gb", 0)), float(summary["teacher_forced"].get("peak_reserved_vram_gb", 0))])
    config["runtime"]["peak_allocated_vram_gb"] = max(stage_allocated)
    config["runtime"]["peak_reserved_vram_gb"] = max(stage_reserved)
    config["runtime"]["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")
    config["runtime"]["current_device"] = int(torch.cuda.current_device())
    e2.atomic_json(run_dir / "flores_config.json", config)
    atomic_text(run_dir / "traces/COMPLETE", json.dumps({"completed_at_utc": config["runtime"]["completed_at_utc"], "pairs": pair_summaries}, indent=2) + "\n")
    print("FLORES SD, teacher-forced scoring, and H2 morphology stages completed.", flush=True)


if __name__ == "__main__":
    main()
