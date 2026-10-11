#!/usr/bin/env python3
"""Fetch exact NAACL model revisions, with pair-scoped storage and eviction.

HF_TOKEN is read from the environment when present; it is never printed or
written to a manifest. Ministral's duplicate consolidated.safetensors exports
are skipped because Transformers loads the indexed model-*.safetensors files.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import get_token

from src.naacl_experiment import atomic_json, load_config, memory_free_gib, model_path


def _weight_files(model_info) -> list[tuple[str, int]]:
    files = []
    for sibling in model_info.siblings:
        name = sibling.rfilename
        if name.endswith(".safetensors") and (
            name == "model.safetensors" or name.startswith("model-")
        ):
            files.append((name, int(sibling.size or 0)))
    if not files:
        raise RuntimeError(
            f"{model_info.id} has no Transformers model.safetensors files; "
            "refusing an unreviewed checkpoint conversion."
        )
    return files


def _model_dir(config, key: str, root_override: Path | None) -> Path:
    if root_override is None:
        return model_path(config, key)
    spec = config["models"][key]
    return root_override / spec["id"].replace("/", "--") / spec["revision"]


def _resolve_keys(args, config) -> list[str]:
    if args.evict_pair:
        pair = config["pairs"][args.evict_pair]
        return list(dict.fromkeys((pair["draft"], pair["target"])))
    if args.pair:
        pair = config["pairs"][args.pair]
        return list(dict.fromkeys((pair["draft"], pair["target"])))
    if args.model:
        return [args.model]
    if args.all:
        return list(config["models"])
    if args.plan:
        return list(config["models"])
    raise SystemExit("Choose --model KEY, --pair PAIR, --all, --plan, or --evict-pair PAIR")


def _evict(config, keys, root_override) -> None:
    for key in keys:
        path = _model_dir(config, key, root_override)
        if not path.exists():
            print(f"EVICT {key}: already absent ({path})", flush=True)
            continue
        manifest_path = path / "naacl_download_manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError(f"Refusing to evict unmanifested path {path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for weight in manifest.get("weight_files", []):
            weight_path = path / weight["filename"]
            if weight_path.is_symlink() or weight_path.is_file():
                weight_path.unlink()
        # Keep the small config/tokenizer files available for alignment.
        manifest["weights_evicted"] = True
        atomic_json(manifest_path, manifest)
        print(f"EVICT {key}: removed {len(manifest.get('weight_files', []))} weight shards; kept tokenizer/config", flush=True)


def _load_and_smoke(path: Path, model_key: str, config) -> None:
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    from src.naacl_experiment import cleanup_cuda

    spec = config["models"][model_key]
    kwargs = {
        "dtype": torch.bfloat16,
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
        "local_files_only": True,
    }
    model_config = AutoConfig.from_pretrained(path, local_files_only=True)
    if getattr(model_config, "model_type", None) == "mistral3":
        from transformers import Mistral3ForConditionalGeneration

        model_class = Mistral3ForConditionalGeneration
    else:
        model_class = AutoModelForCausalLM
    model = model_class.from_pretrained(path, **kwargs).to("cuda").eval()
    input_ids = torch.tensor([[1]], dtype=torch.long, device="cuda")
    with torch.inference_mode():
        # A one-token forward exercises checkpoint deserialization and the text path.
        if getattr(model_config, "model_type", None) == "mistral3":
            out = model(input_ids=input_ids, use_cache=False)
        else:
            out = model(input_ids=input_ids, use_cache=False)
        if not torch.isfinite(out.logits).all().item():
            raise RuntimeError(f"Non-finite smoke logits for {spec['id']}")
    print(f"SMOKE_LOAD_OK {model_key} {spec['id']}@{spec['revision']} on {torch.cuda.get_device_name(0)}", flush=True)
    del model, out
    cleanup_cuda(torch)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/naacl_experiments.yaml")
    parser.add_argument("--model", choices=None, help="Model key from the config")
    parser.add_argument("--pair", choices=None, help="Download only a pair's two checkpoints")
    parser.add_argument("--all", action="store_true", help="Download all core models if they fit in the selected model root")
    parser.add_argument("--plan", action="store_true", help="Print exact sizes and free-space requirements without downloading")
    parser.add_argument("--evict-pair", help="Remove a pair's weight files after its jobs; tokenizer/config files remain")
    parser.add_argument("--model-root", type=Path, help="Override configured model root")
    parser.add_argument("--smoke-load", action="store_true", help="Load each requested checkpoint and run one GPU forward")
    args = parser.parse_args()
    config = load_config(args.config)
    if args.model and args.model not in config["models"]:
        parser.error(f"unknown model key {args.model!r}")
    if args.pair and args.pair not in config["pairs"]:
        parser.error(f"unknown pair {args.pair!r}")
    if args.evict_pair and args.evict_pair not in config["pairs"]:
        parser.error(f"unknown pair {args.evict_pair!r}")
    if args.evict_pair:
        _evict(config, _resolve_keys(args, config), args.model_root)
        return 0

    keys = _resolve_keys(args, config)
    infos = {}
    api = HfApi(token=get_token())
    for key in keys:
        spec = config["models"][key]
        info = api.model_info(spec["id"], revision=spec["revision"], files_metadata=True)
        if info.sha != spec["revision"]:
            raise RuntimeError(f"Pinned revision changed for {key}: wanted {spec['revision']}, got {info.sha}")
        weights = _weight_files(info)
        infos[key] = (info, weights, sum(size for _, size in weights))

    roots = {}
    for key in keys:
        root = args.model_root or model_path(config, key).parents[1]
        root.mkdir(parents=True, exist_ok=True)
        roots[key] = root
    grouped = {}
    for key, root in roots.items():
        grouped.setdefault(root, 0)
        grouped[root] += infos[key][2]
    for root, size in grouped.items():
        free = memory_free_gib(root)
        print(f"MODEL_STORAGE root={root} free_gib={free:.2f} requested_weight_gib={size / 1024**3:.2f}", flush=True)
    for key, (info, weights, size) in infos.items():
        print(
            f"MODEL {key} repo={info.id} revision={info.sha} gated={info.gated} "
            f"weight_gib={size / 1024**3:.2f} shards={len(weights)}",
            flush=True,
        )
    if args.plan:
        return 0
    # Leave room for Hub metadata, model config/tokenizer files, Python, and outputs.
    for root, size in grouped.items():
        required = size / 1024**3
        free = memory_free_gib(root)
        if free < required + 1.0:
            raise RuntimeError(
                f"Need {required + 1.0:.2f} GiB free for selected checkpoints and metadata; "
                f"only {free:.2f} GiB is free at {root}. Select one model or use another model root."
            )

    token = get_token()
    for key in keys:
        spec = config["models"][key]
        info, weights, size = infos[key]
        destination = _model_dir(config, key, args.model_root)
        incomplete_path = destination / "naacl_download_incomplete.json"
        complete_path = destination / "naacl_download_manifest.json"
        if destination.exists() and (destination / "config.json").is_file() and not complete_path.exists() and not incomplete_path.exists():
            raise RuntimeError(f"Destination exists without this downloader's manifest: {destination}")
        destination.mkdir(parents=True, exist_ok=True)
        atomic_json(incomplete_path, {
            "model_key": key,
            "model_id": spec["id"],
            "revision": info.sha,
            "download_in_progress": True,
        })
        names = [s.rfilename for s in info.siblings]
        allow = [name for name in names if not name.endswith((".safetensors", ".bin", ".pt", ".pth"))]
        allow.extend(name for name, _ in weights)
        print(f"DOWNLOAD_START {key} files={len(allow)} gated={info.gated}", flush=True)
        snapshot_download(
            repo_id=spec["id"],
            revision=spec["revision"],
            local_dir=destination,
            allow_patterns=allow,
            token=token,
        )
        manifest = {
            "model_key": key,
            "model_id": spec["id"],
            "revision": info.sha,
            "gated": info.gated,
            "weight_bytes": size,
            "weight_files": [{"filename": name, "bytes": size} for name, size in weights],
            "transformers": __import__("transformers").__version__,
            "torch": __import__("torch").__version__,
            "weights_evicted": False,
        }
        atomic_json(destination / "naacl_download_manifest.json", manifest)
        incomplete_path.unlink(missing_ok=True)
        print(f"DOWNLOAD_COMPLETE {key} local_path={destination}", flush=True)
        if args.smoke_load:
            _load_and_smoke(destination, key, config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
