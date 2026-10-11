"""Shared settings and fast, resumable I/O for the NAACL experiment plan."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/naacl_experiments.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def model_path(config: dict[str, Any], model_key: str) -> Path:
    per_model_env = f"NAACL_MODEL_DIR_{model_key.upper()}"
    root = Path(os.environ.get(per_model_env, os.environ.get("NAACL_MODEL_DIR", config["paths"]["model_root"])))
    model = config["models"][model_key]
    return root / model["id"].replace("/", "--") / model["revision"]


def output_root(config: dict[str, Any], override: str | Path | None = None) -> Path:
    path = Path(override) if override else ROOT / config["paths"]["outputs"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_json(path: str | Path, value: Any) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, default=_json_default) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def atomic_parquet(rows: list[dict[str, Any]], path: str | Path) -> None:
    import pandas as pd

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.name + ".tmp")
    pd.DataFrame(rows).to_parquet(
        temporary,
        index=False,
        engine="pyarrow",
        compression="zstd",
        compression_level=3,
    )
    temporary.replace(destination)


def read_parquet_records(path: str | Path) -> list[dict[str, Any]]:
    import pandas as pd

    return pd.read_parquet(path).to_dict(orient="records")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_prompts(config: dict[str, Any], split: str) -> list[dict[str, Any]]:
    from .table1_data import load_prompt_pool

    pool_path = ROOT / config["data"]["prompts"]
    rows = load_prompt_pool(pool_path)
    if split == "all":
        selected = rows[:40_000]
    elif split == "common":
        selected = rows[:20_000]
    elif split == "extra":
        selected = rows[20_000:40_000]
    else:
        raise ValueError(f"Unknown prompt split {split!r}; expected common, all, or extra")
    return selected


def prompt_ids_sha256(rows: list[dict[str, Any]]) -> str:
    ids = "\n".join(str(row["doc_id"]) for row in rows)
    return hashlib.sha256(ids.encode("utf-8")).hexdigest()


def parquet_shard_paths(directory: Path, shard_size: int) -> list[Path]:
    if not directory.exists():
        return []
    return sorted(directory.glob(f"shard-*-of-{shard_size:05d}.parquet"))


def prompt_shards(rows: list[dict[str, Any]], shard_size: int):
    for start in range(0, len(rows), shard_size):
        yield start // shard_size, rows[start : start + shard_size]


def require_model_snapshot(path: Path) -> None:
    manifest = path / "naacl_download_manifest.json"
    if (
        not path.is_dir()
        or not (path / "config.json").is_file()
        or not manifest.is_file()
        or json.loads(manifest.read_text(encoding="utf-8")).get("weights_evicted")
    ):
        raise FileNotFoundError(
            f"Model snapshot is missing at {path}. Run scripts/download_models.py first."
        )


def memory_free_gib(path: Path) -> float:
    return shutil.disk_usage(path).free / (1024**3)


def cleanup_cuda(torch: Any) -> None:
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def load_tokenizer(config: dict[str, Any], model_key: str, snapshot: Path | None = None):
    from transformers import AutoTokenizer

    spec = config["models"][model_key]
    path = snapshot or model_path(config, model_key)
    backend = spec.get("tokenizer_backend", "auto")
    options: dict[str, Any] = {"local_files_only": True, "use_fast": True}
    if backend == "tokenizers":
        from transformers import PreTrainedTokenizerFast

        tokenizer = PreTrainedTokenizerFast.from_pretrained(
            path,
            fix_mistral_regex=bool(spec.get("fix_mistral_regex", False)),
            **options,
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(path, **options)
    if tokenizer.eos_token_id is None:
        raise RuntimeError(f"Model {model_key} tokenizer has no EOS token")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model(
    config: dict[str, Any],
    model_key: str,
    *,
    device: str = "cuda",
    snapshot: Path | None = None,
):
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM

    path = snapshot or model_path(config, model_key)
    require_model_snapshot(path)
    dtype = torch.bfloat16
    kwargs: dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "attn_implementation": config["inference"].get("attention_backend", "sdpa"),
        "local_files_only": True,
    }
    major = int(transformers.__version__.split(".", 1)[0])
    kwargs["dtype" if major >= 5 else "torch_dtype"] = dtype
    model_config = AutoConfig.from_pretrained(path, local_files_only=True)
    if getattr(model_config, "model_type", None) == "mistral3":
        from transformers import Mistral3ForConditionalGeneration

        model_class = Mistral3ForConditionalGeneration
    else:
        model_class = AutoModelForCausalLM
    model = model_class.from_pretrained(path, **kwargs).to(device).eval()
    return model
