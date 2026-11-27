"""Legacy shard-streaming model loader used by non-T6-b utilities.

The T6-b judge scorer intentionally does not import or use this loader. It
requires a complete local snapshot and calls the repository's standard
``load_model`` implementation.
"""

from __future__ import annotations

import gc
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModelForCausalLM

from scripts.rebuttal.score_cross_family_judge import _ensure_snapshot
from src.naacl_experiment import cleanup_cuda, model_path


def _load_streamed_model(config: dict[str, Any], key: str, api: HfApi, token: str | None):
    """Load pinned safetensors one shard at a time, keeping only the model in RAM."""
    path, _info, weight_files = _ensure_snapshot(config, key, api, token)
    configured_path = model_path(config, key)
    if all((configured_path / filename).is_file() for filename in weight_files):
        model = _from_pretrained(configured_path)
        return model, configured_path, "local_checkpoint"
    existing = [path / filename for filename in weight_files]
    if all(file.is_file() for file in existing):
        model = _from_pretrained(path)
        return model, path, "local_checkpoint"

    model_config = AutoConfig.from_pretrained(path, local_files_only=True)
    if getattr(model_config, "model_type", None) == "mistral3":
        from transformers import Mistral3ForConditionalGeneration

        model_class = Mistral3ForConditionalGeneration
    else:
        model_class = AutoModelForCausalLM
    kwargs: dict[str, Any] = {"attn_implementation": config["inference"].get("attention_backend", "sdpa")}
    major = int(__import__("transformers").__version__.split(".", 1)[0])
    kwargs["dtype" if major >= 5 else "torch_dtype"] = torch.bfloat16
    with torch.device("meta"):
        # Transformers 5.19 exposes Mistral3's constructor as `_from_config`;
        # AutoModel classes expose `from_config`.
        constructor = getattr(model_class, "from_config", None)
        if constructor is None:
            constructor = model_class._from_config
        model = constructor(model_config, **kwargs)
    model.to_empty(device="cpu")

    loaded: set[str] = set()
    tmp_dir = path / ".streaming_weights"
    tmp_dir.mkdir(exist_ok=True)
    for shard_index, filename in enumerate(weight_files):
        print(f"STREAM_WEIGHT_START model={key} shard={shard_index + 1}/{len(weight_files)} name={filename}", flush=True)
        local_file = Path(hf_hub_download(
            repo_id=config["models"][key]["id"],
            filename=filename,
            revision=config["models"][key]["revision"],
            token=token,
            local_dir=tmp_dir,
        ))
        state = load_file(str(local_file), device="cpu")
        if getattr(model_config, "model_type", None) == "mistral3":
            # The published Mistral3 checkpoint has unwrapped component
            # prefixes; the model wrapper registers them under `model.*`.
            remapped = {}
            for name, value in state.items():
                if name.startswith("language_model.model."):
                    name = "model.language_model." + name[len("language_model.model."):]
                elif name.startswith("language_model.lm_head."):
                    name = name[len("language_model."):]
                elif name.startswith("vision_tower."):
                    name = "model." + name
                elif name.startswith("multi_modal_projector."):
                    name = "model." + name
                if name in remapped:
                    raise RuntimeError(f"Checkpoint key remap collision for {key}: {name}")
                remapped[name] = value
            state = remapped
        incompatible = model.load_state_dict(state, strict=False)
        if incompatible.unexpected_keys:
            raise RuntimeError(f"Unexpected checkpoint keys for {key}: {incompatible.unexpected_keys[:8]}")
        loaded.update(state.keys())
        del state
        local_file.unlink(missing_ok=True)
        print(f"STREAM_WEIGHT_COMPLETE model={key} shard={shard_index + 1}/{len(weight_files)}", flush=True)
        gc.collect()

    tied_aliases: set[str] = set()
    if model.config.tie_word_embeddings:
        model.tie_weights()
        input_embeddings = model.get_input_embeddings()
        output_embeddings = model.get_output_embeddings()
        if input_embeddings is not None and output_embeddings is not None and input_embeddings.weight is output_embeddings.weight:
            tied_aliases.add("lm_head.weight")
    expected = set(model.state_dict().keys())
    missing = expected - loaded - tied_aliases
    if missing:
        raise RuntimeError(f"Checkpoint for {key} is missing {len(missing)} model keys: {sorted(missing)[:12]}")
    if any(parameter.is_meta for parameter in model.parameters()):
        raise RuntimeError(f"Checkpoint loading left meta parameters in {key}")
    try:
        model.config.use_cache = False
    except Exception:
        pass
    model = model.to("cuda").eval()
    cleanup_cuda(torch)
    return model, path, "streamed_shards_to_cpu"


def _from_pretrained(path: Path):
    model_config = AutoConfig.from_pretrained(path, local_files_only=True)
    if getattr(model_config, "model_type", None) == "mistral3":
        from transformers import Mistral3ForConditionalGeneration

        model_class = Mistral3ForConditionalGeneration
    else:
        model_class = AutoModelForCausalLM
    major = int(__import__("transformers").__version__.split(".", 1)[0])
    kwargs: dict[str, Any] = {
        "low_cpu_mem_usage": True,
        "attn_implementation": "sdpa",
        "local_files_only": True,
    }
    kwargs["dtype" if major >= 5 else "torch_dtype"] = torch.bfloat16
    return model_class.from_pretrained(path, **kwargs).to("cuda").eval()
