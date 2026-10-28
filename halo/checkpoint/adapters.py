"""Saved PEFT adapters as an artifact: the file layout and its writers and readers, the resume-adapter
refusals and the expert-LoRA shape gates. Producing an adapter from a live sharded model belongs to
:mod:`halo.checkpoint.parallel.adapters`.
"""

from __future__ import annotations

import json
import logging
import os
from collections.abc import Mapping

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from halo.checkpoint.atomic import save_safetensors_file
from halo.checkpoint.format import (
    ADAPTER_BIN_FILE,
    ADAPTER_CONFIG_FILE,
    ADAPTER_SAFETENSORS_FILE,
    ADAPTER_WEIGHT_NAMES,
    RESUME_ADAPTER_DIR,
    RESUME_ADAPTER_MARKER_FILE,
)
from halo.checkpoint.reading import read_checkpoint_config
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from halo.models.unwrapping import PEFT_BASE_MODEL_PREFIX

logger = logging.getLogger(__name__)

# ``peft_type`` markers of the expert-only and the mixed attention+expert adapters. Kept out of PEFT's
# config mapping on purpose: stock PEFT drops unknown state-dict keys silently, so a ``LORA`` label would
# load the attention half and discard every expert delta, while an unknown type makes the load raise.
EXPERT_LORA_PEFT_TYPE = "EXPERT_LORA"
MIXED_EXPERT_LORA_PEFT_TYPE = "LORA_WITH_EP_EXPERT_LORA"
EXPERT_LORA_PEFT_TYPES = frozenset({EXPERT_LORA_PEFT_TYPE, MIXED_EXPERT_LORA_PEFT_TYPE})
# Where a mixed ``adapter_config.json`` describes its expert half; the expert-only one has the fields on top.
EXPERT_LORA_CONFIG_KEY = "ep_expert_lora"
# The fields a LoRA delta is scaled by. A resume under other values would restore every tensor and rescale
# the delta, so the recorded values refuse it; the native expert adapters record the first two.
LORA_SCALING_FIELDS = ("lora_alpha", "use_rslora", "use_dora", "alpha_pattern")
EXPERT_LORA_SCALING_FIELDS = ("lora_alpha", "use_rslora")
# What PEFT appends below the adapted module's own path in a saved key.
_ADAPTER_KEY_SUFFIX_MARKERS = (".lora_", ".modules_to_save", ".base_layer", ".original_module")


def is_expert_lora_key(key: str) -> bool:
    """Whether a saved adapter key is a native EP expert adapter (``<layer>.experts.<attr>.lora_{A,B}``).

    A native adapter stores the whole grouped tensor, so its key ends at ``lora_A``/``lora_B``; stock PEFT
    always appends the tuner suffix, including on per-expert ``ModuleList`` families (Bailing/Ling).
    """
    return ".experts." in key and key.endswith((".lora_A", ".lora_B"))


def expert_lora_config_fields(spec: ExpertLoraSpec) -> dict:
    """The ``adapter_config.json`` fields describing native EP grouped expert adapters: every field that
    changes what the adapter means, since a reload would otherwise rescale every expert delta."""
    return {
        "r": spec.r,
        "lora_alpha": spec.alpha,
        "lora_dropout": spec.dropout,
        "use_rslora": spec.use_rslora,
        "expert_projections": sorted(spec.projections),
    }


def expert_lora_adapter_config(spec: ExpertLoraSpec, base_model_name_or_path: str | None) -> dict:
    """The PEFT-style ``adapter_config.json`` of an expert-only adapter save."""
    return {
        "peft_type": EXPERT_LORA_PEFT_TYPE,
        "base_model_name_or_path": base_model_name_or_path,
        **expert_lora_config_fields(spec),
    }


def write_adapter_config(config: dict, output_dir: str) -> None:
    """Write ``config`` as ``output_dir``'s ``adapter_config.json``."""
    with open(os.path.join(output_dir, ADAPTER_CONFIG_FILE), "w") as fh:
        json.dump(config, fh, indent=2)


def write_mixed_adapter_config(peft_config, expert_lora_spec: ExpertLoraSpec | None, output_dir: str) -> None:
    """Write the attention ``LoraConfig`` re-typed :data:`MIXED_EXPERT_LORA_PEFT_TYPE`, so stock PEFT refuses
    the hybrid adapter instead of loading its attention half, with the expert half described under
    :data:`EXPERT_LORA_CONFIG_KEY`."""
    peft_config.save_pretrained(output_dir)
    config = read_adapter_config(output_dir)
    config["peft_type"] = MIXED_EXPERT_LORA_PEFT_TYPE
    if expert_lora_spec is not None:
        config[EXPERT_LORA_CONFIG_KEY] = expert_lora_config_fields(expert_lora_spec)
    write_adapter_config(config, output_dir)


def write_adapter_file(state: dict, output_dir: str) -> None:
    """Write adapter tensors as ``adapter_model.safetensors``, or as ``adapter_model.bin`` where
    safetensors refuses them."""
    st_path = os.path.join(output_dir, ADAPTER_SAFETENSORS_FILE)
    # safetensors refuses a view (a gathered DTensor adapter, a ``modules_to_save`` clone), which would
    # send the whole adapter to the .bin fallback.
    state = {key: tensor.contiguous() for key, tensor in state.items()}
    try:
        save_safetensors_file(state, st_path)
    except Exception as e:
        logger.warning(f"safetensors adapter save failed: {e}, using pytorch format")
        # The .bin lands first, then the safetensors leftover goes: every reader prefers that name, so
        # leaving it would resume or merge the old adapter.
        torch.save(state, os.path.join(output_dir, ADAPTER_BIN_FILE))
        try:
            if os.path.exists(st_path):
                os.remove(st_path)
        except OSError as remove_error:
            logger.warning(
                f"could not remove stale {ADAPTER_SAFETENSORS_FILE} after the .bin fallback: "
                f"{remove_error} — readers prefer it over the fresh {ADAPTER_BIN_FILE}"
            )


def adapter_weight_paths(adapter_dir: str) -> tuple[str, ...]:
    """The adapter weight files a directory may carry, in PEFT's load-preference order (``.bin`` last)."""
    return tuple(os.path.join(adapter_dir, name) for name in ADAPTER_WEIGHT_NAMES)


def has_adapter_weight_file(directory: str) -> bool:
    """Whether a directory holds an adapter weight file, from stats alone."""
    return any(os.path.isfile(path) for path in adapter_weight_paths(directory))


def read_adapter_config(adapter_dir: str, *, missing_ok: bool = False) -> dict | None:
    """Parse an adapter's ``adapter_config.json``; ``None`` only for an absent file under ``missing_ok``."""
    return read_checkpoint_config(adapter_dir, missing_ok=missing_ok, filename=ADAPTER_CONFIG_FILE)


def read_adapter_keys(adapter_dir: str) -> list[str]:
    """The tensor names in an adapter's first weight file present, read without tensor data; empty when
    the directory holds none."""
    path = next((path for path in adapter_weight_paths(adapter_dir) if os.path.isfile(path)), None)
    if path is None:
        return []
    if path.endswith(ADAPTER_SAFETENSORS_FILE):
        with safe_open(path, framework="pt") as handle:
            return list(handle.keys())
    return list(torch.load(path, map_location="meta", weights_only=True))


def adapter_module_paths(adapter_dir: str) -> set[str]:
    """The base model's module paths a saved adapter addresses, read off its tensor keys.

    A key with no PEFT segment is a parameter saved whole (``save_embedding_layers``); its module is the
    key minus that leaf.
    """
    paths: set[str] = set()
    for key in read_adapter_keys(adapter_dir):
        name = key.removeprefix(PEFT_BASE_MODEL_PREFIX)
        cut = min((at for marker in _ADAPTER_KEY_SUFFIX_MARKERS if (at := name.find(marker)) >= 0), default=-1)
        paths.add(name[:cut] if cut >= 0 else name.rpartition(".")[0])
    return paths


def read_adapter_file(path: str) -> dict:
    """The adapter tensors at ``path``: safetensors, or the legacy pickle PEFT still writes."""
    if path.endswith(ADAPTER_SAFETENSORS_FILE):
        return load_file(path)
    return torch.load(path, map_location="cpu", weights_only=True)


def resume_adapter_dir(checkpoint_dir: str) -> str | None:
    """The adapter directory a marked checkpoint resumes from, or ``None``; stat-only, for rank-0 probes."""
    if not os.path.isfile(os.path.join(checkpoint_dir, RESUME_ADAPTER_MARKER_FILE)):
        return None
    return os.path.join(checkpoint_dir, RESUME_ADAPTER_DIR)


def missing_resume_adapter_reason(checkpoint_dir: str) -> str:
    """The refusal for a marked checkpoint whose :data:`RESUME_ADAPTER_DIR` holds no adapter file."""
    return (
        f"{checkpoint_dir} is marked to resume from its adapter ({RESUME_ADAPTER_MARKER_FILE}), but "
        f"{os.path.join(checkpoint_dir, RESUME_ADAPTER_DIR)} holds no adapter file, so the adapters would "
        f"resume from initialization. Resume from a complete checkpoint."
    )


def resume_adapter_on_own_weights_reason(checkpoint_dir: str) -> str:
    """The refusal for resuming a marked checkpoint onto a model built from its own merged weights."""
    return (
        f"{checkpoint_dir} holds merged weights, which already carry the adapter delta, and resume "
        f"restores the unmerged adapters from {os.path.join(checkpoint_dir, RESUME_ADAPTER_DIR)} onto the "
        f"base model. This model was loaded from the checkpoint itself, so the delta would apply twice. "
        f"Load the model from the base the run started from (the training scripts keep "
        f"model_name_or_path there for this checkpoint)."
    )


def unmarked_merged_checkpoint_reason(checkpoint_dir: str) -> str:
    """The refusal for an adapter run resuming merged weights that carry no resume adapter."""
    return (
        f"{checkpoint_dir} holds merged weights without its resume adapter (no {RESUME_ADAPTER_MARKER_FILE}: "
        f"a torn save, or one written without it), and this run trains adapters. Resuming would restart "
        f"them from initialization under their restored optimizer state, on weights that may already "
        f"hold their trained delta. Resume from a checkpoint that carries its resume adapter, or start a "
        f"new run from its merged weights (model_name_or_path: {checkpoint_dir}, without "
        f"resume_from_checkpoint)."
    )


def lora_scaling_mismatch(adapter_dir: str, attention: Mapping | None, expert: Mapping | None = None) -> str | None:
    """Why the adapter in ``adapter_dir`` was trained at another LoRA scaling than the live run, or None.

    ``attention`` and ``expert`` are the live configs of the halves the run trains, held to the saved
    ``adapter_config.json``; a missing config is a reason too, since nothing else records the scaling.
    """
    if attention is None and expert is None:
        return None
    saved = read_adapter_config(adapter_dir, missing_ok=True)
    if saved is None:
        return (
            f"{os.path.join(adapter_dir, ADAPTER_CONFIG_FILE)} is missing, so the LoRA scaling the adapter "
            f"beside it was trained at cannot be held to this run's. Resume from a complete checkpoint."
        )
    halves = [("", saved, attention, LORA_SCALING_FIELDS)]
    halves.append(("expert ", saved.get(EXPERT_LORA_CONFIG_KEY, saved), expert, EXPERT_LORA_SCALING_FIELDS))
    changed = {
        f"{half}{field}: saved {recorded.get(field)!r}, live {live.get(field)!r}": field
        for half, recorded, live, fields in halves
        if live is not None
        for field in fields
        if recorded.get(field) != live.get(field)
    }
    if not changed:
        return None
    return (
        f"the adapter at {adapter_dir} was trained at another LoRA scaling than this run ({'; '.join(changed)}): "
        f"its tensors would restore by name and shape and every delta would be rescaled. Resume with the "
        f"{' / '.join(dict.fromkeys(changed.values()))} it was saved with."
    )


def reject_expert_lora_adapter(adapter_dir: str) -> None:
    """Refuse an adapter directory carrying native EP expert-LoRA deltas, which ``merge_and_unload`` drops.

    The verdict follows the shape, not just the marker: an absent or unmarked config falls back to a
    tensor-key scan. Every tool that merges a saved adapter runs this.
    """
    peft_type = (read_adapter_config(adapter_dir, missing_ok=True) or {}).get("peft_type")
    if peft_type in EXPERT_LORA_PEFT_TYPES:
        mixed = peft_type == MIXED_EXPERT_LORA_PEFT_TYPE
    else:
        keys = read_adapter_keys(adapter_dir)
        expert_keys = [key for key in keys if is_expert_lora_key(key)]
        if not expert_keys:
            return
        # Any non-expert adapter tensor beside them is the mixed shape.
        mixed = len(expert_keys) != len(keys)
    # ``merge_expert_lora_on_save`` folds both shapes; ``mixed`` only selects the wording.
    if mixed:
        shape = (
            f"holds a MIXED attention+expert LoRA adapter (peft_type={MIXED_EXPERT_LORA_PEFT_TYPE}): "
            "alongside the stock PEFT attention tensors it carries native EP grouped expert deltas"
        )
        folds = "the expert deltas inside each family's expert gather, and the attention deltas with them"
    else:
        shape = (
            f"is a native EP expert-LoRA adapter (peft_type={EXPERT_LORA_PEFT_TYPE}): grouped "
            "*.experts.<attr>.lora_{A,B} tensors"
        )
        folds = "the expert deltas inside each family's expert gather"
    raise ValueError(
        f"{adapter_dir} {shape} — which a PEFT merge silently drops. Re-save the run with "
        f"merge_expert_lora_on_save=True: the gathered EP save folds {folds}, producing a merged "
        "servable HF checkpoint."
    )


def expert_lora_checkpoint_keys(attr: str) -> tuple[str, str]:
    """The saved keys of expert weight ``attr``'s two native adapters, relative to its EP layer."""
    return f"experts.{attr}.lora_A", f"experts.{attr}.lora_B"
