"""The standalone tools' model-directory contract: source resolver, input gates, size preflight, non-weight copy, a
shard merge's resume state, the gated load per ``--task``, the fp32 norm cast, the MTP tail a class drops and the one
full-model write."""

import contextlib
import logging
import os
import shutil
from collections.abc import Collection, Iterable, Mapping, Sequence

import torch
from accelerate.utils import is_peft_model
from huggingface_hub import is_offline_mode, snapshot_download
from transformers import AutoModel, AutoModelForSequenceClassification
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import GENERATION_CONFIG_NAME

from halo.checkpoint.atomic import apply_umask_mode, is_atomic_staging_file
from halo.checkpoint.config_export import finalize_exported_config
from halo.checkpoint.format import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_WEIGHT_NAMES,
    FOREIGN_EXPORT_SUFFIXES,
    OPTIMIZER_META_FILE,
    OPTIMIZER_SHARD_GLOB,
    RESUME_SIDECAR_FILES,
    RESUME_SIDECAR_PREFIXES,
    RUN_INTERNAL_PREFIX,
    UNSUFFIXED_RESUME_ENTRIES,
    WEIGHT_DUMP_DIRS,
    WEIGHT_DUMP_IGNORE_PATTERNS,
    WEIGHT_FILE_SUFFIXES,
    is_per_rank_resume_file,
    optimizer_shard_filename,
)
from halo.checkpoint.hub_layout import save_pretrained_layout
from halo.checkpoint.model_card import tag_exported_model_card
from halo.checkpoint.reading import (
    checkpoint_shard_files,
    iter_checkpoint_tensors,
    read_checkpoint_config,
    resolve_checkpoint_weights,
)
from halo.checkpoint.weights_source import config_schema_source
from halo.checkpoint.writing import sweep_after_full_save
from halo.hardware import available_host_ram_bytes
from halo.model_loading.checkpoint_coverage import from_pretrained_verified
from halo.model_loading.model_preparation import auto_load_model, sanitize_generation_config
from halo.models.mtp.head import mtp_head_path
from halo.models.mtp.spec import (
    declared_tail_fields,
    find_mtp_head_spec,
    stored_head_prefix,
    stored_tail_keys,
    tail_load_patterns,
    unbuilt_tail_keys,
)
from halo.models.param_roles import declared_fp32_pins, fp32_pin_matches, is_normalization_module

logger = logging.getLogger(__name__)

# None resolves from the config (``auto_load_model``), as AutoModelForCausalLM would drop a VLM's vision tower.
TASK_MODEL_CLASSES = {
    "causal_lm": None,
    "classification": AutoModelForSequenceClassification,
    "base": AutoModel,
}
# The tasks a PEFT adapter is trained for: a bare backbone has no adapter task type.
ADAPTER_TASKS = ("causal_lm", "classification")

# Tokenizer files transformers 5 reads but no longer writes. Each overrides its fresh neighbour on load,
# so a copy carried over from the source must not outlive a fresh save.
_LEGACY_TOKENIZER_SIDECARS = ("special_tokens_map.json", "added_tokens.json", "chat_template.json")


def is_non_artifact_entry(name: str, *, is_dir: bool) -> bool:
    """Whether a top-level entry stays out of an artifact: any run-internal ``_*`` entry, a hidden subdirectory, a
    nested ``checkpoint-N`` (resume state) or a vendor weight dump."""
    if name.startswith(RUN_INTERNAL_PREFIX):
        return True
    return is_dir and (name.startswith((".", f"{PREFIX_CHECKPOINT_DIR}-")) or name in WEIGHT_DUMP_DIRS)


def copy_checkpoint_aux_files(
    input_dir: str,
    output_dir: str,
    *,
    include_resume_sidecars: bool = True,
    carried: Collection[str] = (),
    verbose: bool = False,
) -> None:
    """Copy non-weight files to ``output_dir`` and tag the model card, keeping resume state unless
    ``include_resume_sidecars`` is off (an artifact that is a new base). Subdirectories copy whole unless they are
    :func:`is_non_artifact_entry`; ``carried`` names are the caller's to bring. ``output_dir`` inside ``input_dir``
    raises."""
    input_root = os.path.realpath(input_dir)
    if os.path.commonpath([input_root, os.path.realpath(output_dir)]) == input_root:
        raise ValueError(
            f"copy_checkpoint_aux_files: output_dir {output_dir!r} is inside input_dir "
            f"{input_dir!r} — the subdirectory copy would recurse into the destination. Write the "
            f"artifact to a directory outside the source checkpoint."
        )
    for name in os.listdir(input_dir):
        src = os.path.join(input_dir, name)
        is_dir = os.path.isdir(src)
        if (
            name in carried
            or is_non_artifact_entry(name, is_dir=is_dir)
            or (name in UNSUFFIXED_RESUME_ENTRIES and not include_resume_sidecars)
        ):
            continue
        if is_dir:
            shutil.copytree(src, os.path.join(output_dir, name), dirs_exist_ok=True)
            if verbose:
                logger.info(f"Copied: {name}/")
            continue
        skip_as_weight = (
            name.endswith(WEIGHT_FILE_SUFFIXES)
            or name.endswith(FOREIGN_EXPORT_SUFFIXES)
            or name.endswith(".safetensors.index.json")
        )
        keep_as_sidecar = include_resume_sidecars and (
            name in RESUME_SIDECAR_FILES or name.startswith(RESUME_SIDECAR_PREFIXES)
        )
        if (skip_as_weight and not keep_as_sidecar) or is_atomic_staging_file(name):
            continue
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, name))
            if verbose:
                logger.info(f"Copied: {name}")
    tag_exported_model_card(output_dir, source_dir=input_dir)


def first_holder(name: str, input_dirs: Sequence[str]) -> str | None:
    """The first input holding ``name``, or None; two holding it at different sizes raise (a torn copy)."""
    holders = [d for d in input_dirs if os.path.isfile(os.path.join(d, name))]
    if len({os.path.getsize(os.path.join(d, name)) for d in holders}) > 1:
        raise ValueError(f"{name} differs in size across {holders}: a torn copy. Re-copy that node's directory.")
    return holders[0] if holders else None


def resume_state_sources(input_dirs: Sequence[str]) -> dict[str, str]:
    """``{filename: input holding it}`` for the resume state a merge of one save's directories carries: every per-rank
    file and, last, the optimizer meta, so the merged directory resumes exactly on the save's topology behind its
    fingerprint gates. Raises on a size conflict, differing metas, or a meta rank whose optimizer shard no input
    holds."""
    names = {name for input_dir in input_dirs for name in os.listdir(input_dir) if is_per_rank_resume_file(name)}
    sources = {name: first_holder(name, input_dirs) for name in sorted(names)}
    meta_paths = [os.path.join(input_dir, OPTIMIZER_META_FILE) for input_dir in input_dirs]
    metas = [
        torch.load(path, map_location="cpu", weights_only=True) if os.path.isfile(path) else None
        for path in meta_paths
    ]
    for input_dir, meta in zip(input_dirs[1:], metas[1:], strict=True):
        if meta != metas[0]:
            raise ValueError(
                f"{input_dir} and {input_dirs[0]} carry different {OPTIMIZER_META_FILE}: they are not node "
                f"directories of one save, or one node's optimizer save did not complete."
            )
    if metas[0] is None:
        return sources
    sources[OPTIMIZER_META_FILE] = input_dirs[0]
    missing = [rank for rank in range(metas[0]["num_ranks"]) if optimizer_shard_filename(rank) not in sources]
    if missing:
        raise ValueError(
            f"{OPTIMIZER_META_FILE} vouches for {metas[0]['num_ranks']} optimizer shards, but no input holds the "
            f"optimizer shards of ranks {missing}. Bring every rank's {OPTIMIZER_SHARD_GLOB} into the input (a "
            f"PP save on per-node storage: pass every node's directory), or delete {OPTIMIZER_META_FILE} and "
            f"every {OPTIMIZER_SHARD_GLOB} from the input to merge the weights without optimizer state (a "
            f"resume then warm-restarts)."
        )
    return sources


def _file_versions(directory: str, names: tuple[str, ...]) -> dict[str, tuple[int, int]]:
    """``{name: (inode, mtime_ns)}`` for the ``names`` present in ``directory``; a rewrite changes it."""
    versions = {}
    for name in names:
        try:
            stat = os.stat(os.path.join(directory, name))
        except FileNotFoundError:
            continue
        versions[name] = (stat.st_ino, stat.st_mtime_ns)
    return versions


def load_task_model(model_path: str, task: str, *, excuse_task_head: bool, text_only: bool = False, **load_kwargs):
    """``model_path`` loaded as ``task``'s class, its missing keys gated. ``excuse_task_head`` lets the class's added
    head be absent, which only the caller can judge; ``text_only`` applies to the causal-LM resolution and raises
    elsewhere."""
    model_class = TASK_MODEL_CLASSES[task]
    if model_class is None:
        return auto_load_model(model_path, excuse_task_head=excuse_task_head, text_only=text_only, **load_kwargs)
    if text_only:
        raise ValueError(f"task={task!r} loads one class; text_only applies to causal_lm only.")
    return from_pretrained_verified(model_class, model_path, excuse_task_head=excuse_task_head, **load_kwargs)


def cast_norms_to_fp32(model) -> None:
    """Cast every normalization leaf to fp32; leaves only, as a norm-classed container must not drag its children."""
    logger.info("Setting specific layers to appropriate precision...")
    for module in model.modules():
        if is_normalization_module(module) and next(module.children(), None) is None:
            module.to(torch.float32)


def save_full_checkpoint(
    model,
    output_dir: str,
    processing_class=None,
    source_dir=None,
    *,
    include_resume_sidecars: bool = True,
    carried_tensors: Mapping[str, torch.Tensor] | None = None,
    **save_kwargs,
) -> None:
    """``save_pretrained`` plus a tool output's non-weight files, for full models only. The order is the contract:
    ``source_dir``'s files copy before the save so the live config and tokenizer overwrite them, stale weights are
    swept after it, and carried legacy tokenizer files the fresh processing class did not rewrite are dropped.
    ``carried_tensors`` are written beside the model's own."""
    if is_peft_model(model):
        raise ValueError(
            "save_full_checkpoint writes full model directories; an unmerged PEFT adapter save must "
            "call model.save_pretrained directly (it writes adapter files only — nothing to sweep)."
        )
    os.makedirs(output_dir, exist_ok=True)
    if source_dir is not None and os.path.isdir(source_dir):
        copy_checkpoint_aux_files(source_dir, output_dir, include_resume_sidecars=include_resume_sidecars)
    sanitize_generation_config(model)
    if carried_tensors:
        save_kwargs["state_dict"] = {**model.state_dict(), **carried_tensors}
    with save_pretrained_layout(model):
        model.save_pretrained(output_dir, **save_kwargs)
    if not model.can_generate():
        # A copied generation config would advertise sampling defaults this head has no forward for.
        with contextlib.suppress(FileNotFoundError):
            os.remove(os.path.join(output_dir, GENERATION_CONFIG_NAME))
    finalize_exported_config(model.config, output_dir, source=source_dir or config_schema_source(model))
    if processing_class is not None:
        carried = _file_versions(output_dir, _LEGACY_TOKENIZER_SIDECARS)
        processing_class.save_pretrained(output_dir)
        for name, version in carried.items():
            if _file_versions(output_dir, (name,)).get(name) == version:
                os.remove(os.path.join(output_dir, name))
    sweep_after_full_save(output_dir)
    apply_umask_mode(output_dir)


def resolve_checkpoint_source(model_id: str, revision: str | None = None) -> str:
    """A local directory as-is; a Hub id through the download cache, cache-only when offline, as a call that may fetch
    lists a pinned commit's files through the Hub. The fetch leaves vendor weight dumps behind, which no tool reads."""
    if os.path.isdir(model_id):
        return model_id
    logger.info(f"Resolving {model_id} through the HF cache (HF_HOME)...")
    return snapshot_download(
        model_id,
        revision=revision,
        local_files_only=is_offline_mode(),
        ignore_patterns=list(WEIGHT_DUMP_IGNORE_PATTERNS),
    )


def _class_load_ignores(model) -> tuple[str, ...]:
    return getattr(type(model), "_keys_to_ignore_on_load_unexpected", None) or ()


def _warn_on_dropped_mtp_tail(model, model_id: str, output_dir: str, weight_map, *, tool: str) -> None:
    tail = sorted(
        {
            *stored_tail_keys(weight_map, _class_load_ignores(model)),
            *unbuilt_tail_keys(weight_map, model.config, model.state_dict()),
        }
    )
    if not tail:
        return
    declared = declared_tail_fields(model.config)
    logger.warning(
        f"{tool}: {model_id} stores {len(tail)} MTP head tensor(s) (e.g. {tail[0]!r}) that {type(model).__name__} drops "
        f"at load, so {output_dir} carries no head while its config keeps {declared or 'its tail fields'}, and vLLM "
        f"refuses method=mtp on it."
    )


def carried_mtp_tail(
    model, model_id: str, output_dir: str, *, tool: str, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """The MTP head ``model``'s class dropped at load, for the save to write back at ``dtype``, so a conversion keeps a
    head the source trained. The trunk copies a release stores come off the live embeddings, which a vocabulary patch
    grows. A tail with no spec stays dropped, as nothing says which of its tensors follow the vocabulary, and warns with
    the config field its output keeps; so does a class with no output embedding for a head to share, and a quantized
    source, whose tail holds no float tensors to cast. Header-only until a head is carried; a family with no spec whose
    class names no tail and whose config declares none never looks at the source, as a Hub id's snapshot fetches files
    its load did not need."""
    spec = find_mtp_head_spec(model.config)
    if spec is None and not tail_load_patterns(_class_load_ignores(model)) and not declared_tail_fields(model.config):
        return {}
    source = resolve_checkpoint_source(model_id)
    weight_map = resolve_checkpoint_weights(source).weight_map
    quantized = (read_checkpoint_config(source, missing_ok=True) or {}).get("quantization_config")
    if spec is None or quantized or model.get_output_embeddings() is None:
        _warn_on_dropped_mtp_tail(model, model_id, output_dir, weight_map, tool=tool)
        return {}
    path = mtp_head_path(model, spec)
    prefix = stored_head_prefix(weight_map, model.config, path)
    if prefix is None:
        return {}
    embeddings = {"input": model.get_input_embeddings(), "output": model.get_output_embeddings()}
    copies = {
        f"{path}.{key}": embeddings[copied].weight.detach().to("cpu", dtype).clone()
        for key, copied in spec.trunk_copies
    }
    pinned = fp32_pin_matches(weight_map, declared_fp32_pins(model)) if spec.tail_keeps_fp32_pins else frozenset()
    tail = {
        f"{path}.{key[len(prefix) :]}": tensor.to(dtype)
        if tensor.is_floating_point() and key not in pinned
        else tensor
        for key, tensor in iter_checkpoint_tensors(
            source, predicate=lambda key: key.startswith(prefix) and key not in copies
        )
    }
    logger.info(
        f"{tool}: carried the {len(tail)} MTP head tensor(s) {model_id} stores under {path}.* to {output_dir}."
    )
    return {**tail, **copies}


def reject_in_place_conversion(input_dir: str, output_dir: str) -> None:
    """Refuse an output that is its input, since every tool writer deletes the source shards it does not overwrite. An
    existing output is compared by identity too, as ``realpath`` does not see through a bind mount."""
    if os.path.realpath(input_dir) == os.path.realpath(output_dir) or (
        os.path.exists(input_dir) and os.path.exists(output_dir) and os.path.samefile(input_dir, output_dir)
    ):
        raise ValueError(
            f"input and output directory are the same path (input {input_dir}, output {output_dir}). These "
            f"conversions are not in-place: writing the result deletes the source shards it does not overwrite, "
            f"so the original checkpoint would be destroyed. Write to a new directory."
        )


def reject_sibling_adapter(input_dir: str) -> None:
    """Refuse a shard merge with a PEFT adapter beside the shards, its config alone included: the aux copy carries
    ``adapter_config.json`` but not the weights, so the merged directory would advertise an adapter it lacks and
    ``from_pretrained`` would load it as one."""
    adapter_files = [
        name for name in (*ADAPTER_WEIGHT_NAMES, ADAPTER_CONFIG_FILE) if os.path.isfile(os.path.join(input_dir, name))
    ]
    if adapter_files:
        raise ValueError(
            f"{input_dir} holds a PEFT adapter ({', '.join(adapter_files)}) beside its shards. The "
            f"merge writes base weights only and copies {ADAPTER_CONFIG_FILE} across, so the merged "
            f"directory would claim an adapter whose weights are missing. Move the adapter out and "
            f"merge it separately (scripts/after_training/merge_peft_adapters.py against the merged "
            f"base), or re-save gathered."
        )


def _nearest_existing_path(path: str) -> str:
    """``path`` resolved, or its nearest existing ancestor, whose volume a directory yet to be created lands on."""
    target = os.path.realpath(path)
    while not os.path.exists(target):
        target = os.path.dirname(target)
    return target


def cross_device_bytes(paths: Iterable[str], output_dir: str) -> int:
    """Estimated bytes of ``paths`` that :func:`~halo.checkpoint.atomic.link_or_copy_file` copies rather than links:
    those on another filesystem. Undercounts where the output's filesystem refuses a link anyway (a second bind mount,
    a protected hard link, a project-quota boundary)."""
    device = os.stat(_nearest_existing_path(output_dir)).st_dev
    return sum(status.st_size for status in map(os.stat, paths) if status.st_dev != device)


def preflight_resource_warning(tool: str, output_dir: str, *, disk_bytes: int | None, ram_bytes: int | None) -> None:
    """Warn when estimated peak host RAM or output disk exceeds what is free; ``None`` skips a side."""
    if ram_bytes:
        available = available_host_ram_bytes()
        if available is not None and ram_bytes > available:
            logger.warning(
                f"{tool}: estimated peak host RAM ~{ram_bytes / 1e9:.1f} GB exceeds available "
                f"~{available / 1e9:.1f} GB (/proc/meminfo MemAvailable). The process may be OOM-killed; "
                f"free host memory or run on a larger host. Continuing — this is an estimate."
            )
    if disk_bytes:
        free = shutil.disk_usage(_nearest_existing_path(output_dir)).free
        if disk_bytes > free:
            logger.warning(
                f"{tool}: the output volume for {output_dir} has ~{free / 1e9:.1f} GB free but "
                f"the artifact needs ~{disk_bytes / 1e9:.1f} GB. The write may fail with ENOSPC; free disk "
                f"or write the output to a larger volume. Continuing — this is an estimate."
            )


def preflight_model_load_resources(
    weights_dir: str | None,
    output_dir: str,
    *,
    tool: str,
    device_map: str | None = None,
) -> None:
    """Size preflight for a whole-checkpoint ``from_pretrained``, from the source's safetensors bytes. Silent for a hub
    id, a weightless directory and a per-rank one, whose usability is the tool's own gates' call."""
    if not (weights_dir and os.path.isdir(weights_dir)):
        return
    try:
        shards = checkpoint_shard_files(weights_dir)
    except (FileNotFoundError, ValueError):
        return
    total = sum(os.path.getsize(shard) for shard in shards)
    lands_on_host = device_map in (None, "cpu")
    preflight_resource_warning(tool, output_dir, disk_bytes=total, ram_bytes=total if lands_on_host else None)
