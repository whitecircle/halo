"""The standalone tools' model-directory contract: the source resolver, the input gates, the size preflight,
the non-weight copy, the resume state a shard merge carries, the gated load of the head class per ``--task``,
the fp32 norm cast and the one full-model write every tool owes its output directory."""

import contextlib
import logging
import os
import shutil
from collections.abc import Sequence
from fnmatch import fnmatch

import torch
from accelerate.utils import is_peft_model
from huggingface_hub import snapshot_download
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
    WEIGHT_FILE_SUFFIXES,
    optimizer_shard_filename,
)
from halo.checkpoint.hub_layout import save_pretrained_layout
from halo.checkpoint.model_card import tag_exported_model_card
from halo.checkpoint.reading import checkpoint_shard_files
from halo.checkpoint.weights_source import config_schema_source
from halo.checkpoint.writing import sweep_after_full_save
from halo.hardware import available_host_ram_bytes
from halo.model_loading.checkpoint_coverage import from_pretrained_verified
from halo.model_loading.model_preparation import auto_load_model, sanitize_generation_config
from halo.models.param_roles import is_normalization_module

logger = logging.getLogger(__name__)

# ``--task`` → the class a tool loads a checkpoint with; None resolves it from the config (``auto_load_model``:
# AutoModelForCausalLM would drop a VLM's vision tower). One table, so every load goes through the coverage gate.
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


def copy_checkpoint_aux_files(
    input_dir: str, output_dir: str, *, include_resume_sidecars: bool = True, verbose: bool = False
) -> None:
    """Copy a checkpoint's non-weight files from ``input_dir`` to ``output_dir``, then tag the model card.

    Skips the top-level weight files and indexes the caller writes fresh, but keeps the resume state a
    resume-from-merged run restores unless ``include_resume_sidecars`` is off (an artifact that is a new
    base). Subdirectories copy whole, weights included (a SentenceTransformer module directory), except a
    nested ``checkpoint-N``, a run-internal ``_*`` one, a vendor weight dump and a hidden one; unpublished
    staged files stay behind too. ``output_dir`` inside ``input_dir`` raises: the walk would copy the
    destination into itself.
    """
    input_root = os.path.realpath(input_dir)
    if os.path.commonpath([input_root, os.path.realpath(output_dir)]) == input_root:
        raise ValueError(
            f"copy_checkpoint_aux_files: output_dir {output_dir!r} is inside input_dir "
            f"{input_dir!r} — the subdirectory copy would recurse into the destination. Write the "
            f"artifact to a directory outside the source checkpoint."
        )
    for name in os.listdir(input_dir):
        if name in UNSUFFIXED_RESUME_ENTRIES and not include_resume_sidecars:
            continue
        src = os.path.join(input_dir, name)
        if os.path.isdir(src):
            if name.startswith((".", RUN_INTERNAL_PREFIX, f"{PREFIX_CHECKPOINT_DIR}-")) or name in WEIGHT_DUMP_DIRS:
                continue
            shutil.copytree(src, os.path.join(output_dir, name), dirs_exist_ok=True)
            if verbose:
                print(f"Copied: {name}/")  # noqa: T201 — CLI-facing helper; the merge scripts report via print
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
                print(f"Copied: {name}")  # noqa: T201 - CLI-facing helper; the merge scripts report via print
    tag_exported_model_card(output_dir, source_dir=input_dir)


def first_holder(name: str, input_dirs: Sequence[str]) -> str | None:
    """The first input holding ``name``, or None; two holding it at different sizes raise (a torn copy)."""
    holders = [d for d in input_dirs if os.path.isfile(os.path.join(d, name))]
    if len({os.path.getsize(os.path.join(d, name)) for d in holders}) > 1:
        raise ValueError(f"{name} differs in size across {holders}: a torn copy. Re-copy that node's directory.")
    return holders[0] if holders else None


def resume_state_sources(input_dirs: Sequence[str]) -> dict[str, str]:
    """``{filename: input holding it}`` for the resume state a merge of ``input_dirs``, the directories of one
    save, carries: every per-rank file (optimizer shards, RNG and prefetch states, one per global rank) and,
    last, the optimizer meta every node writes alike. Carried, they resume the merged directory exactly on the
    save's topology, behind the save's own fingerprint gates.

    Raises:
        ValueError: a per-rank file held at different sizes, inputs whose optimizer metas differ, or a rank
            of the meta whose optimizer shard no input holds.
    """
    names = {
        name
        for input_dir in input_dirs
        for name in os.listdir(input_dir)
        if fnmatch(name, OPTIMIZER_SHARD_GLOB) or name.startswith(RESUME_SIDECAR_PREFIXES)
    }
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
    """``model_path`` loaded as ``task``'s class (:data:`TASK_MODEL_CLASSES`), its missing keys gated.

    ``excuse_task_head`` lets the head the class adds over the backbone be absent, which only the caller can
    judge (an adapter that saves it, an encoder base under the widest causal-LM class). ``text_only``, the class
    an adapter may be trained through on a multimodal checkpoint, applies to the causal-LM resolution; a task
    with one class raises ``ValueError`` on it.
    """
    model_class = TASK_MODEL_CLASSES[task]
    if model_class is None:
        return auto_load_model(model_path, excuse_task_head=excuse_task_head, text_only=text_only, **load_kwargs)
    if text_only:
        raise ValueError(f"task={task!r} loads one class; text_only applies to causal_lm only.")
    return from_pretrained_verified(model_class, model_path, excuse_task_head=excuse_task_head, **load_kwargs)


def cast_norms_to_fp32(model) -> None:
    """Cast every normalization leaf to fp32 (a bf16 body with fp32 norms). Leaves only: a container that
    classifies as a norm must not drag its children up with it."""
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
    **save_kwargs,
) -> None:
    """``save_pretrained`` plus the non-weight files a conversion tool's output directory needs.

    The order is the contract: ``source_dir``'s files are copied before the save so the live config and
    tokenizer overwrite them, stale weight files are swept only after it, and carried legacy tokenizer
    files the fresh processing class did not rewrite are dropped. The save writes the layout the saved
    config declares (:func:`~halo.checkpoint.hub_layout.save_pretrained_layout`). Full models only.
    """
    if is_peft_model(model):
        raise ValueError(
            "save_full_checkpoint writes full model directories; an unmerged PEFT adapter save must "
            "call model.save_pretrained directly (it writes adapter files only — nothing to sweep)."
        )
    os.makedirs(output_dir, exist_ok=True)
    if source_dir is not None and os.path.isdir(source_dir):
        copy_checkpoint_aux_files(source_dir, output_dir, include_resume_sidecars=include_resume_sidecars)
    sanitize_generation_config(model)
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
    """A local checkpoint directory as-is; a Hub repo id resolved through the download cache."""
    if os.path.isdir(model_id):
        return model_id
    logger.info(f"Resolving {model_id} through the HF cache (HF_HOME)...")
    return snapshot_download(model_id, revision=revision)


def reject_in_place_conversion(input_dir: str, output_dir: str) -> None:
    """Refuse a conversion whose output directory is its own input: every tool writer ends in
    :func:`~halo.checkpoint.writing.remove_stale_checkpoint_files`, which deletes the source shards."""
    if os.path.realpath(input_dir) == os.path.realpath(output_dir):
        raise ValueError(
            f"input and output directory are the same path ({output_dir}). These conversions are not "
            f"in-place: writing the result deletes the source shards it does not overwrite, so the "
            f"original checkpoint would be destroyed. Write to a new directory."
        )


def reject_sibling_adapter(input_dir: str) -> None:
    """Refuse a shard merge whose input holds a PEFT adapter beside the shards.

    The aux copy carries ``adapter_config.json`` but not the adapter weights, so the merged directory
    would advertise an adapter it does not hold; no supported run produces this pairing.
    """
    adapter_weights = [name for name in ADAPTER_WEIGHT_NAMES if os.path.isfile(os.path.join(input_dir, name))]
    if adapter_weights:
        raise ValueError(
            f"{input_dir} holds a PEFT adapter ({', '.join(adapter_weights)}) beside its shards. The "
            f"merge writes base weights only and copies {ADAPTER_CONFIG_FILE} across, so the merged "
            f"directory would claim an adapter whose weights are missing. Move the adapter out and "
            f"merge it separately (scripts/after_training/merge_peft_adapters.py against the merged "
            f"base), or re-save gathered."
        )


def preflight_resource_warning(tool: str, output_dir: str, *, disk_bytes: int | None, ram_bytes: int | None) -> None:
    """Warn when ``tool``'s estimated peak host RAM or output disk exceeds what the host has free.

    Warn-only because both figures are estimates; ``None`` skips a side.
    """
    if ram_bytes:
        available = available_host_ram_bytes()
        if available is not None and ram_bytes > available:
            print(  # noqa: T201 — CLI-facing preflight; the conversion tools report via print
                f"WARNING: {tool}: estimated peak host RAM ~{ram_bytes / 1e9:.1f} GB exceeds available "
                f"~{available / 1e9:.1f} GB (/proc/meminfo MemAvailable). The process may be OOM-killed; "
                f"free host memory or run on a larger host. Continuing — this is an estimate."
            )
    if disk_bytes:
        # The output directory may not exist yet; the nearest existing ancestor is on the same volume.
        target = os.path.realpath(output_dir)
        while not os.path.exists(target):
            target = os.path.dirname(target)
        free = shutil.disk_usage(target).free
        if disk_bytes > free:
            print(  # noqa: T201 — CLI-facing preflight; the conversion tools report via print
                f"WARNING: {tool}: the output volume for {output_dir} has ~{free / 1e9:.1f} GB free but "
                f"the artifact needs ~{disk_bytes / 1e9:.1f} GB. The write may fail with ENOSPC; free disk "
                f"or write the output to a larger volume. Continuing — this is an estimate."
            )


def preflight_model_load_resources(
    weights_dir: str | None,
    output_dir: str,
    *,
    tool: str,
    device_map: str | None = None,
    writes_full_model: bool = True,
) -> None:
    """Size preflight for a tool loading a whole checkpoint through ``from_pretrained``, estimated from the
    source's safetensors bytes.

    Silent for a hub id, a weightless directory and a per-rank-sharded one: whether the input is usable
    is the tool's own gates' call.
    """
    if not (weights_dir and os.path.isdir(weights_dir)):
        return
    try:
        shards = checkpoint_shard_files(weights_dir)
    except (FileNotFoundError, ValueError):
        return
    total = sum(os.path.getsize(shard) for shard in shards)
    lands_on_host = device_map in (None, "cpu")
    preflight_resource_warning(
        tool, output_dir, disk_bytes=total if writes_full_model else None, ram_bytes=total if lands_on_host else None
    )
