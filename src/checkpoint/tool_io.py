"""Checkpoint-directory I/O shared by the standalone conversion tools in ``scripts/``.

Covers which safetensors files make up a checkpoint, whether the requested conversion is writable,
the full-model write path, the staged-publish helpers for a tool that may be replacing the only
copy, and the training-state sidecars an exported artifact carries. Exported ``config.json``
contents live in :mod:`src.checkpoint.config_export`, saved-adapter shape in
:mod:`src.checkpoint.adapters`, and the tools' CLI flags in :mod:`scripts._common`.
"""

import contextlib
import glob
import json
import logging
import math
import os
import shutil
from collections.abc import Callable, Iterator
from typing import Any

import torch
from accelerate.utils import is_peft_model
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers.core_model_loading import PrefixChange
from transformers.integrations.finegrained_fp8 import Fp8Dequantize
from transformers.utils import CONFIG_NAME, GENERATION_CONFIG_NAME

from src.checkpoint.config_export import (
    checkpoint_source_ref,
    finalize_exported_config,
)
from src.checkpoint.format import (
    ADAPTER_CONFIG_FILE,
    ADAPTER_WEIGHT_NAMES,
    EP_SHARD_KEY_RE,
    PROVENANCE_GPT_OSS_SINKS,
    ROUTER_BALANCING_BIASES_FILE,
    SAFETENSORS_FAMILY_GLOB,
    SAFETENSORS_INDEX_FILE,
    SAFETENSORS_WEIGHTS_FILE,
    TRAINING_PROVENANCE_FILE,
    copy_checkpoint_aux_files,
    is_sharded_checkpoint,
    registry_weight_conversions,
    resolve_checkpoint_weights,
    sweep_after_full_save,
)
from src.hardware import available_host_ram_bytes
from src.models.loading.model_preparation import sanitize_generation_config
from src.models.moe_balancing import apply_router_balancing_sidecar, is_balancing_state_key
from src.models.patches.gpt_oss_sinks import SinksPolicy, apply_sinks_policy

logger = logging.getLogger(__name__)

# Tokenizer files transformers 5 reads but no longer writes. Each overrides its fresh neighbour on
# load (pad/eos ids, the added-token set, the processor's chat template), so a copy carried over
# from the source must not outlive a fresh save.
_LEGACY_TOKENIZER_SIDECARS = ("special_tokens_map.json", "added_tokens.json", "chat_template.json")

# Header dtypes of full-precision floating tensors: the ones a conversion casts or quantizes.
SAFETENSORS_FLOAT_DTYPES = frozenset({"F64", "F32", "F16", "BF16"})
# Header dtype of an fp8 e4m3 tensor, the storage the fp8 -> bf16 converters dequantize.
FP8_HEADER_DTYPE = "F8_E4M3"

# Sibling suffixes a staged publish writes through: the output is built beside the target and
# swapped in, so an interrupted write never replaces a good directory with a partial one. A failed
# swap prints one of these paths.
STAGING_SUFFIX = ".halo-staging"
DISPLACED_SUFFIX = ".halo-displaced"

# Training-state records an exported artifact carries beside its weights: how the run's routing was
# balanced, and what training-time model state a later merge has to re-apply.
TRAINING_STATE_FILES = (ROUTER_BALANCING_BIASES_FILE, TRAINING_PROVENANCE_FILE)

# safetensors header dtype spellings -> storage bytes per element, for sizing tensors from headers
# alone. These sizes feed warn-only preflights, so an unknown future spelling falls back to
# bf16-sized rather than aborting a valid conversion over an estimate.
_SAFETENSORS_DTYPE_BYTES = {
    "F64": 8,
    "I64": 8,
    "U64": 8,
    "F32": 4,
    "I32": 4,
    "U32": 4,
    "F16": 2,
    "BF16": 2,
    "I16": 2,
    "U16": 2,
    "F8_E4M3": 1,
    "F8_E5M2": 1,
    "I8": 1,
    "U8": 1,
    "BOOL": 1,
}


def save_full_checkpoint(model, output_dir: str, processing_class=None, source_dir=None, **save_kwargs) -> None:
    """``save_pretrained`` plus the non-weight files a conversion tool's output directory needs.

    Order matters: ``source_dir``'s non-weight files are carried over before the save so the live
    config and tokenizer overwrite the copies; stale weight files are swept only after it, under
    :func:`~src.checkpoint.format.sweep_after_full_save`'s failure-safe rule; and the legacy tokenizer
    sidecars the copy carried over are dropped unless the fresh processing class rewrote them, since
    each overrides its fresh neighbour on load.

    A model whose load dequantized an fp8 source is reverted through its registry conversion mapping
    first, because the recorded conversions carry quantizer rewrites ``save_pretrained`` cannot
    invert. ``source_dir`` doubles as the config schema source when the caller passes one.

    Full-model saves only: an unmerged ``PeftModel`` save writes adapter files alone.
    """
    if is_peft_model(model):
        raise ValueError(
            "save_full_checkpoint writes full model directories; an unmerged PEFT adapter save must "
            "call model.save_pretrained directly (it writes adapter files only — nothing to sweep)."
        )
    if _load_dequantized_fp8(model):
        _restore_pristine_weight_conversions(model)
    os.makedirs(output_dir, exist_ok=True)
    if source_dir is not None and os.path.isdir(source_dir):
        copy_checkpoint_aux_files(source_dir, output_dir)
    sanitize_generation_config(model)
    model.save_pretrained(output_dir, **save_kwargs)
    if not model.can_generate():
        # save_pretrained writes no generation config for a non-generating head, so one copied from
        # the source would survive and advertise sampling defaults this artifact has no forward for.
        with contextlib.suppress(FileNotFoundError):
            os.remove(os.path.join(output_dir, GENERATION_CONFIG_NAME))
    finalize_exported_config(model.config, output_dir, source=source_dir or checkpoint_source_ref(model))
    if processing_class is not None:
        carried = _file_versions(output_dir, _LEGACY_TOKENIZER_SIDECARS)
        processing_class.save_pretrained(output_dir)
        for name, version in carried.items():
            if _file_versions(output_dir, (name,)).get(name) == version:
                os.remove(os.path.join(output_dir, name))
    sweep_after_full_save(output_dir)


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


def _load_dequantized_fp8(model) -> bool:
    """Whether ``model``'s load consumed an ``Fp8Dequantize`` conversion.

    A dequantizing load of an fp8 checkpoint leaves that op on ``_weight_conversions``, where
    ``save_pretrained``'s default revert would run its inverse.
    """
    return any(
        isinstance(op, Fp8Dequantize)
        for conversion in (getattr(model, "_weight_conversions", None) or [])
        for op in getattr(conversion, "operations", ())
    )


def _restore_pristine_weight_conversions(model) -> None:
    """Replace the conversions a dequantizing fp8 load recorded with the model's registry mapping.

    The quantizer rewrites the model's converters for the load and adds a dequantizer whose reverse
    is a pass-through with no target split, so ``save_pretrained``'s revert through the recorded list
    fails — while its alternative (``save_original_format=False``) writes an internal layout neither
    the EP lazy loader nor a serving engine reads. The registry mapping is the pristine rename/merge
    chain, and reverting through it writes the hub layout minus the scales. A ``PrefixChange``
    survives only where the load applied one; ``revert_weight_conversion``'s rebuild drops it.
    """
    recorded = getattr(model, "_weight_conversions", None) or []
    prefix_applied = any(isinstance(conversion, PrefixChange) for conversion in recorded)
    model._weight_conversions = registry_weight_conversions(model, keep_prefix_change=prefix_applied) or None


def resolve_checkpoint_source(model_id: str, revision: str | None = None) -> str:
    """A local checkpoint directory as-is; a Hub repo id resolved through the download cache.

    Used by every ``--model_id`` that may name a Hub repo, so the input gates run on a local
    directory whichever spelling was given, and ``HF_HOME`` decides where the download lands.
    """
    if os.path.isdir(model_id):
        return model_id
    logger.info(f"Resolving {model_id} through the HF cache (HF_HOME)...")
    return snapshot_download(model_id, revision=revision)


def clear_staging_path(target: str | os.PathLike, suffix: str = STAGING_SUFFIX) -> str:
    """``target`` + ``suffix``, cleared of whatever an earlier interrupted run left there.

    Staged publish for a tool that may be rewriting the only copy of its input: build the new
    directory here, verify it, swap it onto the target. Anything found here at the start of a run
    belongs to a run that failed before its swap and is cleared; anything left after a failed swap is
    the only complete copy, so the caller must name that path in its error and leave it in place.

    ``suffix`` picks the role: :data:`STAGING_SUFFIX` for the new directory,
    :data:`DISPLACED_SUFFIX` for a previous target rotated aside until the swap completes.
    """
    path = f"{os.fspath(target)}{suffix}"
    shutil.rmtree(path, ignore_errors=True)
    return path


def reject_in_place_conversion(input_dir: str, output_dir: str) -> None:
    """Refuse a conversion whose output directory is its own input.

    Every writer behind these tools (:func:`save_full_checkpoint` through
    :func:`~src.checkpoint.format.sweep_after_full_save`, and
    :meth:`~src.checkpoint.shard_writer.StageShardWriter.close_as_hf_checkpoint`) ends in
    :func:`~src.checkpoint.format.remove_stale_checkpoint_files`, which deletes the
    ``model*.safetensors`` the write does not own. An in-place run therefore destroys the source
    shards, and every post-write step that still reads them (``merge_ep_shards
    --delete_input_shards``, the aux copy) then reads the output.
    """
    if os.path.realpath(input_dir) == os.path.realpath(output_dir):
        raise ValueError(
            f"input and output directory are the same path ({output_dir}). These conversions are not "
            f"in-place: writing the result deletes the source shards it does not overwrite, so the "
            f"original checkpoint would be destroyed. Write to a new directory."
        )


def reject_sibling_adapter(input_dir: str) -> None:
    """Refuse a shard merge whose input holds a PEFT adapter beside the shards.

    The aux copy carries ``adapter_config.json`` but skips ``adapter_model.safetensors``, so the
    merged artifact would declare an adapter it does not hold. The sharded EP save refuses every
    PeftModel outright, so this pairing cannot come out of a supported run, and the merge cannot tell
    whether such shards already carry the delta; guessing either way loses weights.
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


def finalize_merged_checkpoint(
    input_dir: str,
    output_dir: str,
    shard_files: list[str],
    *,
    kind: str,
    verbose: bool,
    delete_input_shards: bool,
) -> None:
    """Carry a merge source's non-weight files across, then optionally drop its input shards.

    Called once the merged weights and their index are on disk, and removes the inputs last, so a
    merge that failed earlier leaves the only copy of the weights where it found them.

    ``generation_config.json`` is copied verbatim because the input is always a directory this
    toolkit's sharded writer produced (the format gate admits nothing else), which emits that file
    only for a generating model off an already-sanitized config; re-deriving those two rules here
    would need the model this merge never loads.
    """
    copy_checkpoint_aux_files(input_dir, output_dir, verbose=verbose)

    if delete_input_shards:
        for shard_file in shard_files:
            os.remove(os.path.join(input_dir, shard_file))
        if verbose:
            print(f"Deleted {len(shard_files)} input shard files (--delete_input_shards)")  # noqa: T201 — CLI-facing

    print(f"\n✓ Merged {kind} checkpoint saved to: {output_dir}")  # noqa: T201 — CLI-facing


def reject_sharded_checkpoint(checkpoint_dir: str) -> None:
    """Refuse a checkpoint directory that does not hold the whole model.

    Three layouts read as whole and are not: a per-rank EP-sharded save (the ordinary index filename
    over partial per-rank slices), the same save killed before its index landed, and an index naming
    shards that are absent (a PP save on non-shared storage, or a half-copied checkpoint). Nothing
    downstream distinguishes them, so every tool that opens a checkpoint directory calls this first.
    """
    if is_sharded_checkpoint(checkpoint_dir):
        raise ValueError(
            f"{checkpoint_dir} is a per-rank sharded checkpoint (EP shards under a reused index). "
            f"Merge it first — scripts/after_training/merge_ep_shards.py — or re-save gathered "
            f"(save_sharded_ep=False)."
        )
    _reject_indexless_ep_shards(checkpoint_dir)
    _reject_absent_index_shards(checkpoint_dir)


def _reject_absent_index_shards(checkpoint_dir: str) -> None:
    """Refuse an index that names shard files the directory does not hold.

    The artifact that lands this way by design is a PP save on a non-shared output filesystem: each
    node holds its own stage's parts under a copy of the world-wide index, so no node's directory is
    loadable until they are gathered. A half-copied checkpoint reads identically, and both would
    otherwise surface as random-initialized keys or a failure inside ``safe_open``.
    """
    if not os.path.isfile(os.path.join(checkpoint_dir, SAFETENSORS_INDEX_FILE)):
        return
    layout = resolve_checkpoint_weights(checkpoint_dir)
    absent = [name for name in layout.shard_files if not os.path.isfile(layout.path(name))]
    if absent:
        raise ValueError(
            f"{checkpoint_dir}/{SAFETENSORS_INDEX_FILE} names {len(absent)} shard file(s) the directory "
            f"does not hold ({', '.join(absent[:4])}{', …' if len(absent) > 4 else ''}). A "
            f"pipeline-parallel save on a non-shared output filesystem leaves each node holding its "
            f"own stage's parts under a copy of the whole-model index — gather every node's "
            f"model-*.safetensors into one directory first. Otherwise the checkpoint was copied "
            f"incompletely; restore the missing shards."
        )


def _reject_indexless_ep_shards(checkpoint_dir: str) -> None:
    """Refuse per-rank EP shards a save wrote before their index landed (header-only peek).

    The sharded EP save writes the shards first and the index last, so a run killed in that window
    leaves partial ``.shard_N`` expert tensors that :func:`is_sharded_checkpoint` reads as fine
    (no index) and :func:`checkpoint_shard_files`' glob fallback would hand over as whole weights.

    Every rank's shard carries its own ``.shard_N`` keys, so the first file settles it, and the peek
    reads the header only. An unparseable header is left to the caller's own read, which reports the
    truncation with its real cause instead of this refusal.
    """
    if os.path.isfile(os.path.join(checkpoint_dir, SAFETENSORS_INDEX_FILE)):
        return
    shards = sorted(glob.glob(os.path.join(checkpoint_dir, SAFETENSORS_FAMILY_GLOB)))
    if not shards:
        return
    try:
        with safe_open(shards[0], framework="pt") as reader:
            partial = next((key for key in reader.keys() if EP_SHARD_KEY_RE.match(key)), None)  # noqa: SIM118
    except Exception:
        return
    if partial is not None:
        raise ValueError(
            f"{checkpoint_dir} holds per-rank EP shards ({os.path.basename(shards[0])} carries the "
            f"partial expert tensor {partial!r}) but no {SAFETENSORS_INDEX_FILE} — the sharded save "
            f"was interrupted between writing the shards and writing their index. Reading it here "
            f"would take each rank's slice for the whole expert bank. Re-run the save, or restore "
            f"the index and merge with scripts/after_training/merge_ep_shards.py."
        )


def _apply_training_provenance(model, adapter_dir: str) -> list[str]:
    """Re-apply training-time model state the adapter's ``training_provenance.json`` records.

    A merge rebuilds the base from the hub, whose GptOss attention sinks are always live, while a
    ``reset_sinks`` run trained its adapter under neutralized ones, so the merged model must be
    neutralized to match or it serves attention the adapter never trained under. Returns the actions
    applied, for the calling tool to report.
    """
    provenance_path = os.path.join(adapter_dir, TRAINING_PROVENANCE_FILE)
    if not os.path.isfile(provenance_path):
        return []
    with open(provenance_path) as fh:
        provenance = json.load(fh)
    actions = []
    if provenance.get(PROVENANCE_GPT_OSS_SINKS) == SinksPolicy.NEUTRALIZED:
        apply_sinks_policy(model, model.config, policy=SinksPolicy.NEUTRALIZED)
        actions.append(
            "Adapter trained with neutralized GptOss attention sinks — neutralized the merged "
            "model's sinks to match (dtype.min, contributes ~0, exactly as trained)."
        )
    return actions


def apply_training_sidecars(model, source_dir: str) -> list[str]:
    """Re-apply to ``model`` what ``source_dir``'s sidecars record about how it was trained.

    Shared by every tool that assembles an exportable model out of a training run's output: the
    training provenance, then the router-balancing state at the precision it was trained in (a bf16
    round trip quantizes away the ~1e-3 sign steps the balancing update writes), filled from
    ``router_balancing_biases.pt`` where the weights cannot carry it at all.

    Returns the actions taken, for the calling tool to report.
    """
    actions = _apply_training_provenance(model, source_dir)

    restored = _balancing_tensors_at_trained_precision(model, source_dir)
    if restored:
        model.load_state_dict(restored, strict=False, assign=True)

    # A source shipping its own weights already holds the trained bias in the native slot (restored
    # above at its trained dtype), and its sidecar is the resume copy, written against the trainer's
    # module tree, which for an EP run is not the hub tree. It travels with the artifact instead.
    sidecar_path = os.path.join(source_dir, ROUTER_BALANCING_BIASES_FILE)
    if _has_checkpoint_weights(source_dir) or not os.path.isfile(sidecar_path):
        return actions
    sidecar = torch.load(sidecar_path, map_location="cpu", weights_only=True)
    applied, skipped = apply_router_balancing_sidecar(model, sidecar)
    if applied:
        actions.append(f"Applied trained router balancing biases into {len(applied)} native slots.")
    if skipped:
        actions.append(
            f"WARNING: {len(skipped)} routers trained a TRANSIENT balancing bias this architecture "
            f"has no checkpoint slot for — the exported model serves without it (near-tied top-k "
            f"picks flip vs training): {skipped[:4]}{'...' if len(skipped) > 4 else ''}"
        )
    return actions


def copy_training_sidecars(source_dir: str, output_dir: str) -> None:
    """Carry :data:`TRAINING_STATE_FILES` from ``source_dir`` into ``output_dir``.

    For writes that do not go through :func:`save_full_checkpoint`: an unmerged PEFT save, or a full
    save whose ``source_dir`` is the base model rather than the adapter directory holding the
    sidecars. That path already copies them, so a tool that passed it their directory must not.
    """
    for name in TRAINING_STATE_FILES:
        source = os.path.join(source_dir, name)
        if os.path.isfile(source):
            shutil.copy2(source, os.path.join(output_dir, name))


def _balancing_tensors_at_trained_precision(model, source_dir: str) -> dict[str, torch.Tensor]:
    """The model's router-balancing tensors as they should be stored, keyed by state-dict name.

    Read back from the source's own shards where it has them (the checkpoint holds the trained dtype
    and the load may have downcast it); where it has none, as in a PEFT adapter directory, the slot
    is given fp32 storage so the fp32 sidecar lands unrounded. Entries already at the right dtype are
    omitted, so an untouched model returns ``{}``.

    The keys come from the registry-derived :func:`is_balancing_state_key`. Stored tensors match on
    the module path rather than the full name, since a family whose export renames its slot stores it
    under the hub spelling while the live tree uses its own.
    """
    state = model.state_dict()
    balancing_keys = sorted(key for key in state if is_balancing_state_key(key))
    if not balancing_keys:
        return {}
    stored = _read_checkpoint_tensors(source_dir, is_balancing_state_key)
    stored_by_module = {key.rsplit(".", 1)[0]: tensor for key, tensor in stored.items()}
    restored: dict[str, torch.Tensor] = {}
    for key in balancing_keys:
        current = state[key]
        tensor = stored_by_module.get(key.rsplit(".", 1)[0], current.float())
        if tensor.dtype != current.dtype:
            restored[key] = tensor.to(device=current.device)
    return restored


def _has_checkpoint_weights(checkpoint_dir: str) -> bool:
    """Whether a directory holds model weights of its own.

    ``False`` for a PEFT adapter directory (adapter files only) and for a hub id that was never
    resolved to a local path: the two cases where an assembled model's trained state can only have
    come from the sidecars beside it.
    """
    try:
        checkpoint_shard_files(checkpoint_dir)
    except (FileNotFoundError, NotADirectoryError):
        return False
    return True


def _read_checkpoint_tensors(checkpoint_dir: str, wanted: Callable[[str], bool]) -> dict[str, torch.Tensor]:
    """The tensors a checkpoint directory stores under a key ``wanted`` accepts.

    ``{}`` when the directory holds no model weights at all (a PEFT adapter directory, a hub id that
    was never resolved to a local path), which the callers here treat as nothing to read back rather
    than as an error.
    """
    try:
        return dict(iter_checkpoint_tensors(checkpoint_dir, predicate=wanted))
    except (FileNotFoundError, NotADirectoryError):
        return {}


def detect_model_type(checkpoint_dir: str) -> str:
    """``config.model_type`` from a checkpoint's ``config.json`` (``""`` when absent or unreadable).

    Used by every family gate here, so a tool resolves the family the way the sharded merge does
    instead of sniffing key spellings. A composite VLM config with no top-level ``model_type`` falls
    back to its ``text_config``, where the language family lives.
    """
    config_path = os.path.join(checkpoint_dir, CONFIG_NAME)
    if not os.path.isfile(config_path):
        return ""
    with open(config_path) as f:
        config = json.load(f)
    return config.get("model_type") or config.get("text_config", {}).get("model_type", "") or ""


def checkpoint_shard_files(checkpoint_dir: str) -> list[str]:
    """The model safetensors shards of a checkpoint, as paths under ``checkpoint_dir``.

    The shared layout cascade runs first: its index leg names exactly the model shards, so a
    co-located ``adapter_model.safetensors`` is never picked up. The ``model*.safetensors`` glob
    fallback is needed only by these standalone tools, since a training resume refuses an indexless
    sharded checkpoint. Raises when neither matches, rather than returning an empty list a caller
    reads as "no tensors"; a legacy ``pytorch_model.bin`` reads as absent.

    The glob leg is safe only because :func:`reject_sharded_checkpoint` runs first, covering the
    indexless window in which it would hand a tool one rank's partial expert slices.
    """
    reject_sharded_checkpoint(checkpoint_dir)

    layout = resolve_checkpoint_weights(checkpoint_dir)
    if layout.shard_files:
        return [layout.path(name) for name in layout.shard_files]

    shards = sorted(glob.glob(os.path.join(checkpoint_dir, SAFETENSORS_FAMILY_GLOB)))
    if shards:
        return shards

    raise FileNotFoundError(f"No '{SAFETENSORS_WEIGHTS_FILE}' or '{SAFETENSORS_INDEX_FILE}' found in {checkpoint_dir}")


def iter_checkpoint_shards(checkpoint_dir: str) -> Iterator[tuple[str, Any]]:
    """Yield ``(shard_path, open reader)`` per shard of a checkpoint, in :func:`checkpoint_shard_files` order.

    The reader is live for its own iteration step only: a caller that needs a shard's keys together
    (a weight/scale co-location check) reads them inside the step; a reader kept past it is closed.
    """
    for shard in checkpoint_shard_files(checkpoint_dir):
        with safe_open(shard, framework="pt") as reader:
            yield shard, reader


def iter_checkpoint_shard_entries(checkpoint_dir: str) -> Iterator[tuple[str, Any, str]]:
    """Yield ``(shard_path, open reader, key)`` for every tensor a checkpoint stores.

    Shared by every tool that reads a checkpoint's tensors, headers or key set: each shard opens
    exactly once and the live reader is handed out, so the caller takes the tensor (``get_tensor``),
    the header (``get_slice``) or neither, keeping a header-only or key-only scan free of tensor
    reads. Lazy: a caller that stops early opens none of the rest.
    """
    for shard, reader in iter_checkpoint_shards(checkpoint_dir):
        for key in reader.keys():  # noqa: SIM118 - safe_open has .keys() but is not a mapping
            yield shard, reader, key


def iter_checkpoint_tensors(
    checkpoint_dir: str, predicate: Callable[[str], bool] | None = None
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(key, tensor)`` for the checkpoint's tensors whose key ``predicate`` accepts.

    ``predicate`` is applied to the key before the tensor is read, so a selective walk never
    materializes the tensors it is about to discard.
    """
    for _shard, reader, key in iter_checkpoint_shard_entries(checkpoint_dir):
        if predicate is None or predicate(key):
            yield key, reader.get_tensor(key)


def header_nbytes(header) -> int:
    """Storage size of a tensor from its ``safe_open`` slice header alone (no data read)."""
    return math.prod(header.get_shape()) * _SAFETENSORS_DTYPE_BYTES.get(header.get_dtype(), 2)


def stored_tensor_nbytes(reader, key: str) -> int:
    """Storage size of one tensor in an open ``safe_open`` reader, from the header alone (no data read)."""
    return header_nbytes(reader.get_slice(key))


def preflight_resource_warning(tool: str, output_dir: str, *, disk_bytes: int | None, ram_bytes: int | None) -> None:
    """Warn when ``tool``'s estimated peak host RAM / output disk exceeds what the host has free.

    Warn-and-continue: both figures are estimates (file sizes stand in for resident bytes, the output
    is assumed input-sized), so failing hard would refuse conversions that fit; the warning exists so
    a large merge does not hit ENOSPC hours in. ``None`` skips a side, as does an unreadable
    ``/proc/meminfo``.
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
    """Size preflight for a tool that loads a whole checkpoint through ``from_pretrained``.

    The estimate is the source's safetensors bytes: the model lands in host RAM unless ``device_map``
    routes it onto devices, and the output is about the source's size on disk
    (``writes_full_model=False`` for adapter-only outputs). Silent for a hub id, a weightless
    directory, or a per-rank-sharded one; whether that input is usable is the tool's own gates'
    decision rather than a warn-only estimate's.
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
        tool,
        output_dir,
        disk_bytes=total if writes_full_model else None,
        ram_bytes=total if lands_on_host else None,
    )
