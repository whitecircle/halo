"""Checkpoint file-format I/O: the on-disk spellings every reader and writer shares, the save-dtype
casts, the layout cascade the readers resolve through, and the state-dict write helpers built on them.

Format layer only, with no ``torch.distributed``, so the standalone ``scripts/after_training/`` tools
and the parallel save paths read and write one artifact layout; rank coordination around these calls
belongs to the caller. What an exported ``config.json`` must contain is
:mod:`src.checkpoint.config_export`, which this module calls but never the other way round.
"""

from __future__ import annotations

import glob
import json
import logging
import os
import re
import shutil
from collections.abc import Mapping
from contextlib import ExitStack
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import torch
from huggingface_hub import split_torch_state_dict_into_shards
from safetensors import safe_open
from safetensors.torch import load_file as _safetensors_load_file
from safetensors.torch import save_file as _safetensors_save_file
from transformers.conversion_mapping import get_model_conversion_mapping
from transformers.core_model_loading import PrefixChange, revert_weight_conversion
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from src.checkpoint.config_export import save_model_config
from src.models.moe_balancing import balancing_param_keys
from src.models.structure import fp32_pinned_param_names, norm_param_keys, strip_peft_adapter_segment

logger = logging.getLogger(__name__)

# Distributed checkpoints (EP, TP) are saved bf16 even when training uses fp32 master weights.
_SAVE_DTYPE = torch.bfloat16

# Per-file cap for gathered safetensors saves, shared by every save path and the arg default.
DEFAULT_MAX_SHARD_SIZE = "5GB"

# One spelling per artifact for every reader/writer; a typo produces no error, only a weightless
# checkpoint or an LR schedule re-warming from step 0.
SAFETENSORS_INDEX_FILE = "model.safetensors.index.json"
# Format marker every safetensors writer stamps; the readers must agree, so it is defined once.
SAFETENSORS_METADATA = {"format": "pt"}
# The index ``metadata.format`` stamp of a per-rank EP save, and one rank's slice of an EP-sharded
# expert tensor. One spelling each: a reader that misses them takes a partial tensor for a whole one.
EP_SHARDED_FORMAT = "ep_sharded"
EP_SHARD_KEY_INFIX = ".shard_"
EP_SHARD_KEY_RE = re.compile(rf"(.+)\{EP_SHARD_KEY_INFIX}(\d+)$")
# HF's shard filename pattern and the spellings derived from it: the single-file name a one-shard
# save degenerates to, the stale sweep's glob, and the per-rank EP save's own filename and reader.
# A writer and a sweep that disagree orphan shards.
SAFETENSORS_SHARD_PATTERN = "model{suffix}.safetensors"
SAFETENSORS_WEIGHTS_FILE = SAFETENSORS_SHARD_PATTERN.format(suffix="")
SAFETENSORS_FAMILY_GLOB = SAFETENSORS_SHARD_PATTERN.format(suffix="*")
# Both sides escape the same literal, so the reader can only match what the pattern spells.
_EP_SHARD_FILE_RE = re.compile(
    f"^{re.escape(SAFETENSORS_SHARD_PATTERN).replace(re.escape('{suffix}'), r'-\d+-of-\d+')}$"
)
# In-flight part name for a writer that finalizes by renaming: outside the final ``model-{i}-of-{n}``
# pattern so it cannot collide with one, inside the sweep's glob so leftovers cannot outlive it.
HF_STREAM_PART_PREFIX = "model-streaming"
LEGACY_WEIGHTS_FILE = "pytorch_model.bin"
# Every filename that makes a directory a whole-model checkpoint, in the layout cascade's order.
# Read by :func:`has_whole_model_weight_file`, whose safetensors-only mode is the lazy gate's.
WHOLE_MODEL_WEIGHT_FILES = (SAFETENSORS_INDEX_FILE, SAFETENSORS_WEIGHTS_FILE, LEGACY_WEIGHTS_FILE)

SCHEDULER_STATE_FILE = "scheduler.pt"
# HF Trainer's replicated optimizer state, which the sharded modes deliberately replace.
OPTIMIZER_STATE_FILES = ("optimizer.pt", "optimizer.bin")
ROUTER_BALANCING_BIASES_FILE = "router_balancing_biases.pt"

# PEFT adapter artifact filenames. ADAPTER_WEIGHT_NAMES is in load-preference order; PeftAdapterSaver
# falls back to .bin, so detection must accept both.
ADAPTER_SAFETENSORS_FILE = "adapter_model.safetensors"
ADAPTER_BIN_FILE = "adapter_model.bin"
ADAPTER_CONFIG_FILE = "adapter_config.json"
ADAPTER_WEIGHT_NAMES = (ADAPTER_SAFETENSORS_FILE, ADAPTER_BIN_FILE)
# Training-time model state a merge tool cannot recover from the adapter artifacts alone (the GptOss
# sink policy). A sidecar rather than adapter_config.json, so stock PEFT loads the adapter unchanged.
TRAINING_PROVENANCE_FILE = "training_provenance.json"
PROVENANCE_GPT_OSS_SINKS = "gpt_oss_attention_sinks"

# Never carried over: a stray pytorch_model.bin or optimizer*.pt would shadow the fresh safetensors.
_WEIGHT_FILE_SUFFIXES = (".safetensors", ".bin", ".pt")
# Foreign-framework exports, never weights this toolkit reads. The aux copy and the hub-download
# ignore list share this tuple so they cannot disagree.
_FOREIGN_EXPORT_SUFFIXES = (".pth", ".gguf", ".h5", ".msgpack", ".onnx", ".onnx_data", ".tflite", ".ot", ".mlmodel")
# Exempt from that skip: dropping these restarts the LR schedule or zeroes the router biases.
_RESUME_SIDECAR_FILES = (SCHEDULER_STATE_FILE, ROUTER_BALANCING_BIASES_FILE)
# Same exemption by prefix: losing ``rng_state_<rank>.pth`` re-draws every shuffle and dropout mask.
_RESUME_SIDECAR_PREFIXES = ("rng_state",)
# Vendor dumps of the same weights in a raw format: hundreds of GB the aux copy must not duplicate.
# Exact names rather than prefixes, since ``original_adapter_config/`` is aux data the copy must keep.
_WEIGHT_DUMP_DIRS = ("original", "consolidated")
# Hub-download counterpart of copy_checkpoint_aux_files' skip: fetch an aux source minus its weights.
WEIGHT_FILE_IGNORE_PATTERNS = tuple(
    f"*{suffix}" for suffix in (*_WEIGHT_FILE_SUFFIXES, *_FOREIGN_EXPORT_SUFFIXES)
) + tuple(f"{name}/*" for name in _WEIGHT_DUMP_DIRS)
# The tied pair, in the order reconcile_tie_word_embeddings reads them.
_TIE_KEY_SUFFIXES = ("lm_head.weight", "embed_tokens.weight")


def ep_shard_filename(rank: int, world_size: int) -> str:
    """Filename of one rank's slice of a per-rank EP save.

    HF's shard pattern with a 0-based index: this save writes one file per rank, so rank 0 must still
    carry a suffix. :func:`shard_file_name`'s 1-based numbering degenerates to the unsuffixed
    ``model.safetensors`` at a single part, which every reader takes for a whole model.
    """
    return SAFETENSORS_SHARD_PATTERN.format(suffix=f"-{rank:05d}-of-{world_size:05d}")


def is_ep_shard(name: str) -> bool:
    """Whether ``name`` is a per-rank EP save slice.

    Narrower than a ``.safetensors`` glob: a sibling ``adapter_model.safetensors`` or a stale
    ``model.safetensors`` must not be swept into a merge.
    """
    return _EP_SHARD_FILE_RE.match(name) is not None


def shard_file_name(part: int, total_parts: int) -> str:
    """HF's own name for part ``part`` of ``total_parts``; the unsharded file when there is only one.

    ``split_torch_state_dict_into_shards`` is handed :data:`SAFETENSORS_SHARD_PATTERN` and fills the
    same suffix, so a streamed checkpoint is indistinguishable from a ``save_pretrained`` one.
    """
    suffix = "" if total_parts == 1 else f"-{part:05d}-of-{total_parts:05d}"
    return SAFETENSORS_SHARD_PATTERN.format(suffix=suffix)


def cast_to_save_dtype(t: torch.Tensor) -> torch.Tensor:
    """Cast a tensor to the distributed save dtype (bf16) if it is floating point.

    A blanket cast; save paths that hold the live model use :func:`save_dtype_caster` instead, so
    normalization params keep their trained dtype.
    """
    if torch.is_tensor(t) and t.is_floating_point() and t.dtype != _SAVE_DTYPE:
        return t.to(dtype=_SAVE_DTYPE)
    return t


def cast_state_dict_to_save_dtype(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Cast all floating-point tensors in a state dict to the distributed save dtype."""
    return {k: cast_to_save_dtype(v) for k, v in state.items()}


def save_dtype_caster(model: torch.nn.Module):
    """``cast(name, tensor)`` for checkpoint saves that hold the live model.

    Floating tensors go to the save dtype except three tree-derived keep-sets that hold their trained
    dtype: the normalization params, the live router-balancing tensors (hub-respelled) and the
    family's fp32 pins. That way a direct EP/TP save of an fp32-master run matches its merged-shards
    save, and the export quantizes neither the balancing state nor a family's declared fp32 modules.

    Keys also match with their PEFT adapter segment stripped: the EP gather feeds this pre-remap
    keys, where a ``modules_to_save`` router spells its bias ``router.modules_to_save.default.bias``.
    """
    keep = norm_param_keys(model) | balancing_param_keys(model) | fp32_pinned_param_names(model)

    def cast(name: str, t: torch.Tensor) -> torch.Tensor:
        return t if name in keep or strip_peft_adapter_segment(name) in keep else cast_to_save_dtype(t)

    return cast


def _has_child_at_prefix(model: torch.nn.Module, dotted_prefix: str) -> bool:
    """Whether ``model`` (under its ``base_model_prefix``, when present) has a submodule at
    ``dotted_prefix``, which decides whether a ``PrefixChange`` revert re-adds a real prefix."""
    node = getattr(model, getattr(model, "base_model_prefix", ""), model)
    for part in dotted_prefix.split("."):
        node = getattr(node, part, None)
        if node is None:
            return False
    return True


def registry_weight_conversions(model: torch.nn.Module, *, keep_prefix_change: bool) -> list:
    """The family's declared conversion mapping: what a load that recorded nothing reverts through.

    Empty for a module carrying no ``config``: the registry is keyed by the config's family.
    """
    if getattr(model, "config", None) is None:
        return []
    pristine = get_model_conversion_mapping(model, add_legacy=False)
    return [c for c in pristine if keep_prefix_change or not isinstance(c, PrefixChange)]


def revert_conversions_for(model: torch.nn.Module) -> list:
    """The conversions a save-side revert, or a hub-namespace weight sync, inverts.

    What the load recorded, minus a ``PrefixChange`` whose stripped prefix is not a child of the
    saved tree: a text-only load of a multimodal checkpoint consumes
    ``PrefixChange(prefix_to_remove="language_model")``, and reverting that at save would re-emit
    wrapper-prefixed keys under a text-only config, which engine loaders keyed on the architectures
    cannot read. A load that recorded nothing falls back to the family's registry mapping minus its
    ``PrefixChange``, as transformers' own revert does.
    """
    load_conversions = getattr(model, "_weight_conversions", None)
    if not load_conversions:
        return registry_weight_conversions(model, keep_prefix_change=False)
    return [
        c
        for c in load_conversions
        if not (
            isinstance(c, PrefixChange) and c.prefix_to_remove and not _has_child_at_prefix(model, c.prefix_to_remove)
        )
    ]


def revert_load_conversions(model: torch.nn.Module, state_dict: dict) -> dict:
    """Map a module-layout state dict back to the hub checkpoint layout before writing.

    transformers loads several MoE families into a module-fused expert layout and reverts it inside
    ``save_pretrained``, which the gathered/TP writers bypass, so without this a wrapper-less MoE
    save emits fused keys that per-expert engine loaders reject (vLLM 0.26.0: GLM-4/LFM-2) or drop
    without error (Laguna). Identity for dense models; EP-gathered dicts never come here.

    Reverts :func:`revert_conversions_for`'s list, restoring the model's own value afterwards. A
    failure warns rather than raising: the pre-revert dict is a loadable checkpoint that only needs
    ``unfuse_moe_experts.py`` before a per-expert engine.
    """
    load_conversions = getattr(model, "_weight_conversions", None)
    model._weight_conversions = revert_conversions_for(model) or None
    try:
        return revert_weight_conversion(model, state_dict)
    except Exception as e:
        logger.warning(
            f"revert_weight_conversion failed ({e}); writing the module-layout state "
            f"dict instead — run scripts/after_training/unfuse_moe_experts.py before "
            f"serving this checkpoint on a per-expert engine."
        )
        return state_dict
    finally:
        model._weight_conversions = load_conversions


def normalize_gathered_state_dict(model: torch.nn.Module, state_dict: dict) -> dict:
    """Bring a gathered state dict to its on-disk form: save-dtype cast, then hub expert layout.

    The order is load-bearing: the caster's keep-set is derived from the module tree, so it uses the
    live key spelling and must run before the revert respells the expert keys. Shared by the FSDP2/CP
    and TP gathered writers, whose artifacts must be byte-identical for the same model.
    """
    cast = save_dtype_caster(model)
    return revert_load_conversions(model, {key: cast(key, tensor) for key, tensor in state_dict.items()})


def sweep_after_full_save(output_dir: str) -> None:
    """Remove weight files a *completed* ``save_pretrained``-style write did not produce.

    Failure-safe by construction: it runs after the save and derives the keep-set from what is then
    on disk (a consistent index keeps its shards, else a single ``model.safetensors`` keeps itself),
    so a save that failed mid-way, torn index included, sweeps nothing and leaves the directory
    intact. Sweeping first would let a failed save destroy the good checkpoint it held.
    """
    index_path = os.path.join(output_dir, SAFETENSORS_INDEX_FILE)
    single_path = os.path.join(output_dir, SAFETENSORS_WEIGHTS_FILE)
    keep: set[str] = set()
    if os.path.isfile(index_path):
        try:
            index = read_checkpoint_index(output_dir)
        except (OSError, json.JSONDecodeError):
            index = {}
        # A per-rank EP/TP index is not a full save's product; accepting its shard list as a
        # keep-set would leave behind a layout these callers must never produce.
        shards = set() if _index_declares_per_rank_shards(index) else set(index.get("weight_map", {}).values())
        # Both layouts present means one is a previous run's leftover and the newer write is this
        # one; without the tie-break a single-file writer (reset_sinks) loses to the stale shards.
        newer_than_single = not os.path.isfile(single_path) or os.path.getmtime(index_path) >= os.path.getmtime(
            single_path
        )
        if shards and newer_than_single and all(os.path.isfile(os.path.join(output_dir, s)) for s in shards):
            keep = shards | {SAFETENSORS_INDEX_FILE}
    if not keep and os.path.isfile(single_path):
        keep = {SAFETENSORS_WEIGHTS_FILE}
    if keep:
        remove_stale_checkpoint_files(output_dir, keep=keep)


def remove_stale_checkpoint_files(output_dir: str, keep: set[str]) -> None:
    """Delete every ``model*.safetensors``/index in ``output_dir`` this save did not write.

    ``from_pretrained`` prefers a single ``model.safetensors`` over the index, so a previous save's
    leftover (a single file where this one sharded, or a higher shard count) would be loaded instead
    of what was just written. A caller that only sometimes owns the directory (``reset_sinks``
    in-place, an unmerged-PEFT save) must gate the call rather than pass ``keep=set()``
    unconditionally.
    """
    for stale in glob.glob(os.path.join(output_dir, SAFETENSORS_FAMILY_GLOB)) + glob.glob(
        os.path.join(output_dir, SAFETENSORS_INDEX_FILE)
    ):
        if os.path.basename(stale) not in keep:
            os.remove(stale)


def write_merged_index(output_dir: str, weight_map: dict[str, str], metadata: Mapping[str, Any]) -> None:
    """Sweep the model files this save did not write, then write the safetensors index over the rest.

    Every sharded save writes its index through here. The sweep runs after the shards and before the
    index: the parts are on disk and ``weight_map`` names exactly the ones this save claims, which is
    the keep-set needed to drop a stale leftover before it can outrank the fresh index.

    ``metadata`` is the index's metadata block verbatim. A non-mapping is refused, since a bare scalar
    would produce ``{"metadata": 12345}``, an index every reader parses and none can use.
    """
    if not isinstance(metadata, Mapping):
        raise TypeError(
            f"write_merged_index metadata must be a mapping written verbatim into the index's "
            f"metadata block (e.g. {{'total_size': N}}), got {type(metadata).__name__}: {metadata!r}"
        )
    remove_stale_checkpoint_files(output_dir, set(weight_map.values()) | {SAFETENSORS_INDEX_FILE})
    index = {"metadata": dict(metadata), "weight_map": weight_map}
    with open(os.path.join(output_dir, SAFETENSORS_INDEX_FILE), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)


def save_sharded_state_dict(
    state_dict: dict[str, torch.Tensor], output_dir: str, max_shard_size: str = DEFAULT_MAX_SHARD_SIZE
) -> None:
    """Write a gathered state dict as HF-standard sharded safetensors.

    Uses HF's own splitter so the output matches ``save_pretrained`` (single ``model.safetensors``,
    or shards + ``model.safetensors.index.json`` loadable one at a time at 100B+).

    Every shard lands as a ``model-streaming-*`` part and is renamed into its final name only once
    all of them are on disk. Writing final names directly is unsafe on a re-save into a populated
    directory: the splitter's names are deterministic, so a same-shard-count re-save that fails after
    shard k leaves new shards 1..k beside old shards k+1..N under the old index, and
    ``from_pretrained`` then loads half of each model. Stale files are swept only at the end.
    """
    state_dict = {k: (v.contiguous() if torch.is_tensor(v) else v) for k, v in state_dict.items()}
    split = split_torch_state_dict_into_shards(
        state_dict,
        filename_pattern=SAFETENSORS_SHARD_PATTERN,
        max_shard_size=max_shard_size,
    )
    parts: dict[str, str] = {}
    for part_index, (filename, keys) in enumerate(split.filename_to_tensors.items(), start=1):
        part = f"{HF_STREAM_PART_PREFIX}-{part_index:05d}.safetensors"
        _safetensors_save_file(
            {k: state_dict[k] for k in keys},
            os.path.join(output_dir, part),
            metadata=SAFETENSORS_METADATA,
        )
        parts[filename] = part
    # Only now, with every part written: renames are metadata operations, so the window in which the
    # directory holds a mix is the loop below rather than the whole I/O-bound write.
    for filename, part in parts.items():
        os.replace(os.path.join(output_dir, part), os.path.join(output_dir, filename))
    if split.is_sharded:
        write_merged_index(output_dir, split.tensor_to_filename, split.metadata)
    else:
        remove_stale_checkpoint_files(output_dir, set(split.filename_to_tensors))


def write_gathered_checkpoint(
    model: torch.nn.Module,
    state_dict: dict,
    output_dir: str,
    max_shard_size: str = DEFAULT_MAX_SHARD_SIZE,
) -> None:
    """Write a gathered state dict as HF-standard (auto-sharded) safetensors. Save rank only.

    For a caller that already holds the whole dict (an injected-LoRA merge); the parallel gathered
    saves stream through :func:`~src.distributed.checkpoint.write.stream_gathered_checkpoint` instead,
    with the same normalization applied chunk by chunk. The ``.bin`` fallback is specific to this
    path: it needs a dict still whole after the safetensors write failed, which a streamed save no
    longer has, and at that scale a single ``torch.save`` is not a writable artifact anyway.

    Only the config write is gated on the model carrying one; the weights take the same normalization
    and layout either way, and a raw ``.bin`` is an artifact no export tool reads.
    """
    if hasattr(model, "config"):
        # FSDP2 sharding can break tied embeddings; keep the config consistent with the tensors.
        reconcile_tie_word_embeddings(model, state_dict)
        save_model_config(model, output_dir)
    # Save dtype and hub expert layout. EP-gathered dicts never arrive here: ``select_checkpoint_saver``
    # routes every ``has_ep_layers`` context to the EP saver, whose gather emits hub-layout expert keys.
    state_dict = normalize_gathered_state_dict(model, state_dict)
    try:
        save_sharded_state_dict(state_dict, output_dir, max_shard_size=max_shard_size)
    except Exception as e:
        logger.warning(f"sharded safetensors save failed: {e}, using pytorch format")
        # The .bin goes on disk first, then the safetensors leftovers are swept: resume prefers an
        # index over the .bin so they must go, but sweeping first would let a second failure destroy
        # the only checkpoint the directory held. The sweep never matches the .bin itself.
        torch.save(state_dict, os.path.join(output_dir, LEGACY_WEIGHTS_FILE))
        try:
            remove_stale_checkpoint_files(output_dir, keep=set())
        except OSError as sweep_error:
            logger.warning(
                f"stale safetensors sweep after the .bin fallback failed: {sweep_error} — "
                f"resume prefers an index over the .bin, so this directory may resume the OLD weights"
            )


def copy_checkpoint_aux_files(
    input_dir: str, output_dir: str, *, include_resume_sidecars: bool = True, verbose: bool = False
) -> None:
    """Copy a checkpoint's non-weight files (config, tokenizer, chat template, remote-code .py, …)
    from ``input_dir`` to ``output_dir`` verbatim.

    Skips every top-level weight file and safetensors index, which the caller writes fresh, but
    preserves the resume sidecars (``scheduler.pt``, ``router_balancing_biases.pt``, ``rng_state_*``)
    a resume-from-merged run restores; ``include_resume_sidecars=False`` drops them, for an artifact
    that describes no single run (an N-way merge).

    Subdirectories are copied whole, weight files included: a SentenceTransformer module directory
    carries weights no caller rewrites, and filtering them out leaves ``modules.json`` pointing at
    modules that no longer exist. Three kinds stay behind: a nested ``checkpoint-N`` (resume state
    rather than the artifact), a vendor weight dump, and anything hidden.

    ``output_dir`` nested inside ``input_dir`` raises: the walk would copy the destination into
    itself until the disk fills.
    """
    input_root = os.path.realpath(input_dir)
    if os.path.commonpath([input_root, os.path.realpath(output_dir)]) == input_root:
        raise ValueError(
            f"copy_checkpoint_aux_files: output_dir {output_dir!r} is inside input_dir "
            f"{input_dir!r} — the subdirectory copy would recurse into the destination. Write the "
            f"artifact to a directory outside the source checkpoint."
        )
    for name in os.listdir(input_dir):
        src = os.path.join(input_dir, name)
        if os.path.isdir(src):
            if name.startswith((".", f"{PREFIX_CHECKPOINT_DIR}-")) or name in _WEIGHT_DUMP_DIRS:
                continue
            shutil.copytree(src, os.path.join(output_dir, name), dirs_exist_ok=True)
            if verbose:
                print(f"Copied: {name}/")  # noqa: T201 - CLI-facing helper; the merge scripts report via print
            continue
        skip_as_weight = (
            name.endswith(_WEIGHT_FILE_SUFFIXES)
            or name.endswith(_FOREIGN_EXPORT_SUFFIXES)
            or name.endswith(".safetensors.index.json")
        )
        keep_as_sidecar = include_resume_sidecars and (
            name in _RESUME_SIDECAR_FILES or name.startswith(_RESUME_SIDECAR_PREFIXES)
        )
        if skip_as_weight and not keep_as_sidecar:
            continue
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, name))
            if verbose:
                print(f"Copied: {name}")  # noqa: T201 - CLI-facing helper; the merge scripts report via print


def read_checkpoint_index(checkpoint_dir: str, *, missing_ok: bool = False) -> dict:
    """Parse a checkpoint's ``model.safetensors.index.json``. One parse for every index reader.

    ``OSError`` / ``json.JSONDecodeError`` propagate by default so each caller applies its own policy
    to a torn index (an empty keep-set, a rank-0 ``False``, a raise naming a non-shared filesystem).
    ``missing_ok`` returns ``{}`` for a directory with no index at all: a checkpoint small enough for
    a single ``model.safetensors`` has none, and the merge scripts must be able to inspect the format
    marker without an unhandled ``ENOENT`` hiding the refusal that names the fix.
    """
    index_path = os.path.join(checkpoint_dir, SAFETENSORS_INDEX_FILE)
    if missing_ok and not os.path.isfile(index_path):
        return {}
    with open(index_path) as f:
        return json.load(f)


def _index_declares_per_rank_shards(index: dict) -> bool:
    """Whether a parsed safetensors index marks a per-rank EP save rather than a gathered one.

    A gathered HF index carries no ``format`` marker, so the gathered/FSDP2 layout reads as False;
    so does a PP save, whose per-stage shards hold complete tensors.
    """
    meta = index.get("metadata", {}) or {}
    return meta.get("format") == EP_SHARDED_FORMAT


def is_sharded_checkpoint(checkpoint_dir: str) -> bool:
    """True when the directory holds a per-rank EP-sharded save (partial tensors under a reused
    index filename), which is loadable only after merging."""
    if not os.path.isfile(os.path.join(checkpoint_dir, SAFETENSORS_INDEX_FILE)):
        return False
    try:
        index = read_checkpoint_index(checkpoint_dir)
    except (OSError, json.JSONDecodeError):
        # Callers decide on rank 0 then broadcast, so raising on a torn index would block peer ranks.
        return False
    return _index_declares_per_rank_shards(index)


@dataclass(frozen=True)
class CheckpointWeights:
    """Where a checkpoint directory's tensors live, as resolved by :func:`resolve_checkpoint_weights`.

    Filenames are relative to ``directory`` (:meth:`path` joins them): ``shard_files`` are the
    safetensors files holding the tensors, ``index`` the parsed index where there is one, and
    ``legacy_bin`` the ``pytorch_model.bin`` a pre-safetensors checkpoint keeps everything in.
    Neither shards nor a legacy bin means the directory holds no whole-model weight file.
    """

    directory: str
    shard_files: tuple[str, ...] = ()
    index: dict | None = None
    legacy_bin: str | None = None

    def path(self, name: str) -> str:
        """Absolute path of one of this layout's filenames."""
        return os.path.join(self.directory, name)

    @cached_property
    def weight_map(self) -> dict[str, str]:
        """``{tensor key: shard filename}`` — which file each of the checkpoint's tensors is in.

        Lazy because the layouts differ in what a key set costs: the index answers from itself,
        while a single-file checkpoint must be opened for its header. ``{}`` for a legacy
        ``pytorch_model.bin``, whose keys only ``torch.load`` can enumerate.
        """
        if self.index is not None:
            return self.index.get("weight_map") or {}
        if not self.shard_files:
            return {}
        single = self.shard_files[0]
        with safe_open(self.path(single), framework="pt") as f:
            return dict.fromkeys(f.keys(), single)


def resolve_checkpoint_weights(checkpoint_dir: str) -> CheckpointWeights:
    """Where a checkpoint directory's tensors live: the layout cascade, stated once.

    Sharded index first (authoritative, and read as the map itself so no shard is opened), then a
    single ``model.safetensors``, then the legacy ``pytorch_model.bin``. Reports what is on disk and
    applies no policy: whether an empty layout, a per-rank sharded index or an index without a
    ``weight_map`` is an error is the caller's decision. A torn index raises through.
    """
    if os.path.isfile(os.path.join(checkpoint_dir, SAFETENSORS_INDEX_FILE)):
        index = read_checkpoint_index(checkpoint_dir)
        shards = tuple(sorted(set((index.get("weight_map") or {}).values())))
        return CheckpointWeights(checkpoint_dir, shards, index=index)
    if os.path.isfile(os.path.join(checkpoint_dir, SAFETENSORS_WEIGHTS_FILE)):
        return CheckpointWeights(checkpoint_dir, (SAFETENSORS_WEIGHTS_FILE,))
    if os.path.isfile(os.path.join(checkpoint_dir, LEGACY_WEIGHTS_FILE)):
        return CheckpointWeights(checkpoint_dir, legacy_bin=LEGACY_WEIGHTS_FILE)
    return CheckpointWeights(checkpoint_dir)


def has_whole_model_weight_file(checkpoint_dir: str, *, safetensors_only: bool = False) -> bool:
    """Whether a directory holds a whole-model weight file, from stats alone.

    Parse-free by design: callers probe on rank 0 and broadcast, so a torn index must read as present
    here and fail in the reader that follows rather than raise on the one rank that looked.
    ``safetensors_only`` drops the legacy ``pytorch_model.bin`` (the lazy gate's narrower question)
    from the same filename list, since a second list minus one name would route a new whole-model
    filename down the eager fallback.
    """
    names = (SAFETENSORS_INDEX_FILE, SAFETENSORS_WEIGHTS_FILE) if safetensors_only else WHOLE_MODEL_WEIGHT_FILES
    return any(os.path.isfile(os.path.join(checkpoint_dir, name)) for name in names)


def load_full_state_dict(checkpoint_dir: str, device: str = "cpu") -> dict[str, torch.Tensor] | None:
    """Load a gathered checkpoint into one full state dict, resolving the sharded layouts.

    Read-side mirror of :func:`save_sharded_state_dict`, accepting the index+shards, single
    ``model.safetensors``, or legacy ``pytorch_model.bin`` layouts. ``None`` when none exist.

    Raises:
        ValueError: the index marks a per-rank EP save (merge it first).
        RuntimeError: the index is present but torn/unreadable. Callers run this on every rank
            after a rank-0 readability probe, so the diagnosis must name the real cause (a partial
            write, or a per-node copy on a non-shared filesystem) rather than surface a bare
            ``JSONDecodeError`` from whichever rank happened to hold the torn copy.
    """
    index_path = os.path.join(checkpoint_dir, SAFETENSORS_INDEX_FILE)
    try:
        layout = resolve_checkpoint_weights(checkpoint_dir)
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"{index_path} is present but unreadable ({e}) — a torn/partial checkpoint, or a "
            f"per-node copy on a non-shared filesystem that this rank sees mid-write. Resume "
            f"from a complete checkpoint."
        ) from e

    if layout.index is not None:
        if _index_declares_per_rank_shards(layout.index):
            raise ValueError(
                f"{checkpoint_dir} is a per-rank sharded checkpoint; it holds partial per-rank "
                f"tensors and cannot be resumed directly. Merge it first with scripts/"
                f"after_training/merge_ep_shards.py, or re-save with save_sharded_ep=False "
                f"(gathered checkpoint)."
            )
        if not layout.shard_files:
            raise RuntimeError(
                f"{index_path} parsed but carries no 'weight_map' — not a safetensors index. "
                f"Resume from a complete checkpoint."
            )
        state_dict: dict[str, torch.Tensor] = {}
        for shard_file in layout.shard_files:
            state_dict.update(_safetensors_load_file(layout.path(shard_file), device=device))
        return state_dict
    if layout.shard_files:
        return _safetensors_load_file(layout.path(layout.shard_files[0]), device=device)
    if layout.legacy_bin is not None:
        return torch.load(layout.path(layout.legacy_bin), map_location=device, weights_only=True)
    return None


def read_checkpoint_key_set(checkpoint: str) -> set[str]:
    """Tensor-key set of a checkpoint dir (sharded/single safetensors or ``pytorch_model.bin``)
    without materializing any tensor data. Empty set when no weight file is present.
    """
    layout = resolve_checkpoint_weights(checkpoint)
    if layout.legacy_bin is not None:
        return set(torch.load(layout.path(layout.legacy_bin), map_location="meta", weights_only=True))
    return set(layout.weight_map)


class StreamingCheckpointReader:
    """One tensor at a time out of a checkpoint dir (sharded/single safetensors, or a legacy bin).

    The read side of the PP save's streaming contract: every rank of a stage reads the same non-EP
    tensors before distributing its own FSDP2 shard, so buffering them into one dict would put
    ``gpus_per_node x`` the stage's bytes on a host at once.

    Construction opens every shard holding a requested key, so a truncated file raises there, before
    the caller's cross-rank consensus and therefore before any collective. A legacy ``.bin`` is one
    pickle, read whole.
    """

    def __init__(self, checkpoint: str, keys):
        keys = set(keys)
        layout = resolve_checkpoint_weights(checkpoint)
        self._legacy: dict[str, torch.Tensor] | None = None
        self._handles: dict[str, Any] = {}
        self._key_to_shard: dict[str, str] = {}
        self._open_shards = ExitStack()

        if layout.legacy_bin is not None:
            state_dict = torch.load(layout.path(layout.legacy_bin), map_location="cpu", weights_only=True)
            self._legacy = {k: state_dict[k] for k in keys & set(state_dict)}
            self.available = set(self._legacy)
            return

        weight_map = layout.weight_map
        self._key_to_shard = {key: weight_map[key] for key in keys & set(weight_map)}
        self.available = set(self._key_to_shard)
        try:
            for shard in sorted(set(self._key_to_shard.values())):
                handle = safe_open(layout.path(shard), framework="pt", device="cpu")
                self._handles[shard] = self._open_shards.enter_context(handle)
        except BaseException:
            self.close()
            raise

    def get(self, key: str) -> torch.Tensor:
        """The checkpoint's tensor for ``key``, read now and owned by the caller."""
        if self._legacy is not None:
            return self._legacy[key]
        return self._handles[self._key_to_shard[key]].get_tensor(key)

    def close(self) -> None:
        """Release every open shard handle. Idempotent."""
        self._open_shards.close()
        self._handles.clear()
        self._legacy = None

    def __enter__(self) -> StreamingCheckpointReader:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()


def read_specific_keys_from_checkpoint(checkpoint: str, keys) -> dict[str, torch.Tensor]:
    """Read only ``keys`` from a checkpoint dir, returning ``{key: cpu tensor}`` for those present.

    Buffering read for the callers whose key set is a handful of tensors; the stage-sized reads go
    through :class:`StreamingCheckpointReader` directly rather than through this dict.
    """
    with StreamingCheckpointReader(checkpoint, keys) as reader:
        return {key: reader.get(key) for key in reader.available}


def _unique_key_by_suffix(state_dict: dict, suffix: str) -> str | None:
    """The single state-dict key equal to ``suffix`` or ending in ``.{suffix}``; None when absent.
    Warns and returns None on an ambiguous (multi-key) match."""
    matches = [k for k in state_dict if k == suffix or k.endswith(f".{suffix}")]
    if len(matches) > 1:
        logger.warning(
            f"reconcile_tie_word_embeddings: multiple state-dict keys match '{suffix}' "
            f"({matches[:4]}); skipping the tie-consistency check for this save."
        )
        return None
    return matches[0] if matches else None


def is_tie_reconcile_key(name: str) -> bool:
    """Whether ``name`` is one of the two keys :func:`reconcile_tie_word_embeddings` compares.

    A streamed save never holds the whole dict, so it keeps exactly these keys aside as they pass.
    """
    return any(name == suffix or name.endswith(f".{suffix}") for suffix in _TIE_KEY_SUFFIXES)


def reconcile_tie_word_embeddings(model, state_dict: dict) -> None:
    """Make ``config.tie_word_embeddings`` consistent with the saved tensors.

    Tied embeddings can diverge during distributed training (FSDP2 shards lm_head/embed_tokens
    independently, an fp32 upcast splits ``.data``, FLCE trains lm_head directly), and the flag flips
    to False when both diverged tensors are present so the trained lm_head is honoured. Keys resolve
    by suffix, covering wrapped/VLM layouts; a still-tied checkpoint carries one of the pair and skips.
    """
    config = getattr(model, "config", None)
    if config is None or not getattr(config, "tie_word_embeddings", False):
        return
    lm_key = _unique_key_by_suffix(state_dict, _TIE_KEY_SUFFIXES[0])
    emb_key = _unique_key_by_suffix(state_dict, _TIE_KEY_SUFFIXES[1])
    if lm_key is None or emb_key is None:
        return
    lm, emb = state_dict[lm_key], state_dict[emb_key]
    if lm.shape != emb.shape or not torch.equal(lm, emb):
        config.tie_word_embeddings = False
        logger.warning(
            f"{lm_key} and {emb_key} diverged during training; "
            f"saving config with tie_word_embeddings=False for checkpoint consistency."
        )
