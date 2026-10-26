"""Checkpoint file format: the on-disk spellings every reader and writer shares, the layout cascade a
directory's weights resolve through, and the reads built on it.

Format layer only — no ``torch.distributed`` and no model, so the standalone ``scripts/after_training/``
tools and the parallel save paths read one artifact layout; rank coordination around these calls belongs to
the caller. Writing weight files is :mod:`src.checkpoint.writing`; the hub's tensor names on save are
:mod:`src.checkpoint.hub_layout`.
"""

from __future__ import annotations

import json
import os
import re
import shutil
from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import torch
from peft.utils import CONFIG_NAME as PEFT_ADAPTER_CONFIG_NAME
from safetensors import safe_open
from safetensors.torch import load_file as _safetensors_load_file
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from src.checkpoint.atomic import is_atomic_staging_file
from src.checkpoint.model_card import tag_exported_model_card

# Per-file cap for gathered safetensors saves, shared by every save path and the arg default.
DEFAULT_MAX_SHARD_SIZE = "5GB"

# One spelling per artifact for every reader/writer: a typo degrades silently (weightless
# checkpoint, LR schedule re-warming from step 0).
SAFETENSORS_INDEX_FILE = "model.safetensors.index.json"
# Format marker every safetensors writer stamps; the readers must agree, so it has one home.
SAFETENSORS_METADATA = {"format": "pt"}
# The index ``metadata.format`` stamp of a per-rank EP save, and one rank's slice of an EP-sharded
# expert tensor. One spelling each: a reader that misses them takes a partial tensor for a whole one.
EP_SHARDED_FORMAT = "ep_sharded"
EP_SHARD_KEY_INFIX = ".shard_"
EP_SHARD_KEY_RE = re.compile(rf"(.+)\{EP_SHARD_KEY_INFIX}(\d+)$")
# HF's shard filename pattern and the spellings DERIVED from it — the single-file name a one-shard
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
# pattern so it can never collide with one, inside the sweep's glob so leftovers cannot outlive it.
STREAM_PART_PREFIX = "model-streaming"
LEGACY_WEIGHTS_FILE = "pytorch_model.bin"
# Every filename that makes a directory a whole-model checkpoint, in the layout cascade's order — one
# list behind :func:`has_whole_model_weight_file`, whose safetensors-only mode is the lazy gate's.
WHOLE_MODEL_WEIGHT_FILES = (SAFETENSORS_INDEX_FILE, SAFETENSORS_WEIGHTS_FILE, LEGACY_WEIGHTS_FILE)

# HF Trainer's own filenames, spelled here because transformers defines them only in
# ``transformers.trainer``, a ~1 s import this leaf must not pay.
TRAINER_STATE_FILE = "trainer_state.json"
SCHEDULER_STATE_FILE = "scheduler.pt"
# HF Trainer's replicated optimizer state, which the sharded modes deliberately replace.
OPTIMIZER_STATE_FILES = ("optimizer.pt", "optimizer.bin")
# Written after every rank's optimizer shard, so its presence vouches for a complete shard set.
OPTIMIZER_META_FILE = "optimizer_meta.pt"
ROUTER_BALANCING_BIASES_FILE = "router_balancing_biases.pt"
# Checkpointed DPO/KTO and offline GRPO reference scores, with validated split identity.
REFERENCE_LOGPS_FILE = "reference_logps.pt"
# An output_dir entry the run keeps beside its checkpoints and never exports: the Trainer's Hub push
# skips ``_*`` and so does the aux copy.
RUN_INTERNAL_PREFIX = "_"
# Run-local reference scratch, excluded from fresh-output-dir validation and Hub uploads.
REFERENCE_CACHE_DIR_NAME = f"{RUN_INTERNAL_PREFIX}reference_cache"
# Where a resume moves the step directories whose save never completed, out of rotation's sight.
INCOMPLETE_CHECKPOINTS_DIR_NAME = f"{RUN_INTERNAL_PREFIX}incomplete_checkpoints"
# The environmental GRPO prefetch's submitted-but-untrained rounds, one file per rank.
PREFETCH_PENDING_PREFIX = "prefetch_pending"

# PEFT adapter artifact filenames. ADAPTER_WEIGHT_NAMES is in load-preference order; PeftAdapterSaver
# falls back to .bin, so detection must accept both.
ADAPTER_SAFETENSORS_FILE = "adapter_model.safetensors"
ADAPTER_BIN_FILE = "adapter_model.bin"
ADAPTER_CONFIG_FILE = PEFT_ADAPTER_CONFIG_NAME
ADAPTER_WEIGHT_NAMES = (ADAPTER_SAFETENSORS_FILE, ADAPTER_BIN_FILE)
# Training-time model state a merge tool cannot recover from the adapter artifacts alone (the GptOss
# sink policy). A sidecar, never adapter_config.json: stock PEFT must load the adapter unchanged.
TRAINING_PROVENANCE_FILE = "training_provenance.json"
PROVENANCE_GPT_OSS_SINKS = "gpt_oss_attention_sinks"
# A merged checkpoint's unmerged adapter, the state it resumes from, in a subdirectory: a root
# ``adapter_config.json`` makes ``from_pretrained`` load it on top of weights already holding its delta.
# The root marker follows once the adapter is complete, and its presence alone is the resume verdict.
RESUME_ADAPTER_DIR = "resume_adapter"
RESUME_ADAPTER_MARKER_FILE = "resume_adapter.json"
# Resume state like the sidecars below, but not weight-suffixed: the aux copy carries them by
# default and drops them by name where ``include_resume_sidecars`` is off. ``trainer_state.json`` is
# the step a resume continues from and what resume detection takes a directory by.
_UNSUFFIXED_RESUME_ENTRIES = (TRAINER_STATE_FILE, RESUME_ADAPTER_DIR, RESUME_ADAPTER_MARKER_FILE)

# Never carried over: a stray pytorch_model.bin / optimizer*.pt would shadow the fresh safetensors.
_WEIGHT_FILE_SUFFIXES = (".safetensors", ".bin", ".pt")
# Foreign-framework exports, never weights this toolkit reads. The aux copy and the hub-download
# ignore list share this ONE tuple so they cannot disagree.
_FOREIGN_EXPORT_SUFFIXES = (".pth", ".gguf", ".h5", ".msgpack", ".onnx", ".onnx_data", ".tflite", ".ot", ".mlmodel")
# Exempt from that skip: dropping these restarts the LR schedule, zeroes the router biases, or leaves
# a precompute resume with no untrained reference to restore.
_RESUME_SIDECAR_FILES = (SCHEDULER_STATE_FILE, ROUTER_BALANCING_BIASES_FILE, REFERENCE_LOGPS_FILE)
# Same exemption by prefix: losing ``rng_state_<rank>.pth`` re-draws every shuffle and dropout mask,
# and losing a rank's pending prefetch round skips the batch it holds.
_RESUME_SIDECAR_PREFIXES = ("rng_state", PREFETCH_PENDING_PREFIX)
# Vendor dumps of the same weights in a raw format (gpt-oss ships ``original/`` and ``metal/``):
# hundreds of GB the aux copy must not duplicate. Exact names rather than prefixes, since
# ``original_adapter_config/`` is aux data the copy must keep.
_WEIGHT_DUMP_DIRS = ("original", "consolidated", "metal")
# Hub-download counterpart of copy_checkpoint_aux_files' skip: fetch an aux source minus its weights.
WEIGHT_FILE_IGNORE_PATTERNS = tuple(
    f"*{suffix}" for suffix in (*_WEIGHT_FILE_SUFFIXES, *_FOREIGN_EXPORT_SUFFIXES)
) + tuple(f"{name}/*" for name in _WEIGHT_DUMP_DIRS)
# A model load's Hub fetch: the repo's top-level files, where its config, tokenizer, weights and remote
# code sit. Hub patterns are fnmatch, whose ``*`` crosses ``/``, so this drops every nested file, the
# weight dumps above included.
HUB_SUBFOLDER_IGNORE_PATTERNS = ("*/*",)


def ep_shard_filename(rank: int, world_size: int) -> str:
    """Filename of one rank's slice of a per-rank EP save.

    HF's shard pattern with a 0-BASED index: this save writes one file per rank, so rank 0 must
    still carry a suffix — :func:`hf_shard_filename`'s 1-based numbering degenerates to the
    unsuffixed ``model.safetensors`` at a single part, which every reader takes for a whole model.
    """
    return SAFETENSORS_SHARD_PATTERN.format(suffix=f"-{rank:05d}-of-{world_size:05d}")


def prefetch_pending_filename(rank: int, world_size: int) -> str:
    """Filename of one rank's pending prefetch rounds.

    The world size is in the name because each rank draws its own prompts: a resume at another width
    finds no file of its own spelling rather than another layout's prompts.
    """
    return f"{PREFETCH_PENDING_PREFIX}-{rank:05d}-of-{world_size:05d}.pt"


def is_ep_shard(name: str) -> bool:
    """Whether ``name`` is a per-rank EP save slice.

    Narrower than a ``.safetensors`` glob on purpose: a sibling ``adapter_model.safetensors`` or a
    stale ``model.safetensors`` must not be swept into a merge.
    """
    return _EP_SHARD_FILE_RE.match(name) is not None


def hf_shard_filename(part: int, total_parts: int) -> str:
    """HF's own name for part ``part`` of ``total_parts`` — the unsharded file when there is only one.

    ``split_torch_state_dict_into_shards`` is handed :data:`SAFETENSORS_SHARD_PATTERN` and fills the
    same suffix, so a streamed checkpoint is indistinguishable from a ``save_pretrained`` one.
    """
    suffix = "" if total_parts == 1 else f"-{part:05d}-of-{total_parts:05d}"
    return SAFETENSORS_SHARD_PATTERN.format(suffix=suffix)


def copy_checkpoint_aux_files(
    input_dir: str, output_dir: str, *, include_resume_sidecars: bool = True, verbose: bool = False
) -> None:
    """Copy a checkpoint's non-weight files (config, tokenizer, chat template, remote-code .py, …)
    from ``input_dir`` to ``output_dir`` verbatim, then tag the output's Hub model card.

    Every tool that builds an export out of a source directory runs this copy, including the ones
    that carry ``config.json`` across as-is and so never reach the config finalizer; the source's
    card rides along and gets the Halo tag. A source card with malformed metadata stays verbatim and
    untagged, with a warning (:func:`~src.checkpoint.model_card.tag_exported_model_card`): no loader
    reads the card, and most callers run this copy after their weight pass.

    Skips every top-level weight file and safetensors index, which the caller writes fresh, but
    preserves the resume state (``trainer_state.json``, ``scheduler.pt``, ``router_balancing_biases.pt``,
    ``reference_logps.pt``, ``rng_state_*``, ``prefetch_pending-*``) a resume-from-merged run restores;
    ``include_resume_sidecars=False`` drops it, for an artifact that is a new base rather than the
    source run's weights (an N-way merge, an adapter folded into its base).

    Subdirectories are copied whole, weight files included: a SentenceTransformer module directory
    carries weights no caller rewrites, and filtering them out leaves ``modules.json`` pointing at
    modules that no longer exist. Four kinds of directory stay behind: a nested ``checkpoint-N``
    (resume state rather than the artifact), a run's own :data:`RUN_INTERNAL_PREFIX` state (its
    reference scratch, the incomplete checkpoints a resume set aside), a vendor weight dump, and a
    hidden one. So does any file a crashed staged write left unpublished
    (:func:`~src.checkpoint.atomic.is_atomic_staging_file`).

    ``output_dir`` nested inside ``input_dir`` raises: the walk would copy the destination into
    itself until the disk fills.

    A merged checkpoint's resume adapter and its marker are resume state too: carried by default,
    dropped with the sidecars.
    """
    input_root = os.path.realpath(input_dir)
    if os.path.commonpath([input_root, os.path.realpath(output_dir)]) == input_root:
        raise ValueError(
            f"copy_checkpoint_aux_files: output_dir {output_dir!r} is inside input_dir "
            f"{input_dir!r} — the subdirectory copy would recurse into the destination. Write the "
            f"artifact to a directory outside the source checkpoint."
        )
    for name in os.listdir(input_dir):
        if name in _UNSUFFIXED_RESUME_ENTRIES and not include_resume_sidecars:
            continue
        src = os.path.join(input_dir, name)
        if os.path.isdir(src):
            if name.startswith((".", RUN_INTERNAL_PREFIX, f"{PREFIX_CHECKPOINT_DIR}-")) or name in _WEIGHT_DUMP_DIRS:
                continue
            shutil.copytree(src, os.path.join(output_dir, name), dirs_exist_ok=True)
            if verbose:
                print(f"Copied: {name}/")  # noqa: T201 — CLI-facing helper; the merge scripts report via print
            continue
        skip_as_weight = (
            name.endswith(_WEIGHT_FILE_SUFFIXES)
            or name.endswith(_FOREIGN_EXPORT_SUFFIXES)
            or name.endswith(".safetensors.index.json")
        )
        keep_as_sidecar = include_resume_sidecars and (
            name in _RESUME_SIDECAR_FILES or name.startswith(_RESUME_SIDECAR_PREFIXES)
        )
        if (skip_as_weight and not keep_as_sidecar) or is_atomic_staging_file(name):
            continue
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, name))
            if verbose:
                print(f"Copied: {name}")  # noqa: T201 - CLI-facing helper; the merge scripts report via print
    tag_exported_model_card(output_dir, source_dir=input_dir)


def read_checkpoint_index(checkpoint_dir: str, *, missing_ok: bool = False) -> dict:
    """Parse a checkpoint's ``model.safetensors.index.json``. One parse for every index reader.

    ``OSError`` / ``json.JSONDecodeError`` propagate by default so each caller applies its own policy
    to a torn index (an empty keep-set, a rank-0 ``False``, a raise naming a non-shared filesystem).
    ``missing_ok`` returns ``{}`` for a directory with no index at all — a checkpoint small enough
    for a single ``model.safetensors`` has none, and the merge scripts must be able to inspect the
    format marker without an unhandled ``ENOENT`` hiding the refusal that names the fix.
    """
    index_path = os.path.join(checkpoint_dir, SAFETENSORS_INDEX_FILE)
    if missing_ok and not os.path.isfile(index_path):
        return {}
    with open(index_path) as f:
        return json.load(f)


def index_declares_ep_shards(index: dict) -> bool:
    """Whether a parsed safetensors index marks a per-rank EP save rather than a gathered one.

    A gathered HF index carries no ``format`` marker, so the gathered/FSDP2 layout reads as False;
    so does a PP save, whose per-stage shards hold complete tensors.
    """
    meta = index.get("metadata", {}) or {}
    return meta.get("format") == EP_SHARDED_FORMAT


def is_sharded_checkpoint(checkpoint_dir: str) -> bool:
    """True when the directory holds a per-rank EP-sharded save (partial tensors under a reused
    index filename) — loadable only after merging, never directly."""
    if not os.path.isfile(os.path.join(checkpoint_dir, SAFETENSORS_INDEX_FILE)):
        return False
    try:
        index = read_checkpoint_index(checkpoint_dir)
    except (OSError, json.JSONDecodeError):
        # Callers decide on rank 0 then broadcast, so raising on a torn index would strand peer ranks.
        return False
    return index_declares_ep_shards(index)


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

    def key_set(self) -> set[str]:
        """Every tensor key the checkpoint holds, without materializing tensor data: a legacy bin is
        read onto the meta device. Empty when the directory holds no weight file."""
        if self.legacy_bin is not None:
            return set(torch.load(self.path(self.legacy_bin), map_location="meta", weights_only=True))
        return set(self.weight_map)


def resolve_checkpoint_weights(checkpoint_dir: str) -> CheckpointWeights:
    """Where a checkpoint directory's tensors live — the one statement of the layout cascade.

    Sharded index first (authoritative, and read as the map itself so no shard is opened), then a
    single ``model.safetensors``, then the legacy ``pytorch_model.bin``. Reports what is on disk and
    applies no policy: whether an empty layout, a per-rank sharded index or an index without a
    ``weight_map`` is an error is the caller's to decide, because the resume gates, the lazy loaders
    and the standalone tools each answer differently. A torn index raises through.
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
    here and fail loudly in the reader that follows rather than raise on the one rank that looked.
    ``safetensors_only`` drops the legacy ``pytorch_model.bin`` — the lazy gate's narrower question —
    off the same filename list, since a second list minus one name would route a new whole-model
    filename down the eager fallback.
    """
    return any(
        os.path.isfile(os.path.join(checkpoint_dir, name))
        for name in WHOLE_MODEL_WEIGHT_FILES
        if not (safetensors_only and name == LEGACY_WEIGHTS_FILE)
    )


def adapter_weight_paths(adapter_dir: str) -> tuple[str, ...]:
    """The adapter weight files a directory may carry, in PEFT's own load-preference order.

    Taken from the :data:`ADAPTER_WEIGHT_NAMES` tuple that declares it: a reader that misses the
    ``.bin`` fallback reads a saved adapter as absent.
    """
    return tuple(os.path.join(adapter_dir, name) for name in ADAPTER_WEIGHT_NAMES)


def has_adapter_weight_file(directory: str) -> bool:
    """Whether a directory holds a PEFT adapter weight file (:func:`adapter_weight_paths`); stat-only,
    like :func:`has_whole_model_weight_file`."""
    return any(os.path.isfile(path) for path in adapter_weight_paths(directory))


def write_resume_adapter_marker(checkpoint_dir: str) -> None:
    """Mark ``checkpoint_dir`` as resuming from its :data:`RESUME_ADAPTER_DIR`, not its weights.

    The caller writes it only once that directory is complete, so a marked checkpoint carries its
    adapter.
    """
    with open(os.path.join(checkpoint_dir, RESUME_ADAPTER_MARKER_FILE), "w") as fh:
        json.dump({"adapter_dir": RESUME_ADAPTER_DIR}, fh, indent=2)


def resume_adapter_dir(checkpoint_dir: str) -> str | None:
    """The adapter directory a marked checkpoint resumes from, or ``None`` for any other checkpoint.

    Stat-only, like :func:`has_whole_model_weight_file`, for the same rank-0-then-broadcast callers.
    """
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


def load_full_state_dict(checkpoint_dir: str, device: str = "cpu") -> dict[str, torch.Tensor] | None:
    """Load a gathered checkpoint into one full state dict — sharded-aware.

    Read-side mirror of :func:`~src.checkpoint.writing.save_sharded_state_dict`, accepting the
    index+shards, single ``model.safetensors``, or legacy ``pytorch_model.bin`` layouts. ``None`` when
    none exist.

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
        if index_declares_ep_shards(layout.index):
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
    if layout.shard_files:
        state_dict: dict[str, torch.Tensor] = {}
        for shard_file in layout.shard_files:
            state_dict.update(_safetensors_load_file(layout.path(shard_file), device=device))
        return state_dict
    if layout.legacy_bin is not None:
        return torch.load(layout.path(layout.legacy_bin), map_location=device, weights_only=True)
    return None


def read_checkpoint_key_set(checkpoint: str) -> set[str]:
    """Tensor-key set of a checkpoint dir (sharded/single safetensors or ``pytorch_model.bin``)
    without materializing any tensor data. Empty set when no weight file is present.
    """
    return resolve_checkpoint_weights(checkpoint).key_set()


class StreamingCheckpointReader:
    """One tensor at a time out of a checkpoint dir (sharded/single safetensors, or a legacy bin).

    The read side of the PP save's streaming contract: every rank of a stage reads the same non-EP
    tensors before distributing its own FSDP2 shard, so buffering them into one dict would put
    ``gpus_per_node ×`` the stage's bytes on a host at once (~280 GB per node for a dense 70B at
    ``pp_size=4``).

    Construction opens every shard holding a requested key (``keys=None`` requests them all), so a
    truncated file raises THERE — before the caller's cross-rank consensus, hence before any
    collective — and :meth:`get` serves from the validated handles at one tensor of peak. A legacy
    ``.bin`` is one pickle, read whole.
    """

    def __init__(self, checkpoint: str, keys: Iterable[str] | None):
        layout = resolve_checkpoint_weights(checkpoint)
        self._legacy: dict[str, torch.Tensor] | None = None
        self._handles: dict[str, Any] = {}
        self._key_to_shard: dict[str, str] = {}
        self._open_shards = ExitStack()

        if layout.legacy_bin is not None:
            state_dict = torch.load(layout.path(layout.legacy_bin), map_location="cpu", weights_only=True)
            self._legacy = state_dict if keys is None else {k: state_dict[k] for k in set(keys) & set(state_dict)}
            self.available = set(self._legacy)
            return

        weight_map = layout.weight_map
        wanted = weight_map.keys() if keys is None else set(keys) & set(weight_map)
        self._key_to_shard = {key: weight_map[key] for key in wanted}
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

    def safetensors_dtype(self, key: str) -> str | None:
        """Stored dtype from the header, without reading the payload; None for legacy pickle files."""
        if self._legacy is not None:
            return None
        return self._handles[self._key_to_shard[key]].get_slice(key).get_dtype()

    def close(self) -> None:
        """Release every open shard handle. Idempotent."""
        self._open_shards.close()
        self._handles.clear()
        self._legacy = None

    def __enter__(self) -> StreamingCheckpointReader:
        return self

    def __exit__(self, *_exc) -> None:
        self.close()
