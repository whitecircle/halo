"""Raw checkpoint reads: the layout cascade a directory's weights resolve through, the streaming reader, and
the standalone tools' shard walks, header sizing and family sniffing. Model-free; rank coordination around
these calls belongs to the caller.
"""

from __future__ import annotations

import glob
import json
import math
import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import ExitStack
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import torch
from safetensors import SafetensorError, safe_open
from safetensors.torch import load_file as _safetensors_load_file
from transformers.utils import CONFIG_NAME

from halo.checkpoint.format import (
    EP_SHARD_KEY_RE,
    LEGACY_WEIGHTS_FILE,
    SAFETENSORS_FAMILY_GLOB,
    SAFETENSORS_INDEX_FILE,
    SAFETENSORS_WEIGHTS_FILE,
    WHOLE_MODEL_WEIGHT_FILES,
    index_declares_ep_shards,
)

# Header dtypes of the full-precision floating tensors a conversion casts or quantizes.
SAFETENSORS_FLOAT_DTYPES = frozenset({"F64", "F32", "F16", "BF16"})
# Header dtype of the fp8 e4m3 storage the fp8 → bf16 converters dequantize.
FP8_HEADER_DTYPE = "F8_E4M3"
# Bytes per element by header dtype. The sizes feed warn-only preflights, so an unknown spelling falls
# back to bf16-sized rather than aborting a valid conversion over an estimate.
_SAFETENSORS_DTYPE_BYTES = {
    **dict.fromkeys(("F64", "I64", "U64"), 8),
    **dict.fromkeys(("F32", "I32", "U32"), 4),
    **dict.fromkeys(("F16", "BF16", "I16", "U16"), 2),
    **dict.fromkeys(("F8_E4M3", "F8_E5M2", "I8", "U8", "BOOL"), 1),
}
# Why a stage finds its own tensors absent from a per-node pipeline checkpoint: each node's directory holds
# only the shards of the stages that node ran.
PER_NODE_PLACEMENT_REMEDY = (
    "A per-node pipeline checkpoint resumes only onto the same topology and rank placement; for "
    "anything else merge every node's directory with scripts/after_training/merge_pp_stages.py first."
)


@dataclass(frozen=True)
class CheckpointWeights:
    """Where a checkpoint directory's tensors live, as resolved by :func:`resolve_checkpoint_weights`.

    Filenames are relative to ``directory``. Neither ``shard_files`` nor ``legacy_bin`` means the
    directory holds no whole-model weight file.
    """

    directory: str
    shard_files: tuple[str, ...] = ()
    index: dict | None = None
    legacy_bin: str | None = None

    def path(self, name: str) -> str:
        return os.path.join(self.directory, name)

    @cached_property
    def weight_map(self) -> dict[str, str]:
        """``{tensor key: shard filename}``; ``{}`` for a legacy bin. Lazy: a single file is opened for its
        header, an index answers from itself."""
        if self.index is not None:
            return self.index.get("weight_map") or {}
        if not self.shard_files:
            return {}
        single = self.shard_files[0]
        with safe_open(self.path(single), framework="pt") as f:
            return dict.fromkeys(f.keys(), single)

    def absent_shard_files(self, keys: Iterable[str] | None = None) -> list[str]:
        """The index-named shard files the directory does not hold, sorted; with ``keys``, only the files
        holding those keys."""
        if keys is None:
            names = set(self.shard_files)
        else:
            wanted = set(keys)
            names = {shard for key, shard in self.weight_map.items() if key in wanted}
        return sorted(name for name in names if not os.path.isfile(self.path(name)))

    def require_safetensors(self, what: str) -> CheckpointWeights:
        """This layout, refusing one with no safetensors shard (legacy-bin-only or weightless)."""
        if not self.shard_files:
            raise FileNotFoundError(
                f"No safetensors {what} found at {self.directory}. "
                f"Expected {SAFETENSORS_INDEX_FILE} or {SAFETENSORS_WEIGHTS_FILE}"
            )
        return self

    def key_set(self) -> set[str]:
        """Every tensor key, read without tensor data (a legacy bin onto the meta device)."""
        if self.legacy_bin is not None:
            return set(torch.load(self.path(self.legacy_bin), map_location="meta", weights_only=True))
        return set(self.weight_map)


class StreamingCheckpointReader:
    """One tensor at a time out of a checkpoint directory (safetensors, or a legacy bin read whole), so
    co-located ranks do not each hold the whole dict. Construction opens every shard holding a requested
    key (``keys=None``: all): a truncated file raises before the caller's consensus and any collective.
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
        """The stored header dtype, without reading the payload; ``None`` for a legacy bin."""
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


def read_checkpoint_index(checkpoint_dir: str, *, missing_ok: bool = False) -> dict:
    """Parse a checkpoint's safetensors index; ``{}`` for an absent one under ``missing_ok``, a raise for a
    torn one."""
    index_path = os.path.join(checkpoint_dir, SAFETENSORS_INDEX_FILE)
    if missing_ok and not os.path.isfile(index_path):
        return {}
    with open(index_path) as f:
        return json.load(f)


def is_ep_sharded_checkpoint(checkpoint_dir: str) -> bool:
    """Whether the directory holds a per-rank EP-sharded save, loadable only after a merge."""
    if not os.path.isfile(os.path.join(checkpoint_dir, SAFETENSORS_INDEX_FILE)):
        return False
    try:
        index = read_checkpoint_index(checkpoint_dir)
    except (OSError, json.JSONDecodeError):
        # Callers decide on rank 0 then broadcast, so raising on a torn index would strand peer ranks.
        return False
    return index_declares_ep_shards(index)


def ep_sharded_checkpoint_reason(checkpoint_dir: str) -> str:
    """The refusal for reading a per-rank EP-sharded save as a whole model."""
    return (
        f"{checkpoint_dir} is a per-rank sharded checkpoint: EP shards under a reused index, each holding "
        f"partial per-rank tensors, so it cannot be read as a whole model. Merge it first with "
        f"scripts/after_training/merge_ep_shards.py, or re-save gathered (save_sharded_ep: false)."
    )


def resolve_checkpoint_weights(checkpoint_dir: str) -> CheckpointWeights:
    """Where a checkpoint directory's tensors live: the index (no shard opened), else a single
    ``model.safetensors``, else the legacy bin. Policy-free: an empty layout or a per-rank index is the
    caller's call; a torn index raises."""
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
    """Whether a directory holds a whole-model weight file (the legacy bin too unless ``safetensors_only``),
    from stats alone: callers probe on rank 0 and broadcast, so a torn index fails in the reader after."""
    return any(
        os.path.isfile(os.path.join(checkpoint_dir, name))
        for name in WHOLE_MODEL_WEIGHT_FILES
        if not (safetensors_only and name == LEGACY_WEIGHTS_FILE)
    )


def read_checkpoint_config(
    checkpoint_dir: str, *, missing_ok: bool = False, filename: str = CONFIG_NAME
) -> dict | None:
    """Parse a checkpoint's JSON config (``config.json`` unless ``filename`` names another); ``None`` only
    for an absent file under ``missing_ok``."""
    path = os.path.join(checkpoint_dir, filename)
    if missing_ok and not os.path.isfile(path):
        return None
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def load_full_state_dict(checkpoint_dir: str, device: str = "cpu") -> dict[str, torch.Tensor] | None:
    """A gathered checkpoint as one full state dict, from any layout of the cascade; ``None`` without one.

    Raises ``ValueError`` for a per-rank EP save and ``RuntimeError`` for a torn index, naming the cause
    on whichever rank holds the torn copy.
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
            raise ValueError(ep_sharded_checkpoint_reason(checkpoint_dir))
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


def _reject_absent_index_shards(checkpoint_dir: str) -> None:
    """Refuse an index naming shard files the directory does not hold: one node's copy of a PP save on a
    non-shared filesystem, or a half-copied checkpoint."""
    if not os.path.isfile(os.path.join(checkpoint_dir, SAFETENSORS_INDEX_FILE)):
        return
    absent = resolve_checkpoint_weights(checkpoint_dir).absent_shard_files()
    if absent:
        raise ValueError(
            f"{checkpoint_dir}/{SAFETENSORS_INDEX_FILE} names {len(absent)} shard file(s) the directory "
            f"does not hold ({', '.join(absent[:4])}{', …' if len(absent) > 4 else ''}). A "
            f"pipeline-parallel save on a non-shared output filesystem leaves each node holding its "
            f"own stage's parts under a copy of the whole-model index — merge every node's directory "
            f"with scripts/after_training/merge_pp_stages.py first. Otherwise the checkpoint was copied "
            f"incompletely; restore the missing shards."
        )


def _reject_indexless_ep_shards(checkpoint_dir: str) -> None:
    """Refuse per-rank EP shards whose save died before their index, by peeking the first file's keys; an
    unparseable header is left to the caller's read."""
    if os.path.isfile(os.path.join(checkpoint_dir, SAFETENSORS_INDEX_FILE)):
        return
    shards = sorted(glob.glob(os.path.join(checkpoint_dir, SAFETENSORS_FAMILY_GLOB)))
    if not shards:
        return
    try:
        with safe_open(shards[0], framework="pt") as reader:
            partial = next((key for key in reader.keys() if EP_SHARD_KEY_RE.match(key)), None)  # noqa: SIM118
    except (SafetensorError, OSError):
        return
    if partial is not None:
        raise ValueError(
            f"{checkpoint_dir} holds per-rank EP shards ({os.path.basename(shards[0])} carries the "
            f"partial expert tensor {partial!r}) but no {SAFETENSORS_INDEX_FILE} — the sharded save "
            f"was interrupted between writing the shards and writing their index. Reading it here "
            f"would take each rank's slice for the whole expert bank. Re-run the save, or restore "
            f"the index and merge with scripts/after_training/merge_ep_shards.py."
        )


def reject_partial_checkpoint(checkpoint_dir: str) -> None:
    """Refuse a checkpoint directory that reads as whole but is not: a per-rank EP save (with or without
    its index) or an index naming absent shards, which ``from_pretrained`` would random-initialize around.
    Every tool that opens a checkpoint directory calls this first.
    """
    if is_ep_sharded_checkpoint(checkpoint_dir):
        raise ValueError(ep_sharded_checkpoint_reason(checkpoint_dir))
    _reject_indexless_ep_shards(checkpoint_dir)
    _reject_absent_index_shards(checkpoint_dir)


def checkpoint_shard_files(checkpoint_dir: str) -> list[str]:
    """A checkpoint's model safetensors shards as paths, from its index else a ``model*.safetensors`` glob
    (safe behind :func:`reject_partial_checkpoint`); raises when it has none, a legacy bin included."""
    reject_partial_checkpoint(checkpoint_dir)

    layout = resolve_checkpoint_weights(checkpoint_dir)
    if layout.shard_files:
        return [layout.path(name) for name in layout.shard_files]

    shards = sorted(glob.glob(os.path.join(checkpoint_dir, SAFETENSORS_FAMILY_GLOB)))
    if shards:
        return shards

    raise FileNotFoundError(f"No '{SAFETENSORS_WEIGHTS_FILE}' or '{SAFETENSORS_INDEX_FILE}' found in {checkpoint_dir}")


def iter_checkpoint_shards(checkpoint_dir: str) -> Iterator[tuple[str, Any]]:
    """Yield ``(shard_path, open reader)`` per shard; the reader is live for its own step only."""
    for shard in checkpoint_shard_files(checkpoint_dir):
        with safe_open(shard, framework="pt") as reader:
            yield shard, reader


def iter_checkpoint_shard_entries(checkpoint_dir: str) -> Iterator[tuple[str, Any, str]]:
    """Yield ``(shard_path, open reader, key)`` for every tensor a checkpoint stores; a header- or key-only
    scan reads no tensor data."""
    for shard, reader in iter_checkpoint_shards(checkpoint_dir):
        for key in reader.keys():  # noqa: SIM118 - safe_open has .keys() but is not a mapping
            yield shard, reader, key


def iter_checkpoint_tensors(
    checkpoint_dir: str, predicate: Callable[[str], bool] | None = None
) -> Iterator[tuple[str, torch.Tensor]]:
    """Yield ``(key, tensor)`` for the keys ``predicate`` accepts, judged before the tensor is read."""
    for _shard, reader, key in iter_checkpoint_shard_entries(checkpoint_dir):
        if predicate is None or predicate(key):
            yield key, reader.get_tensor(key)


def header_numel(header) -> int:
    """Element count from a ``safe_open`` slice header alone."""
    return math.prod(header.get_shape())


def header_nbytes(header) -> int:
    """Storage size from a ``safe_open`` slice header alone."""
    return header_numel(header) * _SAFETENSORS_DTYPE_BYTES.get(header.get_dtype(), 2)


def detect_model_types(checkpoint_dir: str) -> list[str]:
    """The ``model_type`` spellings a checkpoint's ``config.json`` declares, ``text_config``'s (which fixes
    the expert layout) first; ``[]`` without a config, a raise for an unreadable one."""
    config = read_checkpoint_config(checkpoint_dir, missing_ok=True)
    if config is None:
        return []
    candidates = ((config.get("text_config") or {}).get("model_type"), config.get("model_type"))
    return [candidate for candidate in candidates if candidate]


def detect_model_type(checkpoint_dir: str) -> str:
    """The least specific of :func:`detect_model_types` (``""`` without one): the top-level family, or
    the ``text_config`` one of a composite config that declares none."""
    model_types = detect_model_types(checkpoint_dir)
    return model_types[-1] if model_types else ""
