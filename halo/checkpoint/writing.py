"""Model weight files: save-dtype casts, stale-file sweeps, merged index, streaming writer, tie reconcile."""

from __future__ import annotations

import concurrent.futures
import glob
import json
import logging
import os
from collections import deque
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import torch
from huggingface_hub.serialization._base import parse_size_to_int

from halo.checkpoint.atomic import save_safetensors_file
from halo.checkpoint.format import (
    SAFETENSORS_FAMILY_GLOB,
    SAFETENSORS_INDEX_FILE,
    SAFETENSORS_METADATA,
    SAFETENSORS_WEIGHTS_FILE,
    STREAM_PART_PREFIX,
    hf_shard_filename,
    index_declares_ep_shards,
    stream_part_filename,
)
from halo.checkpoint.hub_layout import revert_to_load_layout
from halo.checkpoint.reading import read_checkpoint_index
from halo.env import env_int
from halo.models.config_access import ties_word_embeddings, untie_word_embeddings
from halo.models.moe.balancing import balancing_param_keys, is_balancing_state_key
from halo.models.param_roles import fp32_pinned_names, norm_param_keys
from halo.models.unwrapping import strip_peft_adapter_segment

logger = logging.getLogger(__name__)

# Exports are saved bf16 even when training uses fp32 masters; training checkpoints keep the live dtype.
_SAVE_DTYPE = torch.bfloat16
# The tied pair, in the order reconcile_tie_word_embeddings reads them.
_TIE_KEY_SUFFIXES = ("lm_head.weight", "embed_tokens.weight")
DEFAULT_CHECKPOINT_WRITE_THREADS = 0
DEFAULT_CHECKPOINT_WRITERS_PER_NODE = 1


def cast_to_save_dtype(t: torch.Tensor) -> torch.Tensor:
    """A floating tensor at the save dtype; the blanket cast under :func:`save_dtype_caster`'s keep-sets."""
    if torch.is_tensor(t) and t.is_floating_point() and t.dtype != _SAVE_DTYPE:
        return t.to(dtype=_SAVE_DTYPE)
    return t


def _as_live(_name: str, tensor: torch.Tensor) -> torch.Tensor:
    return tensor


def _caster_keeping(keep: Callable[[str], bool]) -> Callable[[str, torch.Tensor], torch.Tensor]:
    def cast(name: str, t: torch.Tensor) -> torch.Tensor:
        return t if keep(name) else cast_to_save_dtype(t)

    return cast


def save_dtype_caster(model: torch.nn.Module, *, keep_live_dtype: bool = False):
    """``cast(name, tensor)`` for a live-model save: the save dtype except norm params, router balancing and fp32 pins
    (a PEFT-stripped key matches too). ``keep_live_dtype`` (a training checkpoint) casts nothing, for an exact
    resume."""
    if keep_live_dtype:
        return _as_live
    keep = norm_param_keys(model) | balancing_param_keys(model) | fp32_pinned_names(model, include_buffers=True)
    return _caster_keeping(lambda name: name in keep or strip_peft_adapter_segment(name) in keep)


def adapter_save_dtype_caster() -> Callable[[str, torch.Tensor], torch.Tensor]:
    """``cast(name, tensor)`` for an exported adapter; a balancing tensor on a ``modules_to_save`` clone keeps its
    trained dtype, since a bf16-rounded bias flips near-tie top-k picks on serve."""
    return _caster_keeping(lambda name: is_balancing_state_key(strip_peft_adapter_segment(name)))


def normalize_gathered_state_dict(model: torch.nn.Module, state_dict: dict, *, keep_live_dtype: bool = False) -> dict:
    """A gathered state dict on disk: save-dtype cast (keep-sets use live key spellings), then the hub layout."""
    cast = save_dtype_caster(model, keep_live_dtype=keep_live_dtype)
    return revert_to_load_layout(model, {key: cast(key, tensor) for key, tensor in state_dict.items()})


def remove_stale_checkpoint_files(output_dir: str, keep: set[str]) -> None:
    """Delete ``model*.safetensors`` and the index outside ``keep``, since ``from_pretrained`` would load a leftover
    single file over a fresh index. A caller that only sometimes owns the directory must gate the call."""
    for stale in glob.glob(os.path.join(output_dir, SAFETENSORS_FAMILY_GLOB)) + glob.glob(
        os.path.join(output_dir, SAFETENSORS_INDEX_FILE)
    ):
        if os.path.basename(stale) not in keep:
            os.remove(stale)


def sweep_after_full_save(output_dir: str) -> None:
    """Remove the weight files a completed full save did not produce, keeping a consistent index and its shards, else a
    single file; a crashed save sweeps nothing."""
    index_path = os.path.join(output_dir, SAFETENSORS_INDEX_FILE)
    single_path = os.path.join(output_dir, SAFETENSORS_WEIGHTS_FILE)
    keep: set[str] = set()
    if os.path.isfile(index_path):
        try:
            index = read_checkpoint_index(output_dir)
        except (OSError, json.JSONDecodeError):
            index = {}
        # A per-rank EP/TP index is not a full save's product, so its shard list is no keep-set.
        shards = set() if index_declares_ep_shards(index) else set(index.get("weight_map", {}).values())
        # With both layouts the newer is this write's, or a single-file writer (reset_sinks) loses to stale shards.
        newer_than_single = not os.path.isfile(single_path) or os.path.getmtime(index_path) >= os.path.getmtime(
            single_path
        )
        if shards and newer_than_single and all(os.path.isfile(os.path.join(output_dir, s)) for s in shards):
            keep = shards | {SAFETENSORS_INDEX_FILE}
    if not keep and os.path.isfile(single_path):
        keep = {SAFETENSORS_WEIGHTS_FILE}
    if keep:
        remove_stale_checkpoint_files(output_dir, keep=keep)


def write_merged_index(output_dir: str, weight_map: dict[str, str], metadata: Mapping[str, Any]) -> None:
    """Sweep the model files ``weight_map`` does not claim, then write the index with ``metadata`` verbatim."""
    if not isinstance(metadata, Mapping):
        raise TypeError(
            f"write_merged_index metadata must be a mapping written verbatim into the index's "
            f"metadata block (e.g. {{'total_size': N}}), got {type(metadata).__name__}: {metadata!r}"
        )
    remove_stale_checkpoint_files(output_dir, set(weight_map.values()) | {SAFETENSORS_INDEX_FILE})
    index = {"metadata": dict(metadata), "weight_map": weight_map}
    with open(os.path.join(output_dir, SAFETENSORS_INDEX_FILE), "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)


def _unique_key_by_suffix(state_dict: dict, suffix: str) -> str | None:
    """The one key equal to ``suffix`` or ending in ``.{suffix}``; ``None`` when absent or ambiguous."""
    matches = [k for k in state_dict if k == suffix or k.endswith(f".{suffix}")]
    if len(matches) > 1:
        logger.warning(
            f"reconcile_tie_word_embeddings: multiple state-dict keys match '{suffix}' "
            f"({matches[:4]}); skipping the tie-consistency check for this save."
        )
        return None
    return matches[0] if matches else None


def is_tie_reconcile_key(name: str) -> bool:
    """Whether ``name`` is one of the pair :func:`reconcile_tie_word_embeddings` compares, kept by a streamed save."""
    return any(name == suffix or name.endswith(f".{suffix}") for suffix in _TIE_KEY_SUFFIXES)


def reconcile_tie_word_embeddings(model, state_dict: dict) -> None:
    """Clear ``tie_word_embeddings`` when the saved lm_head and embedding diverged (FSDP2 shards the pair independently
    and FLCE trains lm_head directly)."""
    config = getattr(model, "config", None)
    if config is None or not ties_word_embeddings(config):
        return
    lm_key = _unique_key_by_suffix(state_dict, _TIE_KEY_SUFFIXES[0])
    emb_key = _unique_key_by_suffix(state_dict, _TIE_KEY_SUFFIXES[1])
    if lm_key is None or emb_key is None:
        return
    lm, emb = state_dict[lm_key], state_dict[emb_key]
    if lm.shape != emb.shape or not torch.equal(lm, emb):
        untie_word_embeddings(config)
        logger.warning(
            f"{lm_key} and {emb_key} diverged during training; "
            f"saving config with tie_word_embeddings=False for checkpoint consistency."
        )


def _write_part(part: dict[str, torch.Tensor], path: str) -> None:
    """Empties the part before reporting done, so a flush waiting on it never holds it beside the next."""
    try:
        save_safetensors_file(part, path, metadata=SAFETENSORS_METADATA)
    finally:
        part.clear()


def resolve_checkpoint_write_threads() -> int:
    """``HALO_CHECKPOINT_WRITE_THREADS`` as set, before its ``>= 0`` check: the count a job's ranks agree on."""
    return env_int("HALO_CHECKPOINT_WRITE_THREADS", DEFAULT_CHECKPOINT_WRITE_THREADS)


def validate_checkpoint_write_threads(threads: int) -> int:
    """Parts a :class:`StreamingShardWriter` writes in the background (0 writes in line); a negative count raises."""
    if threads < 0:
        raise ValueError(
            f"HALO_CHECKPOINT_WRITE_THREADS={threads}: expected a thread count >= 0 (0 writes each checkpoint part "
            f"in line)."
        )
    return threads


def resolve_checkpoint_writers_per_node() -> int:
    """``HALO_CHECKPOINT_WRITERS_PER_NODE`` as set, before its ``>= 1`` check: the count a job's ranks agree on."""
    return env_int("HALO_CHECKPOINT_WRITERS_PER_NODE", DEFAULT_CHECKPOINT_WRITERS_PER_NODE)


def validate_checkpoint_writers_per_node(writers: int) -> int:
    """Ranks per node that write a gathered save on a shared filesystem; a count below 1 raises."""
    if writers < 1:
        raise ValueError(
            f"HALO_CHECKPOINT_WRITERS_PER_NODE={writers}: expected a writer count >= 1 (1 writes a gathered save from "
            f"each node's first rank)."
        )
    return writers


@dataclass(frozen=True)
class StreamedParts:
    """One writer's finished output: its part files in write order, every key's file (parts and files already in
    place) and the bytes."""

    part_names: tuple[str, ...]
    weight_map: dict[str, str]
    nbytes: int


def publish_hf_parts(output_dir: str, parts: StreamedParts) -> int:
    """Rename the parts to HF's names, then index them or, when one file holds everything, sweep to it; returns the
    bytes. Valid only when ``parts`` is every ``model*.safetensors`` in the directory, since it deletes the rest."""
    if not parts.part_names:
        raise RuntimeError(
            f"Refusing to finalize an empty checkpoint in {output_dir}: no writer staged a tensor, so there is "
            f"nothing to name and the existing files there would be deleted for nothing."
        )
    indexed = len(parts.part_names) > 1 or bool(set(parts.weight_map.values()) - set(parts.part_names))
    renamed = {
        part_name: hf_shard_filename(i, len(parts.part_names), indexed=indexed)
        for i, part_name in enumerate(parts.part_names, start=1)
    }
    for part_name, final in renamed.items():
        os.replace(os.path.join(output_dir, part_name), os.path.join(output_dir, final))
    if indexed:
        weight_map = {key: renamed.get(file, file) for key, file in parts.weight_map.items()}
        write_merged_index(output_dir, weight_map, {"total_size": parts.nbytes})
    else:
        remove_stale_checkpoint_files(output_dir, set(renamed.values()))
    return parts.nbytes


class StreamingShardWriter:
    """Flush tensors to numbered safetensors parts whenever the next would overflow ``max_shard_size``;
    ``enabled=False`` is a no-op for non-writer ranks, which still enter each collective gather. Entered as a context
    manager, finished parts go to background threads (``save_file`` releases the GIL), holding that many parts more; a
    failed write raises from the next flush or close, and the exit waits out every write and logs uncollected
    failures."""

    def __init__(
        self, output_dir: str, max_shard_size: str, *, enabled: bool, filename_prefix: str = STREAM_PART_PREFIX
    ):
        self.output_dir = output_dir
        self.filename_prefix = filename_prefix
        self.max_bytes = parse_size_to_int(max_shard_size)
        self.enabled = enabled
        self._pending: dict[str, torch.Tensor] = {}
        self._pending_bytes = 0
        self._part = 0
        self._weight_map: dict[str, str] = {}
        self._part_names: list[str] = []
        self._total_bytes = 0
        # Files already on disk under their final names, which the index claims as they are.
        self._written_map: dict[str, str] = {}
        self._written_bytes = 0
        # Read on every rank, disabled ones and a PP stage's never-entered writer included, so a refused count raises
        # before a save's first gather.
        self._configured_threads = validate_checkpoint_write_threads(resolve_checkpoint_write_threads())
        self._write_threads = 0
        self._executor: concurrent.futures.ThreadPoolExecutor | None = None
        self._writes: deque[concurrent.futures.Future] = deque()
        self._write_error: BaseException | None = None

    def __enter__(self) -> StreamingShardWriter:
        self._write_threads = self._configured_threads if self.enabled else 0
        if self._write_threads:
            self._executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=self._write_threads, thread_name_prefix="halo-shard-writer"
            )
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._executor is None:
            return
        try:
            if exc_type is None:
                self._settle_writes(0)
            else:
                # The caller's exception propagates; a write failing beside it, possibly its cause, is logged.
                concurrent.futures.wait(self._writes)
                for write in self._writes:
                    if (error := write.exception()) is not None:
                        logger.error(
                            f"A background part write to {self.output_dir} failed while the save was already "
                            f"raising {exc_type.__name__}: {error!r}",
                            exc_info=error,
                        )
                self._writes.clear()
        finally:
            self._executor.shutdown()
            self._executor = None
            self._write_threads = 0

    @property
    def max_resident_bytes(self) -> int:
        """Host bytes of staged parts held at once: the filling part plus one per background write."""
        return self.max_bytes * (1 + self._write_threads)

    def add(self, key: str, tensor: torch.Tensor) -> None:
        """Stage one tensor, flushing first on overflow (HF's rule). A key stages once, as the first may be flushed."""
        if not self.enabled:
            return
        if key in self._weight_map or key in self._pending or key in self._written_map:
            raise RuntimeError(
                f"{key!r} was staged twice for the checkpoint in {self.output_dir}: a streamed part "
                f"already claims it, so the second tensor cannot replace the first. Two chunks of "
                f"this save name the same tensor (a hand-sliced param emitted by the parameter walk "
                f"AND by its own gather is the known shape) — the writer refuses rather than write "
                f"the key twice at two shapes."
            )
        nbytes = tensor.nbytes
        if self._pending and self._pending_bytes + nbytes > self.max_bytes:
            self._flush()
        self._pending[key] = tensor.contiguous()
        self._pending_bytes += nbytes

    def add_written_files(self, weight_map: Mapping[str, str], nbytes: int) -> None:
        """Index files already in place under their final names (an external table's parts); never renamed or swept."""
        if not self.enabled:
            return
        claimed = set(weight_map) & (self._weight_map.keys() | self._pending.keys() | self._written_map.keys())
        if claimed:
            raise RuntimeError(
                f"{sorted(claimed)[:3]} are written twice to the checkpoint in {self.output_dir}: a file already on "
                f"disk and a staged tensor (or another file) claim the same key."
            )
        self._written_map.update(weight_map)
        self._written_bytes += nbytes

    def close_parts(self) -> StreamedParts:
        """Flush the remainder and return the finished parts, once every one is on disk."""
        if not self.enabled:
            return StreamedParts((), {}, 0)
        self._flush()
        self._settle_writes(0)
        return StreamedParts(
            tuple(self._part_names),
            {**self._weight_map, **self._written_map},
            self._total_bytes + self._written_bytes,
        )

    def close(self) -> tuple[dict[str, str], int]:
        """Flush the remainder and return ``(weight_map fragment, bytes written)``, before the index exchange."""
        parts = self.close_parts()
        return parts.weight_map, parts.nbytes

    def close_as_hf_checkpoint(self) -> int:
        """Flush, then :func:`publish_hf_parts`; returns the bytes written. Valid only when this writer owns every
        ``model*.safetensors`` in the directory."""
        return publish_hf_parts(self.output_dir, self.close_parts()) if self.enabled else 0

    def _flush(self) -> None:
        if not self._pending:
            return
        if self._executor is not None:
            self._settle_writes(self._write_threads - 1)
        self._part += 1
        name = stream_part_filename(self.filename_prefix, self._part)
        path = os.path.join(self.output_dir, name)
        if self._executor is None:
            save_safetensors_file(self._pending, path, metadata=SAFETENSORS_METADATA)
        else:
            # The thread gets a dict of its own; a device tensor's D2H runs here, in the caller's stream order.
            part = {key: tensor.cpu() for key, tensor in self._pending.items()}
            self._writes.append(self._executor.submit(_write_part, part, path))
        self._weight_map.update(dict.fromkeys(self._pending, name))
        self._part_names.append(name)
        self._total_bytes += self._pending_bytes
        self._pending = {}
        self._pending_bytes = 0

    def _settle_writes(self, in_flight: int) -> None:
        """Wait until at most ``in_flight`` writes remain; a failed write raises once others stop, and again later."""
        while self._writes and (len(self._writes) > in_flight or self._writes[0].done()):
            error = self._writes.popleft().exception()
            self._write_error = self._write_error or error
        if self._write_error is not None:
            concurrent.futures.wait(self._writes)
            self._writes.clear()
            raise self._write_error


def merge_shard_index(gathered: list[tuple[dict[str, str], int] | None]) -> tuple[dict[str, str], int]:
    """Fold the gathered fragments into one ``(weight_map, total_size)``, rejecting a key claimed by two files, which
    would load holding the wrong writer's tensor. Runs on rank 0 inside ``combine_across_ranks``, so the raise is
    world-uniform. Bytes count once per file, since on per-node storage each node's writer emits the same filenames."""
    weight_map: dict[str, str] = {}
    counted_files: set[str] = set()
    total_size = 0
    for entry in gathered:
        if not entry:
            continue
        fragment, fragment_bytes = entry
        for key, shard in fragment.items():
            if key in weight_map and weight_map[key] != shard:
                raise RuntimeError(
                    f"checkpoint shards disagree on {key!r}: claimed by both {weight_map[key]!r} and "
                    f"{shard!r}. Every key belongs to exactly one writer, so two claims mean the "
                    f"shard split or the global-name map is wrong."
                )
            weight_map[key] = shard
        files = set(fragment.values())
        if not files & counted_files:
            total_size += fragment_bytes
            counted_files |= files
    return weight_map, total_size


def merge_streamed_parts(gathered: list[StreamedParts | None]) -> StreamedParts:
    """Fold the writers' parts, in rank order, into one; a key two writers claim raises (see
    :func:`merge_shard_index`). Runs on rank 0 inside ``combine_across_ranks``."""
    written = [parts for parts in gathered if parts]
    weight_map, nbytes = merge_shard_index([(parts.weight_map, parts.nbytes) for parts in written])
    return StreamedParts(tuple(name for parts in written for name in parts.part_names), weight_map, nbytes)
