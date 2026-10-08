"""Rank consensus shared by a resume: the pick of the checkpoint (:func:`resolve_resume_checkpoint`),
then its halves — :mod:`.loader` (weights), :mod:`.optimizer` (optimizer + scheduler), :mod:`.peft`
(adapters) and the trainer mixin's sidecars.

Every one of them reads a file the whole world must agree on, behind branches that must be taken
identically on every rank. :func:`consensus_read` and :func:`joined_streaming_reader` are those two
reads; the absent-everywhere policy stays with the caller, because only it knows whether
nothing-to-restore is legitimate.
"""

from __future__ import annotations

import logging
import os
import re
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any

import torch.distributed as dist
from transformers.trainer import TRAINER_STATE_NAME
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from src.checkpoint.atomic import fsync_directory
from src.checkpoint.format import INCOMPLETE_CHECKPOINTS_DIR_NAME, StreamingCheckpointReader
from src.distributed.runtime import (
    DeferredRankFailure,
    broadcast_from_rank0,
    fs_aware_save_rank,
    get_global_rank,
    is_global_main_process,
    is_multi_rank_run,
    rank_consensus,
)

logger = logging.getLogger(__name__)

# The step directories HF's own ``get_last_checkpoint`` recognizes, and no others.
CHECKPOINT_DIR_RE = re.compile(rf"^{PREFIX_CHECKPOINT_DIR}-(\d+)$")


def all_ranks_ok(local_ok: bool) -> bool:
    """True when every rank succeeded — see :func:`rank_consensus`."""
    return rank_consensus(local_ok)[0]


def _torn_partial(what: str, checkpoint: str, remedy: str) -> str:
    return (
        f"{what} present on some ranks but missing on others at {checkpoint} — torn/partial save on "
        f"a non-shared filesystem. Resume from a complete checkpoint.{remedy}"
    )


def _torn_corrupt(what: str, checkpoint: str, remedy: str) -> str:
    return (
        f"{what} unreadable on at least one rank at {checkpoint} — torn/corrupt file on a non-shared "
        f"filesystem. Resume from a complete checkpoint.{remedy}"
    )


def consensus_read(
    paths: str | Iterable[str],
    read_fn: Callable[[str], Any],
    *,
    what: str,
    checkpoint: str,
    remedy: str = "",
) -> tuple[Any, str | None]:
    """COLLECTIVE, twice. Agree a resume input is on every rank, then agree every rank read it.

    Returns ``(value, path)``, or ``(None, None)`` when the file is absent on EVERY rank — the one
    outcome the caller decides for itself (a legitimate warm restart, an adapter-less run, nothing
    to restore). Present on a subset, or unreadable anywhere, raises the two diagnostics every
    caller shares; ``remedy`` appends the caller's own way out.

    ``paths`` may name alternatives (an adapter's safetensors or bin); the first candidate present on
    EVERY rank wins. The choice is a world fact, not a local one: a first-locally-present pick lets
    one node read ``adapter_model.bin`` while another reads the safetensors — same tensors, different
    key ORDER — and a loader whose per-key work is a mesh collective then issues them in different
    orders on different ranks, deadlocking under the collective's name rather than the file's.

    Both consensuses are entered unconditionally, never on the right of a short-circuiting ``and``:
    a collective placed there is entered by a subset of the world the day the test on its left stops
    being rank-uniform, which is a hang instead of a diagnostic. The read itself is rank-local and
    its failure is carried to the join rather than raised, so one torn copy cannot strand its peers.
    """
    # One consensus per candidate, in the caller's fixed (rank-uniform) order — the joined verdicts
    # are what makes the pick below identical on every rank.
    candidates = (paths,) if isinstance(paths, str) else tuple(paths)
    presence = [rank_consensus(os.path.isfile(path)) for path in candidates]
    local_path = next((path for path, (present_all, _) in zip(candidates, presence, strict=True) if present_all), None)
    if local_path is None:
        if any(present_any for _present_all, present_any in presence):
            raise RuntimeError(_torn_partial(what, checkpoint, remedy))
        return None, None

    value = None
    read_error: Exception | None = None
    try:
        value = read_fn(local_path)
    except Exception as e:
        read_error = e
        # The FAILING rank logs, which is rarely rank 0.
        logger.warning(f"[rank {get_global_rank()}] Torn/unreadable {what} at {local_path}: {e}")
    if not all_ranks_ok(read_error is None):
        raise RuntimeError(_torn_corrupt(what, checkpoint, remedy)) from read_error
    return value, local_path


@contextmanager
def joined_streaming_reader(checkpoint: str, keys: Iterable[str] | None, *, what: str) -> Iterator[Any]:
    """COLLECTIVE. Open the checkpoint's shards for ``keys`` (``None``: every key), agree every rank opened
    them, close on exit.

    Opening is what validates: a truncated or torn shard raises inside the constructor, HERE, before
    the caller's own collectives — so the verdict is joined before anything downstream can strand a
    peer. The reader then serves one tensor at a time, which is what keeps a stage-sized resume off
    the host's memory.
    """
    reader: StreamingCheckpointReader | None = None
    read_error: Exception | None = None
    try:
        reader = StreamingCheckpointReader(checkpoint, keys)
    except Exception as e:
        read_error = e
        logger.warning(f"[rank {get_global_rank()}] Torn/unreadable {what} at {checkpoint}: {e}")
    try:
        if not all_ranks_ok(read_error is None):
            raise RuntimeError(_torn_corrupt(what, checkpoint, "")) from read_error
        yield reader
    finally:
        if reader is not None:
            reader.close()


def resolve_resume_checkpoint(output_dir: str, requested: str | None = None) -> str | None:
    """COLLECTIVE. The checkpoint a run resumes from, the same on every rank.

    ``requested`` must be complete on every rank; ``None`` picks the newest step directory of
    ``output_dir`` complete on every rank, or returns ``None`` when it holds none. Complete means
    ``trainer_state.json`` is present, which a save publishes last; the step directories are those any
    FS-aware save rank lists, so a copy one node lacks is incomplete. Raises, moving nothing, for a
    missing or incomplete ``requested`` and for step directories none of which is complete.

    Every incomplete step directory is then moved into :data:`INCOMPLETE_CHECKPOINTS_DIR_NAME` on each
    save rank that holds a copy. Left among the ``checkpoint-<N>`` directories, one numbered above this
    run's next saves is the newest by step, the order rotation falls back to on a mount whose mtimes it
    distrusts; it would then keep that one and delete the complete checkpoints.
    """
    names = _world_step_directories(output_dir, requested)
    candidates = [os.path.join(output_dir, name) for name in names]
    complete = [_complete_on_every_rank(path) for path in candidates]
    if requested is not None:
        checkpoint = requested if _complete_on_every_rank(requested) else None
        if checkpoint is None:
            raise RuntimeError(_incomplete_reason([requested]))
    else:
        checkpoint = next((path for path, ok in zip(candidates, complete, strict=True) if ok), None)
        if checkpoint is None and candidates:
            raise RuntimeError(_incomplete_reason(candidates))
    incomplete = [path for path, ok in zip(candidates, complete, strict=True) if not ok]
    if incomplete:
        _set_aside_incomplete(output_dir, incomplete)
    if is_global_main_process():
        if incomplete:
            logger.warning(
                f"Moved {[os.path.basename(path) for path in incomplete]} into "
                f"{os.path.join(output_dir, INCOMPLETE_CHECKPOINTS_DIR_NAME)}: {TRAINER_STATE_NAME} is missing "
                f"on at least one node, so their saves never completed."
            )
        if checkpoint is not None:
            logger.info(f"Resuming from checkpoint: {checkpoint}")
    return checkpoint


def _complete_on_every_rank(checkpoint: str) -> bool:
    """COLLECTIVE. On a non-shared FS each node reads its own copy, so the verdict is a world one."""
    return rank_consensus(os.path.isfile(os.path.join(checkpoint, TRAINER_STATE_NAME)))[0]


def _incomplete_reason(checkpoints: list[str]) -> str:
    return (
        f"Resume checkpoint(s) {checkpoints} are incomplete on at least one node: {TRAINER_STATE_NAME}, "
        f"which a save publishes last, is missing — a save that stopped partway, or a node whose copy is "
        f"incomplete. Resume from an earlier complete checkpoint or remove the incomplete one."
    )


def _local_step_directories(output_dir: str) -> list[str]:
    if not os.path.isdir(output_dir):
        return []
    return [
        name
        for name in os.listdir(output_dir)
        if CHECKPOINT_DIR_RE.match(name) and os.path.isdir(os.path.join(output_dir, name))
    ]


def _world_step_directories(output_dir: str, requested: str | None) -> list[str]:
    """COLLECTIVE. The step directory names any FS-aware save rank lists, newest first.

    Rank 0 merges them and checks ``requested`` exists; a listing or that check failing raises the
    same exception on every rank.
    """
    local: list[str] | Exception = []
    if fs_aware_save_rank():
        try:
            local = _local_step_directories(output_dir)
        except OSError as e:
            local = e
    if is_multi_rank_run():
        gathered = [None] * dist.get_world_size() if is_global_main_process() else None
        dist.gather_object(local, gathered, dst=0)
    else:
        gathered = [local]
    merged: list[str] | Exception | None = None
    if is_global_main_process():
        merged = next((item for item in gathered if isinstance(item, Exception)), None)
        if merged is None and requested is not None and not os.path.isdir(requested):
            merged = ValueError(
                f"resume_from_checkpoint path does not exist: '{requested}'. Fix the path, or set "
                f"resume_from_checkpoint: true to auto-detect the last checkpoint in output_dir (or remove "
                f"it to start from scratch)."
            )
        if merged is None:
            names = {name for item in gathered for name in item}
            merged = sorted(names, key=lambda name: int(CHECKPOINT_DIR_RE.match(name).group(1)), reverse=True)
    merged = broadcast_from_rank0(merged)
    if isinstance(merged, Exception):
        raise merged
    return merged


def _set_aside_incomplete(output_dir: str, checkpoints: list[str]) -> None:
    """COLLECTIVE. Move each save rank's copies of ``checkpoints`` out of the step-directory namespace."""
    guard = DeferredRankFailure(f"Moving the incomplete checkpoints out of {output_dir}")
    if fs_aware_save_rank():
        guard.run(partial(_move_into_holding, output_dir, checkpoints))
    guard.reject()


def _move_into_holding(output_dir: str, checkpoints: list[str]) -> None:
    held = [path for path in checkpoints if os.path.isdir(path)]
    if not held:
        return
    holding = os.path.join(output_dir, INCOMPLETE_CHECKPOINTS_DIR_NAME)
    os.makedirs(holding, exist_ok=True)
    for path in held:
        name = os.path.basename(path)
        target, copy = os.path.join(holding, name), 1
        while os.path.exists(target):
            target, copy = os.path.join(holding, f"{name}.{copy}"), copy + 1
        os.rename(path, target)
    fsync_directory(output_dir)
