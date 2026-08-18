"""Rank consensus shared by the halves of a resume — :mod:`.loader` (weights), :mod:`.optimizer`
(optimizer + scheduler), :mod:`.peft` (adapters) and the trainer mixin's sidecars.

Every one of them reads a file the whole world must agree on, behind branches that must be taken
identically on every rank, and all of them truncate the key lists their diagnostics print at the
same cap. :func:`consensus_read` and :func:`joined_streaming_reader` are those two reads; the
absent-everywhere policy stays with the caller, because only it knows whether nothing-to-restore is
legitimate.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from typing import Any

from src.checkpoint.format import StreamingCheckpointReader
from src.distributed.runtime import get_global_rank, rank_consensus
from src.log import KEY_PREVIEW_COUNT

logger = logging.getLogger(__name__)

# Re-exported so both halves of a resume reach the shared cap through the module they already import.
__all__ = ["KEY_PREVIEW_COUNT", "all_ranks_ok", "consensus_read", "joined_streaming_reader"]


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
def joined_streaming_reader(checkpoint: str, keys: Iterable[str], *, what: str) -> Iterator[Any]:
    """COLLECTIVE. Open the checkpoint's shards for ``keys``, agree every rank opened them, close on exit.

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
