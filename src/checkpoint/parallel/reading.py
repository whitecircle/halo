"""The two reads a resume shares: a file the whole world must agree on (:func:`consensus_read`) and a
checkpoint's shards opened on every rank (:func:`joined_streaming_reader`).

Both join their verdict before any rank branches on it; the absent-everywhere policy stays with the
caller, because only it knows whether nothing-to-restore is legitimate.
"""

from __future__ import annotations

import logging
import os
from collections.abc import Callable, Iterable, Iterator
from contextlib import contextmanager
from functools import partial
from typing import Any

from src.checkpoint.reading import StreamingCheckpointReader, ep_sharded_checkpoint_reason, is_ep_sharded_checkpoint
from src.runtime.consensus import rank0_verdict, rank_consensus
from src.runtime.ranks import get_global_rank

logger = logging.getLogger(__name__)


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

    Returns ``(value, path)``, or ``(None, None)`` when the file is absent on EVERY rank, the one outcome
    the caller decides for itself. Present on a subset, or unreadable anywhere, raises on every rank;
    ``remedy`` appends the caller's way out. Of alternative ``paths`` (an adapter's safetensors or bin)
    the first present on every rank wins: a first-local pick lets two nodes read the same tensors in a
    different key order, and a per-key mesh collective then deadlocks.
    """
    candidates = (paths,) if isinstance(paths, str) else tuple(paths)
    presence = [rank_consensus(os.path.isfile(path)) for path in candidates]
    local_path = next((path for path, present in zip(candidates, presence, strict=True) if present.all_ok), None)
    if local_path is None:
        if any(present.any_ok for present in presence):
            raise RuntimeError(
                f"{what} present on some ranks but missing on others at {checkpoint} — torn/partial save on "
                f"a non-shared filesystem. Resume from a complete checkpoint.{remedy}"
            )
        return None, None

    value = None
    read_error: Exception | None = None
    try:
        value = read_fn(local_path)
    except Exception as e:
        read_error = e
        logger.warning(f"[rank {get_global_rank()}] Torn/unreadable {what} at {local_path}: {e}")
    if not rank_consensus(read_error is None).all_ok:
        raise RuntimeError(_torn_corrupt(what, checkpoint, remedy)) from read_error
    return value, local_path


@contextmanager
def joined_streaming_reader(checkpoint: str, keys: Iterable[str] | None, *, what: str) -> Iterator[Any]:
    """COLLECTIVE. Open the checkpoint's shards for ``keys`` (``None``: every key), agree every rank opened
    them, close on exit. Opening is what validates a torn shard, so the verdict is joined before the
    caller's own collectives; the reader then serves one tensor at a time."""
    reader: StreamingCheckpointReader | None = None
    read_error: Exception | None = None
    try:
        reader = StreamingCheckpointReader(checkpoint, keys)
    except Exception as e:
        read_error = e
        logger.warning(f"[rank {get_global_rank()}] Torn/unreadable {what} at {checkpoint}: {e}")
    try:
        if not rank_consensus(read_error is None).all_ok:
            raise RuntimeError(_torn_corrupt(what, checkpoint, "")) from read_error
        yield reader
    finally:
        if reader is not None:
            reader.close()


def reject_ep_sharded_resume(checkpoint: str) -> None:
    """COLLECTIVE (rank 0's probe). Refuse a resume from a per-rank EP-sharded save, whose intact files hold
    partial tensors."""
    if rank0_verdict(partial(is_ep_sharded_checkpoint, checkpoint), f"Probing {checkpoint} for per-rank EP shards"):
        raise ValueError(ep_sharded_checkpoint_reason(checkpoint))
