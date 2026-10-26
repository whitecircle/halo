"""Opt-in check that a value is identical on every rank of a process group, for divergence triage.

Manual instrumentation: the toolkit ships no call sites, so ``HALO_TP_CONSISTENCY_CHECK=1`` (raise
instead of warn) acts only once a call is added to the code under investigation. Production agreement
goes through :func:`src.distributed.runtime.agree_across_ranks`.
"""

from __future__ import annotations

import hashlib
import logging
import pickle
from typing import Any

import torch
import torch.distributed as dist

from src.env import env_flag

logger = logging.getLogger(__name__)


def _hash_value(value: Any) -> str:
    """Stable hash: tensors as ``(shape, dtype)`` + SHA256 of raw bytes; containers element-wise;
    else the pickle.

    Tensor bytes go through a ``torch.uint8`` view rather than ``.numpy()``: bf16/fp8 have no numpy
    equivalent, and a ``TypeError`` on one rank inside an ``all_gather_object`` would hang its peers.
    """
    hasher = hashlib.sha256()
    if isinstance(value, torch.Tensor):
        hasher.update(repr(value.shape).encode())
        hasher.update(repr(value.dtype).encode())
        raw = value.detach().to("cpu").contiguous().view(-1)
        if raw.numel():
            hasher.update(raw.view(torch.uint8).numpy().tobytes())
    elif isinstance(value, (list, tuple)):
        # Recurse over every element; the type/length prefix separates [x] from [[x]].
        hasher.update(f"{type(value).__name__}:{len(value)}".encode())
        for item in value:
            hasher.update(_hash_value(item).encode())
    else:
        hasher.update(pickle.dumps(value, protocol=4))
    return hasher.hexdigest()


def assert_rank_uniform(
    value: Any,
    *,
    group: dist.ProcessGroup | None,
    label: str,
    strict: bool | None = None,
) -> None:
    """Warn, or raise under ``strict``, when ``value`` differs across the ranks of ``group``.

    COLLECTIVE over ``group``: every rank gathers every rank's digest. Pass ``tuple(t.shape)`` to
    compare only a tensor's shape, the usual first sign of a TP/EP collective deadlock. ``group=None``
    or a one-rank group is a no-op; ``strict`` defaults to ``HALO_TP_CONSISTENCY_CHECK``.
    """
    if group is None:
        return
    world = dist.get_world_size(group=group)
    if world <= 1:
        return
    if strict is None:
        strict = env_flag("HALO_TP_CONSISTENCY_CHECK")

    rank = dist.get_rank(group=group)
    digest = _hash_value(value)
    digests: list[str | None] = [None] * world
    dist.all_gather_object(digests, digest, group=group)

    distinct = {d for d in digests if d is not None}
    if len(distinct) <= 1:
        return

    summary = ", ".join(f"rank{i}={d[:12] if d else '<missing>'}" for i, d in enumerate(digests))
    message = f"[assert_rank_uniform:{label}] divergent values across ranks: {summary}"
    if strict:
        raise RuntimeError(message + f" (rank {rank}: digest={digest[:12]})")
    logger.warning(message)
