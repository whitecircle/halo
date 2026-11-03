"""World-metric plumbing for trainer stand-ins and fake multi-rank folds; :func:`fake_world` makes accumulators flush
as one world's ranks."""

from collections.abc import Sequence
from contextlib import contextmanager
from unittest.mock import patch

import torch

from halo.trainers import world_metrics
from halo.trainers.world_metrics import MismatchErrorFactory, Records, State, WorldMetrics


def attach_world_metrics(host):
    """Give ``host`` the accumulator the trainer methods record into; returns ``host``."""
    host._world_metrics = WorldMetrics()
    return host


def flushed_metrics(host, mode: str = "train") -> dict[str, list[float]]:
    """``host._metrics[mode]`` after a single-process flush of the pending world metrics."""
    host._world_metrics.flush(host._metrics[mode])
    return host._metrics[mode]


def rank_states(records: Records) -> dict[str, tuple[str, State]]:
    """A rank's records folded on the host in float64; the flush folds a tensor's column on its device to the same
    bits."""
    return {
        key: (kind, world_metrics._fold(kind, [tuple(_as_float(value) for value in row) for row in rows]))
        for key, (kind, rows) in records.items()
    }


def _as_float(value) -> float:
    return float(value.double()) if torch.is_tensor(value) else float(value)


def pooled_states(
    payloads: Sequence[dict[str, tuple[str, State]]], key_mismatch_error: MismatchErrorFactory | None = None
) -> dict[str, tuple[str, State]]:
    """World states of ``payloads``: each key folded over the ranks holding it. Raises as the flush does on mixed kinds
    and, with ``key_mismatch_error``, on differing keys."""
    schema = world_metrics._union_schema([world_metrics._schema(payload) for payload in payloads], key_mismatch_error)
    return {
        key: (kind, world_metrics._fold(kind, [payload[key][1] for payload in payloads if key in payload]))
        for key, kind in schema
    }


@contextmanager
def fake_world(*ranks: WorldMetrics):
    """Inside the block, a flush of any of ``ranks`` folds all of them in order. Records are read on entry: record
    first, then flush inside."""
    payloads = [rank_states(rank._records) for rank in ranks]
    with patch.object(
        world_metrics,
        "_world_states",
        lambda records, layout, group, key_mismatch_error: (pooled_states(payloads, key_mismatch_error), layout),
    ):
        yield
