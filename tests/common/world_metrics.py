"""World-metric plumbing for trainer stand-ins and fake multi-rank folds.

A GRPO trainer method records into ``self._world_metrics`` and the step's flush folds it into
``self._metrics[mode]``; :func:`fake_world` makes several accumulators flush as the ranks of one world. The fake
folds each rank's records on the host (:func:`rank_states`) and the ranks' states by :func:`pooled_states` in place
of the collectives, under the flush's own key checks; ``tests/cpu/trainers/test_world_metrics.py`` checks it
against real gloo ranks.
"""

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
    """A rank's records folded on the host: each state column over the records in order with Python's ``sum``,
    ``max`` or ``min`` (a percentile's values concatenated), every tensor read as a float64. The flush folds a
    tensor's column on its device to the same bits."""
    return {
        key: (kind, world_metrics._fold(kind, [tuple(_as_float(value) for value in row) for row in rows]))
        for key, (kind, rows) in records.items()
    }


def _as_float(value) -> float:
    return float(value.double()) if torch.is_tensor(value) else float(value)


def pooled_states(
    payloads: Sequence[dict[str, tuple[str, State]]], key_mismatch_error: MismatchErrorFactory | None = None
) -> dict[str, tuple[str, State]]:
    """The world states of ranks holding ``payloads`` (each a rank's :func:`rank_states`): every key's states folded
    over the ranks that hold it, keys sorted. Raises as the flush does on a key held as two kinds and, with
    ``key_mismatch_error``, on ranks holding records under different keys."""
    schema = world_metrics._union_schema([world_metrics._schema(payload) for payload in payloads], key_mismatch_error)
    return {
        key: (kind, world_metrics._fold(kind, [payload[key][1] for payload in payloads if key in payload]))
        for key, kind in schema
    }


@contextmanager
def fake_world(*ranks: WorldMetrics):
    """Inside the block, a flush of any of ``ranks`` folds all of them, in order, as the world's ranks.

    Their records are read on entry: record first, then flush each rank inside the block.
    """
    payloads = [rank_states(rank._records) for rank in ranks]
    with patch.object(
        world_metrics,
        "_world_states",
        lambda records, layout, group, key_mismatch_error: (pooled_states(payloads, key_mismatch_error), layout),
    ):
        yield
