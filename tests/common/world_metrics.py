"""World-metric plumbing for trainer stand-ins and fake multi-rank folds.

A GRPO trainer method records into ``self._world_metrics`` and the step's flush folds it into
``self._metrics[mode]``; :func:`fake_world` makes several accumulators flush as the ranks of one world. The fake
folds the ranks' states by :func:`pooled_states` in place of the collectives, under the flush's own key checks;
``tests/cpu/trainers/test_world_metrics.py`` checks it against real gloo ranks.
"""

from collections.abc import Sequence
from contextlib import contextmanager
from unittest.mock import patch

from halo.trainers import world_metrics
from halo.trainers.world_metrics import MismatchErrorFactory, State, WorldMetrics


def attach_world_metrics(host):
    """Give ``host`` the accumulator the trainer methods record into; returns ``host``."""
    host._world_metrics = WorldMetrics()
    return host


def flushed_metrics(host, mode: str = "train") -> dict[str, list[float]]:
    """``host._metrics[mode]`` after a single-process flush of the pending world metrics."""
    host._world_metrics.flush(host._metrics[mode])
    return host._metrics[mode]


def pooled_states(
    payloads: Sequence[dict[str, tuple[str, State]]], key_mismatch_error: MismatchErrorFactory | None = None
) -> dict[str, tuple[str, State]]:
    """The world states of ranks holding ``payloads`` (each a rank's ``_local_states()``): every key's states folded
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
    payloads = [rank._local_states() for rank in ranks]
    with patch.object(
        world_metrics,
        "_world_states",
        lambda states, group, key_mismatch_error: pooled_states(payloads, key_mismatch_error),
    ):
        yield
