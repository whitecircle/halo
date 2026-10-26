"""World-metric plumbing for trainer stand-ins and fake multi-rank folds.

A GRPO trainer method records into ``self._world_metrics`` and the step's flush folds it into
``self._metrics[mode]``; :func:`fake_world` makes several accumulators flush as the ranks of one world.
"""

from contextlib import contextmanager
from unittest.mock import patch

from halo.trainers import world_metrics
from halo.trainers.world_metrics import WorldMetrics


def attach_world_metrics(host):
    """Give ``host`` the accumulator the trainer methods record into; returns ``host``."""
    host._world_metrics = WorldMetrics()
    return host


def flushed_metrics(host, mode: str = "train") -> dict[str, list[float]]:
    """``host._metrics[mode]`` after a single-process flush of the pending world metrics."""
    host._world_metrics.flush(host._metrics[mode])
    return host._metrics[mode]


@contextmanager
def fake_world(*ranks: WorldMetrics):
    """Inside the block, a flush of any of ``ranks`` gathers all of them, in order, as the world's ranks.

    Their records are read on entry: record first, then flush each rank inside the block.
    """
    payloads = [rank._local_states() for rank in ranks]
    with patch.object(world_metrics, "_gathered", lambda payload, group: payloads):
        yield
