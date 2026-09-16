"""World-metric plumbing for GRPO trainer stand-ins.

A trainer method records its batch-level fractions into ``self._world_metrics`` and the step's
flush folds them into ``self._metrics[mode]``; a stand-in that drives one such method needs the
accumulator and a single-process flush to read the metric the way TRL's ``log`` would.
"""

from src.trainers.grpo.rollout.rollout_metrics import WorldMetrics


def attach_world_metrics(host):
    """Give ``host`` the accumulator the trainer methods record into; returns ``host``."""
    host._world_metrics = WorldMetrics()
    return host


def flushed_metrics(host, mode: str = "train") -> dict[str, list[float]]:
    """``host._metrics[mode]`` after a single-process flush of the pending world metrics."""
    host._world_metrics.flush(host._metrics[mode], gather_fn=list)
    return host._metrics[mode]
