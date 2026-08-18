"""Batch-metric accumulation for trainers that average metrics across micro-batches.

Accumulate per-step metrics in ``store_metrics`` and flush their mean in ``log``. Mixed in before
``DistributedTrainerMixin`` so its ``log`` flushes ahead of ``super().log``.
"""

from collections import defaultdict
from typing import Literal

import torch


class StoredMetricsMixin:
    """``store_metrics`` + a ``log`` that averages and flushes them. Backing dict is created lazily."""

    @property
    def _stored_metrics(self) -> dict:
        cached = self.__dict__.get("_stored_metrics_cache")
        if cached is None:
            cached = defaultdict(lambda: defaultdict(list))
            self.__dict__["_stored_metrics_cache"] = cached
        return cached

    def store_metrics(
        self, metrics: dict[str, float | torch.Tensor], train_eval: Literal["train", "eval"] = "train"
    ) -> None:
        """Accumulate metrics for later mean-aggregation at ``log`` time.

        Values may be floats or detached 0-dim tensors; tensors stay on device until ``log`` drains
        them, so storing one adds no per-step host sync.
        """
        for key, value in metrics.items():
            self._stored_metrics[train_eval][key].append(value)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Average the stored batch metrics into ``logs``, then delegate up the MRO.

        Eval-bucket keys gain an ``eval_`` prefix (TRL convention) unless already prefixed; without
        it, eval flushes land on the same series as the train metrics.
        """
        train_eval = "train" if "loss" in logs else "eval"
        prefix = "eval_" if train_eval == "eval" else ""
        for key, values in self._stored_metrics[train_eval].items():
            name = key if key.startswith("eval_") else f"{prefix}{key}"
            logs[name] = torch.stack([torch.as_tensor(v, dtype=torch.float32) for v in values]).mean().item()
        self._stored_metrics[train_eval].clear()
        return super().log(logs, start_time)
