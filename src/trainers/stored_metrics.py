"""Batch-metric accumulation for trainers that average metrics across micro-batches.

Accumulate per-step metrics in ``store_metrics`` and flush their row-weighted world mean in ``log``.
Mixed in BEFORE ``DistributedTrainerMixin`` so its ``log`` flushes before ``super().log``.
"""

from collections import defaultdict
from functools import cached_property
from typing import Literal

import torch

from src.trainers.world_metrics import WorldMetrics


class StoredMetricsMixin:
    """``store_metrics`` + a ``log`` that folds them across ranks, one lazily created :class:`WorldMetrics` per mode."""

    @cached_property
    def _stored_metrics(self) -> defaultdict[str, WorldMetrics]:
        return defaultdict(WorldMetrics)

    def store_metrics(
        self,
        metrics: dict[str, float | torch.Tensor],
        train_eval: Literal["train", "eval"] = "train",
        rows: int | torch.Tensor = 1,
    ) -> None:
        """Accumulate metrics for the row-weighted world mean ``log`` reports.

        Each value is a mean over this rank's ``rows`` real rows (an eval split's final-round padding
        excluded; ``rows=0`` adds nothing), as a float or a detached 0-dim tensor kept on device until
        ``log``. The key set must not depend on the batch: ``log`` reduces every name over the world.
        """
        if torch.as_tensor(rows).numel() != 1:
            raise ValueError(f"store_metrics rows must be one count, got shape {tuple(torch.as_tensor(rows).shape)}")
        for key, value in metrics.items():
            if torch.is_tensor(value) and value.numel() != 1:
                raise ValueError(
                    f"store_metrics value {key!r} has shape {tuple(value.shape)}: store a scalar (a mean over the "
                    f"batch's rows), not a per-row tensor."
                )
            self._stored_metrics[train_eval].mean(key, value, rows)

    def store_batch_metrics(
        self, metrics: dict[str, float | torch.Tensor], train_eval: Literal["train", "eval"], real_rows: int
    ) -> None:
        """:meth:`store_metrics` for one loss batch whose split's own rows are ``real_rows``
        (:meth:`eval_split_rows`): an eval batch weighs those rows, a train micro-batch weighs 1."""
        self.store_metrics(metrics, train_eval=train_eval, rows=real_rows if train_eval == "eval" else 1)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """COLLECTIVE. Fold the world means of the stored batch metrics into ``logs``, then delegate up.

        One :meth:`WorldMetrics.flush` that refuses a key mismatch: ranks that stored metrics must store the same
        names (a rank that ran no batch of this mode abstains), and a key whose rows sum to zero is not logged.
        Eval keys gain TRL's ``eval_`` prefix unless already prefixed.
        """
        train_eval = "train" if "loss" in logs else "eval"
        prefix = "eval_" if train_eval == "eval" else ""
        means = self._stored_metrics[train_eval].flush(
            key_mismatch_error=lambda keys_by_rank: RuntimeError(
                f"Stored {train_eval} metrics differ across ranks: {keys_by_rank}. store_metrics must name its "
                f"metrics from configuration, never from the batch, because log reduces each name over the world."
            )
        )
        for key, mean in means.items():
            logs[key if key.startswith("eval_") else f"{prefix}{key}"] = mean
        return super().log(logs, start_time)
