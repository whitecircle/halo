"""Batch-metric accumulation across micro-batches. Mix in BEFORE ``DistributedTrainerMixin`` so its ``log`` flushes
before ``super().log``."""

from collections import defaultdict
from functools import cached_property
from typing import Literal

import torch

from halo.trainers.metrics.world_metrics import WorldMetrics


class StoredMetricsMixin:
    @cached_property
    def _stored_metrics(self) -> defaultdict[str, WorldMetrics]:
        return defaultdict(WorldMetrics)

    def store_metrics(
        self,
        metrics: dict[str, float | torch.Tensor],
        train_eval: Literal["train", "eval"] = "train",
        rows: int | torch.Tensor = 1,
    ) -> None:
        """Each value is a mean over this rank's ``rows`` real rows (eval final-round padding excluded; ``rows=0`` adds
        nothing). The key set must not depend on the batch, since ``log`` reduces every name over the world."""
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
        self.store_metrics(metrics, train_eval=train_eval, rows=real_rows if train_eval == "eval" else 1)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """COLLECTIVE. Fold the world means into ``logs``. Ranks that stored metrics must store the same names (a rank
        that ran no batch of this mode abstains); a key whose rows sum to zero is not logged."""
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
