"""Batch-metric accumulation for trainers that average metrics across micro-batches.

Accumulate per-step metrics in ``store_metrics`` and flush their row-weighted world mean in ``log``.
Mixed in BEFORE ``DistributedTrainerMixin`` so its ``log`` flushes before ``super().log``.
"""

from collections import defaultdict
from typing import Literal

import torch
import torch.distributed as dist

from src.distributed.runtime import agree_across_ranks, collective_device, is_multi_rank_run


def agreed_metric_keys(keys: list[str], train_eval: str) -> list[str]:
    """The key set every rank that stored metrics agrees on. COLLECTIVE.

    One :func:`agree_across_ranks` all-reduce on the common step. A rank that stored nothing (it ran
    no batch of this mode: ``even_batches`` off, a sharded streaming split) holds zero rows, not
    another key set: it abstains, then learns the keys from the first rank holding them, so it can
    contribute zero sums of the right width. Two ranks that stored different sets raise on EVERY
    rank: the reduce that follows sums one fixed-width row per key, so one rank's sums would be
    paired with a peer's names, or hang on a width mismatch.
    """
    agreement = agree_across_ranks(keys, abstain=not keys)
    if not agreement.agreed:
        gathered: list[list[str] | None] = [None] * dist.get_world_size()
        dist.all_gather_object(gathered, keys)
        stored = {rank: rank_keys for rank, rank_keys in enumerate(gathered) if rank_keys}
        raise RuntimeError(
            f"Stored {train_eval} metrics differ across ranks: {stored}. store_metrics must name its "
            f"metrics from configuration, never from the batch, because log reduces each name over the world."
        )
    if agreement.first_present is None:
        return []
    if not agreement.any_abstained:
        return keys
    shared = [keys]
    dist.broadcast_object_list(shared, src=agreement.first_present)
    return shared[0]


def _weighted_sums(entries: list[tuple[float | torch.Tensor, int | torch.Tensor]], device) -> torch.Tensor:
    """``(Σ value·rows, Σ rows)`` over one key's entries; an entry of no rows adds nothing, NaN included."""
    if not entries:
        return torch.zeros(2, dtype=torch.float32, device=device)
    values = torch.stack(
        [torch.as_tensor(value, dtype=torch.float32, device=device).reshape(()) for value, _ in entries]
    )
    rows = torch.stack(
        [torch.as_tensor(count, dtype=torch.float32, device=device).reshape(()) for _, count in entries]
    )
    return torch.stack([torch.where(rows > 0, values * rows, 0.0).sum(), rows.sum()])


class StoredMetricsMixin:
    """``store_metrics`` + a ``log`` that reduces and flushes them. Backing dict is created lazily."""

    @property
    def _stored_metrics(self) -> dict:
        cached = self.__dict__.get("_stored_metrics_cache")
        if cached is None:
            cached = defaultdict(lambda: defaultdict(list))
            self.__dict__["_stored_metrics_cache"] = cached
        return cached

    def store_metrics(
        self,
        metrics: dict[str, float | torch.Tensor],
        train_eval: Literal["train", "eval"] = "train",
        rows: int | torch.Tensor = 1,
    ) -> None:
        """Accumulate metrics for the row-weighted world mean ``log`` reports.

        Each value is a mean over ``rows`` of this rank's rows, and ``rows`` is its weight in that
        mean. A batch carrying rows that are not the split's own — an eval split's final-round
        padding (:meth:`eval_split_rows`) — computes its values over the real rows and passes their
        count; a batch of padding alone passes ``rows=0`` and adds nothing. The key set must not
        depend on the batch: ``log`` reduces every name over the world. Values may be floats or
        detached 0-dim tensors; tensors stay on device until ``log`` drains them, so storing one
        adds no per-step host sync. A value or ``rows`` of more than one element raises: weighted, it
        would broadcast and log a sum.
        """
        if torch.as_tensor(rows).numel() != 1:
            raise ValueError(f"store_metrics rows must be one count, got shape {tuple(torch.as_tensor(rows).shape)}")
        for key, value in metrics.items():
            if torch.is_tensor(value) and value.numel() != 1:
                raise ValueError(
                    f"store_metrics value {key!r} has shape {tuple(value.shape)}: store a scalar (a mean over the "
                    f"batch's rows), not a per-row tensor."
                )
            self._stored_metrics[train_eval][key].append((value, rows))

    def store_batch_metrics(
        self, metrics: dict[str, float | torch.Tensor], train_eval: Literal["train", "eval"], real_rows: int
    ) -> None:
        """:meth:`store_metrics` for one loss batch whose split's own rows are ``real_rows``
        (:meth:`eval_split_rows`): an eval batch weighs those rows, a train micro-batch weighs 1."""
        self.store_metrics(metrics, train_eval=train_eval, rows=real_rows if train_eval == "eval" else 1)

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        """Fold the world means of the stored batch metrics into ``logs``, then delegate up the MRO.

        Collective: ``log`` runs on every rank, and the bucket it drains is rank-local — a rank's own
        micro-batches, which a log of rank 0's bucket alone would pass off as the run's. The key sets
        are agreed first (:func:`agreed_metric_keys`; a rank that stored nothing weighs 0), then ONE
        fixed-width all-reduce of ``(Σ value·rows, Σ rows)`` per sorted key yields every mean at
        once. TP/CP/PP siblings store the same rows alike, so they weight their replica equally. A
        key whose rows sum to zero world-wide is not logged.

        Eval-bucket keys gain an ``eval_`` prefix (TRL convention) unless already prefixed; without
        it, eval flushes land on the same series as the train metrics.
        """
        train_eval = "train" if "loss" in logs else "eval"
        prefix = "eval_" if train_eval == "eval" else ""
        for key, mean in self._drain_stored_metrics(train_eval).items():
            logs[key if key.startswith("eval_") else f"{prefix}{key}"] = mean
        return super().log(logs, start_time)

    def _drain_stored_metrics(self, train_eval: str) -> dict[str, float]:
        """Reduce and clear one mode's bucket: ``{key: world mean}``. Collective on a multi-rank run."""
        bucket = self._stored_metrics[train_eval]
        keys = agreed_metric_keys(sorted(bucket), train_eval)
        if not keys:
            return {}
        device = collective_device()
        sums = torch.stack([_weighted_sums(bucket.get(key, []), device) for key in keys])
        bucket.clear()
        if is_multi_rank_run():
            dist.all_reduce(sums, op=dist.ReduceOp.SUM)
        return {key: weighted / rows for key, (weighted, rows) in zip(keys, sums.tolist(), strict=True) if rows > 0}
