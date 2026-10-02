"""World-level step metrics for the GRPO trainers, folded from rank-local counts.

TRL's ``GRPOTrainer.log`` averages each process's own ``_metrics`` list and only the main process
reports, so a value computed over one rank's rows is logged as if it were the batch's. The helpers
here send each rank's sums instead and derive the metric from the world totals.
"""

from collections import defaultdict
from collections.abc import Callable, MutableMapping, Sequence

import torch
from accelerate.utils import gather_object

from src.distributed.runtime import current_device

Count = torch.Tensor | float | int


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


# How a WorldMetrics entry folds, by wire kind: each rank sends its sums, the fold adds them column-wise
# and derives the metric from the world totals. A maximum is the one kind that is not a sum.
_FRACTION = "fraction"
_MAXIMUM = "max"
_EFFECTIVE_SAMPLE_FRAC = "ess"
_COVARIANCE = "cov"
_FOLDS: dict[str, Callable[..., float]] = {
    _FRACTION: _ratio,
    # (Σw)² / (n Σw²): 1 for uniform weights, toward 1/n as one weight dominates, 0 when all are zero.
    _EFFECTIVE_SAMPLE_FRAC: lambda w, w2, n: _ratio(w * w, n * w2),
    # E[xy] - E[x] E[y] over the pooled samples.
    _COVARIANCE: lambda xy, x, y, n: _ratio(xy, n) - _ratio(x, n) * _ratio(y, n),
}


def gathered_fractions(
    pairs: Sequence[tuple[Count, Count]], gather_fn: Callable[[torch.Tensor], torch.Tensor]
) -> list[float]:
    """World-level ``numerator / denominator`` of each pair from per-rank counts in ONE collective.

    Call on every rank. Per-rank fractions cannot be averaged — their denominators differ, and TRL's
    ``log`` would report the main process's alone. A world denominator of 0 reads 0.
    """
    # A pair of plain numbers would otherwise stack on CPU and hand the accelerator's NCCL gather a
    # CPU tensor; the device the trainer computes on is the right default.
    device = next((v.device for pair in pairs for v in pair if isinstance(v, torch.Tensor)), current_device())
    local = torch.stack([torch.as_tensor(v, device=device).double() for pair in pairs for v in pair])
    counts = gather_fn(local).view(-1, 2 * len(pairs)).sum(dim=0)
    return (counts[0::2] / counts[1::2].clamp(min=1)).tolist()


class WorldMetrics:
    """Per-step accumulator for batch-level fractions, means, maxima, effective sample sizes and
    covariances.

    Sites record the local sums (or maximum) each metric is derived from here, and :meth:`flush` folds
    every rank's entries in ONE collective. Recording issues no collective and no host sync, so a site
    behind a data-dependent gate is safe: a rank that never reaches it contributes nothing to that key.
    A step that raises between a record and its flush ends the run — every such raise in the trainer
    is rank-uniform and fatal — so no entry survives into the next step.
    """

    def __init__(self) -> None:
        self._pending: dict[str, tuple[str, tuple[Count, ...]]] = {}

    def fraction(self, key: str, numerator: Count, denominator: Count) -> None:
        """World ``numerator / denominator``; a sum over a count is the batch mean."""
        self._record(key, _FRACTION, numerator, denominator)

    def maximum(self, key: str, value: Count) -> None:
        self._record(key, _MAXIMUM, value)

    def effective_sample_frac(self, key: str, weights: torch.Tensor, mask: torch.Tensor) -> None:
        """World normalized effective sample size ``(Σw)² / (n Σw²)`` of ``weights`` where ``mask`` holds."""
        mask = mask.bool()
        w = torch.where(mask, weights.double(), 0.0)
        self._record(key, _EFFECTIVE_SAMPLE_FRAC, w.sum(), (w * w).sum(), mask.sum())

    def covariance(self, key: str, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> None:
        """World covariance of ``x`` and ``y`` over the elements where ``mask`` holds."""
        # where, not a product: a non-finite value outside the mask would turn the sum into NaN.
        mask = mask.bool()
        x, y = torch.where(mask, x.double(), 0.0), torch.where(mask, y.double(), 0.0)
        self._record(key, _COVARIANCE, (x * y).sum(), x.sum(), y.sum(), mask.sum())

    def _record(self, key: str, kind: str, *values: Count) -> None:
        if key in self._pending:
            raise ValueError(f"{key} was already recorded this step; a key folds one entry per rank")
        self._pending[key] = (kind, values)

    def flush(
        self, target: MutableMapping[str, list[float]], gather_fn: Callable[[list], list] = gather_object
    ) -> None:
        """Fold the pending entries across ranks into ``target`` (one TRL ``_metrics[mode]`` dict).

        COLLECTIVE — every rank calls it once per step at the same point. The key set is the union
        of what any rank recorded; a metric whose world count is 0 reads 0.
        """
        pending, self._pending = self._materialized(), {}
        merged: dict[str, list[tuple[str, tuple[float, ...]]]] = defaultdict(list)
        for rank_entries in gather_fn([pending]):
            for key, entry in rank_entries.items():
                merged[key].append(entry)
        for key in sorted(merged):
            entries = merged[key]
            kinds = {kind for kind, _ in entries}
            if len(kinds) != 1:
                raise ValueError(f"{key} was recorded as {sorted(kinds)} on different ranks")
            (kind,) = kinds
            if kind == _MAXIMUM:
                value = max(values[0] for _, values in entries)
            else:
                value = _FOLDS[kind](*(sum(column) for column in zip(*(values for _, values in entries), strict=True)))
            target.setdefault(key, []).append(value)

    def _materialized(self) -> dict[str, tuple[str, tuple[float, ...]]]:
        """The pending entries as floats, every recorded tensor read in one host sync."""
        tensors = [v for _, values in self._pending.values() for v in values if isinstance(v, torch.Tensor)]
        read = iter(
            torch.stack([t.detach().to(tensors[0].device).double() for t in tensors]).tolist() if tensors else ()
        )

        def as_float(value: Count) -> float:
            return next(read) if isinstance(value, torch.Tensor) else float(value)

        return {key: (kind, tuple(as_float(v) for v in values)) for key, (kind, values) in self._pending.items()}
