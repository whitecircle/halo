"""World-level metrics folded from rank-local sums.

TRL's ``log`` averages each process's own ``_metrics`` list and only the main process reports, so a value
computed over one rank's rows is logged as if it were the batch's. Each rank records its sums instead, and one
collective per flush derives every metric from the world totals.
"""

import math
from collections import defaultdict
from collections.abc import Callable, MutableMapping, Sequence
from itertools import chain

import torch
import torch.distributed as dist

from halo.runtime.collectives import collective_device
from halo.runtime.ranks import is_multi_rank_run

Count = torch.Tensor | float | int
State = tuple[float, ...]


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _report_moments(key: str, total: float, squares: float, count: float, low: float, high: float) -> dict:
    if not count:
        return {}
    mean = total / count
    std = math.sqrt(max(squares / count - mean * mean, 0.0))
    return {f"{key}_mean": mean, f"{key}_std": std, f"{key}_min": low, f"{key}_max": high, f"{key}_range": high - low}


def _report_percentile(key: str, *pairs: float) -> dict:
    """The nearest-rank percentile of the pooled ``(q, value)`` pairs, ``q`` the same in every pair."""
    q, ordered = pairs[0], sorted(pairs[1::2])
    rank = max(1, min(len(ordered), math.ceil(q / 100.0 * len(ordered))))
    return {key: ordered[rank - 1]}


# Per kind: the fold of each state column, over a rank's records and then over the ranks (None: the states are
# concatenated), and what the folded state reports.
_KINDS: dict[str, tuple[tuple[Callable, ...] | None, Callable[..., dict]]] = {
    "fraction": ((sum, sum), lambda key, numerator, denominator: {key: _ratio(numerator, denominator)}),
    "mean": ((sum, sum), lambda key, weighted, weight: {key: weighted / weight} if weight else {}),
    "sum": ((sum,), lambda key, total: {key: total}),
    "max": ((max,), lambda key, value: {key: value}),
    "ess": ((sum, sum, sum), lambda key, w, w2, n: {key: _ratio(w * w, n * w2)}),
    "cov": ((sum, sum, sum, sum), lambda key, xy, x, y, n: {key: _ratio(xy, n) - _ratio(x, n) * _ratio(y, n)}),
    "moments": ((sum, sum, sum, min, max), _report_moments),
    "percentile": (None, _report_percentile),
}


def _fold(kind: str, states: Sequence[State]) -> State:
    """One ``kind`` state from several: a rank's records, or the ranks' states."""
    folds = _KINDS[kind][0]
    if folds is None:
        return tuple(chain.from_iterable(states))
    return tuple(fold(column) for fold, column in zip(folds, zip(*states, strict=True), strict=True))


def _gathered(payload: dict, group: dist.ProcessGroup | None) -> list[dict]:
    """COLLECTIVE over ``group`` (the world when None): every rank's ``payload``, in rank order."""
    if not is_multi_rank_run():
        return [payload]
    gathered: list = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, payload, group=group)
    return gathered


def world_sums(values: Sequence[Count]) -> list[float]:
    """Each of ``values`` summed over the world, in one all-reduce of ``len(values)`` numbers.

    COLLECTIVE — every rank calls it with the same number of values, a rank with nothing to report
    sending zeros. For a counter kept across steps, read where TRL's ``log`` would report one rank's.
    """
    device = collective_device()
    local = torch.stack([torch.as_tensor(v, device=device).double() for v in values])
    if is_multi_rank_run():
        dist.all_reduce(local)
    return local.tolist()


class WorldMetrics:
    """Rank-local metric records, folded across ranks by :meth:`flush`.

    Recording issues no collective and no host sync, so a site behind a data-dependent gate is safe; records of a
    key accumulate until the flush, which folds every rank's records per kind:

    - ``fraction``: Σ numerator / Σ denominator, 0 at no denominator; ``mean``: Σ value·weight / Σ weight, a record
      of no weight adding nothing (NaN included) and a key of no world weight not reported; ``sum``; ``maximum``;
    - ``effective_sample_frac``: (Σw)² / (n Σw²) of the masked weights (1 when uniform, toward 1/n as one weight
      dominates, 0 when all are zero); ``covariance``: E[xy] - E[x] E[y] of the masked pairs;
    - ``moments``: ``<key>_{mean,std,min,max,range}`` of the masked values, std = sqrt(E[x²] - E[x]²), not reported
      without a sample; ``percentile``: the nearest-rank percentile, the one kind that ships raw values.
    """

    def __init__(self) -> None:
        self._records: dict[str, tuple[str, list[tuple[Count, ...]]]] = {}

    def __bool__(self) -> bool:
        return bool(self._records)

    def fraction(self, key: str, numerator: Count, denominator: Count) -> None:
        self._record(key, "fraction", numerator, denominator)

    def mean(self, key: str, value: Count, weight: Count = 1) -> None:
        if isinstance(value, torch.Tensor):
            # where, not a product: a NaN mean over no rows must add nothing.
            value = value.detach().double()
            weighted = torch.where(torch.as_tensor(weight, device=value.device) > 0, value * weight, 0.0)
        else:
            weighted = value * weight if weight > 0 else 0.0
        self._record(key, "mean", weighted, weight)

    def sum(self, key: str, value: Count) -> None:
        self._record(key, "sum", value)

    def maximum(self, key: str, value: Count) -> None:
        self._record(key, "max", value)

    def effective_sample_frac(self, key: str, weights: torch.Tensor, mask: torch.Tensor) -> None:
        mask = mask.bool()
        w = torch.where(mask, weights.double(), 0.0)
        self._record(key, "ess", w.sum(), (w * w).sum(), mask.sum())

    def covariance(self, key: str, x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> None:
        # where, not a product: a non-finite value outside the mask would turn the sum into NaN.
        mask = mask.bool()
        x, y = torch.where(mask, x.double(), 0.0), torch.where(mask, y.double(), 0.0)
        self._record(key, "cov", (x * y).sum(), x.sum(), y.sum(), mask.sum())

    def moments(self, key: str, values: torch.Tensor, mask: torch.Tensor) -> None:
        mask, values = mask.bool(), values.detach().double()
        inside = torch.where(mask, values, 0.0)
        low, high = torch.where(mask, values, math.inf).min(), torch.where(mask, values, -math.inf).max()
        self._record(key, "moments", inside.sum(), (inside * inside).sum(), mask.sum(), low, high)

    def percentile(self, key: str, value: Count, q: float) -> None:
        """One value of the population whose ``q``-th percentile (``q`` in [0, 100]) the key reports."""
        self._record(key, "percentile", q, value)

    def _record(self, key: str, kind: str, *values: Count) -> None:
        recorded, records = self._records.setdefault(key, (kind, []))
        if recorded != kind:
            raise ValueError(f"{key} was recorded as {recorded} and as {kind}")
        records.append(values)

    def flush(
        self,
        target: MutableMapping[str, list[float]] | None = None,
        *,
        group: dist.ProcessGroup | None = None,
        key_mismatch_error: Callable[[dict[int, list[str]]], Exception] | None = None,
    ) -> dict[str, float]:
        """COLLECTIVE over ``group`` (the world when None): every rank calls it at the same point, records or not.

        Returns the world metrics, keys sorted, each also appended to ``target[key]`` when given (TRL's
        ``_metrics[mode]``). The keys are the union of what any rank recorded; with ``key_mismatch_error`` every
        rank holding records must name the same keys (a rank with none abstains), else every rank raises
        ``key_mismatch_error({rank: sorted keys})``.
        """
        states, self._records = self._local_states(), {}
        gathered = _gathered(states, group)
        if key_mismatch_error is not None:
            keys_by_rank = {rank: sorted(rank_states) for rank, rank_states in enumerate(gathered) if rank_states}
            if len({tuple(keys) for keys in keys_by_rank.values()}) > 1:
                raise key_mismatch_error(keys_by_rank)
        merged: dict[str, list[tuple[str, State]]] = defaultdict(list)
        for rank_states in gathered:
            for key, entry in rank_states.items():
                merged[key].append(entry)
        world: dict[str, float] = {}
        for key in sorted(merged):
            kinds = {kind for kind, _ in merged[key]}
            if len(kinds) != 1:
                raise ValueError(f"{key} was recorded as {sorted(kinds)} on different ranks")
            kind = kinds.pop()
            world.update(_KINDS[kind][1](key, *_fold(kind, [state for _, state in merged[key]])))
        if target is not None:
            for key, value in world.items():
                target.setdefault(key, []).append(value)
        return world

    def _local_states(self) -> dict[str, tuple[str, State]]:
        """Each key's records folded into one state of floats, every recorded tensor read in one host sync."""
        tensors = [
            v for _, records in self._records.values() for record in records for v in record if torch.is_tensor(v)
        ]
        device = next((t.device for t in tensors if t.device.type != "cpu"), torch.device("cpu"))
        read = iter(
            torch.stack([t.detach().reshape(()).to(device, torch.float64) for t in tensors]).tolist()
            if tensors
            else ()
        )
        states = {}
        for key, (kind, records) in self._records.items():
            floats = [tuple(next(read) if torch.is_tensor(v) else float(v) for v in record) for record in records]
            states[key] = (kind, _fold(kind, floats))
        return states
