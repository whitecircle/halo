"""World-level metrics folded from rank-local sums.

TRL's ``log`` averages each process's own ``_metrics`` list and only the main process reports, so a value
computed over one rank's rows is logged as if it were the batch's. Each rank records its sums instead, and a
flush folds them where they lie and derives every metric from the world totals in fixed-width all-reduces read
back once; only a key the ranks have not agreed on yet, or a percentile's raw values, travels in an object gather.
"""

import math
from collections import defaultdict
from collections.abc import Callable, Iterable, MutableMapping, Sequence
from itertools import chain

import torch
import torch.distributed as dist

from halo.runtime.collectives import collective_device
from halo.runtime.ranks import is_multi_rank_run

Count = torch.Tensor | float | int
State = tuple[float, ...]
# A flush's sorted ``(key, kind)`` pairs, which fix the layout of its reduced columns.
Schema = tuple[tuple[str, str], ...]
MismatchErrorFactory = Callable[[dict[int, list[str]]], Exception]
# Per key, its kind and its records in order, each record one value per state column.
Records = dict[str, tuple[str, list[tuple[Count, ...]]]]

# The state-column folds one MAX all-reduce carries, each at its sign: a min crosses the ranks as the max of its
# negation. Every other column is summed.
_MAX_SIGNS: dict[Callable, float] = {max: 1.0, min: -1.0}


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


def _pooled(kind: str) -> bool:
    """Whether ``kind``'s states concatenate across the ranks, raw values a fixed-width reduce cannot carry."""
    return _KINDS[kind][0] is None


def _fold(kind: str, states: Sequence[State]) -> State:
    """One ``kind`` state from several: a rank's records, or the ranks' states."""
    folds = _KINDS[kind][0]
    if folds is None:
        return tuple(chain.from_iterable(states))
    return tuple(fold(column) for fold, column in zip(folds, zip(*states, strict=True), strict=True))


def _home_device(values: Iterable[Count]) -> torch.device | None:
    """Where ``values``' tensors lie: the first one off the CPU, else the CPU; None without a tensor."""
    tensors = [value for value in values if torch.is_tensor(value)]
    if not tensors:
        return None
    return next((tensor.device for tensor in tensors if tensor.device.type != "cpu"), torch.device("cpu"))


def _staged(numbers: list[float], device: torch.device) -> torch.Tensor:
    """``numbers`` as a float64 vector on ``device``; to a GPU through pinned memory, asynchronously, since a
    pageable host-to-device copy synchronizes the stream."""
    pinned = device.type == "cuda"
    return torch.tensor(numbers, dtype=torch.float64, pin_memory=pinned).to(device, non_blocking=pinned)


def _column(values: Sequence[Count], device: torch.device) -> torch.Tensor:
    """``values`` as one float64 vector on ``device`` with no host sync: each tensor cast where it lies, every
    number in one :func:`_staged` copy."""
    numbers = [float(value) for value in values if not torch.is_tensor(value)]
    if len(numbers) == len(values):
        return _staged(numbers, device)
    staged = iter(_staged(numbers, device)) if numbers else iter(())
    return torch.stack(
        [
            value.detach().reshape(()).to(device, torch.float64) if torch.is_tensor(value) else next(staged)
            for value in values
        ]
    )


def _compensated_sum(rows: torch.Tensor) -> torch.Tensor:
    """Each column of ``rows`` summed top to bottom with Neumaier's compensation, bit for bit what Python's float
    ``sum`` returns for the column as a list: its compensation is added only when nonzero and finite."""
    total, compensation = torch.zeros_like(rows[0]), torch.zeros_like(rows[0])
    for row in rows:
        stepped = total + row
        compensation += torch.where(total.abs() >= row.abs(), (total - stepped) + row, (row - stepped) + total)
        total = stepped
    return torch.where((compensation != 0) & compensation.isfinite(), total + compensation, total)


def _first_max(rows: torch.Tensor) -> torch.Tensor:
    """Each column's maximum as Python's ``max`` takes it: a later value replaces the running one only when greater,
    so the first of equal values, and a leading NaN, stay."""
    best = rows[0]
    for row in rows[1:]:
        best = torch.where(row > best, row, best)
    return best


class _RankFold:
    """This rank's records folded per state column in record order, as Python folds the column's values as a list:
    a column of numbers on the host, one holding a tensor on that tensor's device, so no tensor is read back. A min
    is held negated, as the MAX all-reduce carries it."""

    def __init__(self, records: Records) -> None:
        self._folded: dict[tuple[str, int], Count] = {}
        on_device: list[tuple[tuple[str, int], Callable, list[Count]]] = []
        for key, (kind, rows) in records.items():
            for column, fold in enumerate(_KINDS[kind][0] or ()):
                values = [row[column] for row in rows]
                if any(torch.is_tensor(value) for value in values):
                    on_device.append(((key, column), fold, values))
                else:
                    folded = fold(float(value) for value in values)
                    self._folded[(key, column)] = -folded if fold is min else folded
        self.device = _home_device(value for _, _, values in on_device for value in values)
        if on_device:
            self._fold_on_device(on_device)

    def columns(self, layout: Schema) -> tuple[list[Count], list[Count]]:
        """``layout``'s summed and its max-folded state columns, in order: this rank's folded value, or what a rank
        without the key adds (0, -inf)."""
        sums: list[Count] = []
        maxes: list[Count] = []
        for key, kind in layout:
            for column, fold in enumerate(_KINDS[kind][0] or ()):
                value = self._folded.get((key, column))
                if fold in _MAX_SIGNS:
                    maxes.append(-math.inf if value is None else value)
                else:
                    sums.append(0.0 if value is None else value)
        return sums, maxes

    def _fold_on_device(self, columns: list[tuple[tuple[str, int], Callable, list[Count]]]) -> None:
        """Fold ``columns`` in one grid of their values, padded to the longest with what leaves a fold unchanged:
        the summed columns first, then the max-folded ones, each min negated into a max."""
        ordered = sorted(columns, key=lambda column: (column[1] is not sum, column[1] is min))
        summed = sum(fold is sum for _, fold, _ in ordered)
        first_min = len(ordered) - sum(fold is min for _, fold, _ in ordered)
        pads = [0.0 if fold is sum else math.inf if fold is min else -math.inf for _, fold, _ in ordered]
        depth = max(len(values) for _, _, values in ordered)
        cells = [
            values[row] if row < len(values) else pad
            for row in range(depth)
            for (_, _, values), pad in zip(ordered, pads, strict=True)
        ]
        grid = _column(cells, self.device).view(depth, len(ordered))
        grid[:, first_min:].neg_()
        parts = [_compensated_sum(grid[:, :summed])] if summed else []
        if summed < len(ordered):
            parts.append(_first_max(grid[:, summed:]))
        folded = torch.cat(parts)
        for index, (address, _, _) in enumerate(ordered):
            self._folded[address] = folded[index]


def _gathered(payload: object, group: dist.ProcessGroup | None) -> list:
    """COLLECTIVE over ``group`` (the world when None): every rank's ``payload``, in rank order."""
    if not is_multi_rank_run():
        return [payload]
    gathered: list = [None] * dist.get_world_size(group)
    dist.all_gather_object(gathered, payload, group=group)
    return gathered


def _schema(records: Records) -> Schema:
    """A rank's sorted ``(key, kind)`` pairs; empty when it recorded nothing."""
    return tuple(sorted((key, kind) for key, (kind, _) in records.items()))


def _union_schema(schemas: Sequence[Schema], key_mismatch_error: MismatchErrorFactory | None) -> Schema:
    """Every key any rank holds, at its one kind; raises ``ValueError`` on a key held as two kinds. With
    ``key_mismatch_error`` the ranks holding records must hold the same keys, else it raises
    ``key_mismatch_error({rank: sorted keys})``."""
    if key_mismatch_error is not None:
        keys_by_rank = {rank: [key for key, _ in schema] for rank, schema in enumerate(schemas) if schema}
        if len({tuple(keys) for keys in keys_by_rank.values()}) > 1:
            raise key_mismatch_error(keys_by_rank)
    kinds: dict[str, set[str]] = defaultdict(set)
    for key, kind in chain.from_iterable(schemas):
        kinds[key].add(kind)
    union = []
    for key in sorted(kinds):
        if len(kinds[key]) != 1:
            raise ValueError(f"{key} was recorded as {sorted(kinds[key])} on different ranks")
        union.append((key, *kinds[key]))
    return tuple(union)


def _pooled_states(records: Records) -> dict[str, tuple[str, State]]:
    """Each pooled-kind key's records concatenated as floats, every tensor among them read in one host sync."""
    pooled = {key: entry for key, entry in records.items() if _pooled(entry[0])}
    tensors = [value for _, rows in pooled.values() for row in rows for value in row if torch.is_tensor(value)]
    read = iter(_column(tensors, _home_device(tensors)).tolist() if tensors else ())
    return {
        key: (kind, tuple(next(read) if torch.is_tensor(value) else float(value) for row in rows for value in row))
        for key, (kind, rows) in pooled.items()
    }


def _all_reduce_columns(
    rank_fold: _RankFold, header: list[float], layout: Schema, group: dist.ProcessGroup | None
) -> tuple[list[float], list[float], list[float]]:
    """COLLECTIVE over ``group``: ``header`` and ``layout``'s summed columns in one float64 SUM all-reduce, its
    max-folded ones in one MAX all-reduce, read back in one host sync; empty lists with nothing to reduce."""
    sums, maxes = rank_fold.columns(layout)
    if not (header or sums or maxes):
        return [], [], []
    columns = _column(header + sums + maxes, rank_fold.device or collective_device(group))
    summed = len(header) + len(sums)
    if is_multi_rank_run():
        columns = columns.to(collective_device(group))
        if summed:
            dist.all_reduce(columns[:summed], group=group)
        if maxes:
            dist.all_reduce(columns[summed:], op=dist.ReduceOp.MAX, group=group)
    reduced = columns.tolist()
    return reduced[: len(header)], reduced[len(header) : summed], reduced[summed:]


def _decoded(
    layout: Schema, sums: list[float], maxes: list[float], holders: Sequence[float] | None = None
) -> dict[str, tuple[str, State]]:
    """``layout``'s world states read off its reduced columns, in layout order; with ``holders``, each key's count
    of ranks holding it, only the keys some rank holds."""
    summed, maxed = iter(sums), iter(maxes)
    world: dict[str, tuple[str, State]] = {}
    for index, (key, kind) in enumerate(layout):
        state = tuple(
            _MAX_SIGNS[fold] * next(maxed) if fold in _MAX_SIGNS else next(summed) for fold in _KINDS[kind][0] or ()
        )
        if holders is None or holders[index]:
            world[key] = (kind, state)
    return world


def _reduce_over_layout(
    rank_fold: _RankFold,
    schema: Schema,
    layout: Schema,
    group: dist.ProcessGroup | None,
    key_mismatch_error: MismatchErrorFactory | None,
) -> dict[str, tuple[str, State]] | None:
    """COLLECTIVE over ``group``: the world states over the agreed ``layout``, or None on every rank when a rank's
    keys do not fit it or, with ``key_mismatch_error``, the ranks holding records hold different keys."""
    held = {key for key, _ in schema}
    header = [float(not set(schema) <= set(layout)), float(bool(schema)), *(float(key in held) for key, _ in layout)]
    (misfits, recording, *holders), sums, maxes = _all_reduce_columns(rank_fold, header, layout, group)
    if misfits or (key_mismatch_error is not None and any(0 < count < recording for count in holders)):
        return None
    return _decoded(layout, sums, maxes, holders)


def _reduce_over_gathered_schemas(
    rank_fold: _RankFold,
    records: Records,
    schema: Schema,
    group: dist.ProcessGroup | None,
    key_mismatch_error: MismatchErrorFactory | None,
) -> tuple[dict[str, tuple[str, State]], Schema]:
    """COLLECTIVE over ``group``: the world states over the union of the ranks' keys, gathered with every rank's
    pooled states, and that union; :func:`_union_schema` raises on every rank alike."""
    gathered = _gathered((schema, _pooled_states(records)), group)
    union = _union_schema([rank_schema for rank_schema, _ in gathered], key_mismatch_error)
    _, sums, maxes = _all_reduce_columns(rank_fold, [], union, group)
    world = _decoded(union, sums, maxes)
    for key, kind in union:
        if _pooled(kind):
            world[key] = (kind, _fold(kind, [pooled[key][1] for _, pooled in gathered if key in pooled]))
    return world, union


def _world_states(
    records: Records,
    layout: Schema,
    group: dist.ProcessGroup | None,
    key_mismatch_error: MismatchErrorFactory | None,
) -> tuple[dict[str, tuple[str, State]], Schema]:
    """COLLECTIVE over ``group`` (the world when None): each key's state folded over the ranks, keys sorted, and the
    layout the next flush reduces over.

    ``layout`` is the ``(key, kind)`` set the ranks agreed on last. While every rank's keys fit it, one SUM
    all-reduce carries each rank's verdict, the keys it holds and every summed column, one MAX all-reduce every max
    and negated min (a rank without the key adding 0 or -inf), and one host read brings both back: no object
    collective, and no width or host work growing with the world. A key outside it, a key mismatch or a percentile
    takes one object gather of the ranks' keys instead, and the next flush reuses their union.
    """
    rank_fold, schema = _RankFold(records), _schema(records)
    if is_multi_rank_run() and not any(_pooled(kind) for _, kind in layout):
        world = _reduce_over_layout(rank_fold, schema, layout, group, key_mismatch_error)
        if world is not None:
            return world, layout
    world, union = _reduce_over_gathered_schemas(rank_fold, records, schema, group, key_mismatch_error)
    return world, tuple(sorted({**dict(layout), **dict(union)}.items()))


def world_sums(values: Sequence[Count]) -> list[float]:
    """Each of ``values`` summed over the world, in one all-reduce of ``len(values)`` numbers read back once.

    COLLECTIVE — every rank calls it with the same number of values, a rank with nothing to report
    sending zeros. For a counter kept across steps, read where TRL's ``log`` would report one rank's.
    """
    local = _column(values, _home_device(values) or collective_device())
    if is_multi_rank_run():
        local = local.to(collective_device())
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

    A NaN reaching a ``maximum`` or a ``moments`` min or max may be dropped: neither Python's ``max`` within a rank
    nor the backend's MAX across ranks propagates it reliably. A sum does, so the moments mean and std report it.
    """

    def __init__(self) -> None:
        self._records: Records = {}
        # Per group, the (key, kind) layout its ranks agreed on last, which the next flush reduces over.
        self._layouts: dict[dist.ProcessGroup | None, Schema] = {}

    def __bool__(self) -> bool:
        return bool(self._records)

    def fraction(self, key: str, numerator: Count, denominator: Count) -> None:
        self._record(key, "fraction", numerator, denominator)

    def mean(self, key: str, value: Count, weight: Count = 1) -> None:
        if isinstance(value, torch.Tensor):
            value = value.detach().double()
            if torch.is_tensor(weight):
                # where, not a product: a NaN mean over no rows must add nothing.
                weighted = torch.where(torch.as_tensor(weight, device=value.device) > 0, value * weight, 0.0)
            else:
                weighted = value * weight if weight > 0 else torch.zeros_like(value)
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
        key_mismatch_error: MismatchErrorFactory | None = None,
    ) -> dict[str, float]:
        """COLLECTIVE over ``group`` (the world when None): every rank calls it at the same point, records or not.

        Returns the world metrics, keys sorted, each also appended to ``target[key]`` when given (TRL's
        ``_metrics[mode]``). The keys are the union of what any rank recorded; with ``key_mismatch_error`` every
        rank holding records must name the same keys (a rank with none abstains), else every rank raises
        ``key_mismatch_error({rank: sorted keys})``.
        """
        records, self._records = self._records, {}
        states, self._layouts[group] = _world_states(records, self._layouts.get(group, ()), group, key_mismatch_error)
        world: dict[str, float] = {}
        for key, (kind, state) in states.items():
            world.update(_KINDS[kind][1](key, *state))
        if target is not None:
            for key, value in world.items():
                target.setdefault(key, []).append(value)
        return world
