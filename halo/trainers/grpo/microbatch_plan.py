"""Token-budget micro-batch plans of a GRPO round: the bins of rows one packed backbone call runs, the bin count and
row width the world agrees on, and the slot layout TRL's buffer splits."""

import dataclasses
import math
from collections.abc import Sequence

from halo.runtime.consensus import rank_max

# A bin holds at least this many tokens: the next-token shift leaves nothing in a one-token row.
MIN_ROW_TOKENS = 2


@dataclasses.dataclass(frozen=True)
class MicrobatchPlan:
    """``bins`` are round-row indices, heaviest bin first, over the ``round_rows`` rows of the round (the dummy rows
    that pad it hold no bin). ``rows_per_bin`` is the world's widest bin, which every slot pads to with filler
    (``-1``) rows, since TRL splits its buffer into equal chunks and gathers per-slot tensors."""

    bins: tuple[tuple[int, ...], ...]
    rows_per_bin: int
    slots: int
    round_rows: int

    @property
    def num_bins(self) -> int:
        return len(self.bins)

    @property
    def bin_sizes(self) -> list[int]:
        return [len(rows) for rows in self.bins]

    @property
    def order(self) -> list[int]:
        """The bins' rows in the order the no-grad forwards run them."""
        return [row for rows in self.bins for row in rows]


def first_fit_decreasing(lengths: Sequence[int], budget: int) -> list[list[int]]:
    """Longest first (ties by index) into the first bin with room; a row over ``budget`` takes a bin of its own."""
    loads: list[int] = []
    bins: list[list[int]] = []
    for row in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        target = next((i for i, load in enumerate(loads) if load + lengths[row] <= budget), len(bins))
        if target == len(bins):
            loads.append(0)
            bins.append([])
        loads[target] += lengths[row]
        bins[target].append(row)
    return bins


def balanced_partition(lengths: Sequence[int], num_bins: int) -> list[list[int]]:
    """Exactly ``num_bins`` bins: longest first into the lightest (longest-processing-time), ties to the lowest bin."""
    loads = [0] * num_bins
    bins: list[list[int]] = [[] for _ in range(num_bins)]
    for row in sorted(range(len(lengths)), key=lambda i: (-lengths[i], i)):
        target = loads.index(min(loads))
        loads[target] += lengths[row]
        bins[target].append(row)
    return bins


def _heaviest(lengths: Sequence[int], bins: list[list[int]]) -> int:
    return max((sum(lengths[row] for row in rows) for rows in bins), default=0)


def local_bin_count(lengths: Sequence[int], budget: int, slots: int, ceiling: int | None) -> int:
    """The bins this rank's rows need: first-fit-decreasing at ``budget``, relaxed to ``slots`` balanced bins where
    that is more. ``slots + 1`` is the sentinel for rows no ``slots`` bins can hold within ``ceiling``."""
    count = len(first_fit_decreasing(lengths, budget))
    if count <= slots:
        return count
    limit = max(ceiling or math.inf, max(lengths))
    return slots if _heaviest(lengths, balanced_partition(lengths, slots)) <= limit else slots + 1


def partition_rows(lengths: Sequence[int], num_bins: int, budget: int) -> list[list[int]]:
    """``num_bins`` bins, heaviest first, each row in one. Balanced where no bin then passes ``max(budget, longest
    row)``, else the first-fit bins padded with empty ones; first-fit needing more than ``num_bins`` is relaxed to a
    balanced partition."""
    fitted = first_fit_decreasing(lengths, budget)
    bins = balanced_partition(lengths, num_bins)
    if len(fitted) <= num_bins and _heaviest(lengths, bins) > max(budget, max(lengths, default=0)):
        bins = fitted + [[] for _ in range(num_bins - len(fitted))]
    bins = [sorted(rows) for rows in bins]
    return sorted(bins, key=lambda rows: -sum(lengths[row] for row in rows))


def agree_microbatch_geometry(
    lengths: Sequence[int], budget: int, *, slots: int, ceiling: int | None, round_rows: int
) -> MicrobatchPlan:
    """COLLECTIVE. ``lengths`` are this rank's real rows' attended tokens (at least :data:`MIN_ROW_TOKENS`). Every rank
    gets the world's bin count (at least one, so a round with no row anywhere still runs the backward and its
    collectives) and its own rows re-partitioned into that many bins, then the world's widest bin. Raises on every
    rank when some rank's rows fit no ``slots`` bins of ``ceiling`` tokens."""
    bins_needed, tokens = rank_max([local_bin_count(lengths, budget, slots, ceiling), sum(lengths)])
    if bins_needed > slots:
        raise ValueError(
            f"A rank holds {tokens} row tokens this round, more than steps_per_generation ({slots}) packed "
            f"micro-batches of at most {ceiling} tokens (the DeepEP dispatch ceiling) can carry. Raise "
            f"steps_per_generation, or lower the rollouts per step or the row length cap."
        )
    bins = partition_rows(lengths, max(bins_needed, 1), budget)
    (rows_per_bin,) = rank_max([max(len(rows) for rows in bins)])
    return MicrobatchPlan(tuple(tuple(rows) for rows in bins), max(rows_per_bin, 1), slots, round_rows)


def build_slot_layout(plan: MicrobatchPlan) -> tuple[list[int], list[int]]:
    """``(slot_rows, slot_active)``, each ``slots * rows_per_bin`` long and in TRL's chunk order. The ``slots -
    num_bins`` leading slots are no-ops (all ``-1``, inactive) so the window's last backward is a real one; each real
    slot lists its bin's rows, filler ``-1`` after."""
    width = plan.rows_per_bin
    idle = plan.slots - plan.num_bins
    slot_rows = [-1] * (idle * width)
    slot_active = [0] * (idle * width)
    for rows in plan.bins:
        slot_rows += [*rows, *([-1] * (width - len(rows)))]
        slot_active += [1] * width
    return slot_rows, slot_active
