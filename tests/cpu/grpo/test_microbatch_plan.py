#!/usr/bin/env python
"""The planner of packed GRPO micro-batches: bins within the budget, the world's agreed bin count and width, and the
slot layout that leaves a window's last slot real."""

import pytest
import torch
import trl.trainer.grpo_trainer as trl_grpo_trainer

import halo.trainers.grpo.microbatch_plan as plan_module
from halo.trainers.grpo.microbatch_plan import (
    MicrobatchPlan,
    agree_microbatch_geometry,
    balanced_partition,
    build_slot_layout,
    first_fit_decreasing,
    local_bin_count,
    partition_rows,
)
from tests.common.length_deal import (
    PACKED_DENSE_BUDGET,
    PACKED_DENSE_ROW_TOKENS,
    PACKED_MOE_BUDGET,
    PACKED_MOE_ROW_TOKENS,
    PACKED_SLOTS,
)

SLOTS = 8


def _lengths(rows: int, seed: int, low: int = 2, high: int = 900) -> list[int]:
    return torch.randint(low, high, (rows,), generator=torch.Generator().manual_seed(seed)).tolist()


class _World:
    """The two ``rank_max`` calls of ``agree_microbatch_geometry`` answered from every rank's lengths, so one rank's
    plan is computed as it would be beside its peers."""

    def __init__(self, per_rank: list[list[int]], budget: int, ceiling: int | None):
        self.per_rank, self.budget, self.ceiling = per_rank, budget, ceiling
        self.calls = 0
        self.bins = 0

    def rank_max(self, values: list[int]) -> list[int]:
        self.calls += 1
        if self.calls == 1:
            counts = [local_bin_count(lengths, self.budget, SLOTS, self.ceiling) for lengths in self.per_rank]
            self.bins = max(max(counts), 1)
            return [max(counts), max(sum(lengths) for lengths in self.per_rank)]
        widest = [max(len(rows) for rows in partition_rows(l, self.bins, self.budget)) for l in self.per_rank]
        return [max(widest)]


def _plan_world(per_rank, budget, monkeypatch, ceiling=None) -> list[MicrobatchPlan]:
    plans = []
    for lengths in per_rank:
        monkeypatch.setattr(plan_module, "rank_max", _World(per_rank, budget, ceiling).rank_max)
        plans.append(
            agree_microbatch_geometry(lengths, budget, slots=SLOTS, ceiling=ceiling, round_rows=len(lengths) + 3)
        )
    return plans


def _every_row_once(plan: MicrobatchPlan, rows: int) -> bool:
    return sorted(plan.order) == list(range(rows))


@pytest.mark.parametrize("seed", range(6))
def test_every_row_sits_in_exactly_one_bin_within_the_budget(seed):
    lengths = _lengths(40, seed)
    budget = 1500
    bins = partition_rows(lengths, len(first_fit_decreasing(lengths, budget)), budget)
    assert sorted(row for rows in bins for row in rows) == list(range(40))
    limit = max(budget, max(lengths))
    assert all(sum(lengths[row] for row in rows) <= limit for rows in bins)


def test_a_row_over_the_budget_keeps_a_bin_of_its_own():
    bins = first_fit_decreasing([30, 2000, 40], 100)
    assert [1] in bins and sorted(row for rows in bins for row in rows) == [0, 1, 2]


def test_the_plan_is_a_pure_function_of_the_lengths():
    """Tensor-parallel siblings plan alone from the same lengths and must agree without a collective."""
    lengths = _lengths(33, 3)
    assert partition_rows(lengths, 4, 2000) == partition_rows(list(lengths), 4, 2000)
    ties = [10] * 9
    assert partition_rows(ties, 3, 40) == partition_rows(ties, 3, 40)


@pytest.mark.parametrize("seed", range(4))
def test_balancing_leaves_bins_no_further_apart_than_the_longest_row(seed):
    lengths = _lengths(37, seed)
    loads = [sum(lengths[row] for row in rows) for rows in balanced_partition(lengths, 5)]
    assert max(loads) - min(loads) <= max(lengths)


def test_every_rank_runs_the_worlds_bin_count_and_width(monkeypatch):
    per_rank = [_lengths(12, 1), _lengths(30, 2, high=500), _lengths(3, 3)]
    plans = _plan_world(per_rank, 1200, monkeypatch)
    needed = max(len(first_fit_decreasing(lengths, 1200)) for lengths in per_rank)
    assert {plan.num_bins for plan in plans} == {needed}
    assert len({plan.rows_per_bin for plan in plans}) == 1
    for plan, lengths in zip(plans, per_rank, strict=True):
        assert _every_row_once(plan, len(lengths))
        assert max(plan.bin_sizes) <= plan.rows_per_bin
    assert sum(plans[2].bin_sizes) == 3 and plans[2].num_bins == needed


def test_a_round_with_no_row_anywhere_still_runs_one_inert_bin(monkeypatch):
    plans = _plan_world([[], []], 1000, monkeypatch)
    assert all(plan.num_bins == 1 and plan.rows_per_bin == 1 and plan.bins == ((),) for plan in plans)
    rows, active = build_slot_layout(plans[0])
    assert rows == [-1] * SLOTS and active[-1] == 1


def test_a_multi_turn_round_that_needs_more_bins_than_slots_is_relaxed_never_refused(monkeypatch):
    """``R_pad / S`` rows per slot and a small budget: first-fit wants more than ``S`` bins and today's run works."""
    lengths = _lengths(64, 5, low=300, high=900)
    assert len(first_fit_decreasing(lengths, 1000)) > SLOTS
    (plan,) = _plan_world([lengths], 1000, monkeypatch, ceiling=20_000)
    assert plan.num_bins == SLOTS and _every_row_once(plan, 64)


def test_a_rank_past_every_slot_of_the_ceiling_raises_on_every_rank(monkeypatch):
    heavy, light = _lengths(80, 7, low=800, high=900), _lengths(4, 8)
    for lengths in (heavy, light):
        world = _World([heavy, light], 1000, 4096)
        monkeypatch.setattr(plan_module, "rank_max", world.rank_max)
        with pytest.raises(ValueError, match=r"more than steps_per_generation \(8\)"):
            agree_microbatch_geometry(lengths, 1000, slots=SLOTS, ceiling=4096, round_rows=len(lengths))


def _slot_chunks(rows: list[int], active: list[int], plan: MicrobatchPlan):
    width = plan.rows_per_bin
    return [(rows[i * width : (i + 1) * width], active[i * width : (i + 1) * width]) for i in range(plan.slots)]


def _assert_last_slot_is_real(active: list[int], width: int) -> None:
    assert all(active[-width:]), "the window's last slot is a no-op: its backward would end the window on nothing"


def test_the_layout_puts_the_no_ops_first_and_gathers_back_to_the_rows():
    plan = MicrobatchPlan(bins=((4, 1, 0), (3, 2), (5,)), rows_per_bin=3, slots=SLOTS, round_rows=6)
    rows, active = build_slot_layout(plan)
    chunks = _slot_chunks(rows, active, plan)
    idle = SLOTS - plan.num_bins
    assert all(r == [-1, -1, -1] and a == [0, 0, 0] for r, a in chunks[:idle])
    assert [a for _, a in chunks[idle:]] == [[1, 1, 1]] * plan.num_bins
    assert [[r for r in chunk if r >= 0] for chunk, _ in chunks[idle:]] == [list(rows) for rows in plan.bins]
    _assert_last_slot_is_real(active, plan.rows_per_bin)
    # TRL splits every tensor of the buffer by one chunk size, so each carries one entry per slot row.
    assert len(rows) == len(active) == SLOTS * plan.rows_per_bin


def test_a_layout_that_ends_on_a_no_op_fails_the_last_slot_check():
    """Negative control of the check above: no-ops last, as the rejected design would place them."""
    plan = MicrobatchPlan(bins=((0, 1), (2,)), rows_per_bin=2, slots=4, round_rows=3)
    _, active = build_slot_layout(plan)
    trailing = active[-2 * plan.num_bins :] + active[: -2 * plan.num_bins]
    _assert_last_slot_is_real(active, 2)
    with pytest.raises(AssertionError, match="last slot is a no-op"):
        _assert_last_slot_is_real(trailing, 2)


def test_the_layout_survives_trls_own_split():
    plan = MicrobatchPlan(bins=((0, 1), (2,)), rows_per_bin=2, slots=4, round_rows=3)
    rows, active = build_slot_layout(plan)
    chunks = trl_grpo_trainer.split_tensor_dict(
        {"slot_rows": torch.tensor(rows), "slot_active": torch.tensor(active)}, plan.slots
    )
    assert [c["slot_rows"].tolist() for c in chunks] == [[-1, -1], [-1, -1], [0, 1], [2, -1]]
    assert [c["slot_active"].tolist() for c in chunks] == [[0, 0], [0, 0], [1, 1], [1, 1]]


@pytest.mark.parametrize(
    ("row_tokens", "budget"),
    [(PACKED_DENSE_ROW_TOKENS, PACKED_DENSE_BUDGET), (PACKED_MOE_ROW_TOKENS, PACKED_MOE_BUDGET)],
    ids=["dense", "moe"],
)
def test_the_gpu_suites_rounds_fit_between_two_bins_and_the_slots_without_relaxing(row_tokens, budget):
    """The packed suites assert ``1 < num_bins < slots`` to keep a no-op slot and a rows-per-bin rescale off 1; a
    budget whose first-fit needs more bins than slots is relaxed to ``slots`` and would leave neither."""
    fitted = len(first_fit_decreasing(row_tokens, budget))
    assert 1 < fitted < PACKED_SLOTS
    assert local_bin_count(row_tokens, budget, PACKED_SLOTS, None) == fitted


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
