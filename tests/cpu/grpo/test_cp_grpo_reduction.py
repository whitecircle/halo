#!/usr/bin/env python
"""Real Gloo proof for GRPO's autograd-aware per-row CP sum."""

from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from src.distributed.context_parallel.autograd import cp_sum_rows
from src.distributed.context_parallel.config import CPConfig
from src.trainers.grpo.objective.offline import offline_loss, offline_loss_normalizer, offline_loss_numerator
from tests.common.gloo import run_gloo_ranks

COEFFICIENTS = torch.tensor([[0.0, 0.0, 0.0], [2.0, -3.0, 4.0], [-1.0, 5.0, 2.0], [3.0, 1.0, -2.0]])
ROW_WEIGHTS = torch.tensor([0.5, 2.0, -1.0])
INITIAL_WEIGHT = 1.25
LR = 0.1
TOKEN_COEFFICIENTS = torch.tensor(
    [
        [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
        [[1.0, -2.0, 3.0], [4.0, -1.0, 2.0], [2.0, 1.0, -3.0]],
        [[-3.0, 2.0, 1.0], [1.0, 5.0, -1.0], [-2.0, 4.0, 3.0]],
        [[2.0, 4.0, -1.0], [-2.0, 1.0, 3.0], [5.0, -1.0, 2.0]],
    ]
)
SUPERVISION = torch.tensor(
    [
        [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 1, 0], [1, 0, 0], [0, 0, 0]],
        [[0, 1, 1], [1, 1, 1], [1, 0, 0]],
        [[1, 0, 0], [0, 1, 1], [1, 1, 0]],
    ],
    dtype=torch.float32,
)
GROUP_SIZES = torch.tensor([2, 2, 3])
MAX_COMPLETION_LENGTH = 12


class _ForwardOnlySum(torch.autograd.Function):
    """Mutation control: correct SUM value but the backward collective is deliberately absent."""

    @staticmethod
    def forward(ctx, values, group):
        out = values.detach().clone()
        dist.all_reduce(out, op=dist.ReduceOp.SUM, group=group)
        return out

    @staticmethod
    def backward(ctx, upstream):
        return upstream, None


def _worker(rank: int, cp_size: int) -> None:
    cp_config = CPConfig(cp_size=cp_size, world_size=cp_size, gpus_per_node=cp_size)
    weight = torch.nn.Parameter(torch.tensor(INITIAL_WEIGHT))
    local_rows = weight * COEFFICIENTS[rank]
    global_rows = cp_sum_rows(local_rows, cp_config)

    baseline = torch.nn.Parameter(torch.tensor(INITIAL_WEIGHT))
    baseline_rows = baseline * COEFFICIENTS[:cp_size].sum(dim=0)
    torch.testing.assert_close(global_rows, baseline_rows, atol=0, rtol=0)
    assert global_rows.requires_grad, "forward SUM detached the per-row loss from its local graph"

    (global_rows * ROW_WEIGHTS).sum().backward()
    (baseline_rows * ROW_WEIGHTS).sum().backward()

    local_expected = cp_size * (COEFFICIENTS[rank] * ROW_WEIGHTS).sum()
    torch.testing.assert_close(weight.grad, local_expected, atol=0, rtol=0)
    dist.all_reduce(weight.grad, op=dist.ReduceOp.SUM)
    weight.grad /= cp_size  # Halo's world-wide mean parameter-gradient synchronization.
    torch.testing.assert_close(weight.grad, baseline.grad, atol=0, rtol=0)

    with torch.no_grad():
        weight.add_(weight.grad, alpha=-LR)
        baseline.add_(baseline.grad, alpha=-LR)
    torch.testing.assert_close(weight, baseline, atol=0, rtol=0)

    # Mutation sensitivity: rank 0 has no local targets, so a missing forward SUM would
    # return all zeros there; a missing backward SUM would leave every rank's gradient low
    # by CP (or zero on rank 0). The correct result above cannot pass either mutation.
    if rank == 0:
        assert (global_rows.detach() - local_rows.detach()).abs().max() >= 2.0
        assert baseline.grad.abs() >= 1.0
    mutated = torch.nn.Parameter(torch.tensor(INITIAL_WEIGHT))
    bad_rows = _ForwardOnlySum.apply(mutated * COEFFICIENTS[rank], cp_config.process_group)
    (bad_rows * ROW_WEIGHTS).sum().backward()
    dist.all_reduce(mutated.grad, op=dist.ReduceOp.SUM)
    mutated.grad /= cp_size
    assert (mutated.grad - baseline.grad).abs() >= 1.0

    # An entirely ignored row still must take the collective on every rank and preserve a
    # graph-connected zero, so one rank cannot bypass peers during backward.
    weight.grad = None
    all_ignored = cp_sum_rows(weight * torch.zeros_like(ROW_WEIGHTS), cp_config)
    all_ignored.sum().backward()
    torch.testing.assert_close(all_ignored, torch.zeros_like(ROW_WEIGHTS), atol=0, rtol=0)
    torch.testing.assert_close(weight.grad, torch.zeros_like(weight), atol=0, rtol=0)


@pytest.mark.parametrize("cp_size", [2, 4])
def test_autograd_sum_matches_unsplit_row_loss_gradients_and_optimizer_step(cp_size):
    run_gloo_ranks(_worker, cp_size, cp_size, pg_timeout=timedelta(seconds=40))


def test_cp1_is_an_identity_in_fp32():
    rows = torch.nn.Parameter(torch.tensor([1.0, -2.0], dtype=torch.bfloat16))
    result = cp_sum_rows(rows, None)
    assert result.dtype == torch.float32
    torch.testing.assert_close(result, rows.float(), atol=0, rtol=0)
    result.sum().backward()
    torch.testing.assert_close(rows.grad, torch.ones_like(rows), atol=0, rtol=0)


def _reference_offline_loss(losses, mask, loss_type):
    """Plain full-row offline formula, independent of the CP reducer under test."""
    group_weights = 1.0 / GROUP_SIZES.float()
    numerators = (losses * mask).sum(dim=1) * group_weights
    counts = mask.sum(dim=1)
    if loss_type == "grpo":
        return (numerators / counts.clamp(min=1)).sum() / group_weights.sum()
    if loss_type == "bnpo":
        return numerators.sum() / (counts * group_weights).sum().clamp(min=1)
    return numerators.sum() / (group_weights.sum() * MAX_COMPLETION_LENGTH)


def _offline_worker(rank: int, cp_size: int) -> None:
    cp_config = CPConfig(cp_size=cp_size, world_size=cp_size, gpus_per_node=cp_size)
    full_coefficients = TOKEN_COEFFICIENTS[:cp_size].permute(1, 0, 2).reshape(3, -1)
    full_mask = SUPERVISION[:cp_size].permute(1, 0, 2).reshape(3, -1)
    for loss_type in ("grpo", "bnpo", "dr_grpo"):
        weight = torch.nn.Parameter(torch.tensor(INITIAL_WEIGHT))
        local_loss = weight * TOKEN_COEFFICIENTS[rank]
        got = offline_loss(
            local_loss,
            SUPERVISION[rank],
            GROUP_SIZES,
            loss_type=loss_type,
            max_completion_length=MAX_COMPLETION_LENGTH,
            row_token_counts=full_mask.sum(dim=1),
            cp_config=cp_config,
        )
        reference_weight = torch.nn.Parameter(torch.tensor(INITIAL_WEIGHT))
        want = _reference_offline_loss(reference_weight * full_coefficients, full_mask, loss_type)
        torch.testing.assert_close(got, want, atol=1e-6, rtol=1e-6)

        got.backward()
        want.backward()
        dist.all_reduce(weight.grad, op=dist.ReduceOp.SUM)
        weight.grad /= cp_size
        torch.testing.assert_close(weight.grad, reference_weight.grad, atol=1e-6, rtol=1e-6)

        with torch.no_grad():
            weight.add_(weight.grad, alpha=-LR)
            reference_weight.add_(reference_weight.grad, alpha=-LR)
        torch.testing.assert_close(weight, reference_weight, atol=1e-6, rtol=1e-6)

        # Rank 0 has no supervised tokens. Dropping the numerator's reduction produces a local zero
        # instead of the full-row objective, while an SFT-style extra ×CP mis-scales it.
        if rank == 0:
            assert got.detach().abs() > 0.01, "fixture cannot detect a dropped forward collective"
            assert (got.detach() * cp_size - want.detach()).abs() > 0.01


@pytest.mark.parametrize("cp_size", [2, 4])
def test_offline_objectives_match_full_rows_after_mean_grad_sync(cp_size):
    run_gloo_ranks(_offline_worker, cp_size, cp_size, pg_timeout=timedelta(seconds=40))


def test_offline_loss_rejects_invalid_row_shapes_and_dr_grpo_window():
    local = torch.ones(2, 3)
    mask = torch.ones_like(local)
    with pytest.raises(ValueError, match="one supervision mask"):
        offline_loss(
            local, mask[:, :2], torch.tensor([2, 2]), loss_type="grpo", max_completion_length=3, cp_config=None
        )
    with pytest.raises(ValueError, match="positive max_completion_length"):
        offline_loss(local, mask, torch.tensor([2, 2]), loss_type="dr_grpo", max_completion_length=0, cp_config=None)
    with pytest.raises(ValueError, match="one supervised-token count"):
        offline_loss(
            local,
            mask,
            torch.tensor([2, 2]),
            loss_type="grpo",
            max_completion_length=3,
            row_token_counts=torch.ones(3),
        )
    # A shard's own mask undercounts its rows; refused before the numerator's collective.
    with pytest.raises(ValueError, match="complete row_token_counts"):
        offline_loss(
            local,
            mask,
            torch.tensor([2, 2]),
            loss_type="bnpo",
            max_completion_length=3,
            cp_config=SimpleNamespace(cp_size=2),
        )


@pytest.mark.parametrize("loss_type", ["grpo", "bnpo", "dr_grpo"])
def test_local_and_pipeline_microbatch_reductions_match_the_full_row_oracle(loss_type):
    full = TOKEN_COEFFICIENTS.permute(1, 0, 2).reshape(3, -1).clone().requires_grad_()
    mask = SUPERVISION.permute(1, 0, 2).reshape(3, -1)
    expected = _reference_offline_loss(full, mask, loss_type)
    local = offline_loss(full, mask, GROUP_SIZES, loss_type=loss_type, max_completion_length=MAX_COMPLETION_LENGTH)
    numerator = sum(
        offline_loss_numerator(full[i : i + 1], mask[i : i + 1], GROUP_SIZES[i : i + 1], loss_type=loss_type)
        for i in range(3)
    )
    pipeline = numerator / offline_loss_normalizer(
        mask.sum(dim=1), GROUP_SIZES, loss_type=loss_type, max_completion_length=MAX_COMPLETION_LENGTH
    )
    for actual in (local, pipeline):
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(
            torch.autograd.grad(actual, full, retain_graph=True)[0],
            torch.autograd.grad(expected, full, retain_graph=True)[0],
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
