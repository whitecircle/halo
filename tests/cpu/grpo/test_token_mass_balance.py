#!/usr/bin/env python
"""Token-mass balance of GRPO advantages. A group's advantages sum to zero but their token-weighted sum does not, so
``balance_token_mass`` scales the heavier sign down until the round's net push is zero."""

import types
from collections import defaultdict

import pytest
import torch

from halo.trainers.grpo.grpo_rules import reject_unbalanceable_token_mass
from halo.trainers.grpo.objective.loss_inputs import (
    NEGATIVE_ONLY_MASS_KEY,
    NET_TOKEN_MASS_KEY,
    TOKEN_MASS_SCALE_KEY,
    record_token_mass,
)
from halo.trainers.grpo.online import DistributedOnlineGRPOTrainer


def _local(x):
    return x


def _balance(advantages, weights, gather_fn=_local):
    return record_token_mass(advantages, weights.unsqueeze(1), None, gather_fn, defaultdict(list), enabled=True)


def _net(advantages, weights):
    """The round's net share of token-weighted advantage, in [-1, 1]."""
    return float((advantages * weights).sum() / (advantages.abs() * weights).sum())


def test_long_failures_shrink_the_negatives_until_the_push_nets_to_zero():
    advantages = torch.tensor([0.8, 0.8, -0.8, -0.8])
    weights = torch.tensor([100.0, 100.0, 300.0, 300.0])  # failures three times as long as solves
    balance = _balance(advantages, weights)
    assert balance.net == pytest.approx((160 - 480) / 640)
    assert balance.positive_scale == 1.0 and balance.scale == pytest.approx(1 / 3)
    balanced = balance.apply(advantages)
    assert _net(balanced, weights) == pytest.approx(0.0, abs=1e-6)
    assert torch.equal(balanced[:2], advantages[:2]), "the lighter sign keeps its advantages"


def test_long_solves_shrink_the_positives_instead():
    advantages = torch.tensor([1.0, -0.5, -0.5])
    weights = torch.tensor([400.0, 100.0, 100.0])
    balance = _balance(advantages, weights)
    assert balance.net > 0 and balance.negative_scale == 1.0
    assert _net(balance.apply(advantages), weights) == pytest.approx(0.0, abs=1e-6)


def test_balancing_keeps_every_sign_and_the_order_within_a_sign():
    advantages = torch.tensor([1.2, 0.3, -0.1, -0.9, -2.0])
    weights = torch.tensor([50.0, 80.0, 400.0, 500.0, 900.0])
    balanced = _balance(advantages, weights).apply(advantages)
    assert torch.equal(torch.sign(balanced), torch.sign(advantages))
    negatives = balanced[advantages < 0]
    assert torch.equal(torch.argsort(negatives), torch.argsort(advantages[advantages < 0]))


@pytest.mark.parametrize(
    ("advantages", "net"),
    [(torch.tensor([0.5, 0.2]), 1.0), (torch.tensor([-0.5, -0.2]), -1.0)],
    ids=["positive-only", "negative-only"],
)
def test_a_round_with_mass_on_one_sign_only_trains_nothing_on_its_advantages(advantages, net):
    """Nothing on the other side cancels it. Kept, a round whose positives the IS masks all dropped would push down
    every sampled token at full strength."""
    balance = _balance(advantages, torch.full_like(advantages, 10.0))
    assert balance.net == net and balance.scale == 0.0
    assert torch.equal(balance.apply(advantages), torch.zeros_like(advantages))


def test_the_heavier_sign_shrinks_continuously_to_nothing_as_the_lighter_side_empties():
    """A round whose lighter side keeps one token at a tiny ratio trains almost nothing; at zero it must not jump."""
    advantages = torch.tensor([1.0, -1.0, -1.0])
    scales = [
        _balance(advantages, torch.tensor([positive_tokens, 500.0, 500.0])).negative_scale
        for positive_tokens in (1.0, 1e-3, 1e-6, 0.0)
    ]
    assert scales[:3] == [pytest.approx(1 / 1000), pytest.approx(1e-6), pytest.approx(1e-9)]
    assert scales[3] == 0.0


def test_a_round_with_no_mass_keeps_its_advantages():
    advantages = torch.zeros(3)
    balance = _balance(advantages, torch.full_like(advantages, 10.0))
    assert balance.net == 0.0 and balance.scale == 1.0
    assert torch.equal(balance.apply(advantages), advantages)


@pytest.mark.parametrize(
    ("advantages", "ratio_at"),
    [
        (torch.tensor([float("nan"), -0.5]), None),
        (torch.tensor([0.5, float("inf")]), None),
        (torch.tensor([0.5, -0.5]), float("nan")),
    ],
    ids=["nan-advantage", "inf-advantage", "nan-ratio"],
)
@pytest.mark.parametrize("enabled", [True, False], ids=["on", "off"])
def test_a_mass_that_is_not_finite_raises_on_every_rank(advantages, ratio_at, enabled):
    """Every rank reads the same world sum, so all raise together. Carried on, its scales would turn every row of the
    heavier sign to NaN."""
    loss_mask = torch.ones(2, 4)
    ratio = torch.ones(2, 4)
    if ratio_at is not None:
        ratio[1, 3] = ratio_at
    with pytest.raises(RuntimeError, match="not finite"):
        record_token_mass(advantages, loss_mask, ratio, _local, defaultdict(list), enabled=enabled)


def test_every_rank_takes_the_scales_of_the_whole_round():
    """One rank holds only the short solves, the other only the long failures; the round balances over both."""
    rank0 = (torch.tensor([0.8, 0.8]), torch.tensor([100.0, 100.0]))
    rank1 = (torch.tensor([-0.8, -0.8]), torch.tensor([300.0, 300.0]))
    # What each rank hands the collective, recorded off a first pass; the gather then returns them all.
    sent = []
    for a, w in (rank0, rank1):
        _balance(a, w, lambda local: sent.append(local) or local)
    world = torch.cat(sent)
    scales = [_balance(a, w, lambda _local: world) for a, w in (rank0, rank1)]
    assert scales[0] == scales[1]
    assert scales[0].negative_scale == pytest.approx(1 / 3)


def test_rows_replicated_across_tensor_parallel_ranks_change_nothing():
    advantages = torch.tensor([0.6, -0.6])
    weights = torch.tensor([100.0, 250.0])
    once = _balance(advantages, weights)
    twice = _balance(advantages, weights, lambda x: torch.cat([x, x]))
    assert once == twice


def _env_round():
    """Rows: a solve (one short row), a long failure (two rows) and a failure the trust region masked (IS ratio 0)."""
    advantages = torch.tensor([1.0, -0.5, -0.5, -0.5])
    loss_mask = torch.zeros(4, 8)
    loss_mask[0, :2] = 1  # the solve: 2 tokens
    loss_mask[1, :8] = 1  # the failure's two rows: 8 + 4 tokens
    loss_mask[2, :4] = 1
    loss_mask[3, :8] = 1  # the masked failure's row
    ratio = torch.ones(4, 8)
    ratio[3] = 0.0
    return advantages, loss_mask, ratio


_CUT_TURN = torch.tensor([False, False, True, False])  # the failure's 4-token row, a negative-only one


def test_a_token_weighs_its_place_in_the_loss_times_its_is_ratio():
    """The masked failure's tokens carry no gradient and so no mass: the push is 2 positive against 6
    negative, the negatives shrink by a third, and the round nets to zero."""
    advantages, loss_mask, ratio = _env_round()
    metrics = defaultdict(list)
    balance = record_token_mass(advantages, loss_mask, ratio, _local, metrics, enabled=True)
    assert balance is not None
    assert metrics[NET_TOKEN_MASS_KEY] == [pytest.approx((2 - 6) / 8)]
    assert metrics[TOKEN_MASS_SCALE_KEY] == [pytest.approx(1 / 3)]
    balanced = balance.apply(advantages)
    assert torch.allclose(balanced, torch.tensor([1.0, -0.5 / 3, -0.5 / 3, -0.5 / 3]))
    assert _net(balanced, (loss_mask * ratio).sum(dim=1)) == pytest.approx(0.0, abs=1e-6)


def test_with_the_balance_off_the_net_mass_is_still_logged():
    advantages, loss_mask, ratio = _env_round()
    metrics = defaultdict(list)
    assert record_token_mass(advantages, loss_mask, ratio, _local, metrics, enabled=False) is None
    assert metrics[NET_TOKEN_MASS_KEY] == [pytest.approx(-0.5)]
    assert TOKEN_MASS_SCALE_KEY not in metrics


def test_the_balance_weighs_a_negative_only_row_and_nets_the_whole_round_to_zero():
    """The 4-token failing row is a cut turn, trained on its negative advantage only. Left out of the balance, its 2 of
    mass would stay the round's whole net push once the solve's 2 cancelled the other failure's 4. Weighed in, the
    solve's 2 against all 6 negative shrinks every negative to a third, the cut turn's too, and the whole round nets to
    zero. The logged net stays the trainable rows'."""
    advantages, loss_mask, ratio = _env_round()
    metrics = defaultdict(list)
    balance = record_token_mass(advantages, loss_mask, ratio, _local, metrics, enabled=True, negative_only=_CUT_TURN)
    assert metrics[NET_TOKEN_MASS_KEY] == [pytest.approx((2 - 4) / 6)]
    assert metrics[TOKEN_MASS_SCALE_KEY] == [pytest.approx(1 / 3)]
    assert metrics[NEGATIVE_ONLY_MASS_KEY] == [pytest.approx((2 / 3) / 4)]

    balanced = balance.apply(advantages)
    assert torch.allclose(balanced, torch.tensor([1.0, -0.5 / 3, -0.5 / 3, -0.5 / 3]))
    weights = (loss_mask * ratio).sum(dim=1)
    assert float((balanced * weights).sum()) == pytest.approx(0.0, abs=1e-6), "the whole round nets to zero"
    assert _net(balanced[~_CUT_TURN], weights[~_CUT_TURN]) > 0, "the trainable rows alone keep a push up"


def test_with_the_balance_off_the_net_mass_is_the_trainable_rows_and_the_share_is_plain():
    advantages, loss_mask, ratio = _env_round()
    metrics = defaultdict(list)
    assert (
        record_token_mass(advantages, loss_mask, ratio, _local, metrics, enabled=False, negative_only=_CUT_TURN)
        is None
    )
    assert metrics[NET_TOKEN_MASS_KEY] == [pytest.approx((2 - 4) / 6)]
    assert metrics[NEGATIVE_ONLY_MASS_KEY] == [pytest.approx(2 / 8)]

    metrics = defaultdict(list)
    record_token_mass(advantages, loss_mask, ratio, _local, metrics, enabled=True)
    assert NEGATIVE_ONLY_MASS_KEY not in metrics, "a trainer with no negative-only rows logs no share"


def test_a_round_with_no_positive_mass_trains_nothing_under_the_balance():
    """With no positive mass to cancel it, every negative, the cut turn's included, shrinks to zero."""
    _, loss_mask, ratio = _env_round()
    advantages = torch.tensor([0.0, -0.5, -0.5, -0.5])  # the solve's positive advantage gone
    metrics = defaultdict(list)
    balance = record_token_mass(advantages, loss_mask, ratio, _local, metrics, enabled=True, negative_only=_CUT_TURN)
    assert metrics[TOKEN_MASS_SCALE_KEY] == [0.0] and metrics[NEGATIVE_ONLY_MASS_KEY] == [0.0]
    assert torch.equal(balance.apply(advantages), torch.zeros(4))
    metrics = defaultdict(list)
    record_token_mass(advantages, loss_mask, ratio, _local, metrics, enabled=False, negative_only=_CUT_TURN)
    assert metrics[NEGATIVE_ONLY_MASS_KEY] == [pytest.approx(2 / 6)]


@pytest.mark.parametrize("seed", range(20))
def test_on_any_round_the_balance_nets_the_whole_trained_mass_to_zero_and_the_logs_add_up(seed):
    """Random rows, lengths, IS ratios and cut turns. Balanced, the round's whole push (each row's advantage times its
    IS-weighted tokens, a cut turn's negative advantage only) is zero, no sign flips, nothing grows; off, the logged net
    and share give that push as ``(1 - s) * net - s``."""
    generator = torch.Generator().manual_seed(seed)
    rows = 12
    advantages = torch.randn(rows, generator=generator)
    loss_mask = (torch.rand(rows, 16, generator=generator) < torch.rand(rows, 1, generator=generator)).float()
    ratio = torch.rand(rows, 16, generator=generator) * 2
    negative_only = torch.rand(rows, generator=generator) < 0.3
    trained = ~negative_only | (advantages < 0)  # a cut turn trains on a negative advantage only
    weights = (loss_mask * ratio).sum(dim=1) * trained

    def push(values):
        return float((values * weights).sum() / (values.abs() * weights).sum())

    metrics = defaultdict(list)
    record_token_mass(advantages, loss_mask, ratio, _local, metrics, enabled=False, negative_only=negative_only)
    net, share = metrics[NET_TOKEN_MASS_KEY][0], metrics[NEGATIVE_ONLY_MASS_KEY][0]
    assert (1 - share) * net - share == pytest.approx(push(advantages), abs=1e-5)

    balance = record_token_mass(advantages, loss_mask, ratio, _local, defaultdict(list), True, negative_only)
    balanced = balance.apply(advantages)
    assert float((balanced * weights).sum()) == pytest.approx(0.0, abs=1e-4)
    assert torch.all((torch.sign(balanced) == torch.sign(advantages)) | (balanced == 0))
    assert torch.all(balanced.abs() <= advantages.abs() + 1e-7)
    assert (
        min(balance.positive_scale, balance.negative_scale)
        <= 1.0
        == max(balance.positive_scale, balance.negative_scale)
    )


def _grpo_args(loss_type="dapo", top_entropy_quantile=1.0, off_policy_mask_threshold=None):
    return types.SimpleNamespace(
        loss_type=loss_type,
        top_entropy_quantile=top_entropy_quantile,
        off_policy_mask_threshold=off_policy_mask_threshold,
    )


@pytest.mark.parametrize(
    "args",
    [
        _grpo_args("grpo"),
        _grpo_args("sapo"),
        _grpo_args("bnpo"),
        _grpo_args("vespo"),
        _grpo_args(top_entropy_quantile=0.2),
        _grpo_args(off_policy_mask_threshold=0.5),
    ],
    ids=["grpo", "sapo", "bnpo", "vespo", "entropy-mask", "off-policy-mask"],
)
def test_a_loss_where_token_mass_is_not_the_pull_is_refused(args):
    with pytest.raises(ValueError, match="balance_token_mass"):
        reject_unbalanceable_token_mass(args)


@pytest.mark.parametrize("loss_type", ["cispo", "dapo", "dr_grpo"])
def test_the_token_sum_losses_take_the_balance(loss_type):
    reject_unbalanceable_token_mass(_grpo_args(loss_type))


def _online_host(balance: bool, training: bool = True):
    host = types.SimpleNamespace(
        _balance_token_mass=balance,
        _fuses_round=False,
        model=types.SimpleNamespace(training=training),
        accelerator=types.SimpleNamespace(gather=_local, process_index=0),
        _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
        # An earlier batch's record, then the two advantages TRL logged for this one.
        _logs={"advantages": [0.3, 0.8, -0.8]},
    )
    for name in ("_install_advantages", "_local_slice"):
        setattr(host, name, getattr(DistributedOnlineGRPOTrainer, name).__get__(host))
    return DistributedOnlineGRPOTrainer._apply_token_mass_balance.__get__(host), host


def _online_result():
    """A short solve (2 loss tokens) and a long failure (4 tokens, two of them at IS ratio 0.5): 1.6
    positive mass against 2.4 negative."""
    return {
        "advantages": torch.tensor([0.8, -0.8]),
        "completion_mask": torch.tensor([[1, 1, 0, 0], [1, 1, 1, 1]]),
        "importance_sampling_ratio": torch.tensor([[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.5, 0.5]]),
    }


def test_the_online_trainer_balances_on_trls_loss_mask_and_is_ratio_and_realigns_its_record():
    apply, host = _online_host(balance=True)
    result = _online_result()
    apply(result)
    assert torch.allclose(result["advantages"], torch.tensor([0.8, -0.8 * 2 / 3]))
    assert host._logs["advantages"] == pytest.approx([0.3, 0.8, -0.8 * 2 / 3]), "only this batch's tail is rewritten"
    assert host._metrics["train"][NET_TOKEN_MASS_KEY] == [pytest.approx(-0.2)]


def test_the_online_trainer_leaves_eval_batches_and_an_off_knob_alone():
    apply, host = _online_host(balance=True, training=False)
    result = _online_result()
    apply(result)
    assert torch.equal(result["advantages"], torch.tensor([0.8, -0.8])) and not host._metrics["train"]
    apply, host = _online_host(balance=False)
    result = _online_result()
    apply(result)
    assert torch.equal(result["advantages"], torch.tensor([0.8, -0.8]))
    assert host._metrics["train"][NET_TOKEN_MASS_KEY] == [pytest.approx(-0.2)]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
