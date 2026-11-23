#!/usr/bin/env python
"""MaxRL advantages (``use_maxrl``, arXiv:2602.02710): each reward minus its group mean, divided by that mean, and 0
for a group whose mean is 0.
- On a Bernoulli policy the per-sample gradient estimate is exactly unbiased for the truncated maximum-likelihood
  objective, by enumeration over every success count.
- Hand values, with a placeholder that must not move the mean; an all-zero group gives 0; a negative reward raises,
  and so does a rank whose own rewards are fine when a peer holds one.
- Online GRPO's hook reaches the advantages and the completions record; environmental GRPO's step uses the mean divisor.
- Every conflicting knob is refused with its own message; score centering, the degenerate-group drop and the token-mass
  balance resolve beside it."""

import ast
import math
import types
from collections import defaultdict, deque

import pytest
import torch

from halo.configs.grpo import RLRRConfig
from halo.configs.grpo_environmental import AsyncRolloutConfig
from halo.runtime.consensus import RankConsensus
from halo.trainers.grpo.environmental.environmental_rules import (
    reject_negative_reward_terms_under_maxrl,
    reject_unsupported_score_centering,
)
from halo.trainers.grpo.environmental.step_batch import TrajectoryRowMap
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.grpo_rules import reject_conflicting_advantage_hooks, reject_conflicting_maxrl
from halo.trainers.grpo.objective import advantages as advantages_module
from halo.trainers.grpo.objective.advantages import group_relative_advantages, maxrl_zero_mean_groups
from halo.trainers.grpo.objective.loss_inputs import MAXRL_ZERO_MEAN_GROUP_FRAC_KEY
from halo.trainers.grpo.online import DistributedOnlineGRPOTrainer
from tests.common.repo import REPO_ROOT

ONLINE_SCRIPT = REPO_ROOT / "scripts/training/grpo/online.py"


def _maxrl(rewards, num_generations, valid=None):
    return group_relative_advantages(
        torch.as_tensor(rewards, dtype=torch.float64),
        num_generations,
        "none",
        valid_mask=None if valid is None else torch.as_tensor(valid),
        normalizer="mean",
    )


# ------------------------------------------------------------------------------------------------------- estimator


@pytest.mark.parametrize("n", [4, 8, 16])
@pytest.mark.parametrize("p", [1 / 10, 1 / 3, 7 / 10])
def test_the_estimator_is_unbiased_for_the_truncated_ml_objective_on_a_bernoulli_policy(n, p):
    """E[A_j d/dp log pi(y_j)] over a group of ``n`` draws equals d/dp of J = -sum_{k<n} (1-p)^k / k, which is
    (1 - (1-p)^(n-1)) / p. Exchangeable members, so each success count k contributes the group's average term. A
    changed divisor, or a zero-mean group given anything but 0, breaks the equality."""
    expected = (1 - (1 - p) ** (n - 1)) / p
    score = {1.0: 1 / p, 0.0: -1 / (1 - p)}
    total = 0.0
    for k in range(n + 1):
        rewards = [1.0] * k + [0.0] * (n - k)
        advantages = _maxrl(rewards, n)
        per_sample = sum(a * score[r] for a, r in zip(advantages.tolist(), rewards, strict=True)) / n
        total += math.comb(n, k) * p**k * (1 - p) ** (n - k) * per_sample
    assert total == pytest.approx(expected, rel=1e-12, abs=0)


def test_std_normalized_grpo_is_not_that_estimator():
    """The control: the same enumeration under GRPO's std divisor misses the closed form."""
    n, p = 8, 1 / 3
    score = {1.0: 1 / p, 0.0: -1 / (1 - p)}
    total = 0.0
    for k in range(n + 1):
        rewards = torch.tensor([1.0] * k + [0.0] * (n - k), dtype=torch.float64)
        advantages = group_relative_advantages(rewards, n, "group")
        per_sample = sum(a * score[r] for a, r in zip(advantages.tolist(), rewards.tolist(), strict=True)) / n
        total += math.comb(n, k) * p**k * (1 - p) ** (n - k) * per_sample
    assert total != pytest.approx((1 - (1 - p) ** (n - 1)) / p, rel=1e-3)


# ------------------------------------------------------------------------------------------------------------ math


def test_one_success_in_sixteen_gets_fifteen():
    advantages = _maxrl([1.0] + [0.0] * 15, 16)
    assert advantages[0].item() == pytest.approx(15.0)
    assert advantages[1:].tolist() == pytest.approx([-1.0] * 15)


def test_hand_values_with_a_placeholder_that_does_not_move_the_mean():
    """Group one has a placeholder worth 5: the mean is 1/3 over the three valid members, not 1.5 over four."""
    advantages = _maxrl([1.0, 0.0, 0.0, 5.0, 2.0, 1.0, 1.0, 0.0], 4, valid=[True, True, True, False, *[True] * 4])
    assert advantages[:3].tolist() == pytest.approx([2.0, -1.0, -1.0])
    assert advantages[4:].tolist() == pytest.approx([1.0, 0.0, 0.0, -1.0])


def test_a_group_with_no_success_gets_zero_and_counts_as_zero_mean():
    rewards = torch.tensor([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    advantages = _maxrl(rewards, 4)
    assert torch.equal(advantages[:4], torch.zeros(4, dtype=torch.float64))
    assert torch.isfinite(advantages).all()
    assert maxrl_zero_mean_groups(rewards, 4, torch.ones(8, dtype=torch.bool)).tolist() == [True, False]


def test_a_negative_reward_raises_but_not_on_a_placeholder():
    with pytest.raises(ValueError, match="use_maxrl needs non-negative rewards.*1 of 4"):
        _maxrl([1.0, -0.5, 0.0, 0.0], 4)
    _maxrl([1.0, -0.5, 0.0, 0.0], 4, valid=[True, False, True, True])


def test_a_peer_ranks_negative_reward_raises_here_too(monkeypatch):
    """The verdict is agreed across ranks. On the environmental path the rewards are still rank-local here, so a rank
    whose own rewards are non-negative must raise with the peer that holds a negative one, or it heads into the next
    collective alone and the peers hang."""
    monkeypatch.setattr(advantages_module, "rank_consensus", lambda ok: RankConsensus(False, ok))
    with pytest.raises(ValueError, match=r"use_maxrl needs non-negative rewards.*this rank: 0 of 4"):
        _maxrl([1.0, 0.0, 0.5, 0.0], 4)


def test_the_mean_divisor_refuses_a_std_scaling():
    with pytest.raises(ValueError, match="replaces scale_rewards='group'"):
        group_relative_advantages(torch.tensor([1.0, 0.0]), 2, "group", normalizer="mean")


# ---------------------------------------------------------------------------------------------------------- online

G = 4
REWARDS = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
UNSCORABLE = torch.tensor([False, False, False, True, False, False, False, False])


class _OnlineStub:
    """Borrows the real gate and hook off the trainer without constructing one (needs vLLM)."""

    def __init__(self, *, use_maxrl=False):
        self._use_maxrl = use_maxrl
        self._rlrr_config = None
        self._drop_degenerate_groups = False
        self._scale_rewards_std_floor = 0.0
        self.num_generations = G
        self.args = types.SimpleNamespace(scale_rewards="none")
        self.accelerator = types.SimpleNamespace(process_index=0)
        self._metrics = {"train": defaultdict(list)}
        self._logs = {"advantages": deque([0.0] * len(REWARDS), maxlen=len(REWARDS))}

    _recomputes_from_gathered_rewards = DistributedOnlineGRPOTrainer._recomputes_from_gathered_rewards
    _apply_maxrl_advantages = DistributedOnlineGRPOTrainer._apply_maxrl_advantages
    _install_advantages = DistributedOnlineGRPOTrainer._install_advantages
    _local_slice = DistributedOnlineGRPOTrainer._local_slice

    def _gathered_rewards(self):
        return REWARDS.clone(), UNSCORABLE.clone()


def test_maxrl_alone_arms_the_reward_stash():
    assert _OnlineStub(use_maxrl=True)._recomputes_from_gathered_rewards is True
    assert _OnlineStub()._recomputes_from_gathered_rewards is False


def test_the_online_hook_reaches_the_advantages_and_the_completions_record():
    """The unscorable placeholder stays out of its group's mean and gets 0, as TRL gives it."""
    stub = _OnlineStub(use_maxrl=True)
    result = {"advantages": torch.zeros(len(REWARDS))}
    stub._apply_maxrl_advantages(result)
    expected = torch.tensor([2.0, -1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert torch.allclose(result["advantages"], expected)
    assert torch.allclose(torch.tensor(list(stub._logs["advantages"])), expected)
    assert stub._metrics["train"][MAXRL_ZERO_MEAN_GROUP_FRAC_KEY] == [0.5]


def test_the_online_hook_is_off_without_the_knob():
    stub = _OnlineStub()
    result = {"advantages": torch.ones(len(REWARDS))}
    stub._apply_maxrl_advantages(result)
    assert torch.equal(result["advantages"], torch.ones(len(REWARDS)))
    assert MAXRL_ZERO_MEAN_GROUP_FRAC_KEY not in stub._metrics["train"]


def _online_args(**overrides):
    return types.SimpleNamespace(
        **{
            "scale_rewards": "none",
            "multi_objective_aggregation": "sum_then_normalize",
            "loss_type": "dapo",
            "top_entropy_quantile": 1.0,
            "off_policy_mask_threshold": None,
            **overrides,
        }
    )


def _resolve_online(grpo_args, rlrr=None, std_floor=0.0, drop=False, balance=False):
    reject_conflicting_advantage_hooks(
        grpo_args,
        rlrr,
        std_floor=std_floor,
        drop_degenerate_groups=drop,
        balance_token_mass=balance,
        recomputes_from_gathered_rewards=True,
        use_maxrl=True,
    )


@pytest.mark.parametrize(
    ("args", "kwargs", "match"),
    [
        ({"scale_rewards": "group"}, {}, r"use_maxrl with scale_rewards='group'.*Set scale_rewards: none"),
        ({"scale_rewards": "batch"}, {}, r"use_maxrl with scale_rewards='batch'"),
        ({}, {"std_floor": 0.05}, r"use_maxrl with scale_rewards_std_floor=0.05"),
        ({}, {"rlrr": RLRRConfig()}, "use_maxrl and use_rlrr cannot be combined"),
        ({"multi_objective_aggregation": "normalize_then_sum"}, {}, r"use_maxrl recompute rewards"),
    ],
)
def test_online_refuses_each_conflict_with_its_own_message(args, kwargs, match):
    with pytest.raises(ValueError, match=match):
        _resolve_online(_online_args(**args), **kwargs)


def test_online_resolves_beside_the_degenerate_drop_and_the_token_mass_balance():
    _resolve_online(_online_args(), drop=True, balance=True)


def test_the_online_script_hands_use_maxrl_to_the_trainer():
    """A script that drops it from the trainer call parses ``--use_maxrl`` and trains on TRL's std advantages."""
    tree = ast.parse(ONLINE_SCRIPT.read_text())
    call = next(
        node for node in ast.walk(tree) if isinstance(node, ast.Call) and getattr(node.func, "id", "") == "trainer_cls"
    )
    passed = {kw.arg: ast.unparse(kw.value) for kw in call.keywords if kw.arg}
    assert passed.get("use_maxrl") == "args.use_maxrl", passed


# --------------------------------------------------------------------------------------------------- environmental


class _WorldMetrics:
    def __init__(self):
        self.fractions = {}

    def fraction(self, key, numerator, denominator):
        self.fractions[key] = (int(numerator), int(denominator))


def _env_step(rewards, num_generations, mode, valid=None, use_maxrl=True):
    me = types.SimpleNamespace(
        args=types.SimpleNamespace(scale_rewards="none"),
        async_config=types.SimpleNamespace(scale_rewards_std_floor=0.0, use_maxrl=use_maxrl),
        _world_metrics=_WorldMetrics(),
    )
    valid = [True] * len(rewards) if valid is None else valid
    rollouts = [
        types.SimpleNamespace(
            trajectory=types.SimpleNamespace(truncated=False), counts_toward_baseline=v, total_reward=r
        )
        for r, v in zip(rewards.tolist(), valid, strict=True)
    ]
    rows = TrajectoryRowMap(rollouts, [1] * len(rewards), 0, False)
    _state, traj_advantages, _rows = DistributedAsyncEnvironmentalGRPOTrainer._step_advantages(
        me, rewards, rows, 0, num_generations, mode
    )
    return traj_advantages, me._world_metrics.fractions


def test_the_environmental_step_divides_by_the_valid_mean_and_logs_zero_mean_groups_in_train():
    rewards = torch.tensor([1.0, 0.0, 0.0, 9.0, 0.0, 0.0, 0.0, 0.0])
    valid = [True, True, True, False, True, True, True, True]
    advantages, fractions = _env_step(rewards, 4, "train", valid=valid)
    assert advantages[:3].tolist() == pytest.approx([2.0, -1.0, -1.0])
    assert advantages[4:].tolist() == [0.0, 0.0, 0.0, 0.0]
    assert fractions == {MAXRL_ZERO_MEAN_GROUP_FRAC_KEY: (1, 2)}
    _advantages, eval_fractions = _env_step(rewards, 4, "eval", valid=valid)
    assert eval_fractions == {}


def test_the_environmental_step_keeps_the_mean_baseline_without_the_knob():
    advantages, fractions = _env_step(torch.tensor([1.0, 0.0, 0.0, 0.0]), 4, "train", use_maxrl=False)
    assert advantages.tolist() == pytest.approx([0.75, -0.25, -0.25, -0.25])
    assert fractions == {}


@pytest.mark.parametrize(
    ("price_on", "floor", "match"),
    [
        (True, 0.0, "use_maxrl with reasoning_price on: it subtracts"),
        (False, 0.2, "use_maxrl with reasoning_floor on: it subtracts"),
        (True, 0.2, "use_maxrl with reasoning_price and reasoning_floor on: they subtract"),
    ],
)
def test_the_reasoning_term_refusal_names_each_term_that_is_on(price_on, floor, match):
    """The construction path is pinned in ``test_env_trainer_wiring.py``. This pins the wording per term."""
    with pytest.raises(ValueError, match=match):
        reject_negative_reward_terms_under_maxrl(types.SimpleNamespace(reasoning_floor=floor), price_on=price_on)
    reject_negative_reward_terms_under_maxrl(types.SimpleNamespace(reasoning_floor=0.0), price_on=False)


def test_score_centering_resolves_beside_maxrl_and_the_advantages_stay_maxrls():
    """Score centering replaces each token's score with a centered one and owns its importance weight. The token's
    coefficient is still its trajectory's advantage, which MaxRL sets, so neither knob changes the other's term."""
    config = AsyncRolloutConfig(
        score_centering_top_k=16,
        rollout_backend="sglang",
        train_on_sampled_tokens=True,
        use_chunked_grpo_logprobs=True,
        rollout_top_p=1.0,
        use_maxrl=True,
    )
    reject_unsupported_score_centering(config)
    reject_conflicting_maxrl("none", config.scale_rewards_std_floor)
    advantages, _fractions = _env_step(torch.tensor([1.0, 0.0, 0.0, 0.0]), 4, "train")
    assert advantages.tolist() == pytest.approx([3.0, -1.0, -1.0, -1.0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
