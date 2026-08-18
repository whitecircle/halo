#!/usr/bin/env python
"""``scale_rewards_std_floor`` must actually reach the online-GRPO advantage math.

The floor is the collapse guard: with ``scale_rewards="group"`` (TRL's default) a behaviourally
degenerate group — every reward within a few hundredths — divides its own noise by a near-zero std
and gets full-scale advantages. TRL's own divisor is a fixed ``1e-4``, so the floor can only be
applied by the trainer's own recompute hook.

Two gates guard that hook: the reward stash in ``_calculate_rewards`` and the early return in
``_apply_advantage_shaping``. A knob covered by one but not the other is silently inert: keying both
on ``advantage_shaping`` — ``None`` at the default ``advantage_mode="mean"`` — does exactly that.
These tests pin that both gates honour the floor on its own.

    python tests/cpu/grpo/test_online_std_floor_reaches_advantages.py
"""

from collections import deque

import pytest
import torch

from src.args.mixins import AdvantageShaping
from src.trainers.grpo.objective.advantages import group_relative_advantages
from src.trainers.grpo.online import DistributedGRPOTrainer

G = 4
# Rewards all within 2e-3 of each other: no real signal, std ~1e-3.
DEGENERATE_REWARDS = torch.tensor([0.501, 0.500, 0.499, 0.500])


class _Args:
    scale_rewards = "group"
    multi_objective_aggregation = "sum_then_normalize"


class _Accelerator:
    process_index = 0


class _OnlineStub:
    """Borrows the real gate + hook off the trainer without constructing one (needs vLLM)."""

    def __init__(self, *, std_floor=0.0, shaping=None, rlrr=None, drop_degenerate=False):
        self._scale_rewards_std_floor = std_floor
        self._advantage_shaping = shaping
        self._rlrr_config = rlrr
        self._drop_degenerate_groups = drop_degenerate
        self.num_generations = G
        self.args = _Args()
        self.accelerator = _Accelerator()
        # TRL appends one entry per gathered completion before either hook runs, into a BOUNDED
        # deque (generation_batch_size) — a plain list would support operations production cannot.
        self._logs = {"advantages": deque([0.0] * G, maxlen=G)}

    _recomputes_from_gathered_rewards = DistributedGRPOTrainer._recomputes_from_gathered_rewards
    _apply_advantage_shaping = DistributedGRPOTrainer._apply_advantage_shaping
    _install_advantages = DistributedGRPOTrainer._install_advantages
    _local_slice = DistributedGRPOTrainer._local_slice

    def _gathered_rewards(self):
        return DEGENERATE_REWARDS.clone(), torch.zeros(G, dtype=torch.bool)


def _trl_baseline_advantages():
    """What TRL computes: group-mean-centred, divided by (std + 1e-4) with no floor."""
    return group_relative_advantages(DEGENERATE_REWARDS.float(), G, "group", std_floor=0.0)


def test_std_floor_alone_arms_the_reward_stash():
    """The stash gate must fire for the floor alone, or the hook sees ``None`` and returns."""
    assert _OnlineStub(std_floor=0.2)._recomputes_from_gathered_rewards is True
    assert _OnlineStub()._recomputes_from_gathered_rewards is False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"shaping": AdvantageShaping(mode="qae")},
        {"rlrr": object()},
        {"drop_degenerate": True},
    ],
)
def test_other_hooks_still_arm_the_stash(kwargs):
    """Widening the gate must not narrow it for the pre-existing consumers."""
    assert _OnlineStub(**kwargs)._recomputes_from_gathered_rewards is True


def test_std_floor_alone_suppresses_degenerate_group_amplification():
    """With only the floor set, the hook must rewrite advantages and shrink them by ~std_floor/std."""
    baseline = _trl_baseline_advantages()
    assert baseline.abs().max() > 0.5, "fixture no longer amplifies; pick a tighter reward spread"

    stub = _OnlineStub(std_floor=0.2)
    result = {"advantages": baseline.clone()}
    stub._apply_advantage_shaping(result)

    assert result["advantages"].abs().max() < 0.05, (
        "scale_rewards_std_floor did not reach the advantages — the shaping hook returned early "
        "because advantage_shaping is None at the default advantage_mode"
    )
    # The floor replaces the tiny std with 0.2, so the amplification drops by that ratio.
    expected = group_relative_advantages(DEGENERATE_REWARDS.float(), G, "group", std_floor=0.2)
    assert torch.allclose(result["advantages"], expected, atol=1e-6)
    # The durable completions record must carry the SAME values the gradient sees, not TRL's.
    assert torch.allclose(torch.tensor(list(stub._logs["advantages"])), expected, atol=1e-6)


def test_a_short_advantage_log_raises_instead_of_misattributing_rows():
    """TRL logs one advantage per gathered completion; a shorter tail cannot be realigned.

    Overwriting the last N entries of a log that holds fewer than N would attribute each row's
    advantage to the wrong completion, and the parquet gives no sign of it.
    """
    stub = _OnlineStub(std_floor=0.2)
    stub._logs["advantages"] = deque([0.0] * (G - 1), maxlen=G - 1)
    with pytest.raises(RuntimeError, match="cannot realign the completions record"):
        stub._apply_advantage_shaping({"advantages": _trl_baseline_advantages()})


def test_no_floor_and_no_shaping_leaves_trl_advantages_untouched():
    """The default path must not be recomputed at all — TRL's values stand."""
    baseline = _trl_baseline_advantages()
    stub = _OnlineStub()
    result = {"advantages": baseline.clone()}
    stub._apply_advantage_shaping(result)
    assert torch.equal(result["advantages"], baseline)


def test_floor_preserves_advantages_when_the_group_has_real_spread():
    """A group whose std exceeds the floor must be unaffected — the floor is a bound, not a rescale."""
    spread = torch.tensor([0.0, 0.4, 0.8, 1.0])

    class _SpreadStub(_OnlineStub):
        def _gathered_rewards(self):
            return spread.clone(), torch.zeros(G, dtype=torch.bool)

    unfloored = group_relative_advantages(spread, G, "group", std_floor=0.0)
    stub = _SpreadStub(std_floor=0.2)
    result = {"advantages": unfloored.clone()}
    stub._apply_advantage_shaping(result)
    assert torch.allclose(result["advantages"], unfloored, atol=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
