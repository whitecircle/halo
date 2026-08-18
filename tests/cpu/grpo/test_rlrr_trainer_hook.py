#!/usr/bin/env python
"""Test DistributedGRPOTrainer's RLRR advantage hook in single-process isolation.

Exercises ``_apply_rlrr_advantages`` (the production online-GRPO integration)
against a fake trainer ``self`` so the gather/slice/recompute logic is validated
without a GPU or distributed init.

Run: python tests/cpu/grpo/test_rlrr_trainer_hook.py
"""

import types
from collections import deque

import pytest
import torch

from src.args.mixins import RLRRConfig
from src.trainers.grpo.online import DistributedGRPOTrainer


def _fake_self(rlrr_config, rewards_per_func, num_generations):
    """Minimal stand-in exposing only what _apply_rlrr_advantages touches.

    accelerator.gather is identity here (single process); process_index=0. The real
    ``_gathered_rewards`` / ``_local_slice`` helpers are exercised (bound below), so the
    stash-aggregation and group-integrity logic is the production code, not a mock.
    """
    me = types.SimpleNamespace(
        _rlrr_config=rlrr_config,
        _last_rewards_per_func=rewards_per_func,
        reward_weights=torch.ones(rewards_per_func.shape[1]),
        num_generations=num_generations,
        model=types.SimpleNamespace(training=True),
        args=types.SimpleNamespace(multi_objective_aggregation="sum_then_normalize"),
        accelerator=types.SimpleNamespace(process_index=0, gather=lambda x: x),
        # TRL appends one entry per gathered completion before the hook runs, into a BOUNDED deque
        # (generation_batch_size) — a plain list would support operations production cannot.
        _logs={"advantages": deque([0.0] * rewards_per_func.shape[0], maxlen=rewards_per_func.shape[0])},
    )
    me._gathered_rewards = types.MethodType(DistributedGRPOTrainer._gathered_rewards, me)
    me._local_slice = types.MethodType(DistributedGRPOTrainer._local_slice, me)
    me._install_advantages = types.MethodType(DistributedGRPOTrainer._install_advantages, me)
    return me


def _result(advantages, completion_lengths):
    # completion_mask: rows of ones of the given lengths (padded).
    max_len = max(completion_lengths)
    mask = torch.zeros(len(completion_lengths), max_len, dtype=torch.long)
    for i, length in enumerate(completion_lengths):
        mask[i, :length] = 1
    return {"advantages": torch.zeros(len(advantages)), "completion_mask": mask}


def test_hook_noop_when_disabled():
    me = _fake_self(None, torch.tensor([[1.0], [0.0]]), 2)
    result = _result([0.0, 0.0], [3, 5])
    before = result["advantages"].clone()
    DistributedGRPOTrainer._apply_rlrr_advantages(me, result)
    assert torch.equal(result["advantages"], before), "disabled RLRR must not touch advantages"


def test_hook_recomputes_advantages():
    # One group of 4: rewards [1,1,0,0] -> correct/correct/incorrect/incorrect.
    rewards_per_func = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    me = _fake_self(RLRRConfig(mode="hrr"), rewards_per_func, 4)
    result = _result([0.0] * 4, completion_lengths=[5, 5, 5, 5])
    DistributedGRPOTrainer._apply_rlrr_advantages(me, result)
    adv = result["advantages"]
    assert adv.shape == (4,), adv.shape
    # Correct responses must end up with higher advantage than incorrect ones.
    assert adv[:2].min() > adv[2:].max(), adv


def test_hook_all_correct_group_keeps_signal_via_length():
    # All correct, distinct lengths -> length re-ranking yields non-zero advantages.
    rewards_per_func = torch.tensor([[1.0], [1.0], [1.0], [1.0]])
    me = _fake_self(RLRRConfig(mode="hrr", lam=4.0), rewards_per_func, 4)
    result = _result([0.0] * 4, completion_lengths=[2, 10, 20, 30])
    DistributedGRPOTrainer._apply_rlrr_advantages(me, result)
    adv = result["advantages"]
    assert adv.std() > 0, f"all-correct group should retain signal, got {adv}"
    assert adv[0] > adv[-1], adv  # shortest correct ranked best


def test_build_rlrr_config_from_args():
    """The RLVR script args build a usable RLRRConfig only when use_rlrr is set."""
    from src.args.rlvr_online_grpo_args import RLVROnlineGRPOScriptArguments

    assert RLVROnlineGRPOScriptArguments(use_rlrr=False).build_rlrr_config() is None
    cfg = RLVROnlineGRPOScriptArguments(
        use_rlrr=True, rlrr_mode="prr", rlrr_tau=0.2, rlrr_correctness_clip=False
    ).build_rlrr_config()
    assert isinstance(cfg, RLRRConfig)
    assert cfg.mode == "prr" and cfg.tau == 0.2 and cfg.correctness_clip is False


def test_hook_noop_in_eval_mode():
    # Eval mode (model.training=False) must leave advantages untouched.
    rewards_per_func = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    me = _fake_self(RLRRConfig(mode="hrr"), rewards_per_func, 4)
    me.model.training = False
    result = _result([0.0] * 4, completion_lengths=[5, 5, 5, 5])
    before = result["advantages"].clone()
    DistributedGRPOTrainer._apply_rlrr_advantages(me, result)
    assert torch.equal(result["advantages"], before), "eval mode must not reshape advantages"


def test_rlrr_advantages_reach_the_completions_log():
    """The durable completions record must show the advantages RLRR actually trained on.

    TRL fills ``_logs["advantages"]`` with its own group-normalized values before the hook runs, so
    without the tail overwrite the parquet/console record reports a set no gradient ever saw — the
    exact numbers a user reads when diagnosing an RLRR run.
    """
    rewards_per_func = torch.tensor([[1.0], [1.0], [0.0], [0.0]])
    me = _fake_self(RLRRConfig(mode="hrr"), rewards_per_func, 4)
    trl_logged = deque([-99.0] * 4, maxlen=4)  # TRL's own values, already appended
    me._logs = {"advantages": trl_logged}
    result = _result([0.0] * 4, completion_lengths=[5, 5, 5, 5])

    DistributedGRPOTrainer._apply_rlrr_advantages(me, result)

    assert list(trl_logged) != [-99.0] * 4, "RLRR advantages never reached _logs — the record is TRL's, not ours"
    # approx: the log keeps the float64 gathered values, the trained tensor is cast to the result dtype.
    assert list(trl_logged) == pytest.approx(result["advantages"].tolist(), rel=1e-6), (
        f"logged advantages must be the trained ones: {trl_logged} vs {result['advantages'].tolist()}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
