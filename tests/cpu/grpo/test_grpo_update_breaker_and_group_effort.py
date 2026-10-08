"""Unit tests for three env-GRPO stability mechanisms (no GPU / no live model):

1. ``_update_breaker_tripped`` — the IS trust-region circuit breaker (``skip_update_masked_frac``):
   when the mask stages (geo-band / veto / OPSM) have zeroed the ratio of more than the configured
   fraction of IS-corrected trajectories, the step must be declared broken so the caller zeroes the
   whole step's policy gradient (training on the surviving selection-biased sample amplifies the
   drift). Below the threshold, and with the knob unset, the verdict must be False. The verdict is
   returned, not applied, so its EFFECT is read off the real batch build
   (``tests/common/env_grpo_batch.py``): both advantage tensors zeroed, the completions record and the
   step diagnostics written after that. The same build pins what the step around the breaker trains:
   the empty-step halt, the token-mass balance and the engine's forced closes.

2. Group-level reasoning-effort conditioning — ``_stamp_group_efforts`` (trainer side) draws ONE
   effort per generation group and stamps it into every member's rollout context;
   ``bind_episode_effort`` (Ray-actor side) must prefer that context level over the env's own
   per-episode draw. Together they keep GRPO group members identically conditioned, so the effort
   lottery can never become intra-group advantage noise.

3. The ``_build_training_tensors`` phase helpers — three of them issue collectives behind config- or
   mode-derived gates, so a data-dependent early ``return`` in any helper would let one rank skip a
   collective its peers enter (a watchdog hang at scale). No single-process run can show a rank skipping
   a collective, so these are read off the source tree.

    python tests/cpu/grpo/test_grpo_update_breaker_and_group_effort.py
"""

import ast
import inspect
import math
import pathlib
import textwrap
import types
from collections import defaultdict

import pytest
import torch
from accelerate import PartialState

from src.configs.async_training_config import AsyncTrainingConfig, ISMaskConfig
from src.environments.base import VALID_REASONING_EFFORTS, stable_reasoning_effort
from src.environments.episode import bind_episode_effort
from src.trainers.grpo import environmental
from src.trainers.grpo.environmental import (
    EMPTY_ROLLOUT_STEP_LIMIT,
    BatchBuildFence,
    DistributedAsyncEnvironmentalGRPOTrainer,
)
from src.trainers.grpo.objective.application import (
    NEGATIVE_ONLY_MASS_KEY,
    NET_TOKEN_MASS_KEY,
    TOKEN_MASS_SCALE_KEY,
)
from tests.common.env_grpo_batch import batch_host, build, episode, row

PartialState()  # the breaker warns through accelerate's logger, which refuses to log without it


def _breaker_host(threshold):
    """Minimal stand-in exposing exactly what ``_update_breaker_tripped`` reads."""
    host = types.SimpleNamespace(
        _skip_update_masked_frac=threshold,
        _breaker_tripped_this_step=False,
        accelerator=types.SimpleNamespace(gather=lambda x: x),  # single-process: identity
        _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
    )
    return DistributedAsyncEnvironmentalGRPOTrainer._update_breaker_tripped.__get__(host), host


def _masked_batch(masked_trajs: list[bool], tokens_per_row: int = 4):
    """One row per trajectory; a masked trajectory has ALL corrected-token ratios zeroed."""
    n = len(masked_trajs)
    ratio = torch.ones(n, tokens_per_row)
    for i, masked in enumerate(masked_trajs):
        if masked:
            ratio[i] = 0.0
    eff_corrected = torch.ones(n, tokens_per_row, dtype=torch.bool)
    traj_row_ids = torch.arange(n)
    advantages = torch.randn(n)
    return ratio, eff_corrected, traj_row_ids, advantages


def test_breaker_fires_above_threshold():
    breaker, host = _breaker_host(0.5)
    ratio, corr, ids, _adv = _masked_batch([True, True, True, False])
    assert breaker(ratio, corr, ids, 4, "train") is True, "masked frac (0.75) > threshold must break the step"
    assert host._metrics["train"]["sampling/update_skipped"] == [1.0]
    assert host._metrics["train"]["sampling/is_masked_traj_frac"] == [0.75]


def test_breaker_inert_below_threshold():
    breaker, host = _breaker_host(0.5)
    ratio, corr, ids, _adv = _masked_batch([True, False, False, False])
    assert breaker(ratio, corr, ids, 4, "train") is False, "the step must stand below the threshold"
    assert host._metrics["train"]["sampling/update_skipped"] == [0.0]
    assert host._metrics["train"]["sampling/is_masked_traj_frac"] == [0.25]


def test_breaker_partial_token_masking_is_not_a_masked_trajectory():
    # A trajectory with ANY surviving corrected token is not fully masked (a per-token zero such as
    # ``zero_engine_forced_closes`` masks tokens, not trajectories) — it must not count toward the
    # breaker fraction.
    breaker, host = _breaker_host(0.5)
    ratio, corr, ids, _adv = _masked_batch([False, False])
    ratio[0, :3] = 0.0  # 3 of 4 tokens masked, 1 survives
    assert breaker(ratio, corr, ids, 2, "train") is False
    assert host._metrics["train"]["sampling/is_masked_traj_frac"] == [0.0]


def test_breaker_disabled_and_eval_are_noops():
    for threshold, mode in ((None, "train"), (0.5, "eval")):
        breaker, host = _breaker_host(threshold)
        ratio, corr, ids, _adv = _masked_batch([True, True])
        assert breaker(ratio, corr, ids, 2, mode) is False
        assert not host._metrics["train"]["sampling/update_skipped"]
        assert not host._metrics["eval"]["sampling/update_skipped"]


def test_breaker_ignores_dummy_rows():
    # Dummy padding rows (traj_row_ids == -1) must not enter the fraction.
    breaker, host = _breaker_host(0.5)
    ratio, corr, ids, _adv = _masked_batch([True, False])
    ids = torch.tensor([0, 1, -1, -1])
    ratio = torch.cat([ratio, torch.zeros(2, ratio.shape[1])])
    corr = torch.cat([corr, torch.ones(2, corr.shape[1], dtype=torch.bool)])
    assert breaker(ratio, corr, ids, 2, "train") is False
    assert host._metrics["train"]["sampling/is_masked_traj_frac"] == [0.5]


def test_breaker_trips_on_masked_token_share():
    """One masked trajectory in four is 25% by count but can carry most of the step's tokens; the
    breaker reads the token share too, or a step that lost its long trajectories trains on the
    short survivors alone."""
    tripped, host = _breaker_host(0.4)
    ratio = torch.ones(4, 20)
    ratio[0] = 0.0
    eff_corrected = torch.zeros(4, 20, dtype=torch.bool)
    eff_corrected[0] = True  # 20 corrected tokens, all masked
    eff_corrected[1:, :2] = True  # 6 corrected tokens, all surviving
    assert tripped(ratio, eff_corrected, torch.arange(4), 4, "train")
    assert host._metrics["train"]["sampling/is_masked_traj_frac"][-1] == pytest.approx(0.25)
    assert host._metrics["train"]["sampling/is_masked_token_frac"][-1] == pytest.approx(20 / 26)
    assert host._breaker_tripped_this_step


def test_breaker_below_both_fractions_leaves_the_optimizer_skip_unarmed():
    tripped, host = _breaker_host(0.4)
    ratio, eff_corrected, traj_row_ids, _ = _masked_batch([True, False, False, False])
    assert not tripped(ratio, eff_corrected, traj_row_ids, 4, "train")
    assert host._metrics["train"]["sampling/is_masked_token_frac"][-1] == pytest.approx(0.25)
    assert not host._breaker_tripped_this_step


def test_a_tripped_breaker_drops_the_gradients_on_every_optimizer_step_of_the_round():
    """A zeroed loss still hands the optimizer zero gradients, on which Adam steps by momentum; the
    skip must set every grad to None (what makes an optimizer skip a parameter) on EVERY optimizer
    step the round feeds — a generation round spans ``num_iterations`` of them, all trained on the
    zeroed advantages — so the skip must not consume the flag. Only the next round's verdict does."""
    calls = []
    host = types.SimpleNamespace(
        _breaker_tripped_this_step=True,
        optimizer=types.SimpleNamespace(zero_grad=lambda set_to_none: calls.append(set_to_none)),
    )
    skip = DistributedAsyncEnvironmentalGRPOTrainer._skip_optimizer_step_if_breaker_tripped.__get__(host)
    skip()
    skip()
    assert calls == [True, True]
    assert host._breaker_tripped_this_step is True
    host._breaker_tripped_this_step = False
    skip()
    assert calls == [True, True]


def test_none_gradients_leave_adam_state_and_weights_untouched():
    """The skip relies on an optimizer contract: a parameter with ``grad=None``
    is skipped outright, so neither its weights nor its moments move even with momentum built up."""
    param = torch.nn.Parameter(torch.ones(4))
    opt = torch.optim.AdamW([param], lr=0.1, weight_decay=0.0)
    param.grad = torch.ones(4)
    opt.step()
    moved = param.detach().clone()
    exp_avg = opt.state[param]["exp_avg"].clone()
    opt.zero_grad(set_to_none=True)
    opt.step()
    assert torch.equal(param.detach(), moved)
    assert torch.equal(opt.state[param]["exp_avg"], exp_avg)
    param.grad = torch.zeros(4)
    opt.step()
    assert not torch.equal(param.detach(), moved), "a ZERO gradient still steps the weights by momentum"


def test_toolkit_optimizers_skip_none_gradients():
    """Guards the same contract for the toolkit's own optimizers, whose step loops are hand-written."""
    for name in ("adamw_bf16", "flash_adamw", "muon"):
        source = (pathlib.Path("src/optimizers") / f"{name}.py").read_text()
        assert "grad is None" in source or "grad is not None" in source, f"{name}: no None-grad guard in the step loop"


def _armed(threshold: float | None) -> DistributedAsyncEnvironmentalGRPOTrainer:
    """A trainer shell after ``_arm_update_breaker``, the step ``__init__`` takes, recording its callbacks."""
    host = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    host.async_config = AsyncTrainingConfig(skip_update_masked_frac=threshold)
    host.callbacks = []
    host.add_callback = host.callbacks.append
    host._arm_update_breaker()
    return host


def test_the_optimizer_skip_hook_is_attached_only_behind_the_breaker_knob():
    """Without the knob no verdict trips, so the hook would only run on every optimizer step for nothing;
    with it, the one hook turns a tripped verdict into this trainer's skipped optimizer step."""
    assert _armed(None).callbacks == []
    host = _armed(0.3)
    (hook,) = host.callbacks
    assert host._skip_update_masked_frac == 0.3 and host._breaker_tripped_this_step is False
    calls = []
    host.optimizer = types.SimpleNamespace(zero_grad=lambda set_to_none: calls.append(set_to_none))
    hook.on_pre_optimizer_step(None, None, None)
    assert calls == [], "an untripped round steps the optimizer"
    host._breaker_tripped_this_step = True
    hook.on_pre_optimizer_step(None, None, None)
    assert calls == [True]


# --- what the batch build trains around the breaker -------------------------------------------------

_SAMPLED = -1.0
_FORCED_CLOSE = 77


def _vetoed_round(threshold: float):
    """A solve and a failure of one group, both IS-corrected; the solve's one token sits so far below its
    sampling log-prob that the veto masks its trajectory: half the corrected trajectories and tokens."""
    rows = [row([6], sampling=[_SAMPLED]), row([7], sampling=[_SAMPLED])]
    policy = torch.tensor([[_SAMPLED - 3.0], [_SAMPLED]])
    host = batch_host(
        rows,
        policy,
        training=True,
        is_correction=True,
        is_mask_config=ISMaskConfig(veto_min=0.5),
        skip_update_masked_frac=threshold,
        save_completions=True,
    )
    return host, build(host, [episode(1.0), episode(0.0)])


def test_a_tripped_breaker_zeroes_the_trained_and_the_recorded_advantages():
    """The verdict alone trains nothing away: a tripped step zeroes the per-row advantages the loss reads
    AND the per-trajectory ones the durable completions record reports. Below the threshold both stand."""
    host, batch = _vetoed_round(0.4)
    assert host._metrics["train"]["sampling/update_skipped"] == [1.0]
    assert torch.equal(batch["advantages"], torch.zeros(2))
    assert list(host._logs["advantages"]) == [0.0, 0.0]

    host, batch = _vetoed_round(0.9)
    assert host._metrics["train"]["sampling/update_skipped"] == [0.0]
    assert batch["advantages"][0] > 0 > batch["advantages"][1]
    assert list(host._logs["advantages"]) == batch["advantages"].tolist()


def test_the_step_diagnostics_read_the_advantages_the_breaker_left():
    """``logps/advantage_cov`` describes the update the step takes: zero on a tripped step, whatever the
    advantages were before the breaker zeroed them (the solve's low log-prob against its positive advantage)."""
    host, _ = _vetoed_round(0.4)
    assert host._metrics["train"]["logps/advantage_cov"] == [0.0]
    host, _ = _vetoed_round(0.9)
    assert host._metrics["train"]["logps/advantage_cov"][0] < 0


def _gathered_with_a_peer(monkeypatch, rewards: list[float], state: list[int]) -> None:
    """The world gather of a two-rank run whose peer holds ``rewards`` and per-trajectory ``state``
    (1 valid, 0 invalid)."""
    peer = {torch.float32: torch.tensor(rewards), torch.int8: torch.tensor(state, dtype=torch.int8)}
    monkeypatch.setattr(environmental, "gather", lambda t: torch.cat([t, peer[t.dtype]]))


def _failed_round(training: bool):
    host = batch_host([row([6]), row([7])], torch.full((2, 1), _SAMPLED), training=training)
    host._empty_rollout_steps = EMPTY_ROLLOUT_STEP_LIMIT - 1
    return host, [episode(0.0, valid=False), episode(0.0, valid=False)]


def test_the_empty_step_halt_reads_the_world_not_this_rank(monkeypatch):
    """A rank whose own episodes all failed trains on while a peer's survived: judged on its local mask it
    would raise alone and leave the peers in the next collective. Empty world-wide, the step halts."""
    host, failed = _failed_round(training=True)
    _gathered_with_a_peer(monkeypatch, rewards=[1.0, 0.0], state=[1, 1])
    build(host, failed)
    assert host._empty_rollout_steps == 0

    host, failed = _failed_round(training=True)
    _gathered_with_a_peer(monkeypatch, rewards=[0.0, 0.0], state=[0, 0])
    with pytest.raises(RuntimeError, match="no valid episode anywhere in the world"):
        build(host, failed)


def test_an_empty_eval_round_never_halts_the_run(monkeypatch):
    host, failed = _failed_round(training=False)
    _gathered_with_a_peer(monkeypatch, rewards=[0.0, 0.0], state=[0, 0])
    build(host, failed)
    assert host._empty_rollout_steps == EMPTY_ROLLOUT_STEP_LIMIT - 1


def _mass_round(balance: bool, training: bool = True):
    """A one-token solve and a three-token failure whose tokens the IS correction weighs at 1/2."""
    rows = [row([6], sampling=[_SAMPLED]), row([7, 8, 9], sampling=[_SAMPLED] * 3)]
    policy = torch.tensor([[_SAMPLED] * 3, [_SAMPLED + math.log(0.5)] * 3])
    host = batch_host(
        rows, policy, training=training, is_correction=True, balance_token_mass=balance, save_completions=True
    )
    return host, build(host, [episode(1.0), episode(0.0)])


def test_the_token_mass_balance_weighs_the_corrected_ratio_and_reaches_both_advantage_sets():
    """The failure's three tokens at IS ratio 1/2 weigh 3/2 against the solve's one, so the negatives
    shrink to 2/3: on the per-row advantages the loss reads and the per-trajectory ones the record reports."""
    _, plain = _mass_round(balance=False)
    host, balanced = _mass_round(balance=True)
    expected = plain["advantages"] * torch.tensor([1.0, 2 / 3])
    assert torch.allclose(balanced["advantages"], expected)
    assert list(host._logs["advantages"]) == pytest.approx(expected.tolist())


def test_the_token_mass_balance_leaves_a_negative_only_row_raw_and_records_the_trainable_rows():
    """Per turn: a one-token solve against a failure of a 2-token turn and a 4-token cut turn. The balance
    weighs the solve's 1/2 against the failure's trainable 1 and halves the negatives there; the cut turn
    keeps its raw -1/2 and its 2 of mass is the round's net push. The record carries the failure's trainable
    turn."""
    solve, failure, cut = row([6]), row([7, 8]), row([9, 10, 11, 12], negative_only=True)
    host = batch_host(
        [solve, failure, cut],
        torch.zeros(3, 4),
        training=True,
        scale_rewards="none",
        balance_token_mass=True,
        save_completions=True,
    )
    host._train_on_sampled_tokens = True
    host._tokenize_step_rows = lambda results: [[solve], [failure, cut]]
    batch = build(host, [episode(1.0), episode(0.0)])
    assert batch["advantages"].tolist() == pytest.approx([0.5, -0.25, -0.5])
    assert list(host._logs["advantages"]) == pytest.approx([0.5, -0.25])
    metrics = host._metrics["train"]
    assert metrics[NET_TOKEN_MASS_KEY] == [pytest.approx((0.5 - 1) / 1.5)]
    assert metrics[NEGATIVE_ONLY_MASS_KEY] == [pytest.approx(2 / 3)]


def test_an_eval_round_is_never_balanced():
    _, plain = _mass_round(balance=False, training=False)
    host, batch = _mass_round(balance=True, training=False)
    assert torch.equal(batch["advantages"], plain["advantages"])
    assert TOKEN_MASS_SCALE_KEY not in host._metrics["eval"]


def test_an_engine_forced_close_trains_no_gradient():
    """The reasoning close the engine appended at the budget (sampling log-prob 0) carries ratio 0 whatever
    the correction computed for it, or it trains with its episode's advantage and teaches the model to stop
    closing its reasoning; the turn's own tokens keep their ratio."""
    rows = [row([6, 7, _FORCED_CLOSE], sampling=[_SAMPLED, _SAMPLED, 0.0]), row([8, 9, 10], sampling=[_SAMPLED] * 3)]
    host = batch_host(
        rows, torch.full((2, 3), _SAMPLED), training=True, is_correction=True, forced_close_ids=(_FORCED_CLOSE,)
    )
    ratio = build(host, [episode(1.0), episode(0.0)])["importance_sampling_ratio"]
    assert ratio[0, 2] == 0.0
    assert torch.equal(ratio[0, :2], torch.ones(2)) and torch.equal(ratio[1], torch.ones(3))


def _method_ast(name: str) -> ast.FunctionDef:
    source = textwrap.dedent(inspect.getsource(getattr(DistributedAsyncEnvironmentalGRPOTrainer, name)))
    return ast.parse(source).body[0]


def _build_training_tensors_ast() -> ast.FunctionDef:
    return _method_ast("_build_training_tensors")


def _calls(stmt: ast.stmt, name: str) -> list[ast.Call]:
    """The calls of ``name`` in ``stmt``, as a method (``x.name(...)``) or a bare function (``name(...)``)."""
    return [
        node
        for node in ast.walk(stmt)
        if isinstance(node, ast.Call) and (getattr(node.func, "attr", None) or getattr(node.func, "id", None)) == name
    ]


# The phase helpers ``_build_training_tensors`` is partitioned into, in call order.
_PHASE_HELPERS = (
    "_build_rollout_rewards",
    "_assemble_rollout_routing",
    "_recompute_logps_and_routing_masks",
    "_apply_is_correction",
    "_narrow_masks_and_normalizer",
    "_score_is_correction",
)


def test_build_training_tensors_routes_through_every_phase_helper():
    """A helper the parent no longer calls pins nothing, so the early-return check below would go vacuous."""
    fn = _build_training_tensors_ast()
    missing = [name for name in _PHASE_HELPERS if not _calls(fn, name)]
    assert not missing, f"phase helpers no longer called: {missing}"


def test_world_metrics_flush_once_after_every_recording_site_with_no_return_between():
    """The step's rank-local counts fold in ONE collective at the end of ``_build_training_tensors``:
    a second flush or a return before it strands the peers in the fold, and a site recording after it
    logs its count a step late."""
    fn = _build_training_tensors_ast()
    flush_at = [i for i, stmt in enumerate(fn.body) if _calls(stmt, "flush")]
    assert len(flush_at) == 1, f"_build_training_tensors flushes the world metrics {len(flush_at)} times"
    recording = ("fraction", "maximum", "effective_sample_frac", "covariance", "_record_step_diagnostics")
    recorded_at = [i for i, stmt in enumerate(fn.body) if any(_calls(stmt, name) for name in recording)]
    logged_at = [i for i, stmt in enumerate(fn.body) if _calls(stmt, "_populate_completion_logs")]
    assert recorded_at and max(recorded_at) < flush_at[0], "a count recorded after the flush logs a step late"
    assert max(logged_at) < flush_at[0], "the flush follows the completions record, the last phase every rank runs"
    between = fn.body[min(recorded_at) : flush_at[0]]
    assert not any(isinstance(node, ast.Return) for stmt in between for node in ast.walk(stmt)), (
        "a return between a recording site and the flush skips the collective on that path"
    )


def test_phase_helpers_never_early_return():
    """Each phase helper returns exactly once, as its final statement. Three of them issue collectives
    (the uniform raise, the recompute forward's EP dispatch, the normalizer gather), so a
    data-dependent early-out would let one rank skip a collective its peers enter."""
    for name in _PHASE_HELPERS:
        fn = _method_ast(name)
        returns = [node for node in ast.walk(fn) if isinstance(node, ast.Return)]
        assert len(returns) == 1 and fn.body[-1] is returns[0], (
            f"{name} returns from {len(returns)} site(s); its only return must be the final statement"
        )


def _effort_host(num_generations: int, training: bool = True, num_generations_eval: int = 1):
    host = types.SimpleNamespace(
        num_generations=num_generations,
        num_generations_eval=num_generations_eval,
        model=types.SimpleNamespace(training=training),
        _batch_errors=BatchBuildFence(),
    )
    return DistributedAsyncEnvironmentalGRPOTrainer._stamp_group_efforts.__get__(host), host


def _group_prompts(rows: int, group: int) -> list[str]:
    """Group-expanded prompts, as the RepeatSampler hands them: ``group`` consecutive rows per problem."""
    return [f"problem {i // group}" for i in range(rows)]


def test_stamp_group_efforts_uniform_within_group():
    stamp, _ = _effort_host(4)
    contexts = [None] * 12  # 3 groups of 4
    stamp(_group_prompts(12, 4), contexts)
    levels = [ctx["reasoning_effort"] for ctx in contexts]
    assert all(lv in VALID_REASONING_EFFORTS for lv in levels)
    for start in range(0, 12, 4):
        assert len(set(levels[start : start + 4])) == 1, f"group at {start} mixes efforts: {levels}"


def test_stamp_group_efforts_preserves_existing_context_keys():
    stamp, _ = _effort_host(2)
    contexts = [{"answer": "42"}, None]
    stamp(_group_prompts(2, 2), contexts)
    assert contexts[0]["answer"] == "42"
    assert contexts[0]["reasoning_effort"] == contexts[1]["reasoning_effort"]


def test_stamp_group_efforts_records_a_split_group_rather_than_raising():
    """Every batch-construction failure goes through the rank-uniform fence, so a rank-local raise
    here would strand the peers in the caller's all-reduce. The failure is RECORDED and raised
    uniformly one call later (the collective half is pinned in
    ``test_env_ragged_eval_batch_uniform_raise.py``)."""
    stamp, host = _effort_host(4)
    contexts = [None] * 6
    stamp(_group_prompts(6, 4), contexts)
    assert "multiple of the group size" in host._batch_errors.reason
    assert contexts == [None] * 6, "a refused batch must not be half-stamped"


def test_stamp_group_efforts_eval_uses_eval_group_size():
    stamp, _ = _effort_host(4, training=False, num_generations_eval=1)
    contexts = [None] * 3  # not a multiple of 4 — fine in eval (group size 1)
    stamp(_group_prompts(3, 1), contexts)
    assert all(ctx["reasoning_effort"] in VALID_REASONING_EFFORTS for ctx in contexts)


def test_training_groups_of_one_problem_still_draw_their_levels_at_random():
    """The stable draw is the eval's alone: a training group's level is the lottery the effort-conditioned
    policy learns over, so one problem drawn in many rounds must see more than one level."""
    stamp, _ = _effort_host(2)
    contexts = [None] * 60
    stamp(["the same problem"] * 60, contexts)
    assert len({ctx["reasoning_effort"] for ctx in contexts}) > 1


def test_an_eval_scores_each_problem_at_one_level_whatever_the_draw_order():
    """Every eval round must put a problem at the same level, or the checkpoints' per-level scores (and the
    level mix behind the headline one) are not comparable."""
    stamp, _ = _effort_host(4, training=False, num_generations_eval=1)
    prompts = [f"problem {i}" for i in range(60)]
    first, second = [None] * 60, [None] * 60
    stamp(prompts, first)
    stamp(list(reversed(prompts)), second)
    by_problem = {p: c["reasoning_effort"] for p, c in zip(prompts, first, strict=False)}
    assert all(by_problem[p] == c["reasoning_effort"] for p, c in zip(reversed(prompts), second, strict=False))
    assert set(by_problem.values()) == set(VALID_REASONING_EFFORTS), "the draw still spreads over every level"
    assert by_problem["problem 0"] == stable_reasoning_effort([{"role": "user", "content": "problem 0"}])


def test_bind_episode_effort_prefers_the_context_level():
    def level(context, effort):
        env = types.SimpleNamespace(reasoning_effort=effort, thinking_budget_for_effort=lambda level: None)
        return bind_episode_effort(context, env, max_tokens=1000).level

    assert level({"reasoning_effort": "high"}, "random") == "high"
    # No context level → falls back to the env setting ('random' draws a concrete level).
    assert level({}, "random") in VALID_REASONING_EFFORTS
    assert level(None, "random") in VALID_REASONING_EFFORTS
    # Fixed env setting passes through; no setting at all → None.
    assert level(None, "low") == "low"
    assert level(None, None) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
