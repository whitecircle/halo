"""Three env-GRPO stability mechanisms: the IS trust-region breaker, group-level reasoning-effort conditioning, and the
``_build_training_tensors`` phase helpers, read off the source: one process cannot show a rank skip a collective."""

import ast
import inspect
import math
import pathlib
import textwrap
import types
from collections import defaultdict

import pytest
import torch

from halo.configs.grpo_environmental import AsyncRolloutConfig, ISRMaskConfig
from halo.environments.episodes.reasoning_effort import bind_episode_effort, stable_reasoning_effort
from halo.environments.rollout_config import VALID_REASONING_EFFORTS
from halo.trainers.grpo.environmental import step_batch
from halo.trainers.grpo.environmental.loss.importance_sampling import _BreakerOptimizerSkipCallback
from halo.trainers.grpo.environmental.step_batch import EMPTY_ROLLOUT_STEP_LIMIT, BatchBuildFence
from halo.trainers.grpo.environmental.trainer import DistributedAsyncEnvironmentalGRPOTrainer
from halo.trainers.grpo.objective.loss_inputs import (
    NEGATIVE_ONLY_MASS_KEY,
    NET_TOKEN_MASS_KEY,
    TOKEN_MASS_SCALE_KEY,
)
from tests.common.env_grpo_batch import batch_host, build, episode, row


class _TrainerStandIn(types.SimpleNamespace):
    """Weak-referenceable, as the trainer the breaker's hook holds is."""


def _skip_hook(host):
    callback = _BreakerOptimizerSkipCallback(host)
    return lambda: callback.on_pre_optimizer_step(None, None, None)


def _breaker_host(threshold):
    host = types.SimpleNamespace(
        async_config=types.SimpleNamespace(skip_update_masked_frac=threshold),
        _breaker_tripped_this_step=False,
        accelerator=types.SimpleNamespace(gather=lambda x: x),
        _metrics={"train": defaultdict(list), "eval": defaultdict(list)},
    )
    return DistributedAsyncEnvironmentalGRPOTrainer._update_breaker_tripped.__get__(host), host


def _masked_batch(masked_trajs: list[bool], tokens_per_row: int = 4):
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
    # A per-token zero such as ``zero_engine_forced_closes`` masks tokens, not trajectories.
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
    breaker, host = _breaker_host(0.5)
    ratio, corr, ids, _adv = _masked_batch([True, False])
    ids = torch.tensor([0, 1, -1, -1])
    ratio = torch.cat([ratio, torch.zeros(2, ratio.shape[1])])
    corr = torch.cat([corr, torch.ones(2, corr.shape[1], dtype=torch.bool)])
    assert breaker(ratio, corr, ids, 2, "train") is False
    assert host._metrics["train"]["sampling/is_masked_traj_frac"] == [0.5]


def test_breaker_trips_on_masked_token_share():
    """One masked trajectory in four can carry most of the step's tokens; without the token share, a step that lost its
    long trajectories trains on the short survivors alone."""
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
    """A zeroed loss still hands the optimizer zero gradients, on which Adam steps by momentum. Grads must be None on
    every optimizer step the round feeds, so the skip must not consume the flag; only the next round's verdict does."""
    calls = []
    host = _TrainerStandIn(
        _breaker_tripped_this_step=True,
        optimizer=types.SimpleNamespace(zero_grad=lambda set_to_none: calls.append(set_to_none)),
    )
    skip = _skip_hook(host)
    skip()
    skip()
    assert calls == [True, True]
    assert host._breaker_tripped_this_step is True
    host._breaker_tripped_this_step = False
    skip()
    assert calls == [True, True]


def test_none_gradients_leave_adam_state_and_weights_untouched():
    """The skip relies on this optimizer contract, even with momentum built up."""
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
    """The toolkit's own optimizers have hand-written step loops."""
    for name in ("adamw_bf16", "flash_adamw", "muon"):
        source = (pathlib.Path("halo/optimizers") / f"{name}.py").read_text()
        assert "grad is None" in source or "grad is not None" in source, f"{name}: no None-grad guard in the step loop"


def _armed(threshold: float | None) -> DistributedAsyncEnvironmentalGRPOTrainer:
    host = object.__new__(DistributedAsyncEnvironmentalGRPOTrainer)
    host.async_config = AsyncRolloutConfig(skip_update_masked_frac=threshold)
    host.callbacks = []
    host.add_callback = host.callbacks.append
    host._arm_update_breaker()
    return host


def test_the_optimizer_skip_hook_is_attached_only_behind_the_breaker_knob():
    assert _armed(None).callbacks == []
    host = _armed(0.3)
    (hook,) = host.callbacks
    assert host._breaker_tripped_this_step is False
    calls = []
    host.optimizer = types.SimpleNamespace(zero_grad=lambda set_to_none: calls.append(set_to_none))
    hook.on_pre_optimizer_step(None, None, None)
    assert calls == [], "an untripped round steps the optimizer"
    host._breaker_tripped_this_step = True
    hook.on_pre_optimizer_step(None, None, None)
    assert calls == [True]


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
        isr_mask_config=ISRMaskConfig(veto_min=0.5),
        skip_update_masked_frac=threshold,
        save_completions=True,
    )
    return host, build(host, [episode(1.0), episode(0.0)])


def test_a_tripped_breaker_zeroes_the_trained_and_the_recorded_advantages():
    """The per-row advantages are what the loss reads, the per-trajectory ones what the completions record reports."""
    host, batch = _vetoed_round(0.4)
    assert host._metrics["train"]["sampling/update_skipped"] == [1.0]
    assert torch.equal(batch["advantages"], torch.zeros(2))
    assert list(host._logs["advantages"]) == [0.0, 0.0]

    host, batch = _vetoed_round(0.9)
    assert host._metrics["train"]["sampling/update_skipped"] == [0.0]
    assert batch["advantages"][0] > 0 > batch["advantages"][1]
    assert list(host._logs["advantages"]) == batch["advantages"].tolist()


def test_the_step_diagnostics_read_the_advantages_the_breaker_left():
    """``logps/advantage_cov`` describes the update the step takes: zero on a tripped step."""
    host, _ = _vetoed_round(0.4)
    assert host._metrics["train"]["logps/advantage_cov"] == [0.0]
    host, _ = _vetoed_round(0.9)
    assert host._metrics["train"]["logps/advantage_cov"][0] < 0


def _gathered_with_a_peer(monkeypatch, rewards: list[float], state: list[int]) -> None:
    """The world gather of a two-rank run whose peer holds ``rewards`` and ``state`` (1 valid, 0 invalid)."""
    peer = {torch.float32: torch.tensor(rewards), torch.int8: torch.tensor(state, dtype=torch.int8)}
    monkeypatch.setattr(step_batch, "gather", lambda t: torch.cat([t, peer[t.dtype]]))


def _failed_round(training: bool):
    host = batch_host([row([6]), row([7])], torch.full((2, 1), _SAMPLED), training=training)
    host._empty_rollout_steps = EMPTY_ROLLOUT_STEP_LIMIT - 1
    return host, [episode(0.0, valid=False), episode(0.0, valid=False)]


def test_the_empty_step_halt_reads_the_world_not_this_rank(monkeypatch):
    """Judged on its local mask, a rank whose own episodes all failed would raise alone and leave the peers in the next
    collective."""
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
    """The failure's three tokens at IS ratio 1/2 weigh 3/2 against the solve's one, so the negatives shrink to 2/3."""
    _, plain = _mass_round(balance=False)
    host, balanced = _mass_round(balance=True)
    expected = plain["advantages"] * torch.tensor([1.0, 2 / 3])
    assert torch.allclose(balanced["advantages"], expected)
    assert list(host._logs["advantages"]) == pytest.approx(expected.tolist())


def test_the_token_mass_balance_leaves_a_negative_only_row_raw_and_records_the_trainable_rows():
    """The balance weighs the solve's 1/2 against the failure's trainable 1 and halves the negatives there; the cut
    turn keeps its raw -1/2 and its 2 of mass is the round's net push."""
    solve, failure, cut = row([6]), row([7, 8]), row([9, 10, 11, 12], negative_only=True)
    host = batch_host(
        [solve, failure, cut],
        torch.zeros(3, 4),
        training=True,
        scale_rewards="none",
        balance_token_mass=True,
        save_completions=True,
    )
    host.async_config.train_on_sampled_tokens = True
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
    """Otherwise the close trains with its episode's advantage and teaches the model to stop closing its reasoning."""
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
    return [
        node
        for node in ast.walk(stmt)
        if isinstance(node, ast.Call) and (getattr(node.func, "attr", None) or getattr(node.func, "id", None)) == name
    ]


# The phase helpers ``_build_training_tensors`` is partitioned into, in call order.
_PHASE_HELPERS = (
    "_rows_for_step",
    "_build_rollout_rewards",
    "_pad_rows_across_ranks",
    "_stack_rows",
    "_assemble_rollout_routing",
    "_score_step",
    "_step_advantages",
    "_apply_opsm_stage",
    "_narrow_masks_and_normalizer",
    "_apply_token_mass_balance",
    "_apply_breaker_and_record_metrics",
)
# The helpers those phases call in turn, each holding or gating a collective of its own.
_NESTED_HELPERS = ("_recompute_logps_and_routing", "_apply_is_correction", "_narrow_corrected_mask_and_record")
_RECORDING = ("fraction", "maximum", "effective_sample_frac", "covariance", "_record_step_diagnostics")


def test_build_training_tensors_routes_through_every_phase_helper():
    """A helper the parent does not call pins nothing, so the early-return check below would go vacuous."""
    fn = _build_training_tensors_ast()
    missing = [name for name in _PHASE_HELPERS if not _calls(fn, name)]
    assert not missing, f"phase helpers no longer called: {missing}"
    nested = [_method_ast("_score_step"), _method_ast("_apply_breaker_and_record_metrics")]
    missing = [name for name in _NESTED_HELPERS if not any(_calls(phase, name) for phase in nested)]
    assert not missing, f"nested helpers no longer called: {missing}"


def test_rewards_are_built_once_tokenizing_has_settled_the_baseline():
    """Tokenizing drops an episode the template cannot re-render from the baseline, and the failed-group waiver of
    the reasoning terms reads the baseline: a reward built first would count a dropped solve for its group."""
    fn = _build_training_tensors_ast()
    tokenized_at = [i for i, stmt in enumerate(fn.body) if _calls(stmt, "_rows_for_step")]
    rewards_at = [i for i, stmt in enumerate(fn.body) if _calls(stmt, "_build_rollout_rewards")]
    assert tokenized_at and rewards_at and max(tokenized_at) < min(rewards_at)
    assert _calls(_method_ast("_rows_for_step"), "_tokenize_step_rows"), "the rows phase no longer tokenizes"


def test_world_metrics_flush_once_after_every_recording_site_with_no_return_between():
    """The step's rank-local counts fold in one collective, in the last phase of ``_build_training_tensors``: a second
    flush or a return before it strands the peers, and a site recording after it logs its count a step late."""
    parent = _build_training_tensors_ast()
    assert not _calls(parent, "flush"), "the batch build flushes outside its metrics phase"
    phase_at = [i for i, stmt in enumerate(parent.body) if _calls(stmt, "_apply_breaker_and_record_metrics")]
    recorded_at = [i for i, stmt in enumerate(parent.body) if any(_calls(stmt, name) for name in _RECORDING)]
    assert len(phase_at) == 1 and max(recorded_at) < phase_at[0], "a count recorded after the metrics phase"
    assert not any(isinstance(node, ast.Return) for stmt in parent.body[: phase_at[0]] for node in ast.walk(stmt))
    fn = _method_ast("_apply_breaker_and_record_metrics")
    flush_at = [i for i, stmt in enumerate(fn.body) if _calls(stmt, "flush")]
    assert len(flush_at) == 1, f"the metrics phase flushes the world metrics {len(flush_at)} times"
    recorded_at = [i for i, stmt in enumerate(fn.body) if any(_calls(stmt, name) for name in _RECORDING)]
    logged_at = [i for i, stmt in enumerate(fn.body) if _calls(stmt, "_populate_completion_logs")]
    assert recorded_at and max(recorded_at) < flush_at[0], "a count recorded after the flush logs a step late"
    assert max(logged_at) < flush_at[0], "the flush follows the completions record, the last phase every rank runs"
    assert not any(isinstance(node, ast.Return) for stmt in fn.body[: flush_at[0]] for node in ast.walk(stmt)), (
        "a return before the flush skips the collective on that path"
    )


def test_phase_helpers_never_early_return():
    """Several issue collectives (the uniform raise, row padding, the recompute's EP dispatch, the normalizer and
    token-mass gathers, the metrics fold), so a data-dependent early-out would let a rank skip one its peers enter."""
    for name in _PHASE_HELPERS + _NESTED_HELPERS:
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
    """As the RepeatSampler hands them: ``group`` consecutive rows per problem."""
    return [f"problem {i // group}" for i in range(rows)]


def test_stamp_group_efforts_uniform_within_group():
    stamp, _ = _effort_host(4)
    contexts = [None] * 12
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
    """A rank-local raise here would strand the peers in the caller's all-reduce;
    ``test_env_ragged_eval_batch_uniform_raise.py`` pins the collective half."""
    stamp, host = _effort_host(4)
    contexts = [None] * 6
    stamp(_group_prompts(6, 4), contexts)
    assert "multiple of the group size" in host._batch_errors.reason
    assert contexts == [None] * 6, "a refused batch must not be half-stamped"


def test_stamp_group_efforts_eval_uses_eval_group_size():
    stamp, _ = _effort_host(4, training=False, num_generations_eval=1)
    contexts = [None] * 3  # not a multiple of 4, fine in eval (group size 1)
    stamp(_group_prompts(3, 1), contexts)
    assert all(ctx["reasoning_effort"] in VALID_REASONING_EFFORTS for ctx in contexts)


def test_training_groups_of_one_problem_still_draw_their_levels_at_random():
    """A training group's level is the lottery the effort-conditioned policy learns over."""
    stamp, _ = _effort_host(2)
    contexts = [None] * 60
    stamp(["the same problem"] * 60, contexts)
    assert len({ctx["reasoning_effort"] for ctx in contexts}) > 1


def test_an_eval_scores_each_problem_at_one_level_whatever_the_draw_order():
    """Otherwise the checkpoints' per-level scores, and the level mix behind the headline one, are not comparable."""
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
        env = types.SimpleNamespace(reasoning_effort=effort)
        return bind_episode_effort(context, env, max_tokens=1000).level

    assert level({"reasoning_effort": "high"}, "random") == "high"
    # No context level falls back to the env setting ('random' draws a concrete level).
    assert level({}, "random") in VALID_REASONING_EFFORTS
    assert level(None, "random") in VALID_REASONING_EFFORTS
    assert level(None, "low") == "low"
    assert level(None, None) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
