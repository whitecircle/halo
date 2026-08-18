"""Unit tests for three env-GRPO stability mechanisms (no GPU / no live model):

1. ``_update_breaker_tripped`` — the IS trust-region circuit breaker (``skip_update_masked_frac``):
   when the mask stages (geo-band / veto / OPSM) have zeroed the ratio of more than the configured
   fraction of IS-corrected trajectories, the step must be declared broken so the caller zeroes the
   whole step's policy gradient (training on the surviving selection-biased sample amplifies the
   drift). Below the threshold, and with the knob unset, the verdict must be False. The verdict is
   returned, not applied, so the EFFECT is pinned separately on the caller's own body: both
   advantage tensors zeroed, and the durable completions record written after that — neither is
   reachable without a live model and a rollout server, so both are read off the source tree.

2. Group-level reasoning-effort conditioning — ``_stamp_group_efforts`` (trainer side) draws ONE
   effort per generation group and stamps it into every member's rollout context;
   ``resolve_episode_effort`` (Ray-actor side) must prefer that context level over the env's own
   per-episode draw. Together they keep GRPO group members identically conditioned, so the effort
   lottery can never become intra-group advantage noise.

3. The ``_build_training_tensors`` phase helpers — three of them issue collectives behind config- or
   mode-derived gates, so a data-dependent early ``return`` in any helper would let one rank skip a
   collective its peers enter (a watchdog hang at scale). Read off the source tree, like (1).

    python tests/cpu/grpo/test_grpo_update_breaker_and_group_effort.py
"""

import ast
import inspect
import textwrap
import types
from collections import defaultdict

import pytest
import torch
from accelerate import PartialState

from src.environments.base import VALID_REASONING_EFFORTS
from src.environments.episode import resolve_episode_effort
from src.trainers.grpo.environmental import DistributedAsyncEnvironmentalGRPOTrainer

PartialState()  # the breaker warns through accelerate's logger, which refuses to log without it


def _breaker_host(threshold):
    """Minimal stand-in exposing exactly what ``_update_breaker_tripped`` reads."""
    host = types.SimpleNamespace(
        _skip_update_masked_frac=threshold,
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
    # A trajectory with ANY surviving corrected token is not fully masked (token-band masks tokens,
    # not trajectories) — it must not count toward the breaker fraction.
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


def _method_ast(name: str) -> ast.FunctionDef:
    source = textwrap.dedent(inspect.getsource(getattr(DistributedAsyncEnvironmentalGRPOTrainer, name)))
    return ast.parse(source).body[0]


def _build_training_tensors_ast() -> ast.FunctionDef:
    return _method_ast("_build_training_tensors")


def _breaker_branch(fn: ast.FunctionDef) -> ast.If:
    """The ``if self._update_breaker_tripped(...)`` node, wherever it is nested."""
    for node in ast.walk(fn):
        if (
            isinstance(node, ast.If)
            and isinstance(node.test, ast.Call)
            and isinstance(node.test.func, ast.Attribute)
            and node.test.func.attr == "_update_breaker_tripped"
        ):
            return node
    raise AssertionError("_build_training_tensors no longer branches on _update_breaker_tripped")


def _calls(stmt: ast.stmt, attr: str) -> list[ast.Call]:
    return [
        node
        for node in ast.walk(stmt)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == attr
    ]


def _rebinds_to_zeros(stmt: ast.stmt, name: str) -> bool:
    """``<name> = ...zeros_like(<name>)`` — a REBIND, not an in-place ``zero_()``: both tensors are
    graph-carrying, so zeroing them in place would either error or corrupt the surrounding autograd."""
    if not (isinstance(stmt, ast.Assign) and [t.id for t in stmt.targets if isinstance(t, ast.Name)] == [name]):
        return False
    call = stmt.value
    if not (isinstance(call, ast.Call) and call.args):
        return False
    func = call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", None)
    return func == "zeros_like" and isinstance(call.args[0], ast.Name) and call.args[0].id == name


def test_a_tripped_breaker_zeroes_both_advantage_tensors():
    """The verdict alone trains nothing away: a tripped step must zero the per-ROW advantages (the
    policy gradient) AND the per-TRAJECTORY ones (what the durable record reports). Dropping either
    assignment leaves a broken step training its un-zeroed gradient, or logging advantages no
    gradient ever saw."""
    branch = _breaker_branch(_build_training_tensors_ast())
    for name in ("local_advantages", "traj_advantages"):
        assert any(_rebinds_to_zeros(stmt, name) for stmt in branch.body), (
            f"the tripped branch no longer rebinds {name} to zeros:\n{ast.unparse(branch)}"
        )


def test_the_completions_record_is_written_after_the_breaker():
    """The durable record must report the advantages the gradient saw. Populated ahead of the
    breaker it reports a set no gradient ever saw — the drift the online trainer's
    ``_install_advantages`` guard exists to prevent on its own path."""
    fn = _build_training_tensors_ast()
    branch = _breaker_branch(fn)
    breaker_at = next(i for i, stmt in enumerate(fn.body) if branch in ast.walk(stmt))
    logged_at = [i for i, stmt in enumerate(fn.body) if _calls(stmt, "_populate_completion_logs")]

    assert logged_at, "_build_training_tensors no longer populates the completions record"
    assert min(logged_at) > breaker_at, (
        "the completions record is populated before the breaker can zero this step's advantages"
    )
    returned_at = [i for i, stmt in enumerate(fn.body) if any(isinstance(n, ast.Return) for n in ast.walk(stmt))]
    assert not [i for i in returned_at if breaker_at < i < min(logged_at)], (
        "a return between the breaker and the completions record skips the record on that path"
    )


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


def test_phase_helpers_never_early_return():
    """Each phase helper returns exactly once, as its final statement. Three of them issue collectives
    (the uniform raise, the recompute forward's EP dispatch, the empty-step all_reduce and normalizer
    gather), so a data-dependent early-out would let one rank skip a collective its peers enter."""
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
        _batch_build_error=None,
    )
    return DistributedAsyncEnvironmentalGRPOTrainer._stamp_group_efforts.__get__(host), host


def test_stamp_group_efforts_uniform_within_group():
    stamp, _ = _effort_host(4)
    contexts = [None] * 12  # 3 groups of 4
    stamp(contexts)
    levels = [ctx["reasoning_effort"] for ctx in contexts]
    assert all(lv in VALID_REASONING_EFFORTS for lv in levels)
    for start in range(0, 12, 4):
        assert len(set(levels[start : start + 4])) == 1, f"group at {start} mixes efforts: {levels}"


def test_stamp_group_efforts_preserves_existing_context_keys():
    stamp, _ = _effort_host(2)
    contexts = [{"answer": "42"}, None]
    stamp(contexts)
    assert contexts[0]["answer"] == "42"
    assert contexts[0]["reasoning_effort"] == contexts[1]["reasoning_effort"]


def test_stamp_group_efforts_records_a_split_group_rather_than_raising():
    """A ragged batch reaches a SUBSET of the DP ranks, so a rank-local raise here would strand the
    peers in the caller's all-reduce. The failure is RECORDED and raised uniformly one call later
    (the collective half is pinned in ``test_env_ragged_eval_batch_uniform_raise.py``)."""
    stamp, host = _effort_host(4)
    contexts = [None] * 6
    stamp(contexts)
    assert "multiple of the group size" in host._batch_build_error
    assert contexts == [None] * 6, "a refused batch must not be half-stamped"


def test_stamp_group_efforts_eval_uses_eval_group_size():
    stamp, _ = _effort_host(4, training=False, num_generations_eval=1)
    contexts = [None] * 3  # not a multiple of 4 — fine in eval (group size 1)
    stamp(contexts)
    assert all(ctx["reasoning_effort"] in VALID_REASONING_EFFORTS for ctx in contexts)


def test_resolve_episode_effort_prefers_context():
    env = types.SimpleNamespace(reasoning_effort="random")
    assert resolve_episode_effort({"reasoning_effort": "high"}, env) == "high"
    # No context level → falls back to the env setting ('random' draws a concrete level).
    assert resolve_episode_effort({}, env) in VALID_REASONING_EFFORTS
    assert resolve_episode_effort(None, env) in VALID_REASONING_EFFORTS
    # Fixed env setting passes through; no setting at all → None.
    assert resolve_episode_effort(None, types.SimpleNamespace(reasoning_effort="low")) == "low"
    assert resolve_episode_effort(None, types.SimpleNamespace(reasoning_effort=None)) is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
