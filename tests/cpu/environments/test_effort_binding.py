#!/usr/bin/env python
"""CPU tests: both rollout drivers bind an episode's reasoning effort through ONE seam.

``bind_episode_effort`` (``src/environments/episode.py``) resolves the episode's level once and turns the
env's per-level CoT budget into the turn's token caps. The Ray actor and the offline eval runner must
both go through it: a driver that resolves the level itself, or skips the budget, generates under a
different contract than the one the policy was trained on.

Both drivers are exercised for real — the actor via the class ``@ray.remote`` wraps (no cluster), the
eval runner with its generation call stubbed — so the tests fail if either one stops binding.

    python tests/cpu/environments/test_effort_binding.py
"""

import sys
from types import SimpleNamespace

import pytest

import src.environments.eval_runner as eval_runner
import src.environments.ray_actors as ray_actors
from src.environments import episode
from src.environments.base import VALID_REASONING_EFFORTS
from src.environments.engine_wire import generation_control_fields
from src.environments.envs.protocols.native import NativeToolUseEnvironment
from src.environments.tools.definitions import NativeToolRegistry

_MAX_TOKENS = 20000
# The reasoning-end marker's id under the episode scope: one no other id in a fake turn carries.
_END = 151668


class _PlainEnv(NativeToolUseEnvironment):
    """Tool-less env binding no per-level budget (the ``BaseEnvironment`` default)."""

    def __init__(self, **kwargs):
        super().__init__(tool_registry=NativeToolRegistry(), max_turns=4, **kwargs)


class _BudgetEnv(_PlainEnv):
    """…and one that binds a distinct CoT budget per level, like the coding envs."""

    BUDGETS = {"low": 1000, "medium": 2000, "high": 4000}

    def thinking_budget_for_effort(self, effort):
        return self.BUDGETS.get(effort)


def _text_turn(text="the answer", token_ids=None):
    return SimpleNamespace(
        answer=text, finish_reason="stop", completion_tokens=3, tool_calls=None, reasoning=None, token_ids=token_ids
    )


def _tool_turn(token_ids=None):
    """A turn that calls a tool no registry has: the step is spent, the episode stays open."""
    call = SimpleNamespace(id="c0", function=SimpleNamespace(name="nonexistent_tool", arguments="{}"))
    return SimpleNamespace(
        answer="",
        finish_reason="tool_calls",
        completion_tokens=3,
        tool_calls=[call],
        reasoning=None,
        token_ids=token_ids,
    )


def _actor_text_turn(token_ids=None):
    return ray_actors.TurnGeneration(
        text="the answer", tool_calls=[], reasoning="", tokens=3, finish_reason="stop", token_ids=token_ids
    )


def _actor_tool_turn(token_ids=None):
    """The actor-side twin of ``_tool_turn``: a call to a tool no registry has keeps the episode open."""
    call = {"id": "c0", "type": "function", "function": {"name": "nonexistent_tool", "arguments": "{}"}}
    return ray_actors.TurnGeneration(
        text="", tool_calls=[call], reasoning="", tokens=3, finish_reason="tool_calls", token_ids=token_ids
    )


def _ids_closing_reasoning_after(reasoning_tokens: int) -> list[int]:
    """Sampled ids whose reasoning closes with the marker as its ``reasoning_tokens``-th token."""
    return [7] * (reasoning_tokens - 1) + [_END] + [9] * 5


async def _drive_eval(monkeypatch, env, context, *, responses, config=None):
    """Run the eval driver over ``responses``; return its captured request kwargs and trajectory."""
    calls = []

    async def fake_generate(**kwargs):
        calls.append(kwargs)
        return responses[len(calls) - 1]

    monkeypatch.setattr(eval_runner, "generate_openai_response", fake_generate)
    traj = await eval_runner.run_episode(
        env,
        "solve it",
        dict(context or {}),
        None,
        rollout=config or ray_actors.RolloutConfig(model_name="m", temperature=0.7, max_tokens=_MAX_TOKENS),
    )
    return calls, traj


async def _drive_actor(env_cls, context, config, generations=None):
    """Run the real ``EnvironmentActor`` episode loop off-cluster over ``generations`` (one text turn
    when omitted); return its per-turn ``(config, level, budget, payload)`` records and the result.

    ``@ray.remote`` wraps the class and keeps the original at ``__ray_metadata__.modified_class``;
    ``__init__`` only assigns attributes, so this exercises the genuine ``run_episode`` binding. The
    transport is replaced below the payload builder, so each record's payload is the request the
    actor would have sent for that turn.
    """
    cls = ray_actors.EnvironmentActor.__ray_metadata__.modified_class
    actor = cls.__new__(cls)
    actor.__init__(actor_id=0, env_type=(env_cls, {}), env_config={})
    seen = []

    async def fake_client():
        return None

    async def fake_generate(client, url, messages, cfg, reasoning_effort=None, reasoning_budget=None):
        payload = actor._build_payload(messages, cfg, reasoning_effort, reasoning_budget)
        seen.append((cfg, reasoning_effort, reasoning_budget, payload))
        return generations[len(seen) - 1] if generations else _actor_text_turn()

    actor._get_http_client = fake_client
    actor._generate = fake_generate
    result = await actor.run_episode("solve it", dict(context or {}), "http://x", config)
    return seen, result


async def test_both_drivers_bind_the_same_caps(monkeypatch):
    """The level's budget must reach BOTH drivers' generation contract, identically."""
    context = {"reasoning_effort": "high"}
    config = ray_actors.RolloutConfig(max_tokens=_MAX_TOKENS)

    seen, result = await _drive_actor(_BudgetEnv, context, config)
    assert result.error is None, result.error
    (actor_cfg, actor_level, actor_budget, _) = seen[0]
    calls, eval_traj = await _drive_eval(monkeypatch, _BudgetEnv(), context, responses=[_text_turn()], config=config)

    # Actor side: the env's per-level budget replaces the (unset) global CoT cap, turn cap untouched,
    # and the template is told the same number the engine enforces.
    assert (actor_level, actor_cfg.max_thinking_tokens, actor_cfg.max_tokens) == ("high", 4000, _MAX_TOKENS)
    assert actor_budget == 4000
    # Eval side: byte-identical generation-control fields to the ones the actor's payload carries —
    # built from the ACTOR's own per-episode config, so this compares the two drivers, not the helper
    # with itself. The eval must not compute the CoT budget and then drop it from the request.
    assert calls[0]["extra_body"] == generation_control_fields(actor_cfg, actor_level)
    assert calls[0]["extra_body"]["reasoning_effort"] == "high"
    assert calls[0]["extra_body"]["thinking_token_budget"] == 4000
    # …and the budget also reaches the template as a variable, the one channel a prompt can state it through.
    assert calls[0]["extra_body"]["chat_template_kwargs"] == {"reasoning_budget": 4000}
    assert calls[0]["max_tokens"] == actor_cfg.max_tokens
    # …and both leave the episode carrying the budget it ran under: the trainer's re-render reads it
    # off the trajectory, the eval records it in the trajectory file.
    assert (eval_traj.reasoning_effort, eval_traj.reasoning_budget) == ("high", 4000)
    assert (result.trajectory.reasoning_effort, result.trajectory.reasoning_budget) == ("high", 4000)
    recorded = eval_runner.serialize_trajectory(eval_traj)
    assert (recorded["reasoning_effort"], recorded["reasoning_budget"]) == ("high", 4000)


async def test_env_without_a_level_budget_keeps_the_global_caps(monkeypatch):
    """No per-level budget: the run's own caps stand — the level still steers the template."""
    context = {"reasoning_effort": "high"}
    config = ray_actors.RolloutConfig(max_tokens=_MAX_TOKENS, max_thinking_tokens=8192)

    seen, result = await _drive_actor(_PlainEnv, context, config)
    assert result.error is None, result.error
    (actor_cfg, _, _, _) = seen[0]
    calls, eval_traj = await _drive_eval(monkeypatch, _PlainEnv(), context, responses=[_text_turn()])

    assert (actor_cfg.max_thinking_tokens, actor_cfg.max_tokens) == (8192, _MAX_TOKENS)
    assert result.trajectory.reasoning_budget == 8192
    assert calls[0]["max_tokens"] == _MAX_TOKENS
    assert eval_traj.reasoning_budget is None  # eval declares no global CoT cap
    assert eval_traj.reasoning_effort == "high"


async def test_random_level_is_drawn_once_for_the_whole_episode(monkeypatch):
    """A 'random' env level is one draw per EPISODE: every turn, and the recorded budget, share it.

    Re-resolving per turn would steer consecutive turns at different efforts and leave a budget that
    belongs to no turn — the trap the once-per-episode seam exists to prevent. Counted, not inferred
    from the drawn levels: a per-turn re-draw agrees with itself one time in three.
    """
    env = _BudgetEnv(reasoning_effort="random")
    bindings = []
    real_bind = eval_runner.bind_episode_effort

    def counting_bind(*args, **kwargs):
        bound = real_bind(*args, **kwargs)
        bindings.append(bound)
        return bound

    monkeypatch.setattr(eval_runner, "bind_episode_effort", counting_bind)
    calls, traj = await _drive_eval(monkeypatch, env, None, responses=[_tool_turn(), _text_turn()])

    levels = {call["extra_body"]["reasoning_effort"] for call in calls}
    assert len(calls) == 2, "the tool turn must not end the episode, or this proves nothing"
    assert len(bindings) == 1, "the level must be bound once per episode, not once per turn"
    assert len(levels) == 1
    level = levels.pop()
    assert level in VALID_REASONING_EFFORTS
    assert traj.reasoning_effort == level
    assert traj.reasoning_budget == _BudgetEnv.BUDGETS[level]


def _episode_config(**overrides):
    return ray_actors.RolloutConfig(
        max_tokens=_MAX_TOKENS,
        max_thinking_tokens=4000,
        thinking_budget_scope="episode",
        thinking_turn_reserve=256,
        reasoning_end_token_id=_END,
        chat_template_kwargs={"reasoning_budget_scope": "episode"},
        **overrides,
    )


async def test_episode_scope_narrows_each_turns_engine_cap_to_what_the_budget_has_left(monkeypatch):
    """Three turns each spending 1500 reasoning tokens of the level's 4000-token episode budget: the
    engine cap walks 4000 → 2500 → 1000 while every request keeps stating the episode's 4000 and its
    scope to the template and asks for the ids the spend is read off — on both drivers identically."""
    context = {"reasoning_effort": "high"}
    config = _episode_config()
    ids = _ids_closing_reasoning_after(1500)
    expected_caps = [4000, 2500, 1000]
    stated = {"reasoning_budget_scope": "episode", "reasoning_budget": 4000}

    gens = [_actor_tool_turn(ids), _actor_tool_turn(ids), _actor_text_turn(ids)]
    seen, result = await _drive_actor(_BudgetEnv, context, config, generations=gens)
    assert result.error is None, result.error
    assert [cfg.max_thinking_tokens for cfg, _, _, _ in seen] == expected_caps
    assert [payload["thinking_token_budget"] for _, _, _, payload in seen] == expected_caps
    assert {budget for _, _, budget, _ in seen} == {4000}
    assert [payload["chat_template_kwargs"] for _, _, _, payload in seen] == [stated] * 3
    assert all(payload["return_token_ids"] is True for _, _, _, payload in seen)
    # 4500 spent of 4000: a further turn would have reasoned only its reserve.
    assert result.trajectory.info["thinking_budget_exhausted"] is True
    assert result.metrics["episode/thinking_budget_exhausted"] == 1.0
    assert (result.trajectory.reasoning_effort, result.trajectory.reasoning_budget) == ("high", 4000)

    responses = [_tool_turn(ids), _tool_turn(ids), _text_turn(token_ids=ids)]
    calls, eval_traj = await _drive_eval(monkeypatch, _BudgetEnv(), context, responses=responses, config=config)
    assert [call["extra_body"]["thinking_token_budget"] for call in calls] == expected_caps
    assert [call["extra_body"]["chat_template_kwargs"] for call in calls] == [stated] * 3
    assert all(call["extra_body"]["return_token_ids"] is True for call in calls)
    assert eval_traj.info["thinking_budget_exhausted"] is True
    assert (eval_traj.reasoning_effort, eval_traj.reasoning_budget) == ("high", 4000)

    # Anti-vacuity for the flag: an episode that stops well inside its budget is recorded as such.
    calls, kept = await _drive_eval(monkeypatch, _BudgetEnv(), context, responses=[_text_turn(token_ids=ids)])
    assert kept.info.get("thinking_budget_exhausted") is None, "the per-turn scope records no verdict"
    _, kept = await _drive_eval(
        monkeypatch, _BudgetEnv(), context, responses=[_text_turn(token_ids=ids)], config=config
    )
    assert kept.info["thinking_budget_exhausted"] is False


async def test_episode_scope_refuses_a_turn_without_sampled_ids():
    """The spend is read off the sampled ids; a turn that arrives without them cannot be counted, and
    counting it as zero would hand every later turn the full budget — the loophole the scope closes.
    The same fake turn drives every per-turn-scope test above without error."""
    seen, result = await _drive_actor(_BudgetEnv, {"reasoning_effort": "high"}, _episode_config())
    assert result.error is not None and "sampled token ids" in result.error
    assert len(seen) == 1, "the episode must stop at the turn it cannot count"


def test_episode_scope_refuses_a_level_with_nothing_to_share():
    """An episode budget is the level's ``thinking_tokens`` or the run's ceiling; with neither there is
    no total for the turns to share, and binding one silently would run the scope as uncapped reasoning
    while the template promised a budget. The ceiling alone is a complete budget."""
    context = {"reasoning_effort": "high"}
    with pytest.raises(ValueError, match="nothing to share"):
        episode.bind_episode_effort(context, _PlainEnv(), max_tokens=_MAX_TOKENS, scope="episode")
    ceiling_only = episode.bind_episode_effort(
        context, _PlainEnv(), max_tokens=_MAX_TOKENS, max_thinking_tokens=8192, scope="episode"
    )
    assert (ceiling_only.thinking_budget, ceiling_only.turn_thinking_cap(0)) == (8192, 8192)
    # The per-turn scope has nothing to share and binds the uncapped run as before.
    assert episode.bind_episode_effort(context, _PlainEnv(), max_tokens=_MAX_TOKENS).thinking_budget is None


async def test_the_eval_refuses_a_scope_gap_before_generating(monkeypatch):
    """The eval runs the trainer's scope gate: an env that resolves no level under a ceiling-less episode
    scope would otherwise fail every episode at its first turn, each recorded as a zero-reward error
    sample, and the run would report a score of zero."""
    calls = []

    async def fake_generate(**kwargs):
        calls.append(kwargs)
        return _text_turn(token_ids=_ids_closing_reasoning_after(10))

    monkeypatch.setattr(eval_runner, "generate_openai_response", fake_generate)
    config = ray_actors.RolloutConfig(
        max_tokens=_MAX_TOKENS, thinking_budget_scope="episode", reasoning_end_token_id=_END
    )
    examples = [{"prompt": "solve it", "context": {}}]
    with pytest.raises(ValueError, match="nothing to share"):
        await eval_runner.collect_results(_BudgetEnv(), examples, None, rollout=config)
    assert calls == [], "the gate must refuse before the first request"
    # Anti-vacuity: the same contract runs once the env sets a level every episode can bind.
    results = await eval_runner.collect_results(_BudgetEnv(reasoning_effort="high"), examples, None, rollout=config)
    assert "error" not in results[0]["samples"][0] and len(calls) == 1


def test_drivers_share_one_seam():
    # A re-forked local copy is exactly how the two drivers drifted apart before.
    assert eval_runner.bind_episode_effort is episode.bind_episode_effort
    assert ray_actors.bind_episode_effort is episode.bind_episode_effort


def test_context_level_wins_over_the_env_setting():
    # GRPO's group baseline assumes one conditioning per group: the trainer stamps it into the context.
    env = _BudgetEnv(reasoning_effort="low")
    bound = episode.bind_episode_effort({"reasoning_effort": "high"}, env, max_tokens=_MAX_TOKENS)
    assert (bound.level, bound.thinking_budget) == ("high", 4000)
    assert episode.bind_episode_effort(None, env, max_tokens=_MAX_TOKENS).level == "low"
    unset = episode.bind_episode_effort(None, _BudgetEnv(), max_tokens=_MAX_TOKENS)
    assert (unset.level, unset.thinking_budget, unset.max_tokens) == (None, None, _MAX_TOKENS)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
