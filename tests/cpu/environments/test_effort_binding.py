#!/usr/bin/env python
"""Both rollout drivers bind an episode's reasoning effort through ``bind_episode_effort``, so they generate under
the contract the policy trained on. The output budget and the answer-room bound ride the same seam. The actor
runs via the class ``@ray.remote`` wraps (no cluster), the eval runner with its generation call stubbed."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

import halo.environments.eval_runner as eval_runner
import halo.environments.ray_actors as ray_actors
from halo.environments import reasoning_effort
from halo.environments.engine_wire import generation_control_fields
from halo.environments.episode_keys import OUTPUT_BUDGET_EXHAUSTED_KEY, REWARD_COMPONENTS_KEY
from halo.environments.messages import Trajectory
from halo.environments.protocols.native_tool_use import NativeToolUseEnvironment
from halo.environments.rollout_config import VALID_REASONING_EFFORTS
from halo.environments.tools.definitions import NativeTool, NativeToolRegistry
from tests.common.fake_chat_client import chat_reply

_MAX_TOKENS = 20000
# The reasoning-end marker's id the per-turn reasoning count reads up to: one no other id in a fake turn carries.
_END = 151668


def _echo_registry() -> NativeToolRegistry:
    """One no-op tool, so a tool turn is an ordinary productive turn that keeps the episode open."""
    registry = NativeToolRegistry()
    registry.register(NativeTool(name="echo", description="echo", parameters=[], handler=lambda: "ok"))
    return registry


class _PlainEnv(NativeToolUseEnvironment):
    """Env with one no-op tool binding no per-level budget (the ``EnvironmentBase`` default)."""

    def __init__(self, **kwargs):
        super().__init__(tool_registry=_echo_registry(), max_turns=4, **kwargs)


class _BudgetEnv(_PlainEnv):
    """Binds a distinct CoT budget per level, like the coding envs."""

    BUDGETS = {"low": 1000, "medium": 2000, "high": 4000}

    def thinking_budget_for_effort(self, effort):
        return self.BUDGETS.get(effort)


def _text_turn(text="the answer", tokens=3, finish_reason="stop"):
    return chat_reply(text, finish_reason=finish_reason, completion_tokens=tokens)


def _tool_turn(tokens=3, name="echo"):
    """A turn that calls ``name``: the no-op tool spends the step and keeps the episode open, and a name the
    registry lacks is a turn whose every call is refused."""
    call = SimpleNamespace(id="c0", function=SimpleNamespace(name=name, arguments="{}"))
    return chat_reply("", finish_reason="tool_calls", completion_tokens=tokens, tool_calls=[call])


def _actor_text_turn(token_ids=None, tokens=3, text="the answer"):
    return ray_actors.TurnGeneration(
        text=text, tool_calls=[], reasoning="", tokens=tokens, finish_reason="stop", token_ids=token_ids
    )


def _actor_tool_turn(token_ids=None, tokens=3, name="echo"):
    call = {"id": "c0", "type": "function", "function": {"name": name, "arguments": "{}"}}
    return ray_actors.TurnGeneration(
        text="", tool_calls=[call], reasoning="", tokens=tokens, finish_reason="tool_calls", token_ids=token_ids
    )


def _ids_closing_reasoning_after(reasoning_tokens: int) -> list[int]:
    """Sampled ids whose reasoning closes with the marker as its ``reasoning_tokens``-th token."""
    return [7] * (reasoning_tokens - 1) + [_END] + [9] * 5


async def _drive_eval(monkeypatch, env, context, *, responses, config=None):
    calls = []

    async def fake_chat(client, **request):
        calls.append(request)
        return responses[len(calls) - 1]

    monkeypatch.setattr(eval_runner, "chat_completion", fake_chat)
    traj = await eval_runner.run_episode(
        env,
        "solve it",
        dict(context or {}),
        None,
        rollout=config or ray_actors.RolloutConfig(model_name="m", temperature=0.7, max_tokens=_MAX_TOKENS),
    )
    return calls, traj


async def _drive_actor(env_cls, context, config, generations=None, env_kwargs=None):
    """Run the real ``EnvironmentActor`` episode loop off-cluster over ``generations``, returning per-turn
    ``(config, level, budget, payload)`` records and the result. ``@ray.remote`` keeps the class at
    ``__ray_metadata__.modified_class``; the transport is replaced below the payload builder."""
    cls = ray_actors.EnvironmentActor.__ray_metadata__.modified_class
    actor = cls.__new__(cls)
    actor.__init__(actor_id=0, env_type=(env_cls, env_kwargs or {}), env_config={})
    seen = []

    async def fake_generate(
        client, url, messages, cfg, reasoning_effort=None, reasoning_budget=None, budget_soft=False
    ):
        payload = actor._build_payload(messages, cfg, reasoning_effort, reasoning_budget, budget_soft)
        seen.append((cfg, reasoning_effort, reasoning_budget, payload))
        return generations[len(seen) - 1] if generations else _actor_text_turn()

    actor._http_client = None
    actor._generate = fake_generate
    result = await actor.run_episode("solve it", dict(context or {}), "http://x", config)
    return seen, result


def _actor_caps(seen):
    return [(payload["max_tokens"], payload.get("thinking_token_budget")) for _, _, _, payload in seen]


def _eval_caps(calls):
    return [(call["max_tokens"], call["extra_body"].get("thinking_token_budget")) for call in calls]


async def test_both_drivers_bind_the_same_caps(monkeypatch):
    context = {"reasoning_effort": "high"}
    config = ray_actors.RolloutConfig(max_tokens=_MAX_TOKENS)

    seen, result = await _drive_actor(_BudgetEnv, context, config)
    assert result.error is None, result.error
    (actor_cfg, actor_level, actor_budget, _) = seen[0]
    calls, eval_traj = await _drive_eval(monkeypatch, _BudgetEnv(), context, responses=[_text_turn()], config=config)

    # Actor side: the per-level budget replaces the unset global cap, and the template is told the number the engine
    # enforces.
    assert (actor_level, actor_cfg.max_thinking_tokens, actor_cfg.max_tokens) == ("high", 4000, _MAX_TOKENS)
    assert actor_budget == 4000
    # Eval side: byte-identical control fields, built from the actor's own config, so this compares the two drivers and
    # not the helper with itself.
    assert calls[0]["extra_body"] == generation_control_fields(actor_cfg, actor_level, actor_budget)
    assert calls[0]["extra_body"]["reasoning_effort"] == "high"
    assert calls[0]["extra_body"]["thinking_token_budget"] == 4000
    # The budget also reaches the template as a variable, the one channel a prompt can state it through.
    assert calls[0]["extra_body"]["chat_template_kwargs"] == {"reasoning_budget": 4000}
    assert calls[0]["max_tokens"] == actor_cfg.max_tokens
    # Both leave the episode carrying the budget: the trainer's re-render reads it off the trajectory, the eval records
    # it in the file.
    assert (eval_traj.reasoning_effort, eval_traj.reasoning_budget) == ("high", 4000)
    assert (result.trajectory.reasoning_effort, result.trajectory.reasoning_budget) == ("high", 4000)
    recorded = eval_runner.serialize_trajectory(eval_traj)
    assert (recorded["reasoning_effort"], recorded["reasoning_budget"]) == ("high", 4000)


async def test_every_turn_of_an_episode_runs_under_the_levels_whole_cap(monkeypatch):
    """Three turns each reasoning 1500 of the level's 4000 all request the same cap and turn total and ask for no
    sampled ids."""
    context = {"reasoning_effort": "high"}
    config = ray_actors.RolloutConfig(max_tokens=_MAX_TOKENS, reasoning_end_token_id=_END)
    ids = _ids_closing_reasoning_after(1500)
    expected = [(_MAX_TOKENS, 4000)] * 3

    gens = [_actor_tool_turn(ids), _actor_tool_turn(ids), _actor_text_turn(ids)]
    seen, result = await _drive_actor(_BudgetEnv, context, config, generations=gens)
    assert result.error is None, result.error
    assert _actor_caps(seen) == expected
    assert [(cfg.max_tokens, cfg.max_thinking_tokens) for cfg, _, _, _ in seen] == expected
    assert [payload["chat_template_kwargs"] for _, _, _, payload in seen] == [{"reasoning_budget": 4000}] * 3
    assert (result.trajectory.reasoning_effort, result.trajectory.reasoning_budget) == ("high", 4000)
    assert OUTPUT_BUDGET_EXHAUSTED_KEY not in result.trajectory.info, "no output budget, no verdict"

    responses = [_tool_turn(), _tool_turn(), _text_turn()]
    calls, eval_traj = await _drive_eval(monkeypatch, _BudgetEnv(), context, responses=responses, config=config)
    assert _eval_caps(calls) == expected
    assert [call["extra_body"] for call in calls] == [
        generation_control_fields(cfg, level, budget) for cfg, level, budget, _ in seen
    ]
    assert all("return_token_ids" not in call["extra_body"] for call in calls)
    assert (eval_traj.reasoning_effort, eval_traj.reasoning_budget) == ("high", 4000)


async def test_env_without_a_level_budget_keeps_the_global_caps(monkeypatch):
    """No per-level budget: the run's own caps stand, and the level still steers the template."""
    context = {"reasoning_effort": "high"}
    config = ray_actors.RolloutConfig(max_tokens=_MAX_TOKENS, max_thinking_tokens=8192)

    seen, result = await _drive_actor(_PlainEnv, context, config)
    assert result.error is None, result.error
    (actor_cfg, _, _, _) = seen[0]
    calls, eval_traj = await _drive_eval(monkeypatch, _PlainEnv(), context, responses=[_text_turn()])

    assert (actor_cfg.max_thinking_tokens, actor_cfg.max_tokens) == (8192, _MAX_TOKENS)
    assert result.trajectory.reasoning_budget == 8192
    assert calls[0]["max_tokens"] == _MAX_TOKENS
    assert eval_traj.reasoning_budget is None
    assert eval_traj.reasoning_effort == "high"


async def test_random_level_is_drawn_once_for_the_whole_episode(monkeypatch):
    """Re-resolving per turn would steer turns at different efforts. Counted, not inferred from the levels: a
    per-turn re-draw agrees with itself one time in three."""
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


async def test_the_actor_records_each_turns_cap_and_reasoning_and_the_eval_its_cap_alone(monkeypatch):
    """``episode/thinking_cap_turns`` reads each training turn's cap and sampled reasoning. The eval asks for no
    ids, so its record carries the cap alone."""
    context = {"reasoning_effort": "high"}
    config = ray_actors.RolloutConfig(max_tokens=_MAX_TOKENS, reasoning_end_token_id=_END)
    ids = _ids_closing_reasoning_after(1500)

    gens = [_actor_tool_turn(ids), _actor_tool_turn(ids), _actor_text_turn(ids)]
    _, result = await _drive_actor(_BudgetEnv, context, config, generations=gens)
    actor_turns = [m for m in result.trajectory.messages if m.role == "assistant"]
    assert [(m.thinking_cap, m.reasoning_tokens) for m in actor_turns] == [(4000, 1500)] * 3

    responses = [_tool_turn(), _tool_turn(), _text_turn()]
    _, eval_traj = await _drive_eval(monkeypatch, _BudgetEnv(), context, responses=responses, config=config)
    assert [(m.thinking_cap, m.reasoning_tokens) for m in eval_traj.messages if m.role == "assistant"] == [
        (4000, None)
    ] * 3
    recorded = [m for m in eval_runner.serialize_trajectory(eval_traj)["messages"] if m["role"] == "assistant"]
    assert [m["thinking_cap"] for m in recorded] == [4000] * 3 and all("reasoning_tokens" not in m for m in recorded)


def test_a_level_budget_is_clamped_by_the_runs_cap_which_alone_caps_an_unbudgeted_level():
    """A level's cap never exceeds ``rollout_max_thinking_tokens``; a level without one runs under the run's cap or
    uncapped; a cap filling the turn is refused, since it leaves no answer room."""
    context = {"reasoning_effort": "high"}
    clamped = reasoning_effort.bind_episode_effort(
        context, _BudgetEnv(), max_tokens=_MAX_TOKENS, max_thinking_tokens=3000
    )
    assert (clamped.thinking_budget, clamped.max_tokens) == (3000, _MAX_TOKENS)
    level = reasoning_effort.bind_episode_effort(
        context, _BudgetEnv(), max_tokens=_MAX_TOKENS, max_thinking_tokens=8192
    )
    assert (level.thinking_budget, level.max_tokens) == (4000, _MAX_TOKENS)
    run = reasoning_effort.bind_episode_effort(context, _PlainEnv(), max_tokens=_MAX_TOKENS, max_thinking_tokens=8192)
    assert (run.thinking_budget, run.max_tokens) == (8192, _MAX_TOKENS)
    assert reasoning_effort.bind_episode_effort(context, _PlainEnv(), max_tokens=_MAX_TOKENS).thinking_budget is None
    with pytest.raises(ValueError, match=r"'high' level's thinking_tokens \(4000\) must sit below rollout_max_tokens"):
        reasoning_effort.bind_episode_effort(context, _BudgetEnv(), max_tokens=4000)
    with pytest.raises(ValueError, match="must sit below rollout_max_tokens"):
        reasoning_effort.bind_episode_effort(context, _PlainEnv(), max_tokens=4000, max_thinking_tokens=4000)
    for env in (_BudgetEnv(), _PlainEnv()):
        bound = reasoning_effort.bind_episode_effort(context, env, max_tokens=_MAX_TOKENS, max_episode_tokens=50000)
        assert bound.episode_tokens == 50000
        assert reasoning_effort.bind_episode_effort(context, env, max_tokens=_MAX_TOKENS).episode_tokens is None


async def test_the_eval_refuses_a_level_cap_that_fills_the_turn_before_generating(monkeypatch):
    """A drawable level whose cap fills the turn would fail every episode as a zero-reward error sample, so the eval
    gates before its first request."""
    calls = []

    async def fake_chat(client, **request):
        calls.append(request)
        return _text_turn()

    monkeypatch.setattr(eval_runner, "chat_completion", fake_chat)
    env = _BudgetEnv(reasoning_effort="random")
    examples = [{"prompt": "solve it", "context": {}}]
    too_small = ray_actors.RolloutConfig(model_name="m", max_tokens=4000)
    with pytest.raises(ValueError, match=r"'high' level's thinking_tokens \(4000\) must sit below rollout_max_tokens"):
        await eval_runner.collect_results(env, examples, None, rollout=too_small)
    assert calls == [], "the gate must refuse before the first request"

    def caps(env, **kwargs):
        return {
            level: effort.thinking_budget for level, effort in reasoning_effort.efforts_by_level(env, **kwargs).items()
        }

    assert caps(env, max_tokens=_MAX_TOKENS, max_thinking_tokens=None) == _BudgetEnv.BUDGETS
    assert caps(_BudgetEnv(reasoning_effort="low"), max_tokens=_MAX_TOKENS, max_thinking_tokens=None) == {"low": 1000}
    assert caps(_PlainEnv(), max_tokens=_MAX_TOKENS, max_thinking_tokens=None) == {None: None}
    results = await eval_runner.collect_results(
        env, examples, None, rollout=ray_actors.RolloutConfig(model_name="m", max_tokens=_MAX_TOKENS)
    )
    assert len(calls) == 1 and "error" not in results[0]["samples"][0]


def test_drivers_share_one_seam():
    # A local copy of the seam in either driver lets the two drift apart.
    assert eval_runner.bind_episode_effort is reasoning_effort.bind_episode_effort
    assert ray_actors.bind_episode_effort is reasoning_effort.bind_episode_effort


def test_context_level_wins_over_the_env_setting():
    # GRPO's group baseline assumes one conditioning per group: the trainer stamps it into the context.
    env = _BudgetEnv(reasoning_effort="low")
    bound = reasoning_effort.bind_episode_effort({"reasoning_effort": "high"}, env, max_tokens=_MAX_TOKENS)
    assert (bound.level, bound.thinking_budget) == ("high", 4000)
    assert reasoning_effort.bind_episode_effort(None, env, max_tokens=_MAX_TOKENS).level == "low"
    unset = reasoning_effort.bind_episode_effort(None, _BudgetEnv(), max_tokens=_MAX_TOKENS)
    assert (unset.level, unset.thinking_budget, unset.max_tokens) == (None, None, _MAX_TOKENS)


# What the native protocol charges an episode closed as a max_turns overflow; the output budget's
# truncation must be priced the same.
_OVERFLOW_PRICE = 0.3


def _effort(**overrides) -> reasoning_effort.EpisodeEffort:
    """A 20000-token turn with a 4000-token reasoning cap (16000 of answer room) under a 50000-token
    episode budget."""
    bound = {"level": None, "thinking_budget": 4000, "max_tokens": _MAX_TOKENS, "episode_tokens": 50000}
    merged = {**bound, **overrides}
    merged.setdefault("thinking_cap", merged["thinking_budget"])
    return reasoning_effort.EpisodeEffort(**merged)


def _caps(effort, generated):
    """``turn_caps`` as the pair of request fields it sets, or ``None`` once the budget holds no turn."""
    caps = effort.turn_caps(generated)
    if caps is None:
        return None
    assert set(caps) == {"max_tokens", "max_thinking_tokens"}
    return caps["max_tokens"], caps["max_thinking_tokens"]


def test_turn_caps_narrow_to_what_the_budget_has_left_and_keep_the_answer_room():
    """The total narrows to what is left and the reasoning cap gives up the difference, so the turn keeps its answer
    room whole; the floor is one token, since a cap of 0 would close the reasoning before it opened."""
    effort = _effort()
    # 50000 and exactly 20000 left: the turn's own caps stand.
    for generated in (0, 30000):
        assert _caps(effort, generated) == (_MAX_TOKENS, 4000)
    # 18000 left: the total narrows to it and the reasoning cap gives up the 2000.
    assert _caps(effort, 32000) == (18000, 2000)
    # Exactly the answer room left: a turn still starts, its reasoning cap floored at one token.
    assert _caps(effort, 34000) == (16000, 1)
    assert _caps(effort, 33999) == (16001, 1)
    # One token under the room, and far past the budget: no turn.
    assert _caps(effort, 34001) is None and _caps(effort, 60000) is None
    # No reasoning cap: the room is the whole turn, so only the total could narrow, and it never
    # narrows below one whole turn before the budget holds no turn.
    uncapped = _effort(thinking_budget=None)
    assert _caps(uncapped, 30000) == (_MAX_TOKENS, None)
    assert _caps(uncapped, 30001) is None
    unbounded = _effort(episode_tokens=None)
    assert _caps(unbounded, 10**6) == (_MAX_TOKENS, 4000)


async def test_an_output_budget_ends_the_episode_truncated_when_less_than_a_turn_is_left(monkeypatch):
    """50000 tokens over 20000-token turns: after three turns 1000 remain, under a turn's room, so the episode
    closes truncated with max_turns unused, priced like an overflow."""
    config = ray_actors.RolloutConfig(max_tokens=_MAX_TOKENS, max_episode_tokens=50000)
    sampled = [19000, 11000, 19000]
    expected = [(_MAX_TOKENS, None)] * 3

    gens = [_actor_tool_turn(tokens=n) for n in sampled]
    seen, result = await _drive_actor(
        _PlainEnv, None, config, generations=gens, env_kwargs={"turn_overflow_penalty": _OVERFLOW_PRICE}
    )
    assert result.error is None, result.error
    assert _actor_caps(seen) == expected
    assert [(cfg.max_tokens, cfg.max_thinking_tokens) for cfg, _, _, _ in seen] == expected
    traj = result.trajectory
    assert len(seen) == result.episode_length == 3 < _PlainEnv().max_turns, "the budget, not max_turns, ended it"
    assert traj.done and traj.truncated and not traj.info.get("completed")
    assert result.terminated_naturally is False
    assert traj.info[OUTPUT_BUDGET_EXHAUSTED_KEY] is True
    assert result.metrics["episode/output_budget_exhausted"] == 1.0
    assert result.generation_tokens == sum(sampled) == 49000
    assert traj.info[REWARD_COMPONENTS_KEY]["reward/tool_shaping"] == pytest.approx(-_OVERFLOW_PRICE)

    responses = [_tool_turn(tokens=n) for n in sampled]
    calls, eval_traj = await _drive_eval(
        monkeypatch, _PlainEnv(turn_overflow_penalty=_OVERFLOW_PRICE), None, responses=responses, config=config
    )
    assert _eval_caps(calls) == expected
    assert eval_traj.done and eval_traj.truncated and not eval_traj.info.get("completed")
    assert eval_traj.info[OUTPUT_BUDGET_EXHAUSTED_KEY] is True
    assert eval_traj.info["_eval_stats"]["completion_tokens"] == 49000
    assert eval_traj.info["_eval_stats"]["generations"] == 3
    assert eval_traj.info[REWARD_COMPONENTS_KEY]["reward/tool_shaping"] == pytest.approx(-_OVERFLOW_PRICE)


async def test_a_cut_on_the_last_turn_the_budget_affords_closes_the_episode_as_one_overflow(monkeypatch):
    """The third turn is the last the budget affords and, cut there, cannot be retried: one overflow price, no cut
    price, no nudge after the fragment."""
    config = ray_actors.RolloutConfig(max_tokens=_MAX_TOKENS, max_episode_tokens=50000)
    prices = {"turn_overflow_penalty": 0.3, "length_cutoff_penalty": 0.1}
    cut = SimpleNamespace(**{**vars(_actor_text_turn(tokens=11000)), "text": "half an ans", "finish_reason": "length"})

    gens = [_actor_tool_turn(tokens=19000), _actor_tool_turn(tokens=11000), cut]
    _, result = await _drive_actor(_PlainEnv, None, config, generations=gens, env_kwargs=prices)
    traj = result.trajectory
    assert result.error is None and traj.done and traj.truncated and result.episode_length == 3
    assert traj.info["length_cutoff_turns"] == 1 and traj.messages[-1].role == "assistant"
    assert traj.info[REWARD_COMPONENTS_KEY]["reward/tool_shaping"] == pytest.approx(-0.3)

    eval_cut = _text_turn("half an ans", tokens=11000, finish_reason="length")
    responses = [_tool_turn(tokens=19000), _tool_turn(tokens=11000), eval_cut]
    _, eval_traj = await _drive_eval(monkeypatch, _PlainEnv(**prices), None, responses=responses, config=config)
    assert eval_traj.done and eval_traj.truncated and eval_traj.messages[-1].role == "assistant"
    assert eval_traj.info[REWARD_COMPONENTS_KEY]["reward/tool_shaping"] == pytest.approx(-0.3)


async def test_both_drivers_run_the_turn_after_an_unproductive_one_under_the_levels_whole_cap(monkeypatch):
    """Each leaves the next request on the level's whole caps. A turn that closed its reasoning under its cap reads
    as such, so ``episode/thinking_cap_turns`` counts only the one that reasoned to 4000."""
    context = {"reasoning_effort": "high"}
    config = ray_actors.RolloutConfig(max_tokens=_MAX_TOKENS, reasoning_end_token_id=_END)
    expected = [(_MAX_TOKENS, 4000)] * 4
    # (truncated, empty, calls_rejected): each kind of unproductive turn really happened, then the answer.
    flags = [(True, False, False), (False, True, False), (False, False, True), (False, False, False)]

    cut = SimpleNamespace(**{**vars(_actor_text_turn(tokens=5000)), "text": "half an ans", "finish_reason": "length"})
    gens = [
        cut,
        _actor_text_turn(_ids_closing_reasoning_after(1000), text=""),
        _actor_tool_turn(_ids_closing_reasoning_after(1500), name="absent"),
        _actor_text_turn(_ids_closing_reasoning_after(4000)),
    ]
    seen, result = await _drive_actor(_BudgetEnv, context, config, generations=gens)
    assert result.error is None, result.error
    turns = [m for m in result.trajectory.messages if m.role == "assistant"]
    assert [(m.truncated, m.empty, m.calls_rejected) for m in turns] == flags
    assert _actor_caps(seen) == expected
    assert [m.thinking_cap for m in turns] == [4000] * 4
    assert result.metrics["episode/thinking_cap_turns"] == 1.0

    eval_cut = _text_turn("half an ans", tokens=5000, finish_reason="length")
    responses = [eval_cut, _text_turn(""), _tool_turn(name="absent"), _text_turn()]
    calls, eval_traj = await _drive_eval(monkeypatch, _BudgetEnv(), context, responses=responses, config=config)
    eval_turns = [m for m in eval_traj.messages if m.role == "assistant"]
    assert [(m.truncated, m.empty, m.calls_rejected) for m in eval_turns] == flags
    assert _eval_caps(calls) == expected
    assert [m.thinking_cap for m in eval_turns] == [4000] * 4


async def test_the_eval_refuses_an_output_budget_under_one_turn_before_generating(monkeypatch):
    """An explicit ``--max_tokens`` above the contract's episode budget would let no turn start."""
    calls = []

    async def fake_chat(client, **request):
        calls.append(request)
        return _text_turn()

    monkeypatch.setattr(eval_runner, "chat_completion", fake_chat)
    too_big = ray_actors.RolloutConfig(model_name="m", max_tokens=60000, max_episode_tokens=50000)
    with pytest.raises(
        ValueError, match=r"rollout_max_episode_tokens \(50000\) must be at least rollout_max_tokens \(60000\)"
    ):
        await eval_runner.collect_results(_PlainEnv(), [{"prompt": "solve it", "context": {}}], None, rollout=too_big)
    assert calls == []


async def test_an_output_budget_shrinks_the_reasoning_cap_with_the_total_and_records_the_levels_cap(monkeypatch):
    """4000-token cap, 20000-token turns, 38000 for the episode: the second turn drops to 19000 with a 3000 cap,
    then no third starts. Each turn records the level's 4000, not the narrowed cap."""
    config = ray_actors.RolloutConfig(max_tokens=_MAX_TOKENS, max_thinking_tokens=4000, max_episode_tokens=38000)
    sampled = [19000, 18000]
    expected = [(_MAX_TOKENS, 4000), (19000, 3000)]

    gens = [_actor_tool_turn(tokens=n) for n in sampled]
    seen, result = await _drive_actor(_PlainEnv, None, config, generations=gens)
    assert result.error is None, result.error
    assert _actor_caps(seen) == expected
    assert {budget for _, _, budget, _ in seen} == {4000}, "the template still hears the level's whole cap"
    assert result.trajectory.truncated and result.trajectory.info[OUTPUT_BUDGET_EXHAUSTED_KEY] is True
    assert result.generation_tokens == 37000
    assert [m.thinking_cap for m in result.trajectory.messages if m.role == "assistant"] == [4000, 4000]

    responses = [_tool_turn(tokens=n) for n in sampled]
    calls, eval_traj = await _drive_eval(monkeypatch, _PlainEnv(), None, responses=responses, config=config)
    assert _eval_caps(calls) == expected
    assert [call["extra_body"] for call in calls] == [
        generation_control_fields(cfg, level, budget) for cfg, level, budget, _ in seen
    ]
    assert eval_traj.truncated and eval_traj.info[OUTPUT_BUDGET_EXHAUSTED_KEY] is True
    assert eval_traj.info["_eval_stats"]["completion_tokens"] == 37000
    assert [m.thinking_cap for m in eval_traj.messages if m.role == "assistant"] == [4000, 4000]


async def test_a_narrowed_turn_reads_a_cut_off_its_own_total(monkeypatch):
    """The count is compared with the request's own cap, not the run's turn cap it no longer runs under."""
    config = ray_actors.RolloutConfig(max_tokens=_MAX_TOKENS, max_thinking_tokens=4000, max_episode_tokens=50000)
    full = _tool_turn(tokens=18000)  # the whole 18000 the third request asked for, finish_reason "stop"
    responses = [_tool_turn(tokens=19000), _tool_turn(tokens=13000), full]
    calls, traj = await _drive_eval(monkeypatch, _PlainEnv(), None, responses=responses, config=config)
    assert _eval_caps(calls) == [(_MAX_TOKENS, 4000), (_MAX_TOKENS, 4000), (18000, 2000)]
    assistant = [m for m in traj.messages if m.role == "assistant"]
    assert [m.truncated for m in assistant] == [False, False, True]


async def test_an_episode_inside_its_budget_records_false_and_one_without_a_budget_records_nothing(monkeypatch):
    """Both the flag and the metric exist exactly when a budget was set."""
    budget = ray_actors.RolloutConfig(max_tokens=_MAX_TOKENS, max_episode_tokens=50000)
    _, result = await _drive_actor(_PlainEnv, None, budget, generations=[_actor_text_turn(tokens=19000)])
    assert result.error is None, result.error
    assert result.terminated_naturally and not result.trajectory.truncated
    assert result.trajectory.info[OUTPUT_BUDGET_EXHAUSTED_KEY] is False
    assert result.metrics["episode/output_budget_exhausted"] == 0.0
    assert result.generation_tokens == 19000
    _, kept = await _drive_eval(monkeypatch, _PlainEnv(), None, responses=[_text_turn(tokens=19000)], config=budget)
    assert kept.info[OUTPUT_BUDGET_EXHAUSTED_KEY] is False and not kept.truncated

    unbounded = ray_actors.RolloutConfig(max_tokens=_MAX_TOKENS)
    gens = [_actor_tool_turn(tokens=19000)] * 4
    seen, result = await _drive_actor(_PlainEnv, None, unbounded, generations=gens)
    assert result.error is None, result.error
    assert len(seen) == 4 and result.trajectory.truncated, "max_turns alone ends the episode"
    assert [payload["max_tokens"] for _, _, _, payload in seen] == [_MAX_TOKENS] * 4
    assert OUTPUT_BUDGET_EXHAUSTED_KEY not in result.trajectory.info
    assert "episode/output_budget_exhausted" not in result.metrics
    _, free = await _drive_eval(
        monkeypatch, _PlainEnv(), None, responses=[_tool_turn(tokens=19000)] * 4, config=unbounded
    )
    assert OUTPUT_BUDGET_EXHAUSTED_KEY not in free.info


# A code-contests-like shape: 30000-token turns, per-level caps 8192 / 12288 / 16384, an 81920-token
# episode, and 8192 tokens of answer room past each cap.
_CONTEST_TURN = 30000
_CONTEST_EPISODE = 81920
_ROOM = 8192


class _ContestEnv(_PlainEnv):
    BUDGETS = {"low": 8192, "medium": 12288, "high": 16384}

    def thinking_budget_for_effort(self, effort):
        return self.BUDGETS.get(effort)


def _contest(level, *, answer=_ROOM, episode_tokens=None) -> reasoning_effort.EpisodeEffort:
    return reasoning_effort.bind_episode_effort(
        {"reasoning_effort": level},
        _ContestEnv(),
        max_tokens=_CONTEST_TURN,
        max_episode_tokens=episode_tokens,
        max_answer_tokens=answer,
    )


def test_an_answer_bound_totals_each_turn_at_its_reasoning_cap_plus_the_room():
    """Each level's turn totals its cap plus 8192, not 30000. Where the cap plus the room exceeds the turn,
    ``rollout_max_tokens`` binds; unset, every turn totals 30000."""
    for level, total in {"low": 16384, "medium": 20480, "high": 24576}.items():
        bounded = _contest(level)
        cap = _ContestEnv.BUDGETS[level]
        assert (bounded.turn_total, bounded.answer_room, _caps(bounded, 0)) == (total, _ROOM, (total, cap))
        assert _caps(bounded, 10**6) == (total, cap), "no output budget: nothing narrows"
        unset = _contest(level, answer=None)
        assert (unset.answer_room, _caps(unset, 0)) == (_CONTEST_TURN - cap, (_CONTEST_TURN, cap))
    # 16384 + 20000 overruns the turn: the turn stays 30000 with 13616 of room.
    wide = _contest("high", answer=20000)
    assert (wide.turn_total, wide.answer_room, _caps(wide, 0)) == (_CONTEST_TURN, 13616, (_CONTEST_TURN, 16384))


def test_a_bound_at_or_past_a_levels_own_room_leaves_its_turns_as_unset():
    """A bound at or past a level's own room shrinks none of its turns, since ``rollout_max_tokens`` binds first."""
    for level, cap in _ContestEnv.BUDGETS.items():
        unset = _contest(level, answer=None, episode_tokens=_CONTEST_EPISODE)
        for answer in (_CONTEST_TURN - cap, _CONTEST_TURN - cap + 5000):
            wide = _contest(level, answer=answer, episode_tokens=_CONTEST_EPISODE)
            for generated in range(0, _CONTEST_EPISODE + 1, 997):
                assert wide.turn_caps(generated) == unset.turn_caps(generated), (level, answer, generated)


def test_an_answer_bound_narrows_with_the_output_budget_and_starts_a_turn_only_while_the_room_is_left():
    """Near the end of the episode the total narrows and the cap gives up the difference, so every turn keeps its
    8192; a turn starts while 8192 remain and not one token under."""
    low = _contest("low", episode_tokens=_CONTEST_EPISODE)
    assert _caps(low, 60000) == (16384, 8192), "21920 left: the turn's own caps stand"
    assert _caps(low, 70000) == (11920, 3728)
    high = _contest("high", episode_tokens=_CONTEST_EPISODE)
    assert _caps(high, 60000) == (21920, 13728)
    for effort in (low, _contest("medium", episode_tokens=_CONTEST_EPISODE), high):
        assert _caps(effort, _CONTEST_EPISODE - _ROOM) == (_ROOM, 1)
        last = _CONTEST_EPISODE - _ROOM + 1
        assert _caps(effort, last) is None
        traj = Trajectory(messages=[])
        effort.stamp(traj, last)
        assert traj.info[OUTPUT_BUDGET_EXHAUSTED_KEY] is True
        effort.stamp(traj, last - 1)
        assert traj.info[OUTPUT_BUDGET_EXHAUSTED_KEY] is False
    assert _caps(_contest("low", answer=None, episode_tokens=_CONTEST_EPISODE), 70000) is None


async def test_both_drivers_bound_a_turn_and_the_one_after_a_cut_by_the_answer_room(monkeypatch):
    """A 40000-token budget: the first turn asks 8192 + 8192 and is cut, the next asks the same, the third has 13616
    left with its cap down to 5424, then 7616 remain and the episode closes truncated."""
    context = {"reasoning_effort": "low"}
    config = ray_actors.RolloutConfig(max_tokens=_CONTEST_TURN, max_answer_tokens=_ROOM, max_episode_tokens=40000)
    sampled = [16384, 10000, 6000]
    expected = [(16384, 8192), (16384, 8192), (13616, 5424)]

    cut = SimpleNamespace(**{**vars(_actor_text_turn(tokens=sampled[0])), "text": "half", "finish_reason": "length"})
    gens = [cut, _actor_tool_turn(tokens=sampled[1]), _actor_tool_turn(tokens=sampled[2])]
    seen, result = await _drive_actor(_ContestEnv, context, config, generations=gens)
    assert result.error is None, result.error
    assert _actor_caps(seen) == expected
    traj = result.trajectory
    assert [m.truncated for m in traj.messages if m.role == "assistant"] == [True, False, False]
    assert [m.thinking_cap for m in traj.messages if m.role == "assistant"] == [8192] * 3
    assert len(seen) == 3 < _ContestEnv().max_turns and traj.truncated
    assert traj.info[OUTPUT_BUDGET_EXHAUSTED_KEY] is True
    assert result.generation_tokens == sum(sampled)

    eval_cut = _text_turn("half", tokens=sampled[0], finish_reason="length")
    responses = [eval_cut, _tool_turn(tokens=sampled[1]), _tool_turn(tokens=sampled[2])]
    calls, eval_traj = await _drive_eval(monkeypatch, _ContestEnv(), context, responses=responses, config=config)
    assert _eval_caps(calls) == expected
    assert [m.thinking_cap for m in eval_traj.messages if m.role == "assistant"] == [8192] * 3
    assert eval_traj.truncated and eval_traj.info[OUTPUT_BUDGET_EXHAUSTED_KEY] is True


async def test_an_answer_bound_is_refused_where_a_drawable_level_has_no_reasoning_cap(monkeypatch):
    """The bound counts past a reasoning cap, so an episode without one has nothing to bound."""
    context = {"reasoning_effort": "high"}
    with pytest.raises(ValueError, match=r"rollout_max_answer_tokens \(8192\) bounds .* reasoning_effort='high'"):
        reasoning_effort.bind_episode_effort(context, _PlainEnv(), max_tokens=_MAX_TOKENS, max_answer_tokens=8192)
    capped = reasoning_effort.bind_episode_effort(
        context, _PlainEnv(), max_tokens=_MAX_TOKENS, max_thinking_tokens=4000, max_answer_tokens=8192
    )
    assert (capped.answer_tokens, capped.answer_room) == (8192, 8192)

    calls = []

    async def fake_chat(client, **request):
        calls.append(request)
        return _text_turn()

    monkeypatch.setattr(eval_runner, "chat_completion", fake_chat)
    uncapped = ray_actors.RolloutConfig(model_name="m", max_tokens=_MAX_TOKENS, max_answer_tokens=8192)
    examples = [{"prompt": "solve it", "context": {}}]
    with pytest.raises(ValueError, match="rollout_max_answer_tokens"):
        await eval_runner.collect_results(_PlainEnv(reasoning_effort="random"), examples, None, rollout=uncapped)
    assert calls == [], "the gate must refuse before the first request"
    efforts = reasoning_effort.efforts_by_level(
        _ContestEnv(reasoning_effort="random"),
        max_tokens=_CONTEST_TURN,
        max_thinking_tokens=None,
        max_answer_tokens=_ROOM,
    )
    caps = {level: effort.thinking_cap for level, effort in efforts.items()}
    assert caps == _ContestEnv.BUDGETS, "every drawable level capped: the gate passes"


# The soft shape the code-contests vLLM recipes run: the same per-level budgets, never enforced; the engine closes
# the reasoning only where the answer room is left of the turn.
_SOFT_CEILING = _CONTEST_TURN - _ROOM


def _soft_env(**kwargs):
    profiles = {
        level: {"thinking_tokens": budget, "enforce_thinking_cap": False}
        for level, budget in _ContestEnv.BUDGETS.items()
    }
    return _PlainEnv(reasoning_effort_profiles=profiles, **kwargs)


def _soft(level, *, ceiling=None, answer=_ROOM):
    return reasoning_effort.bind_episode_effort(
        {"reasoning_effort": level},
        _soft_env(),
        max_tokens=_CONTEST_TURN,
        max_thinking_tokens=ceiling,
        max_answer_tokens=answer,
    )


def test_a_soft_budget_is_stated_while_the_engine_closes_only_where_the_answer_room_is_left():
    """A level with ``enforce_thinking_cap: false`` keeps its budget as what the template states and the terms read,
    while the engine closes reasoning only where ``rollout_max_answer_tokens`` is left of the turn."""
    for level, budget in _ContestEnv.BUDGETS.items():
        soft = _soft(level)
        assert (soft.thinking_budget, soft.thinking_cap, soft.budget_soft) == (budget, _SOFT_CEILING, True)
        assert (soft.answer_room, _caps(soft, 0)) == (_ROOM, (_CONTEST_TURN, _SOFT_CEILING))
        enforced = _contest(level)
        assert (enforced.thinking_budget, enforced.thinking_cap, enforced.budget_soft) == (budget, budget, False)
    trajectory = Trajectory()
    _soft("high").stamp(trajectory, 0)
    assert (trajectory.reasoning_budget, trajectory.reasoning_budget_soft) == (16384, True)
    _contest("high").stamp(trajectory, 0)
    assert (trajectory.reasoning_budget, trajectory.reasoning_budget_soft) == (16384, False)


def test_a_run_cap_is_a_soft_budgets_ceiling_and_without_either_bound_nothing_caps_it():
    """``rollout_max_thinking_tokens`` sets the ceiling where a run states one; with neither it nor an answer
    bound, nothing caps the reasoning but the turn."""
    assert _soft("high", ceiling=21000).thinking_cap == 21000
    bare = _soft("high", answer=None)
    assert (bare.thinking_budget, bare.thinking_cap, bare.budget_soft) == (16384, None, True)
    assert _caps(bare, 0) == (_CONTEST_TURN, None)


def test_an_uncapped_soft_turn_needs_its_whole_total_left_of_the_output_budget():
    """Nothing closes the reasoning below the total, so no part of the turn is answer room: a turn starts only while
    the budget holds the whole total. An enforced level narrows instead."""
    bare = reasoning_effort.bind_episode_effort(
        {"reasoning_effort": "high"}, _soft_env(), max_tokens=_CONTEST_TURN, max_episode_tokens=_CONTEST_EPISODE
    )
    assert bare.answer_room == _CONTEST_TURN
    last_start = _CONTEST_EPISODE - _CONTEST_TURN
    assert _caps(bare, last_start) == (_CONTEST_TURN, None)
    assert _caps(bare, last_start + 1) is None
    enforced = _contest("high", answer=None, episode_tokens=_CONTEST_EPISODE)
    assert _caps(enforced, last_start + 1) == (_CONTEST_TURN - 1, 16384 - 1)


def test_a_soft_budget_at_or_past_its_ceiling_is_refused():
    """At or past its ceiling the engine would enforce the budget, so it is no soft budget."""
    with pytest.raises(
        ValueError, match=r"'high' level's soft thinking_tokens \(16384\) must sit below its ceiling \(12000"
    ):
        _soft("high", ceiling=12000)
    with pytest.raises(ValueError, match=r"must sit below its ceiling \(10000"):
        _soft("high", answer=20000)
    with pytest.raises(ValueError, match=r"must sit below its ceiling \(16384"):
        _soft("high", ceiling=16384)
    assert _soft("high", ceiling=16385).thinking_cap == 16385
    with pytest.raises(ValueError, match="must sit below rollout_max_tokens"):
        reasoning_effort.bind_episode_effort(
            {"reasoning_effort": "high"},
            _PlainEnv(reasoning_effort_profiles={"high": {"thinking_tokens": 40000, "enforce_thinking_cap": False}}),
            max_tokens=_CONTEST_TURN,
        )


def test_a_soft_levels_ceiling_at_or_past_the_turn_is_refused():
    """A cap at or past the turn total closes nothing and leaves no answer room."""
    for cap in (_CONTEST_TURN, _CONTEST_TURN + 1):
        with pytest.raises(
            ValueError, match=rf"rollout_max_thinking_tokens \({cap}\) must sit below rollout_max_tokens"
        ):
            _soft("high", ceiling=cap, answer=None)
    assert _soft("high", ceiling=_CONTEST_TURN - 1, answer=None).answer_room == 1


async def test_both_drivers_send_the_ceiling_and_state_the_soft_budget(monkeypatch):
    """Both ask the engine for the ceiling, state the budget flagged soft, and leave the episode carrying it."""
    context = {"reasoning_effort": "high"}
    config = ray_actors.RolloutConfig(max_tokens=_CONTEST_TURN, max_answer_tokens=_ROOM)
    profiles = {"high": {"thinking_tokens": 16384, "enforce_thinking_cap": False}}
    soft_kwargs = {"reasoning_budget": 16384, "reasoning_budget_soft": True}
    seen, result = await _drive_actor(_PlainEnv, context, config, env_kwargs={"reasoning_effort_profiles": profiles})
    assert result.error is None, result.error
    payload = seen[0][3]
    assert (payload["max_tokens"], payload["thinking_token_budget"]) == (_CONTEST_TURN, _SOFT_CEILING)
    assert payload["chat_template_kwargs"] == soft_kwargs
    assert (result.trajectory.reasoning_budget, result.trajectory.reasoning_budget_soft) == (16384, True)
    calls, eval_traj = await _drive_eval(
        monkeypatch,
        _PlainEnv(reasoning_effort_profiles=profiles),
        context,
        responses=[_text_turn()],
        config=replace(config, model_name="m"),
    )
    assert calls[0]["extra_body"]["thinking_token_budget"] == _SOFT_CEILING
    assert calls[0]["extra_body"]["chat_template_kwargs"] == soft_kwargs
    assert eval_runner.serialize_trajectory(eval_traj)["reasoning_budget_soft"] is True


async def test_both_drivers_send_the_ceiling_again_on_the_turn_after_a_cut_under_a_soft_budget(monkeypatch):
    """Under a soft budget the turn after a cut one asks the engine for the ceiling and the whole turn again, and
    both turns record the ceiling as the cap they ran under."""
    context = {"reasoning_effort": "low"}
    config = ray_actors.RolloutConfig(max_tokens=_CONTEST_TURN, max_answer_tokens=_ROOM)
    profiles = {"low": {"thinking_tokens": 8192, "enforce_thinking_cap": False}}
    expected = [(_CONTEST_TURN, _SOFT_CEILING)] * 2

    cut = SimpleNamespace(**{**vars(_actor_text_turn(tokens=30000)), "text": "half", "finish_reason": "length"})
    seen, result = await _drive_actor(
        _PlainEnv,
        context,
        config,
        generations=[cut, _actor_text_turn()],
        env_kwargs={"reasoning_effort_profiles": profiles},
    )
    assert result.error is None, result.error
    turns = [m for m in result.trajectory.messages if m.role == "assistant"]
    assert [m.truncated for m in turns] == [True, False]
    assert _actor_caps(seen) == expected
    assert [m.thinking_cap for m in turns] == [_SOFT_CEILING] * 2

    eval_cut = _text_turn("half", tokens=30000, finish_reason="length")
    calls, eval_traj = await _drive_eval(
        monkeypatch,
        _PlainEnv(reasoning_effort_profiles=profiles),
        context,
        responses=[eval_cut, _text_turn()],
        config=replace(config, model_name="m"),
    )
    eval_turns = [m for m in eval_traj.messages if m.role == "assistant"]
    assert [m.truncated for m in eval_turns] == [True, False]
    assert _eval_caps(calls) == expected
    assert [m.thinking_cap for m in eval_turns] == [_SOFT_CEILING] * 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
