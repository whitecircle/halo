#!/usr/bin/env python
"""CPU tests: both rollout drivers bind an episode's reasoning effort through ONE seam.

``bind_episode_effort`` (``halo/environments/reasoning_effort.py``) resolves the episode's level once and binds the
run's token caps every turn runs under: the reasoning cap is ``rollout_max_thinking_tokens``, else the turn
less its answer room, the same at every level. A level's profile (``reasoning_budget``,
``reasoning_price``) is what the trainer's reasoning terms read; it never reaches the engine or the
template, which hears the level alone. The Ray actor and the offline eval runner must both go through
the seam: a driver that resolves the level itself, or caps a turn its own way, generates under a
different contract than the one the policy was trained on. The episode output budget
(``rollout_max_episode_tokens``) rides the same seam: ``EpisodeEffort.turn_caps`` narrows a turn's
request to what the budget has left and hands back none once it holds no turn, which ends the episode
truncated. So does the answer-room bound (``rollout_max_answer_tokens``): a capped turn's total is at
most its reasoning cap plus the room, so a turn the engine force-closed cannot reason on in its visible
output up to ``rollout_max_tokens``.

Both drivers are exercised for real — the actor via the class ``@ray.remote`` wraps (no cluster), the
eval runner with its generation call stubbed — so the tests fail if either one stops binding.

    python tests/cpu/environments/test_effort_binding.py
"""

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
# Per-level reasoning budgets and prices, each budget distinct from every run cap below, so a budget that
# reached the engine would show as a cap the run never set.
_PROFILES = {
    "low": {"reasoning_budget": 1000, "reasoning_price": 0.008},
    "medium": {"reasoning_budget": 2000, "reasoning_price": 0.006},
    "high": {"reasoning_budget": 3000, "reasoning_price": 0.004},
}
# The run's own template variables: what the nested template kwargs carry, and all they carry.
_TEMPLATE_KWARGS = {"preserve_thinking": True}


def _echo_registry() -> NativeToolRegistry:
    """One no-op tool, so a tool turn is an ordinary productive turn that keeps the episode open."""
    registry = NativeToolRegistry()
    registry.register(NativeTool(name="echo", description="echo", parameters=[], handler=lambda: "ok"))
    return registry


class _PlainEnv(NativeToolUseEnvironment):
    """Env with one no-op tool and empty effort profiles (the ``BaseEnvironment`` default)."""

    def __init__(self, **kwargs):
        super().__init__(tool_registry=_echo_registry(), max_turns=4, **kwargs)


class _ProfiledEnv(_PlainEnv):
    """…and one whose levels each carry a reasoning budget and price, as the code-contests recipes do."""

    def __init__(self, **kwargs):
        kwargs.setdefault("reasoning_effort_profiles", _PROFILES)
        super().__init__(**kwargs)


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
    """Run the real ``EnvironmentActor`` episode loop off-cluster over ``generations`` (one text turn
    when omitted), the env built with ``env_kwargs``; return its per-turn ``(config, level, payload)``
    records and the result.

    ``@ray.remote`` wraps the class and keeps the original at ``__ray_metadata__.modified_class``;
    ``__init__`` only assigns attributes, so this exercises the genuine ``run_episode`` binding. The
    transport is replaced below the payload builder, so each record's payload is the request the
    actor would have sent for that turn.
    """
    cls = ray_actors.EnvironmentActor.__ray_metadata__.modified_class
    actor = cls.__new__(cls)
    actor.__init__(actor_id=0, env_type=(env_cls, env_kwargs or {}), env_config={})
    seen = []

    async def fake_generate(client, url, messages, cfg, reasoning_effort=None):
        payload = actor._build_payload(messages, cfg, reasoning_effort)
        seen.append((cfg, reasoning_effort, payload))
        return generations[len(seen) - 1] if generations else _actor_text_turn()

    actor._http_client = None
    actor._generate = fake_generate
    result = await actor.run_episode("solve it", dict(context or {}), "http://x", config)
    return seen, result


def _actor_caps(seen):
    return [(payload["max_tokens"], payload.get("thinking_token_budget")) for _, _, payload in seen]


def _eval_caps(calls):
    return [(call["max_tokens"], call["extra_body"].get("thinking_token_budget")) for call in calls]


async def test_both_drivers_bind_the_same_caps(monkeypatch):
    """The run's caps must reach BOTH drivers' generation contract, identically, and the level's profile
    neither: the engine is sent the run's 4000-token cap, not the level's 3000-token budget, and the
    template hears the run's own variables and the level, never a count."""
    context = {"reasoning_effort": "high"}
    config = ray_actors.RolloutConfig(
        max_tokens=_MAX_TOKENS, max_thinking_tokens=4000, chat_template_kwargs=_TEMPLATE_KWARGS
    )

    seen, result = await _drive_actor(_ProfiledEnv, context, config)
    assert result.error is None, result.error
    (actor_cfg, actor_level, payload) = seen[0]
    calls, eval_traj = await _drive_eval(monkeypatch, _ProfiledEnv(), context, responses=[_text_turn()], config=config)

    assert (actor_level, actor_cfg.max_thinking_tokens, actor_cfg.max_tokens) == ("high", 4000, _MAX_TOKENS)
    # Eval side: byte-identical generation-control fields to the ones the actor's payload carries —
    # built from the ACTOR's own per-turn config, so this compares the two drivers, not the helper
    # with itself.
    extra_body = calls[0]["extra_body"]
    assert extra_body == generation_control_fields(actor_cfg, actor_level)
    assert extra_body["reasoning_effort"] == payload["reasoning_effort"] == "high"
    assert extra_body["thinking_token_budget"] == payload["thinking_token_budget"] == 4000
    assert extra_body["chat_template_kwargs"] == payload["chat_template_kwargs"] == _TEMPLATE_KWARGS
    assert calls[0]["max_tokens"] == payload["max_tokens"] == _MAX_TOKENS
    # Both leave the episode carrying the level it ran under: the trainer's re-render and reasoning
    # terms read it off the trajectory, the eval records it — and only it — in the trajectory file.
    assert eval_traj.reasoning_effort == result.trajectory.reasoning_effort == "high"
    recorded = eval_runner.serialize_trajectory(eval_traj)
    assert recorded["reasoning_effort"] == "high"
    assert set(recorded) == {"messages", "total_reward", "done", "truncated", "reasoning_effort", "info"}


async def test_every_turn_of_an_episode_runs_under_the_runs_whole_cap(monkeypatch):
    """The reasoning cap is per turn: three turns each reasoning 1500 tokens of the run's 4000 all request
    the same 4000-token cap and the same turn total, add no template variable for the level's budget, and
    the control fields ask for no sampled ids (the eval transport captures nothing) — on both drivers
    identically."""
    context = {"reasoning_effort": "high"}
    config = ray_actors.RolloutConfig(max_tokens=_MAX_TOKENS, max_thinking_tokens=4000, reasoning_end_token_id=_END)
    ids = _ids_closing_reasoning_after(1500)
    expected = [(_MAX_TOKENS, 4000)] * 3

    gens = [_actor_tool_turn(ids), _actor_tool_turn(ids), _actor_text_turn(ids)]
    seen, result = await _drive_actor(_ProfiledEnv, context, config, generations=gens)
    assert result.error is None, result.error
    assert _actor_caps(seen) == expected
    assert [(cfg.max_tokens, cfg.max_thinking_tokens) for cfg, _, _ in seen] == expected
    assert all("chat_template_kwargs" not in payload for _, _, payload in seen), "the run sets no template variable"
    assert result.trajectory.reasoning_effort == "high"
    assert OUTPUT_BUDGET_EXHAUSTED_KEY not in result.trajectory.info, "no output budget, no verdict"

    responses = [_tool_turn(), _tool_turn(), _text_turn()]
    calls, eval_traj = await _drive_eval(monkeypatch, _ProfiledEnv(), context, responses=responses, config=config)
    assert _eval_caps(calls) == expected
    assert [call["extra_body"] for call in calls] == [generation_control_fields(cfg, level) for cfg, level, _ in seen]
    assert all("return_token_ids" not in call["extra_body"] for call in calls)
    assert eval_traj.reasoning_effort == "high"


def _bound(env, level, **caps) -> reasoning_effort.EpisodeEffort:
    return reasoning_effort.bind_episode_effort({"reasoning_effort": level}, env, max_tokens=_MAX_TOKENS, **caps)


def test_every_level_binds_the_runs_caps_whatever_its_profile():
    """A level never caps the engine: under a run cap, an answer bound, an output budget or none, a
    profiled env binds the plain env's contract at each level, and the three levels differ in nothing
    but the level."""
    runs = {
        None: {},
        4000: {"max_thinking_tokens": 4000},
        _MAX_TOKENS - 8192: {"max_answer_tokens": 8192},
        3000: {"max_thinking_tokens": 3000, "max_answer_tokens": 8192, "max_episode_tokens": 50000},
    }
    for cap, caps in runs.items():
        bound = {level: _bound(_ProfiledEnv(), level, **caps) for level in VALID_REASONING_EFFORTS}
        for level, effort in bound.items():
            assert effort == _bound(_PlainEnv(), level, **caps), (level, caps)
            assert (effort.level, effort.thinking_cap) == (level, cap), caps
        assert len({replace(effort, level=None) for effort in bound.values()}) == 1, caps


async def test_without_a_run_cap_neither_driver_caps_a_profiled_levels_reasoning(monkeypatch):
    """No ``rollout_max_thinking_tokens`` and no answer bound: nothing caps the reasoning, though the level's
    profile sets a budget — no driver sends a cap, and no turn records one."""
    context = {"reasoning_effort": "high"}
    config = ray_actors.RolloutConfig(max_tokens=_MAX_TOKENS)

    seen, result = await _drive_actor(_ProfiledEnv, context, config)
    assert result.error is None, result.error
    assert _actor_caps(seen) == [(_MAX_TOKENS, None)]
    assert [m.thinking_cap for m in result.trajectory.messages if m.role == "assistant"] == [None]
    calls, eval_traj = await _drive_eval(monkeypatch, _ProfiledEnv(), context, responses=[_text_turn()])
    assert _eval_caps(calls) == [(_MAX_TOKENS, None)]
    assert [m.thinking_cap for m in eval_traj.messages if m.role == "assistant"] == [None]
    assert eval_traj.reasoning_effort == "high"


async def test_random_level_is_drawn_once_for_the_whole_episode(monkeypatch):
    """A 'random' env level is one draw per EPISODE: every turn, and the recorded level, share it.

    Re-resolving per turn would steer consecutive turns at different efforts and leave a recorded level
    that belongs to no turn — the trap the once-per-episode seam exists to prevent. Counted, not inferred
    from the drawn levels: a per-turn re-draw agrees with itself one time in three.
    """
    env = _ProfiledEnv(reasoning_effort="random")
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
    assert traj.reasoning_effort == bindings[0].level == level


async def test_the_actor_records_each_turns_cap_and_reasoning_and_the_eval_its_cap_alone(monkeypatch):
    """``episode/thinking_cap_turns`` reads each training turn's pair: the run's cap the turn ran under and
    the reasoning it sampled through the close. The eval asks for no ids, so its record carries the cap
    alone."""
    context = {"reasoning_effort": "high"}
    config = ray_actors.RolloutConfig(max_tokens=_MAX_TOKENS, max_thinking_tokens=4000, reasoning_end_token_id=_END)
    ids = _ids_closing_reasoning_after(1500)

    gens = [_actor_tool_turn(ids), _actor_tool_turn(ids), _actor_text_turn(ids)]
    _, result = await _drive_actor(_ProfiledEnv, context, config, generations=gens)
    actor_turns = [m for m in result.trajectory.messages if m.role == "assistant"]
    assert [(m.thinking_cap, m.reasoning_tokens) for m in actor_turns] == [(4000, 1500)] * 3

    responses = [_tool_turn(), _tool_turn(), _text_turn()]
    _, eval_traj = await _drive_eval(monkeypatch, _ProfiledEnv(), context, responses=responses, config=config)
    assert [(m.thinking_cap, m.reasoning_tokens) for m in eval_traj.messages if m.role == "assistant"] == [
        (4000, None)
    ] * 3
    recorded = [m for m in eval_runner.serialize_trajectory(eval_traj)["messages"] if m["role"] == "assistant"]
    assert [m["thinking_cap"] for m in recorded] == [4000] * 3 and all("reasoning_tokens" not in m for m in recorded)


def test_the_reasoning_cap_is_the_run_cap_else_the_turn_less_its_answer_room():
    """``rollout_max_thinking_tokens`` caps every turn's reasoning; without it an answer bound caps it where
    only that room is left of the turn; with neither nothing does. A cap outside ``[1, rollout_max_tokens)``
    is refused: at or past the turn it leaves no answer room and the turn is cut mid-reasoning every time,
    and under one it closes the reasoning before it opens. The output budget rides on every binding."""
    env = _ProfiledEnv()
    assert _bound(env, "high", max_thinking_tokens=8192).thinking_cap == 8192
    assert _bound(env, "high", max_answer_tokens=8192).thinking_cap == _MAX_TOKENS - 8192
    assert _bound(env, "high", max_thinking_tokens=3000, max_answer_tokens=8192).thinking_cap == 3000
    assert _bound(env, "high").thinking_cap is None
    for caps, cap in (
        ({"max_thinking_tokens": _MAX_TOKENS}, _MAX_TOKENS),
        ({"max_thinking_tokens": 0}, 0),
        ({"max_answer_tokens": _MAX_TOKENS}, 0),
        ({"max_answer_tokens": _MAX_TOKENS + 1}, -1),
    ):
        with pytest.raises(
            ValueError, match=rf"reasoning cap \({cap}: .* must sit in \[1, rollout_max_tokens=20000\)"
        ):
            _bound(env, "high", **caps)
    for env in (_ProfiledEnv(), _PlainEnv()):
        assert _bound(env, "high", max_episode_tokens=50000).episode_tokens == 50000
        assert _bound(env, "high").episode_tokens is None


async def test_the_eval_refuses_a_reasoning_cap_that_fills_the_turn_before_generating(monkeypatch):
    """A cap that fills the turn would fail every episode at its first turn, each recorded as a zero-reward
    error sample: the eval gates on it before its first request, as the trainer does at construction — an
    explicit ``--max_tokens`` at or under the training contract's run cap or answer bound included — and a
    run whose cap fits passes the same gate, at every level the env draws."""
    calls = []

    async def fake_chat(client, **request):
        calls.append(request)
        return _text_turn()

    monkeypatch.setattr(eval_runner, "chat_completion", fake_chat)
    env = _ProfiledEnv(reasoning_effort="random")
    examples = [{"prompt": "solve it", "context": {}}]
    for too_small in (
        ray_actors.RolloutConfig(model_name="m", max_tokens=4000, max_thinking_tokens=4000),
        ray_actors.RolloutConfig(model_name="m", max_tokens=8192, max_answer_tokens=8192),
    ):
        with pytest.raises(ValueError, match=r"must sit in \[1, rollout_max_tokens="):
            await eval_runner.collect_results(env, examples, None, rollout=too_small)
    assert calls == [], "the gate must refuse before the first request"

    def caps(env, **kwargs):
        efforts = reasoning_effort.efforts_by_level(env, max_tokens=_MAX_TOKENS, **kwargs)
        return {level: effort.thinking_cap for level, effort in efforts.items()}

    assert caps(env, max_thinking_tokens=4000) == dict.fromkeys(VALID_REASONING_EFFORTS, 4000)
    assert caps(env, max_thinking_tokens=None, max_answer_tokens=8192) == dict.fromkeys(
        VALID_REASONING_EFFORTS, _MAX_TOKENS - 8192
    )
    assert caps(_ProfiledEnv(reasoning_effort="low"), max_thinking_tokens=4000) == {"low": 4000}
    assert caps(_PlainEnv(), max_thinking_tokens=None) == {None: None}
    fits = ray_actors.RolloutConfig(model_name="m", max_tokens=_MAX_TOKENS, max_thinking_tokens=4000)
    results = await eval_runner.collect_results(env, examples, None, rollout=fits)
    assert len(calls) == 1 and "error" not in results[0]["samples"][0]


def test_drivers_share_one_seam():
    # A re-forked local copy would let the two drivers bind different contracts.
    assert eval_runner.bind_episode_effort is reasoning_effort.bind_episode_effort
    assert ray_actors.bind_episode_effort is reasoning_effort.bind_episode_effort


def test_context_level_wins_over_the_env_setting():
    # GRPO's group baseline assumes one conditioning per group: the trainer stamps it into the context.
    env = _ProfiledEnv(reasoning_effort="low")
    stamped = reasoning_effort.bind_episode_effort(
        {"reasoning_effort": "high"}, env, max_tokens=_MAX_TOKENS, max_thinking_tokens=4000
    )
    own = reasoning_effort.bind_episode_effort(None, env, max_tokens=_MAX_TOKENS, max_thinking_tokens=4000)
    assert (stamped.level, own.level) == ("high", "low")
    assert replace(stamped, level="low") == own, "the context changes the level and nothing else"
    unset = reasoning_effort.bind_episode_effort(None, _ProfiledEnv(), max_tokens=_MAX_TOKENS)
    assert (unset.level, unset.thinking_cap, unset.max_tokens) == (None, None, _MAX_TOKENS)


# What the native protocol charges an episode closed as a max_turns overflow; the output budget's
# truncation must be priced the same.
_OVERFLOW_PRICE = 0.3


def _effort(**overrides) -> reasoning_effort.EpisodeEffort:
    """A 20000-token turn with a 4000-token reasoning cap (16000 of answer room) under a 50000-token
    episode budget."""
    bound = {"level": None, "thinking_cap": 4000, "max_tokens": _MAX_TOKENS, "episode_tokens": 50000}
    return reasoning_effort.EpisodeEffort(**{**bound, **overrides})


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
    uncapped = _effort(thinking_cap=None)
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
    assert [(cfg.max_tokens, cfg.max_thinking_tokens) for cfg, _, _ in seen] == expected
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


async def test_both_drivers_run_the_turn_after_an_unproductive_one_under_the_runs_whole_cap(monkeypatch):
    """A cut turn, a turn ended on nothing and a turn whose one call names no tool each leave the next request on
    the run's 4000-token reasoning cap and its whole turn total, and every turn records that cap: a turn that
    closed its reasoning at 1000 or 1500 tokens reads as under its cap, so ``episode/thinking_cap_turns`` counts
    only the one that reasoned to 4000."""
    context = {"reasoning_effort": "high"}
    config = ray_actors.RolloutConfig(max_tokens=_MAX_TOKENS, max_thinking_tokens=4000, reasoning_end_token_id=_END)
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
    seen, result = await _drive_actor(_ProfiledEnv, context, config, generations=gens)
    assert result.error is None, result.error
    turns = [m for m in result.trajectory.messages if m.role == "assistant"]
    assert [(m.truncated, m.empty, m.calls_rejected) for m in turns] == flags
    assert _actor_caps(seen) == expected
    assert [m.thinking_cap for m in turns] == [4000] * 4
    assert result.metrics["episode/thinking_cap_turns"] == 1.0

    eval_cut = _text_turn("half an ans", tokens=5000, finish_reason="length")
    responses = [eval_cut, _text_turn(""), _tool_turn(name="absent"), _text_turn()]
    calls, eval_traj = await _drive_eval(monkeypatch, _ProfiledEnv(), context, responses=responses, config=config)
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


async def test_an_output_budget_shrinks_the_reasoning_cap_with_the_total_and_records_the_runs_cap(monkeypatch):
    """The run's 4000-token reasoning cap under 20000-token turns (16000 of answer room), 38000 for the
    episode: the second turn has 19000 left, so its request drops to 19000 with a 3000 reasoning cap,
    the room kept whole; after it 1000 remain, under the room, and no third turn starts. The narrowed
    caps reach both drivers' requests, the eval's control fields are the actor's own turn by turn, and
    each turn records the run's 4000 — the cap ``episode/thinking_cap_turns`` reads against — not the
    narrowed one."""
    config = ray_actors.RolloutConfig(max_tokens=_MAX_TOKENS, max_thinking_tokens=4000, max_episode_tokens=38000)
    sampled = [19000, 18000]
    expected = [(_MAX_TOKENS, 4000), (19000, 3000)]

    gens = [_actor_tool_turn(tokens=n) for n in sampled]
    seen, result = await _drive_actor(_PlainEnv, None, config, generations=gens)
    assert result.error is None, result.error
    assert _actor_caps(seen) == expected
    assert result.trajectory.truncated and result.trajectory.info[OUTPUT_BUDGET_EXHAUSTED_KEY] is True
    assert result.generation_tokens == 37000
    assert [m.thinking_cap for m in result.trajectory.messages if m.role == "assistant"] == [4000, 4000]

    responses = [_tool_turn(tokens=n) for n in sampled]
    calls, eval_traj = await _drive_eval(monkeypatch, _PlainEnv(), None, responses=responses, config=config)
    assert _eval_caps(calls) == expected
    assert [call["extra_body"] for call in calls] == [generation_control_fields(cfg, level) for cfg, level, _ in seen]
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
    assert [payload["max_tokens"] for _, _, payload in seen] == [_MAX_TOKENS] * 4
    assert OUTPUT_BUDGET_EXHAUSTED_KEY not in result.trajectory.info
    assert "episode/output_budget_exhausted" not in result.metrics
    _, free = await _drive_eval(
        monkeypatch, _PlainEnv(), None, responses=[_tool_turn(tokens=19000)] * 4, config=unbounded
    )
    assert OUTPUT_BUDGET_EXHAUSTED_KEY not in free.info


# --- The answer-room bound ---

# A code-contests-like shape: 30000-token turns, an 81920-token episode, and 8192 tokens of answer room
# past the reasoning cap.
_CONTEST_TURN = 30000
_CONTEST_EPISODE = 81920
_ROOM = 8192
# Run caps a recipe may set, each under the turn.
_RUN_CAPS = (8192, 12288, 16384)
# The cap the answer bound sets without a run cap: the turn less its room.
_ROOM_CAP = _CONTEST_TURN - _ROOM


def _contest(cap, *, answer=_ROOM, episode_tokens=None) -> reasoning_effort.EpisodeEffort:
    return reasoning_effort.bind_episode_effort(
        {"reasoning_effort": "low"},
        _ProfiledEnv(),
        max_tokens=_CONTEST_TURN,
        max_thinking_tokens=cap,
        max_episode_tokens=episode_tokens,
        max_answer_tokens=answer,
    )


def test_an_answer_bound_totals_each_turn_at_its_reasoning_cap_plus_the_room():
    """Under each run cap a turn totals the cap plus 8192, not 30000, so a turn the engine force-closed keeps
    8192 tokens past the close. Where the cap plus the room exceeds the turn, ``rollout_max_tokens`` still
    binds; unset, every turn totals 30000. Without a run cap the bound sets the cap itself, where its room
    is left of the turn, which stays whole."""
    for cap, total in zip(_RUN_CAPS, (16384, 20480, 24576), strict=True):
        bounded = _contest(cap)
        assert (bounded.turn_total, bounded.answer_room, _caps(bounded, 0)) == (total, _ROOM, (total, cap))
        assert _caps(bounded, 10**6) == (total, cap), "no output budget: nothing narrows"
        unset = _contest(cap, answer=None)
        assert (unset.answer_room, _caps(unset, 0)) == (_CONTEST_TURN - cap, (_CONTEST_TURN, cap))
    # 16384 + 20000 overruns the turn: the turn stays 30000 with 13616 of room.
    wide = _contest(16384, answer=20000)
    assert (wide.turn_total, wide.answer_room, _caps(wide, 0)) == (_CONTEST_TURN, 13616, (_CONTEST_TURN, 16384))
    alone = _contest(None)
    assert (alone.turn_total, alone.answer_room, _caps(alone, 0)) == (_CONTEST_TURN, _ROOM, (_CONTEST_TURN, _ROOM_CAP))


def test_a_bound_at_or_past_the_run_caps_own_room_leaves_its_turns_as_unset():
    """A bound at the cap's own room (30000 less the cap) or past it shrinks none of the turns: their caps
    match the unset ones at every point of the episode budget, since ``rollout_max_tokens`` binds first."""
    for cap in _RUN_CAPS:
        unset = _contest(cap, answer=None, episode_tokens=_CONTEST_EPISODE)
        for answer in (_CONTEST_TURN - cap, _CONTEST_TURN - cap + 5000):
            wide = _contest(cap, answer=answer, episode_tokens=_CONTEST_EPISODE)
            for generated in range(0, _CONTEST_EPISODE + 1, 997):
                assert wide.turn_caps(generated) == unset.turn_caps(generated), (cap, answer, generated)


def test_an_answer_bound_narrows_with_the_output_budget_and_starts_a_turn_only_while_the_room_is_left():
    """Near the end of an 81920-token episode the total narrows to what is left and the reasoning cap
    gives up the difference, so every turn keeps its 8192 of room. A turn starts while 8192 remain and
    not one token under, under every run cap and under the bound's own — where unset, an episode under
    the 8192 cap already stopped at 21808 left — and the trajectory's exhausted flag reads the same gate."""
    small = _contest(8192, episode_tokens=_CONTEST_EPISODE)
    assert _caps(small, 60000) == (16384, 8192), "21920 left: the turn's own caps stand"
    assert _caps(small, 70000) == (11920, 3728)
    large = _contest(16384, episode_tokens=_CONTEST_EPISODE)
    assert _caps(large, 60000) == (21920, 13728)
    alone = _contest(None, episode_tokens=_CONTEST_EPISODE)
    assert _caps(alone, 60000) == (21920, 13728)
    for effort in (small, _contest(12288, episode_tokens=_CONTEST_EPISODE), large, alone):
        assert _caps(effort, _CONTEST_EPISODE - _ROOM) == (_ROOM, 1)
        last = _CONTEST_EPISODE - _ROOM + 1
        assert _caps(effort, last) is None
        traj = Trajectory(messages=[])
        effort.stamp(traj, last)
        assert traj.info[OUTPUT_BUDGET_EXHAUSTED_KEY] is True
        effort.stamp(traj, last - 1)
        assert traj.info[OUTPUT_BUDGET_EXHAUSTED_KEY] is False
    assert _caps(_contest(8192, answer=None, episode_tokens=_CONTEST_EPISODE), 70000) is None


async def test_both_drivers_bound_a_turn_and_the_one_after_a_cut_by_the_answer_room(monkeypatch):
    """A low-level episode under the run's 8192-token cap and a 40000-token budget: the first turn asks for
    8192 + 8192 and is cut there; the turn after it asks for the same 8192 + 8192; the third has 13616 left
    and asks for all of it, its cap down to 5424; after it 7616 remain, under the room, so no fourth turn
    starts and the episode closes truncated. Both drivers request the same caps and record the run's cap on
    every turn, the level's 1000-token budget nowhere among them."""
    context = {"reasoning_effort": "low"}
    config = ray_actors.RolloutConfig(
        max_tokens=_CONTEST_TURN, max_thinking_tokens=8192, max_answer_tokens=_ROOM, max_episode_tokens=40000
    )
    sampled = [16384, 10000, 6000]
    expected = [(16384, 8192), (16384, 8192), (13616, 5424)]

    cut = SimpleNamespace(**{**vars(_actor_text_turn(tokens=sampled[0])), "text": "half", "finish_reason": "length"})
    gens = [cut, _actor_tool_turn(tokens=sampled[1]), _actor_tool_turn(tokens=sampled[2])]
    seen, result = await _drive_actor(_ProfiledEnv, context, config, generations=gens)
    assert result.error is None, result.error
    assert _actor_caps(seen) == expected
    traj = result.trajectory
    assert [m.truncated for m in traj.messages if m.role == "assistant"] == [True, False, False]
    assert [m.thinking_cap for m in traj.messages if m.role == "assistant"] == [8192] * 3
    assert len(seen) == 3 < _ProfiledEnv().max_turns and traj.truncated
    assert traj.info[OUTPUT_BUDGET_EXHAUSTED_KEY] is True
    assert result.generation_tokens == sum(sampled)

    eval_cut = _text_turn("half", tokens=sampled[0], finish_reason="length")
    responses = [eval_cut, _tool_turn(tokens=sampled[1]), _tool_turn(tokens=sampled[2])]
    calls, eval_traj = await _drive_eval(monkeypatch, _ProfiledEnv(), context, responses=responses, config=config)
    assert _eval_caps(calls) == expected
    assert [m.thinking_cap for m in eval_traj.messages if m.role == "assistant"] == [8192] * 3
    assert eval_traj.truncated and eval_traj.info[OUTPUT_BUDGET_EXHAUSTED_KEY] is True


async def test_both_drivers_send_the_answer_bounds_cap_again_on_the_turn_after_a_cut(monkeypatch):
    """No run cap: the answer bound caps the reasoning where 8192 of the 30000-token turn is left. Both
    drivers send that cap with the whole turn, again on the turn after a cut one, and both turns record it
    as the cap they ran under; the template hears no count."""
    context = {"reasoning_effort": "low"}
    config = ray_actors.RolloutConfig(max_tokens=_CONTEST_TURN, max_answer_tokens=_ROOM)
    expected = [(_CONTEST_TURN, _ROOM_CAP)] * 2

    cut = SimpleNamespace(**{**vars(_actor_text_turn(tokens=30000)), "text": "half", "finish_reason": "length"})
    seen, result = await _drive_actor(_ProfiledEnv, context, config, generations=[cut, _actor_text_turn()])
    assert result.error is None, result.error
    turns = [m for m in result.trajectory.messages if m.role == "assistant"]
    assert [m.truncated for m in turns] == [True, False]
    assert _actor_caps(seen) == expected
    assert [m.thinking_cap for m in turns] == [_ROOM_CAP] * 2
    assert all("chat_template_kwargs" not in payload for _, _, payload in seen)

    eval_cut = _text_turn("half", tokens=30000, finish_reason="length")
    calls, eval_traj = await _drive_eval(
        monkeypatch,
        _ProfiledEnv(),
        context,
        responses=[eval_cut, _text_turn()],
        config=replace(config, model_name="m"),
    )
    eval_turns = [m for m in eval_traj.messages if m.role == "assistant"]
    assert [m.truncated for m in eval_turns] == [True, False]
    assert _eval_caps(calls) == expected
    assert [m.thinking_cap for m in eval_turns] == [_ROOM_CAP] * 2
    assert all("chat_template_kwargs" not in call["extra_body"] for call in calls)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
