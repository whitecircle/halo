#!/usr/bin/env python
"""CPU tests: both rollout drivers bind an episode's reasoning effort through ONE seam.

``bind_episode_effort`` (``src/environments/episode.py``) resolves the episode's level once and turns the
env's per-level CoT budget into the per-turn reasoning cap every turn runs under. The Ray actor and the
offline eval runner must both go through it: a driver that resolves the level itself, or skips the
budget, generates under a different contract than the one the policy was trained on. The episode
output budget (``rollout_max_episode_tokens``) rides the same seam: ``EpisodeEffort.turn_caps`` narrows
a turn's request to what the budget has left and hands back none once it holds no turn, which ends
the episode truncated.

Both drivers are exercised for real — the actor via the class ``@ray.remote`` wraps (no cluster), the
eval runner with its generation call stubbed — so the tests fail if either one stops binding.

    python tests/cpu/environments/test_effort_binding.py
"""

from types import SimpleNamespace

import pytest

import src.environments.eval_runner as eval_runner
import src.environments.ray_actors as ray_actors
from src.environments import episode
from src.environments.base import OUTPUT_BUDGET_EXHAUSTED_KEY, REWARD_COMPONENTS_KEY, VALID_REASONING_EFFORTS
from src.environments.engine_wire import generation_control_fields
from src.environments.envs.protocols.native import NativeToolUseEnvironment
from src.environments.tools.definitions import NativeToolRegistry

_MAX_TOKENS = 20000
# The reasoning-end marker's id the per-turn reasoning count reads up to: one no other id in a fake turn carries.
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


def _text_turn(text="the answer", token_ids=None, tokens=3):
    return SimpleNamespace(
        answer=text,
        finish_reason="stop",
        completion_tokens=tokens,
        tool_calls=None,
        reasoning=None,
        token_ids=token_ids,
    )


def _tool_turn(token_ids=None, tokens=3):
    """A turn that calls a tool no registry has: the step is spent, the episode stays open."""
    call = SimpleNamespace(id="c0", function=SimpleNamespace(name="nonexistent_tool", arguments="{}"))
    return SimpleNamespace(
        answer="",
        finish_reason="tool_calls",
        completion_tokens=tokens,
        tool_calls=[call],
        reasoning=None,
        token_ids=token_ids,
    )


def _actor_text_turn(token_ids=None, tokens=3):
    return ray_actors.TurnGeneration(
        text="the answer", tool_calls=[], reasoning="", tokens=tokens, finish_reason="stop", token_ids=token_ids
    )


def _actor_tool_turn(token_ids=None, tokens=3):
    """The actor-side twin of ``_tool_turn``: a call to a tool no registry has keeps the episode open."""
    call = {"id": "c0", "type": "function", "function": {"name": "nonexistent_tool", "arguments": "{}"}}
    return ray_actors.TurnGeneration(
        text="", tool_calls=[call], reasoning="", tokens=tokens, finish_reason="tool_calls", token_ids=token_ids
    )


def _ids_closing_reasoning_after(reasoning_tokens: int) -> list[int]:
    """Sampled ids whose reasoning closes with the marker as its ``reasoning_tokens``-th token."""
    return [7] * (reasoning_tokens - 1) + [_END] + [9] * 5


async def _drive_eval(monkeypatch, env, context, *, responses, config=None):
    """Run the eval driver over ``responses``; return its captured request kwargs and trajectory."""
    calls = []

    async def fake_generate(model, messages, **kwargs):
        calls.append({"model": model, "messages": messages, **kwargs})
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


async def _drive_actor(env_cls, context, config, generations=None, env_kwargs=None):
    """Run the real ``EnvironmentActor`` episode loop off-cluster over ``generations`` (one text turn
    when omitted), the env built with ``env_kwargs``; return its per-turn ``(config, level, budget,
    payload)`` records and the result.

    ``@ray.remote`` wraps the class and keeps the original at ``__ray_metadata__.modified_class``;
    ``__init__`` only assigns attributes, so this exercises the genuine ``run_episode`` binding. The
    transport is replaced below the payload builder, so each record's payload is the request the
    actor would have sent for that turn.
    """
    cls = ray_actors.EnvironmentActor.__ray_metadata__.modified_class
    actor = cls.__new__(cls)
    actor.__init__(actor_id=0, env_type=(env_cls, env_kwargs or {}), env_config={})
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


def _actor_caps(seen):
    return [(payload["max_tokens"], payload.get("thinking_token_budget")) for _, _, _, payload in seen]


def _eval_caps(calls):
    return [(call["max_tokens"], call["extra_body"].get("thinking_token_budget")) for call in calls]


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
    assert calls[0]["extra_body"] == generation_control_fields(actor_cfg, actor_level, actor_budget)
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


async def test_every_turn_of_an_episode_runs_under_the_levels_whole_cap(monkeypatch):
    """A thinking budget is per turn: three turns each reasoning 1500 tokens of the level's 4000 all
    request the same 4000-token cap and the same turn total, state the same budget to the template,
    and the control fields ask for no sampled ids (the eval transport captures nothing) — on both
    drivers identically."""
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

    responses = [_tool_turn(ids), _tool_turn(ids), _text_turn(token_ids=ids)]
    calls, eval_traj = await _drive_eval(monkeypatch, _BudgetEnv(), context, responses=responses, config=config)
    assert _eval_caps(calls) == expected
    assert [call["extra_body"] for call in calls] == [
        generation_control_fields(cfg, level, budget) for cfg, level, budget, _ in seen
    ]
    assert all("return_token_ids" not in call["extra_body"] for call in calls)
    assert (eval_traj.reasoning_effort, eval_traj.reasoning_budget) == ("high", 4000)


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


async def test_the_actor_records_each_turns_cap_and_reasoning_and_the_eval_its_cap_alone(monkeypatch):
    """The overlong charge reads each training turn's pair: the cap the turn's level set and the
    reasoning it sampled through the close. The eval prices no charge and asks for no ids, so its
    record carries the cap alone."""
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
    """The level's ``thinking_tokens`` caps every turn, never above ``rollout_max_thinking_tokens``; a
    level without one runs under the run's cap, or uncapped; and a cap that fills the turn is refused,
    since such a turn has no answer room and is cut mid-reasoning every time. The turn total is the
    run's ``max_tokens`` whatever the level, and the output budget rides on every binding."""
    context = {"reasoning_effort": "high"}
    clamped = episode.bind_episode_effort(context, _BudgetEnv(), max_tokens=_MAX_TOKENS, max_thinking_tokens=3000)
    assert (clamped.thinking_budget, clamped.max_tokens) == (3000, _MAX_TOKENS)
    level = episode.bind_episode_effort(context, _BudgetEnv(), max_tokens=_MAX_TOKENS, max_thinking_tokens=8192)
    assert (level.thinking_budget, level.max_tokens) == (4000, _MAX_TOKENS)
    run = episode.bind_episode_effort(context, _PlainEnv(), max_tokens=_MAX_TOKENS, max_thinking_tokens=8192)
    assert (run.thinking_budget, run.max_tokens) == (8192, _MAX_TOKENS)
    assert episode.bind_episode_effort(context, _PlainEnv(), max_tokens=_MAX_TOKENS).thinking_budget is None
    with pytest.raises(ValueError, match=r"'high' level's thinking_tokens \(4000\) must sit below rollout_max_tokens"):
        episode.bind_episode_effort(context, _BudgetEnv(), max_tokens=4000)
    with pytest.raises(ValueError, match="must sit below rollout_max_tokens"):
        episode.bind_episode_effort(context, _PlainEnv(), max_tokens=4000, max_thinking_tokens=4000)
    for env in (_BudgetEnv(), _PlainEnv()):
        bound = episode.bind_episode_effort(context, env, max_tokens=_MAX_TOKENS, max_episode_tokens=50000)
        assert bound.episode_tokens == 50000
        assert episode.bind_episode_effort(context, env, max_tokens=_MAX_TOKENS).episode_tokens is None


async def test_the_eval_refuses_a_level_cap_that_fills_the_turn_before_generating(monkeypatch):
    """A drawable level whose cap fills the turn would fail every episode at its first turn, each recorded
    as a zero-reward error sample: the eval gates on it before its first request, as the trainer does at
    construction, and a run whose levels fit passes the same gate."""
    calls = []

    async def fake_generate(model, messages, **kwargs):
        calls.append(kwargs)
        return _text_turn()

    monkeypatch.setattr(eval_runner, "generate_openai_response", fake_generate)
    env = _BudgetEnv(reasoning_effort="random")
    examples = [{"prompt": "solve it", "context": {}}]
    too_small = ray_actors.RolloutConfig(model_name="m", max_tokens=4000)
    with pytest.raises(ValueError, match=r"'high' level's thinking_tokens \(4000\) must sit below rollout_max_tokens"):
        await eval_runner.collect_results(env, examples, None, rollout=too_small)
    assert calls == [], "the gate must refuse before the first request"
    caps = episode.thinking_caps_by_level
    assert caps(env, max_tokens=_MAX_TOKENS, max_thinking_tokens=None) == _BudgetEnv.BUDGETS
    assert caps(_BudgetEnv(reasoning_effort="low"), max_tokens=_MAX_TOKENS, max_thinking_tokens=None) == {"low": 1000}
    assert caps(_PlainEnv(), max_tokens=_MAX_TOKENS, max_thinking_tokens=None) == {None: None}
    results = await eval_runner.collect_results(
        env, examples, None, rollout=ray_actors.RolloutConfig(model_name="m", max_tokens=_MAX_TOKENS)
    )
    assert len(calls) == 1 and "error" not in results[0]["samples"][0]


def test_drivers_share_one_seam():
    # A local copy of the seam in either driver lets the two drift apart.
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


# --- The episode output budget ---

# What the native protocol charges an episode closed as a max_turns overflow; the output budget's
# truncation must be priced the same.
_OVERFLOW_PRICE = 0.3


def _effort(**overrides) -> episode.EpisodeEffort:
    """A 20000-token turn with a 4000-token reasoning cap (16000 of answer room) under a 50000-token
    episode budget."""
    bound = {"level": None, "thinking_budget": 4000, "max_tokens": _MAX_TOKENS, "episode_tokens": 50000}
    return episode.EpisodeEffort(**{**bound, **overrides})


def _caps(effort, generated):
    """``turn_caps`` as the pair of request fields it sets — exactly the two, since the drivers splat it
    over the turn's ``RolloutConfig`` — or ``None`` once the budget holds no turn."""
    caps = effort.turn_caps(generated)
    if caps is None:
        return None
    assert set(caps) == {"max_tokens", "max_thinking_tokens"}
    return caps["max_tokens"], caps["max_thinking_tokens"]


def test_turn_caps_narrow_to_what_the_budget_has_left_and_keep_the_answer_room():
    """The total narrows to what is left, and the reasoning cap gives up the difference so the turn
    keeps its 16000 of answer room whole; a turn with exactly the room left keeps one reasoning token,
    never a cap of 0, which would close the reasoning before it opened; one token less holds no turn."""
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
    # No budget: nothing narrows, however much was sampled.
    unbounded = _effort(episode_tokens=None)
    assert _caps(unbounded, 10**6) == (_MAX_TOKENS, 4000)


async def test_an_output_budget_ends_the_episode_truncated_when_less_than_a_turn_is_left(monkeypatch):
    """No reasoning cap: 50000 tokens for the episode over 20000-token turns. After three turns 1000
    remain, under a turn's room, so the episode stops with a turn of max_turns still unused — closed
    truncated, not completed, and priced like a max_turns overflow. Both drivers identically, and each
    reports the spend as its token count."""
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
    assert result.success is False
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
    """After a 19000- and an 11000-token turn exactly 20000 of the 50000 remain, one 20000-token turn: the
    third turn is the last the budget affords. Cut there, it cannot be retried, so the episode closes as
    a max_turns overflow would — one overflow price, no cut price, and no nudge dangling after the
    fragment — on both drivers alike."""
    config = ray_actors.RolloutConfig(max_tokens=_MAX_TOKENS, max_episode_tokens=50000)
    prices = {"turn_overflow_penalty": 0.3, "length_cutoff_penalty": 0.1}
    cut = SimpleNamespace(**{**vars(_actor_text_turn(tokens=11000)), "text": "half an ans", "finish_reason": "length"})

    gens = [_actor_tool_turn(tokens=19000), _actor_tool_turn(tokens=11000), cut]
    _, result = await _drive_actor(_PlainEnv, None, config, generations=gens, env_kwargs=prices)
    traj = result.trajectory
    assert result.error is None and traj.done and traj.truncated and result.episode_length == 3
    assert traj.info["length_cutoff_turns"] == 1 and traj.messages[-1].role == "assistant"
    assert traj.info[REWARD_COMPONENTS_KEY]["reward/tool_shaping"] == pytest.approx(-0.3)

    eval_cut = SimpleNamespace(
        **{**vars(_text_turn(tokens=11000)), "answer": "half an ans", "finish_reason": "length"}
    )
    responses = [_tool_turn(tokens=19000), _tool_turn(tokens=11000), eval_cut]
    _, eval_traj = await _drive_eval(monkeypatch, _PlainEnv(**prices), None, responses=responses, config=config)
    assert eval_traj.done and eval_traj.truncated and eval_traj.messages[-1].role == "assistant"
    assert eval_traj.info[REWARD_COMPONENTS_KEY]["reward/tool_shaping"] == pytest.approx(-0.3)


async def test_the_eval_refuses_an_output_budget_under_one_turn_before_generating(monkeypatch):
    """An explicit ``--max_tokens`` above the training contract's episode budget would let no turn start;
    the shared binding seam refuses it before the first request, as the trainer's config does at parse."""
    calls = []

    async def fake_generate(model, messages, **kwargs):
        calls.append(kwargs)
        return _text_turn()

    monkeypatch.setattr(eval_runner, "generate_openai_response", fake_generate)
    too_big = ray_actors.RolloutConfig(model_name="m", max_tokens=60000, max_episode_tokens=50000)
    with pytest.raises(
        ValueError, match=r"rollout_max_episode_tokens \(50000\) must be at least rollout_max_tokens \(60000\)"
    ):
        await eval_runner.collect_results(_PlainEnv(), [{"prompt": "solve it", "context": {}}], None, rollout=too_big)
    assert calls == []


async def test_an_output_budget_shrinks_the_reasoning_cap_with_the_total_and_records_the_levels_cap(monkeypatch):
    """The run's 4000-token reasoning cap under 20000-token turns (16000 of answer room), 38000 for the
    episode: the second turn has 19000 left, so its request drops to 19000 with a 3000 reasoning cap,
    the room kept whole; after it 1000 remain, under the room, and no third turn starts. The narrowed
    caps reach both drivers' requests, the eval's control fields are the actor's own turn by turn, and
    each turn records the level's 4000 — the cap the overlong charge ramps to — not the narrowed one."""
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
    """A turn that consumed its narrowed total is a cut even when the engine labelled it complete: the
    count is compared with the request's own cap, not the run's turn cap it no longer runs under."""
    config = ray_actors.RolloutConfig(max_tokens=_MAX_TOKENS, max_thinking_tokens=4000, max_episode_tokens=50000)
    full = _tool_turn(tokens=18000)  # the whole 18000 the third request asked for, finish_reason "stop"
    responses = [_tool_turn(tokens=19000), _tool_turn(tokens=13000), full]
    calls, traj = await _drive_eval(monkeypatch, _PlainEnv(), None, responses=responses, config=config)
    assert _eval_caps(calls) == [(_MAX_TOKENS, 4000), (_MAX_TOKENS, 4000), (18000, 2000)]
    assistant = [m for m in traj.messages if m.role == "assistant"]
    assert [m.truncated for m in assistant] == [False, False, True]


async def test_an_episode_inside_its_budget_records_false_and_one_without_a_budget_records_nothing(monkeypatch):
    """Anti-vacuity for the flag and the metric: both exist exactly when a budget was set."""
    budget = ray_actors.RolloutConfig(max_tokens=_MAX_TOKENS, max_episode_tokens=50000)
    _, result = await _drive_actor(_PlainEnv, None, budget, generations=[_actor_text_turn(tokens=19000)])
    assert result.error is None, result.error
    assert result.success and not result.trajectory.truncated
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
