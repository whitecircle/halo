#!/usr/bin/env python
"""CPU tests: the environment side of reward terms — grades priced into components that sum to the
reward, external terms settled by the dispatcher after the closing step, the sample a scorer reads,
scorer failures that invalidate the episode (or price it neutral), the veto gate on the objective,
and the guards around a stale or unsettled reward.

Run: python tests/cpu/environments/test_reward_terms_settlement.py  (or pytest)
"""

import asyncio

import pytest

from src.configs.environment_config import EnvironmentConfig
from src.environments import base as base_module
from src.environments.base import (
    EPISODE_ERROR_KEY,
    EPISODE_INVALID_KEY,
    EPISODE_INVALID_REASON_KEY,
    REWARD_COMPONENTS_KEY,
    REWARD_DETAILS_KEY,
    REWARD_ERRORS_KEY,
    REWARD_PENDING_KEY,
    AsyncBaseEnvironment,
    BaseEnvironment,
    EpisodeGrade,
    Message,
    Trajectory,
)
from src.environments.envs.protocols.native import NativeToolUseEnvironment
from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment
from src.environments.episode import EpisodeDispatcher
from src.environments.registry import resolve_environment
from src.environments.tools.definitions import NativeToolRegistry
from src.rewards.samples import render_transcript
from src.rewards.scorers import catalog as scorers_module
from src.rewards.scorers.base import Scorer, ScoreResult
from src.rewards.terms import OBJECTIVE_REWARD_KEY, JudgeTerm, OnError

JUDGE = {
    "source": "judge",
    "name": "quality",
    "weight": 0.5,
    "requirements": [{"name": "clear", "description": "Clear."}],
}
CHECKS = {
    "source": "judge",
    "name": "conduct",
    "checks": [
        {"name": "cheated", "description": "Hard-coded the expected output.", "veto": True},
        {"name": "sloppy", "description": "Left debug prints in."},
    ],
}


class _FakeJudge(Scorer):
    """A scoring judge scores 1.0 unless the final text says ``fail`` (a ``None`` verdict) or ``half``
    (0.5); a veto judge fires ``cheated`` wherever the transcript, reasoning included, says ``cheat``."""

    instances: list["_FakeJudge"] = []

    def __init__(self, term):
        super().__init__(term)
        self.samples = []
        self.verified = 0
        _FakeJudge.instances.append(self)

    async def score_one(self, sample):
        self.samples.append(sample)
        text = sample.completion[-1]["content"] if sample.completion else ""
        if "fail" in text:
            return ScoreResult(None, error="judge down")
        if self.term.is_veto:
            fired = "cheat" in render_transcript(sample.completion, include_reasoning=True)
            metrics = {f"judge/{self.term.name}/{leaf}": float(fired) for leaf in ("cheated", "veto")}
            return ScoreResult(0.0, metrics, detail="fired cheated" if fired else None, veto=fired)
        score = 0.5 if "half" in text else 1.0
        return ScoreResult(score, {"judge/quality/clear": score}, detail=f"graded {text!r}")

    async def verify(self):
        self.verified += 1

    async def aclose(self):
        pass


@pytest.fixture
def fake_judge(monkeypatch):
    monkeypatch.setitem(scorers_module.SCORERS, JudgeTerm, _FakeJudge)
    _FakeJudge.instances.clear()
    return _FakeJudge


def _native(judge=JUDGE, **kwargs):
    return resolve_environment("native_math", {"reward_terms": [{"source": "environment"}, judge], **kwargs})


async def _run(env, answer_text, answer="4"):
    dispatcher = EpisodeDispatcher(env)
    ids, _ = await dispatcher.reset(["What is 2+2?"], [{"answer": answer}])
    steps = await dispatcher.step(ids, [answer_text], [{"answer": answer, "finish_reason": "stop"}])
    return ids[0], steps[0].trajectory


# --- EpisodeGrade ---


@pytest.mark.parametrize("objective", [-0.01, 1.01, float("nan"), True, "1"])
def test_grade_refuses_an_objective_outside_the_unit_interval(objective):
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        EpisodeGrade(objective)


@pytest.mark.parametrize("shaping", [{"objective": 0.1}, {"turn_shaping": 0.1}, {"a/b": 0.1}, {"ok": float("inf")}])
def test_grade_refuses_reserved_or_malformed_shaping(shaping):
    with pytest.raises(ValueError):
        EpisodeGrade(1.0, shaping)


# --- pricing without external terms ---


def test_components_sum_to_the_reward_and_price_the_objective():
    env = resolve_environment(
        "native_math",
        {"reward_terms": [{"source": "environment", "weight": 2.0, "exponent": 2.0}], "no_tool_use_penalty": 0.1},
    )
    ids, _ = env.reset(["What is 2+2?"], [{"answer": "4"}])
    traj = env.step(ids, ["The answer is 4"], [{"answer": "4", "finish_reason": "stop"}])[0].trajectory
    components = traj.info[REWARD_COMPONENTS_KEY]
    assert components == {"reward/turn_shaping": 0.0, "reward/tool_shaping": -0.1, OBJECTIVE_REWARD_KEY: 2.0}
    assert traj.total_reward == pytest.approx(sum(components.values()))
    assert REWARD_PENDING_KEY not in traj.info and "answer" not in traj.info["context"]
    assert env.rollout_metrics(traj)[OBJECTIVE_REWARD_KEY] == 2.0


@pytest.mark.parametrize(("passed", "priced"), [(4, 2.0), (2, 0.0)])
def test_code_contests_grade_is_priced_by_the_term_not_the_environment(passed, priced):
    """The environment hands the term its all-or-nothing grade; the term's weight prices it, and a convex
    exponent cannot turn a half pass into credit (a pass fraction priced at exponent 2 would pay 0.5)."""
    env = CodeContestsEnvironment(
        language="python", reward_terms=[{"source": "environment", "weight": 2.0, "exponent": 2.0}]
    )
    traj = env._reset_single("Print a+b.", {"answer": {"tests": [{"input": "1\n2\n", "output": "3"}]}})
    traj.info.update(
        submission_result="graded",
        tests_total=4,
        tests_passed=passed,
        tests_graded=4,
        tests_ran_ok=4,
        tests_infra_errors=0,
    )
    env._settle_grade(traj, None)
    assert traj.info[REWARD_COMPONENTS_KEY][OBJECTIVE_REWARD_KEY] == pytest.approx(priced)
    with pytest.raises(TypeError, match="pass_fraction_exponent"):
        CodeContestsEnvironment(language="python", pass_fraction_exponent=2.0)


def test_a_stale_compute_reward_override_is_refused():
    class Stale(BaseEnvironment):
        def _reset_single(self, prompt, context=None):
            return Trajectory()

        def _step_single(self, trajectory, action, context=None):
            return trajectory, 0.0, True, False, {}

        def _grade_episode(self, trajectory, context=None):
            return EpisodeGrade(1.0)

        def _compute_reward(self, trajectory, context=None):
            return 1.0

    with pytest.raises(TypeError, match="_compute_reward"):
        Stale()


# --- external terms ---


def test_external_terms_are_pending_until_settled_and_the_dispatcher_settles_them(fake_judge):
    env = _native()
    ids, _ = env.reset(["What is 2+2?"], [{"answer": "4"}])
    traj = env.step(ids, ["It is 4"], [{"answer": "4", "finish_reason": "stop"}])[0].trajectory
    assert traj.info[REWARD_PENDING_KEY] is True
    assert traj.total_reward == pytest.approx(1.0)  # the environment's side only
    assert "answer" in traj.info["context"]  # the grading payload waits for the scorers
    with pytest.raises(RuntimeError, match="settle_async"):
        env.rollout_metrics(traj)

    env.settle(ids)
    assert REWARD_PENDING_KEY not in traj.info
    assert traj.info[REWARD_COMPONENTS_KEY]["reward/quality"] == 0.5
    assert traj.total_reward == pytest.approx(1.5)
    assert "answer" not in traj.info["context"]
    assert traj.info[REWARD_DETAILS_KEY] == {"quality": "graded 'It is 4'"}
    metrics = env.rollout_metrics(traj)
    assert metrics["reward/quality"] == 0.5 and metrics["judge/quality/clear"] == 1.0
    assert metrics["judge/quality/scored"] == 1.0 and metrics["episode/reward_scored"] == 1.0
    # The trainer's residue check: the reward/* keys of the metrics sum exactly to the reward.
    assert sum(v for k, v in metrics.items() if k.startswith("reward/")) == traj.total_reward
    (judge,) = fake_judge.instances
    (sample,) = judge.samples
    assert sample.reference == "4" and sample.final_answer == "It is 4"
    assert sample.completion[-1]["content"] == "It is 4"
    assert sample.prompt[-1]["role"] == "user" and all(m["role"] != "assistant" for m in sample.prompt)


def test_dispatcher_settles_the_closing_step(fake_judge):
    env = _native()
    eid, traj = asyncio.run(_run(env, "half of it"))
    assert REWARD_PENDING_KEY not in traj.info
    assert traj.info[REWARD_COMPONENTS_KEY]["reward/quality"] == 0.25
    assert traj.total_reward == pytest.approx(0.25)  # a wrong answer: objective 0, judge 0.5 x 0.5


def test_a_failed_scorer_invalidates_the_episode_and_contributes_nothing(fake_judge):
    env = _native()
    _, traj = asyncio.run(_run(env, "fail"))
    assert traj.info[EPISODE_INVALID_KEY] is True
    assert traj.info[REWARD_ERRORS_KEY] == {"quality": "judge down"}
    assert traj.info[EPISODE_INVALID_REASON_KEY] == "reward term 'quality' scored nothing: judge down"
    assert traj.info[REWARD_COMPONENTS_KEY]["reward/quality"] == 0.0
    metrics = env.rollout_metrics(traj)
    assert metrics["judge/quality/scored"] == 0.0 and metrics["episode/reward_scored"] == 0.0


def test_a_neutral_term_prices_nothing_on_failure_and_keeps_the_episode_valid(fake_judge):
    """``on_error: neutral``: the term contributes 0 and the episode trains on its other terms, the
    failure still on record and in the metrics."""
    env = _native(judge={**JUDGE, "on_error": "neutral"})
    _, traj = asyncio.run(_run(env, "fail"))
    assert EPISODE_INVALID_KEY not in traj.info and EPISODE_INVALID_REASON_KEY not in traj.info
    assert traj.info[REWARD_ERRORS_KEY] == {"quality": "judge down"}
    assert traj.info[REWARD_COMPONENTS_KEY]["reward/quality"] == 0.0
    metrics = env.rollout_metrics(traj)
    assert metrics["judge/quality/scored"] == 0.0 and metrics["episode/reward_scored"] == 0.0


def test_a_fired_veto_zeroes_the_objective_and_keeps_the_episode_valid(fake_judge):
    """A veto judge gates the environment's grade: a solve it fires on pays 0 for the objective, stays
    a valid episode, and reports the flag; a solve it does not fire on keeps its grade."""
    env = _native(judge=CHECKS)
    ids, _ = env.reset(["What is 2+2?"], [{"answer": "4"}])
    cheating = {"answer": "4", "finish_reason": "stop", "reasoning": "I will cheat and just print 4"}
    traj = env.step(ids, ["It is 4"], [cheating])[0].trajectory
    assert traj.info[REWARD_COMPONENTS_KEY][OBJECTIVE_REWARD_KEY] == 1.0  # solved, before the verdict
    env.settle(ids)
    components = traj.info[REWARD_COMPONENTS_KEY]
    assert components[OBJECTIVE_REWARD_KEY] == 0.0 and components["reward/conduct"] == 0.0
    assert traj.total_reward == pytest.approx(0.0)
    assert EPISODE_INVALID_KEY not in traj.info
    assert traj.info[REWARD_DETAILS_KEY] == {"conduct": "fired cheated"}
    metrics = env.rollout_metrics(traj)
    assert metrics["judge/conduct/veto"] == 1.0 and metrics["judge/conduct/cheated"] == 1.0
    assert metrics["judge/conduct/scored"] == 1.0 and metrics["episode/reward_scored"] == 1.0
    assert sum(v for k, v in metrics.items() if k.startswith("reward/")) == traj.total_reward

    ids, _ = env.reset(["What is 2+2?"], [{"answer": "4"}])
    clean = env.step(ids, ["It is 4"], [{"answer": "4", "finish_reason": "stop"}])[0].trajectory
    env.settle(ids)
    assert clean.info[REWARD_COMPONENTS_KEY][OBJECTIVE_REWARD_KEY] == 1.0
    assert env.rollout_metrics(clean)["judge/conduct/veto"] == 0.0


def test_a_veto_judge_needs_the_environment_term_it_gates():
    with pytest.raises(ValueError, match=r"veto judge term\(s\) \['conduct'\]"):
        resolve_environment("native_math", {"reward_terms": [CHECKS]})


def test_an_episode_its_driver_lost_is_not_sent_to_a_scorer(fake_judge):
    env = _native(max_turns=3)
    ids, _ = env.reset(["What is 2+2?"], [{"answer": "4"}])
    env.get_trajectories(ids)[0].info[EPISODE_ERROR_KEY] = "generation failed"
    traj = env.finalize_truncated(ids)[0].trajectory
    assert REWARD_PENDING_KEY not in traj.info and traj.info[REWARD_COMPONENTS_KEY]["reward/quality"] == 0.0
    assert "answer" not in traj.info["context"]
    assert fake_judge.instances == []


def test_an_async_environment_settles_through_the_dispatcher(fake_judge):
    class Echo(AsyncBaseEnvironment):
        def _reset_single(self, prompt, context=None):
            return self._init_trajectory(prompt, context)

        async def _step_single_async(self, trajectory, action, context=None):
            trajectory.info["completed"] = True
            return trajectory, 0.35, True, False, {}

        def _step_single(self, trajectory, action, context=None):
            raise AssertionError("the async path must be taken")

        def _grade_episode(self, trajectory, context=None):
            return EpisodeGrade(1.0)

    env = Echo(reward_terms=[{"source": "environment", "weight": 2.0}, JUDGE])
    _, traj = asyncio.run(_run(env, "half"))
    assert traj.info[REWARD_COMPONENTS_KEY] == {
        "reward/turn_shaping": 0.35,
        OBJECTIVE_REWARD_KEY: 2.0,
        "reward/quality": 0.25,
    }
    assert traj.total_reward == pytest.approx(2.6)


def test_settle_can_be_called_again_on_a_fresh_loop(fake_judge):
    env = _native()
    for text in ("first", "second"):
        ids, _ = env.reset(["What is 2+2?"], [{"answer": "4"}])
        env.step(ids, [text], [{"answer": "4", "finish_reason": "stop"}])
        env.settle(ids)
        assert REWARD_PENDING_KEY not in env.get_trajectories(ids)[0].info
    assert len(fake_judge.instances) == 2  # the first call's scorer was released with its loop


def test_truncated_episodes_are_settled_too(fake_judge):
    env = _native(max_turns=1)
    dispatcher = EpisodeDispatcher(env)
    ids, _ = asyncio.run(dispatcher.reset(["What is 2+2?"], [{"answer": "4"}]))
    steps = asyncio.run(dispatcher.finalize_truncated(ids))
    traj = steps[0].trajectory
    assert traj.truncated and REWARD_PENDING_KEY not in traj.info
    assert "reward/quality" in traj.info[REWARD_COMPONENTS_KEY]


def test_settle_is_a_no_op_without_pending_episodes(fake_judge):
    env = resolve_environment("native_math", {})
    ids, _ = env.reset(["What is 2+2?"], [{"answer": "4"}])
    env.step(ids, ["4"], [{"answer": "4", "finish_reason": "stop"}])
    env.settle(ids)  # no loop, no scorer, no error
    assert fake_judge.instances == []
    with pytest.raises(ValueError, match="not found"):
        env.settle([999])


def test_verify_backend_probes_every_external_term(fake_judge):
    env = _native()
    env.verify_backend()
    assert [judge.verified for judge in fake_judge.instances] == [1]
    assert resolve_environment("native_math", {}).verify_backend() is None


# --- the sample a scorer reads ---


def test_the_sample_carries_the_turns_with_their_reasoning_and_flags():
    """What a judge reads: the prompt turns, then every policy turn with its reasoning, a cut turn
    flagged as such, the tools the policy could call, the protocol's final answer and the row's
    reference."""
    env = _native(max_turns=3)
    ids, _ = env.reset(["What is 2+2?"], [{"answer": "4"}])
    env.step(ids, ["a thought that ran"], [{"answer": "4", "finish_reason": "length", "reasoning": "cut thought"}])
    env.step(ids, ["It is 4"], [{"answer": "4", "finish_reason": "stop", "reasoning": "two and two"}])
    sample = env._scoring_sample(env.get_trajectories(ids)[0])

    assert all(m["role"] != "assistant" for m in sample.prompt) and sample.prompt[-1]["content"] == "What is 2+2?"
    cut, nudge, final = sample.completion
    assert cut["content"] == "a thought that ran" and cut["reasoning_content"] == "cut thought"
    assert cut["truncated"] is True
    assert nudge == {"role": "user", "content": NativeToolUseEnvironment.LENGTH_CUTOFF_NUDGE}
    assert final["reasoning_content"] == "two and two" and "truncated" not in final
    assert sample.final_answer == "It is 4" and sample.reference == "4"
    assert sample.tools == env.get_tools_schema() and sample.tools


def test_a_tool_less_environment_hands_the_scorer_no_tools():
    env = NativeToolUseEnvironment(tool_registry=NativeToolRegistry(), reward_terms=[{"source": "environment"}, JUDGE])
    ids, _ = env.reset(["q"], [{"answer": "4"}])
    traj = env.step(ids, ["4"], [{"answer": "4", "finish_reason": "stop"}])[0].trajectory
    assert env._scoring_sample(traj).tools is None


def test_the_final_answer_is_the_protocols_own_and_none_for_an_unfinished_episode():
    """ReAct hands its ``Final Answer:``; an episode its turn budget ended has none, so a scorer reads
    a note rather than the fragment that happens to be the last assistant text."""
    react = resolve_environment("react_math", {"reward_terms": [{"source": "environment"}, JUDGE]})
    ids, _ = react.reset(["What is 2+2?"], [{"answer": "4"}])
    react.step(ids, ["Thought: easy\nFinal Answer: 4"], [{"answer": "4", "finish_reason": "stop"}])
    assert react._scoring_sample(react.get_trajectories(ids)[0]).final_answer == "4"

    capped = _native(max_turns=1)
    ids, _ = capped.reset(["What is 2+2?"], [{"answer": "4"}])
    fragment = "a fragment that reads like 4"
    traj = capped.step(ids, [fragment], [{"answer": "4", "finish_reason": "length"}])[0].trajectory
    assert traj.truncated and not traj.info["completed"]
    sample = capped._scoring_sample(traj)
    assert sample.final_answer is None and sample.completion[-1]["content"] == fragment


def test_code_contests_hands_the_scorer_the_submitted_program():
    """The answer is the fenced program, the hidden tests never travel as the reference, and the
    completion keeps the real turns rather than a synthesized answer turn."""
    env = CodeContestsEnvironment(language="python")
    traj = env._reset_single("Print a+b.", {"answer": {"tests": [{"input": "1\n2\n", "output": "3"}]}})
    unsubmitted = env._scoring_sample(traj)
    assert unsubmitted.final_answer is None and unsubmitted.reference is None

    call = {"id": "c0", "function": {"name": "submit_solution", "arguments": '{"code": "print(3)"}'}}
    traj.add_message(Message.assistant("submitting", tool_calls=[call]))
    traj.add_message(Message.tool("Passed 1/1 test cases.", tool_call_id="c0", name="submit_solution"))
    traj.info.update(_submitted_code="print(3)", submission_language="python")
    sample = env._scoring_sample(traj)
    assert sample.final_answer == "```python\nprint(3)\n```"
    assert sample.reference is None
    assert [(m["role"], m.get("tool_calls")) for m in sample.completion] == [("assistant", [call]), ("tool", None)]


# --- config ---


def test_environment_config_parses_terms_and_forwards_them():
    config = EnvironmentConfig(
        environment_type="native_math", rewards=[{"source": "environment", "exponent": 2.0}, JUDGE]
    )
    assert [type(term).__name__ for term in config.reward_terms] == ["EnvironmentTerm", "JudgeTerm"]
    assert config.to_env_config() == {"reward_terms": [{"source": "environment", "exponent": 2.0}, JUDGE]}
    with pytest.raises(ValueError, match="rewards\\[0\\]"):
        EnvironmentConfig(environment_type="native_math", rewards=[{"source": "accuracy"}])
    with pytest.raises(TypeError, match="success_reward"):
        EnvironmentConfig(environment_type="native_math", success_reward=1.0)


def test_environment_config_round_trips_a_veto_judge():
    """A ``checks`` term parses as a veto judge with the defaults a config gives one — priced at 0,
    neutral on error — and reaches the environment as the same term."""
    config = EnvironmentConfig(environment_type="native_math", rewards=[{"source": "environment"}, CHECKS])
    term = config.reward_terms[1]
    assert term.is_veto and term.weight == 0.0 and term.on_error is OnError.NEUTRAL
    env = resolve_environment(config.environment_type, config.to_env_config())
    assert env.reward_terms[1] == term


def test_a_term_cannot_take_a_declared_shaping_name():
    clash = {**JUDGE, "name": "tool_shaping"}
    with pytest.raises(ValueError, match="shaping components of NativeToolUseEnvironment"):
        resolve_environment("native_math", {"reward_terms": [{"source": "environment"}, clash]})
    with pytest.raises(ValueError, match="shaping components of CodeContestsEnvironment"):
        CodeContestsEnvironment(language="python", reward_terms=[{**JUDGE, "name": "submission"}])
    with pytest.raises(ValueError, match="shaping components"):
        resolve_environment("native_math", {"reward_terms": [{**JUDGE, "name": "turn_shaping"}]})


def test_an_undeclared_shaping_component_is_refused():
    class Undeclared(base_module.BaseEnvironment):
        def _reset_single(self, prompt, context=None):
            return self._init_trajectory(prompt, context)

        def _step_single(self, trajectory, action, context=None):
            return trajectory, 0.0, True, False, {}

        def _grade_episode(self, trajectory, context=None):
            return EpisodeGrade(1.0, {"bonus": 0.1})

    env = Undeclared()
    ids, _ = env.reset(["q"], [{}])
    with pytest.raises(ValueError, match="without declaring it in SHAPING_COMPONENTS"):
        env.step(ids, ["a"], [{}])

    class Declared(Undeclared):
        SHAPING_COMPONENTS = ("bonus",)

    env = Declared()
    ids, _ = env.reset(["q"], [{}])
    traj = env.step(ids, ["a"], [{}])[0].trajectory
    assert traj.info[REWARD_COMPONENTS_KEY]["reward/bonus"] == 0.1 and traj.total_reward == pytest.approx(1.1)


def test_a_grade_that_is_not_an_episode_grade_is_refused():
    """``_settle_grade`` reads ``.objective``/``.shaping`` off the return value; a bare float (the
    retired ``_compute_reward`` shape) would otherwise fail deep inside pricing, or — for a duck-typed
    stand-in — price an unvalidated objective straight into the reward."""

    class NotAGrade(base_module.BaseEnvironment):
        def _reset_single(self, prompt, context=None):
            return self._init_trajectory(prompt, context)

        def _step_single(self, trajectory, action, context=None):
            return trajectory, 0.0, True, False, {}

        def _grade_episode(self, trajectory, context=None):
            return 1.0

    env = NotAGrade()
    ids, _ = env.reset(["q"], [{}])
    with pytest.raises(TypeError, match="must return an EpisodeGrade"):
        env.step(ids, ["a"], [{}])


def test_a_shaping_component_declared_by_both_halves_is_refused():
    """The protocol's ``_episode_shaping`` and the env's own grade both contribute shaping under bare
    names. A name produced by both would silently take the grade's value (a plain dict update), losing
    the protocol's — so the collision is refused instead."""

    class Colliding(base_module.BaseEnvironment):
        SHAPING_COMPONENTS = ("bonus",)

        def _reset_single(self, prompt, context=None):
            return self._init_trajectory(prompt, context)

        def _step_single(self, trajectory, action, context=None):
            return trajectory, 0.0, True, False, {}

        def _episode_shaping(self, trajectory):
            return {"bonus": 0.25}

        def _grade_episode(self, trajectory, context=None):
            return EpisodeGrade(1.0, {"bonus": 0.1})

    env = Colliding()
    ids, _ = env.reset(["q"], [{}])
    with pytest.raises(ValueError, match="'reward/bonus' is declared twice"):
        env.step(ids, ["a"], [{}])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
