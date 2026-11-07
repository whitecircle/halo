#!/usr/bin/env python
"""The code-contests evaluation protocols (``eval_protocol``). ``leaderboard`` grades one program per sample with no
scratchpad at every effort level: a config written for this run contradicting it is refused, while a training
config's budgets give way; a level's reasoning budget and price are no pin and stay. ``harness`` pins nothing. The eval
script's ``--reasoning_effort none`` evaluates a non-thinking model at no level, and no level sizes the turn."""

import json
import sys
from types import SimpleNamespace

import pytest

import halo.entrypoints.code_contests_eval as code_contests_eval
import halo.environments.eval_runner as eval_runner
from halo.entrypoints.code_contests_eval import (
    FLAG_OWNED_ENV_KWARGS,
    NO_REASONING_EFFORT,
    contest_meta,
    resolve_env_config,
    resolve_eval_protocol,
    run_trajectory_path,
)
from halo.entrypoints.eval_contract import reject_flag_owned_env_kwargs, rollout_config_from_args
from halo.environments.registry import build_environment
from halo.environments.rollout_config import VALID_REASONING_EFFORTS, RolloutConfig
from halo.environments.tasks.coding.code_contests import (
    DEFAULT_EVAL_PROTOCOL,
    EVAL_PROTOCOL_KNOB_DEFAULTS,
    EVAL_PROTOCOLS,
    CodeContestsEnvironment,
)
from halo.environments.tasks.coding.code_contests_prompts import (
    SCRATCHPAD_BUDGET_SPENT_REPLY,
    SUBMISSION_BUDGET_SPENT_REPLY,
)
from halo.environments.tasks.coding.contest_datasets import ContestSelection
from halo.environments.tasks.coding.verdicts import VERDICT_DETAIL_FULL
from halo.environments.tool_accounting import EPISODE_TOOL_BUDGETS_KEY, TOOL_CALL_COUNTS_KEY
from scripts.environments.inference.eval_code_contests import parse_args
from tests.common.code_contests import (
    SINGLE_TEST_ANSWER,
    StubSandbox,
    call_tool,
    counts_beside_budget_words,
    reset_episode,
    retired_budget_phrases,
)
from tests.common.fake_chat_client import chat_reply

# The shipped recipes' ladder: effort buys submissions and scratchpad runs as well as free reasoning.
_RECIPE_PROFILES = {
    "low": {"reasoning_budget": 40000, "reasoning_price": 0.008, "max_submissions": 1, "max_test_calls": 2},
    "medium": {"reasoning_budget": 50000, "reasoning_price": 0.006, "max_submissions": 2, "max_test_calls": 4},
    "high": {"reasoning_budget": 60000, "reasoning_price": 0.004, "max_submissions": 3, "max_test_calls": 6},
}


def _env(**kwargs) -> CodeContestsEnvironment:
    return CodeContestsEnvironment(language="python", sandbox=StubSandbox(), **kwargs)


def _budgets(env) -> tuple[int, int]:
    return env.max_submissions, env.max_test_calls


def test_leaderboard_pins_one_submission_and_no_scratchpad():
    env = _env(eval_protocol="leaderboard", verdict_detail=VERDICT_DETAIL_FULL)
    assert env.eval_protocol == "leaderboard"
    assert _budgets(env) == (1, 0)
    assert env.grading_spec.verdict_detail == VERDICT_DETAIL_FULL, (
        "the verdict detail is the config's, not the protocol's"
    )
    # The pins bind through the caps alone: the descriptions say nothing of a disabled scratchpad or an
    # only submission, and carry no count.
    schemas = {tool["function"]["name"]: tool["function"]["description"] for tool in env.get_tools_schema()}
    for description in (schemas["python_repl"], schemas["submit_solution"]):
        assert not retired_budget_phrases(description), description
        assert not counts_beside_budget_words(description), description


def test_the_harness_is_the_default_and_pins_nothing():
    default = _env()
    assert default.eval_protocol == DEFAULT_EVAL_PROTOCOL == "harness"
    assert _budgets(default) == (2, 5)
    assert _budgets(_env(eval_protocol="harness", max_submissions=3, max_test_calls=0)) == (3, 0)


@pytest.mark.parametrize("contradiction", [{"max_submissions": 3}, {"max_test_calls": 2}])
def test_a_config_contradicting_a_pin_is_refused(contradiction):
    with pytest.raises(ValueError, match="eval_protocol 'leaderboard' pins"):
        _env(eval_protocol="leaderboard", **contradiction)


def test_a_config_agreeing_with_the_pins_is_accepted():
    assert _budgets(_env(eval_protocol="leaderboard", max_submissions=1, max_test_calls=0)) == (1, 0)


def test_an_unknown_protocol_is_refused():
    with pytest.raises(ValueError, match="eval_protocol must be one of"):
        _env(eval_protocol="pass_at_k")


def test_every_protocol_pins_only_knobs_the_environment_resolves():
    """A pin on any other knob would be dropped without a word, labelling the run with a protocol it does not
    follow."""
    for name, pins in EVAL_PROTOCOLS.items():
        assert set(pins) <= set(EVAL_PROTOCOL_KNOB_DEFAULTS), name


def test_the_leaderboard_pins_hold_at_every_effort_level():
    """A training ladder binds three submissions at ``high``; under the leaderboard the episode still gets one and
    no scratchpad, the level keeps its reasoning budget and price, and the task message states no budget."""
    harness = _env(reasoning_effort_profiles=_RECIPE_PROFILES)
    traj = reset_episode(harness, {"reasoning_effort": "high", **SINGLE_TEST_ANSWER})
    assert traj.info[EPISODE_TOOL_BUDGETS_KEY] == {"python_repl": 6, "submit_solution": 3}

    leaderboard = _env(eval_protocol="leaderboard", reasoning_effort_profiles=_RECIPE_PROFILES)
    for level, profile in _RECIPE_PROFILES.items():
        traj = reset_episode(leaderboard, {"reasoning_effort": level, **SINGLE_TEST_ANSWER})
        assert traj.info[EPISODE_TOOL_BUDGETS_KEY] == {"python_repl": 0, "submit_solution": 1}, level
        terms = (profile["reasoning_budget"], profile["reasoning_price"])
        assert leaderboard.reasoning_terms_for_effort(level) == harness.reasoning_terms_for_effort(level) == terms
        user = next(m for m in reversed(traj.messages) if m.role == "user")
        assert not retired_budget_phrases(user.content), (level, user.content)


def test_a_pinned_profile_key_is_still_validated():
    """The pins supersede a profile's budgets; they do not excuse an invalid one."""
    with pytest.raises(
        ValueError, match=r"reasoning_effort_profiles\['high'\]: max_submissions must be an integer >= 1"
    ):
        _env(eval_protocol="leaderboard", reasoning_effort_profiles={"high": {"max_submissions": 0}})


def test_a_leaderboard_episode_grades_one_program_and_refuses_the_scratchpad():
    env = _env(eval_protocol="leaderboard", reasoning_effort_profiles=_RECIPE_PROFILES)
    traj = reset_episode(env, {"reasoning_effort": "high", **SINGLE_TEST_ANSWER})

    assert call_tool(env, traj, "python_repl") == f"Error: {SCRATCHPAD_BUDGET_SPENT_REPLY}"
    call_tool(env, traj, "submit_solution")
    assert call_tool(env, traj, "submit_solution") == f"Error: {SUBMISSION_BUDGET_SPENT_REPLY}"

    assert traj.info[TOOL_CALL_COUNTS_KEY].get("python_repl", 0) == 0
    assert traj.info[TOOL_CALL_COUNTS_KEY]["submit_solution"] == 1
    assert (traj.info["tests_passed"], traj.info["tests_total"]) == (1, 1)
    assert traj.info["tested_before_submission"] is False


def _scratchpad_turn(env, ids):
    call = {
        "id": "c",
        "function": {"name": "python_repl", "arguments": json.dumps({"code": "print(1)", "stdin": "1"})},
    }
    return env.step(ids, [""], [{"finish_reason": "stop", "tool_calls": [call]}])[0]


def test_a_leaderboard_scratchpad_call_flags_no_turn_while_a_spent_budget_still_does():
    """A leaderboard scratchpad call gets the spent-budget reply and price, but refuses nothing the protocol
    offered, so its turn stays trainable. A harness budget the episode spent still flags the turn."""
    leaderboard = _env(eval_protocol="leaderboard", tool_error_penalty=0.05)
    ids, _ = leaderboard.reset(["solve it"], [SINGLE_TEST_ANSWER])
    step = _scratchpad_turn(leaderboard, ids)
    traj = step.trajectory
    assert traj.messages[-1].content == f"Error: {SCRATCHPAD_BUDGET_SPENT_REPLY}"
    assert step.reward == pytest.approx(-0.05)
    assert not traj.messages[-2].calls_rejected and not traj.messages[-2].untrainable

    harness = _env(max_test_calls=1, tool_error_penalty=0.05)
    ids, _ = harness.reset(["solve it"], [SINGLE_TEST_ANSWER])
    assert not _scratchpad_turn(harness, ids).trajectory.messages[-2].calls_rejected
    step = _scratchpad_turn(harness, ids)
    traj = step.trajectory
    assert traj.messages[-1].content == f"Error: {SCRATCHPAD_BUDGET_SPENT_REPLY}"
    assert step.reward == pytest.approx(-0.05)
    assert traj.messages[-2].calls_rejected and traj.messages[-2].untrainable


def test_the_registry_presets_take_the_protocol():
    env = build_environment("codeforces", {"eval_protocol": "leaderboard", "sandbox": StubSandbox()})
    assert env.grading_spec.comparison == "tokens"
    assert (env.eval_protocol, *_budgets(env)) == ("leaderboard", 1, 0)


def test_a_training_config_s_budgets_give_way_to_the_flag_s_protocol():
    """Top-level budgets are the trained contract as much as the profile ones: both give way."""
    trained = {"max_submissions": 3, "max_test_calls": 6, "reasoning_effort_profiles": _RECIPE_PROFILES}
    eval_protocol, contract = resolve_eval_protocol("leaderboard", trained)
    assert eval_protocol == "leaderboard"
    assert "max_submissions" not in contract and "max_test_calls" not in contract
    assert _budgets(_env(**contract, eval_protocol=eval_protocol)) == (1, 0)
    # Without the flag the trained contract stands whole under the harness.
    assert resolve_eval_protocol(None, trained) == ("harness", trained)


def test_the_meta_records_the_config_the_environment_was_built_from():
    """Every flag overriding the training config is what the meta line records, the env built from the same config."""
    trained = {"eval_protocol": "harness", "language": "python", "reasoning_effort": "low", "max_turns": 9}
    flags = SimpleNamespace(eval_protocol="leaderboard", language="cpp", reasoning_effort="high", max_turns=4)
    env_config = resolve_env_config(flags, trained, {"timeout_per_test": 3})
    env = CodeContestsEnvironment(sandbox=StubSandbox(), **env_config)
    recorded = contest_meta("livecodebench", ContestSelection(), env, env_config["reasoning_effort"], env_config)
    assert recorded["env_kwargs"] == {
        "eval_protocol": "leaderboard",
        "language": "cpp",
        "reasoning_effort": "high",
        "max_turns": 4,
        "timeout_per_test": 3,
    }
    assert recorded["eval_protocol"] == env.eval_protocol == "leaderboard"


def test_a_contradiction_stated_for_this_run_is_refused():
    """A config that names the protocol itself, or ``--env_kwargs`` laid over it, contradicts the pin."""
    own = {"eval_protocol": "leaderboard", "max_submissions": 3}
    eval_protocol, contract = resolve_eval_protocol(None, own)
    assert (eval_protocol, contract) == ("leaderboard", own)
    with pytest.raises(ValueError, match="eval_protocol 'leaderboard' pins"):
        _env(**contract)
    _, contract = resolve_eval_protocol("leaderboard", {"max_submissions": 3})
    with pytest.raises(ValueError, match="eval_protocol 'leaderboard' pins"):
        _env(**{**contract, "eval_protocol": "leaderboard", "max_submissions": 3})


def test_the_eval_script_takes_the_protocol_from_its_flag_only():
    with pytest.raises(SystemExit, match="--eval_protocol, not --env_kwargs"):
        reject_flag_owned_env_kwargs({"eval_protocol": "leaderboard"}, FLAG_OWNED_ENV_KWARGS)


def test_every_env_option_a_flag_sets_is_refused_in_env_kwargs():
    """``--env_kwargs`` is laid over the flags, so ``{"max_turns": 3}`` would silently cap a ``--max_turns 16`` run.
    Read off the resolver, so a flag added without a refusal fails here."""
    flags = SimpleNamespace(max_turns=16, language="python", eval_protocol="harness", reasoning_effort="high")
    flag_set = set(resolve_env_config(flags, {}, {}))
    assert flag_set <= set(FLAG_OWNED_ENV_KWARGS), sorted(flag_set - set(FLAG_OWNED_ENV_KWARGS))
    for key in sorted(flag_set):
        with pytest.raises(SystemExit, match=f"--{key}, not --env_kwargs"):
            reject_flag_owned_env_kwargs({key: 3}, FLAG_OWNED_ENV_KWARGS)
    reject_flag_owned_env_kwargs({"timeout_per_test": 3}, FLAG_OWNED_ENV_KWARGS)


def test_a_default_run_keeps_its_trajectory_file_name():
    """The protocol and the selection name the file only where they depart from the defaults."""
    args = SimpleNamespace(
        save_trajectories=None, trajectory_dir="/runs", model="org/m", adapter="livecodebench", split="test"
    )
    assert run_trajectory_path(args, _env(), ContestSelection()) == "/runs/org-m__livecodebench__test__python.jsonl"
    window = ContestSelection.parse("2025-01-04", "2025-04-06", ["atcoder"])
    assert run_trajectory_path(args, _env(eval_protocol="leaderboard"), window) == (
        "/runs/org-m__livecodebench__test__python__leaderboard__2025-01-04..2025-04-06_atcoder.jsonl"
    )
    assert run_trajectory_path(args, _env(), window) == (
        "/runs/org-m__livecodebench__test__python__2025-01-04..2025-04-06_atcoder.jsonl"
    )


def _parse_flags(monkeypatch, *flags: str):
    monkeypatch.setattr(sys, "argv", ["eval_code_contests.py", "--dataset", "d", "--model", "m", *flags])
    return parse_args()


def test_the_effort_flag_spells_no_level_as_none(monkeypatch):
    """``--env_kwargs`` may not set an option its flag owns, so ``none`` is ``reasoning_effort=None``, over a
    training config's level too."""
    args = _parse_flags(monkeypatch, "--reasoning_effort", "none")
    assert resolve_env_config(args, {}, {})["reasoning_effort"] is None
    assert resolve_env_config(args, {"reasoning_effort": "high"}, {})["reasoning_effort"] is None


@pytest.mark.parametrize("flag", [None, *VALID_REASONING_EFFORTS, NO_REASONING_EFFORT])
def test_without_a_training_config_every_level_samples_under_the_rollouts_default_turn(monkeypatch, flag):
    """A level is stated to the model and never sizes its turn: without ``--max_tokens`` or ``--training_config`` the
    eval gets the training rollout's default turn and no reasoning cap at every level, though each level's profile
    sets a reasoning budget past that turn."""
    profiles = {level: {"reasoning_budget": 60000, "reasoning_price": 0.004} for level in VALID_REASONING_EFFORTS}
    handed = {}

    async def fake_collect_results(env, examples, client, *, rollout, **kwargs):
        handed.update(env=env, rollout=rollout)
        return []

    def stub_sandboxed(env_type, config):
        return build_environment(env_type, {**config, "sandbox": StubSandbox()})

    monkeypatch.setattr(code_contests_eval, "build_environment", stub_sandboxed)
    monkeypatch.setattr(code_contests_eval, "build_examples", lambda *args: [{"prompt": "p", "context": {}}])
    monkeypatch.setattr(code_contests_eval, "build_openai_client", lambda **kwargs: None)
    monkeypatch.setattr(code_contests_eval, "collect_results", fake_collect_results)
    monkeypatch.setattr(code_contests_eval, "report", lambda *args, **kwargs: None)
    monkeypatch.setattr(code_contests_eval, "write_eval_outputs", lambda *args, **kwargs: None)
    effort = () if flag is None else ("--reasoning_effort", flag)
    args = _parse_flags(monkeypatch, *effort, "--env_kwargs", json.dumps({"reasoning_effort_profiles": profiles}))
    code_contests_eval.run_code_contests_eval(args)

    rollout = handed["rollout"]
    assert (rollout.max_tokens, rollout.max_thinking_tokens, rollout.max_answer_tokens) == (
        RolloutConfig.max_tokens,
        None,
        None,
    )
    assert handed["env"].reasoning_terms_for_effort("high") == (60000, 0.004), "the profiles reached the env"


async def test_a_no_level_episode_sends_no_level_and_records_none(monkeypatch):
    """The episode's requests carry no level, budget or template variable, so a server without a reasoning parser
    serves them, and the records hold no level."""
    args = _parse_flags(monkeypatch, "--reasoning_effort", "none")
    env_config = resolve_env_config(args, {}, {})
    env = CodeContestsEnvironment(sandbox=StubSandbox(), **env_config)
    rollout = rollout_config_from_args(args, None, env, default_temperature=0.2)
    calls = []

    async def fake_chat(client, **request):
        calls.append(request)
        return chat_reply("done", completion_tokens=3)

    monkeypatch.setattr(eval_runner, "chat_completion", fake_chat)
    traj = await eval_runner.run_episode(env, "solve it", dict(SINGLE_TEST_ANSWER), None, rollout=rollout)

    assert env.reasoning_effort is None
    filters = {"top_k": rollout.top_k, "min_p": rollout.min_p, "repetition_penalty": rollout.repetition_penalty}
    assert calls and all(call["extra_body"] == filters for call in calls)
    assert eval_runner.serialize_trajectory(traj)["reasoning_effort"] is None
    meta = contest_meta(args.adapter, ContestSelection(), env, env_config["reasoning_effort"], env_config)
    assert meta["reasoning_effort"] is None and meta["env_kwargs"]["reasoning_effort"] is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
