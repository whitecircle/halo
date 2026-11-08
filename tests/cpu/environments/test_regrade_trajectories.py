#!/usr/bin/env python
"""The offline re-grader validates a trajectory's meta record before grading. ``eval_code_contests.py`` stamps
``adapter``/``language`` and ``eval_env.py`` does not; unvalidated, a report-only key would raise ``TypeError``
only after minutes of sandboxed grading."""

import json
import logging
from dataclasses import fields
from pathlib import Path

import pytest

from halo.entrypoints import code_contests_regrade
from halo.entrypoints.code_contests_eval import contest_meta
from halo.environments.tasks.coding.code_contests import CodeContestsEnvironment
from halo.environments.tasks.coding.contest_datasets import ContestSelection
from halo.environments.tasks.coding.grading import GradingSpec
from halo.environments.tasks.coding.verdicts import FloatTolerance

# What eval_code_contests.py stamps (generic eval meta + its meta_extra).
_FULL_META = {
    "env_type": "code_contests",
    "adapter": "code_contests",
    "dataset": "deepmind/code_contests",
    "split": "test",
    "model": "org/model",
    "language": "python",
}
# What eval_env.py stamps: the generic eval meta only.
_RUN_ENV_META = {k: v for k, v in _FULL_META.items() if k not in ("adapter", "language")}


def _write_trajectories(path: Path, meta: dict) -> str:
    lines = [json.dumps({"type": "meta", **meta})]
    lines.append(json.dumps({"type": "episode", "index": 0, "messages": []}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_full_meta_validates():
    assert code_contests_regrade.validate_meta("f.jsonl", dict(_FULL_META)) is None


def test_validate_meta_reports_every_missing_key_and_its_producer():
    with pytest.raises(ValueError) as excinfo:
        code_contests_regrade.validate_meta("f.jsonl", dict(_RUN_ENV_META))

    message = str(excinfo.value)
    assert "adapter" in message and "language" in message
    assert "eval_code_contests.py" in message, "the error must name the producer that stamps the missing keys"


def test_a_meta_without_its_split_is_refused():
    """Guessing a split would grade every episode of a train-split run (HLCE) against another split's problems."""
    with pytest.raises(ValueError, match="split"):
        code_contests_regrade.validate_meta("f.jsonl", {k: v for k, v in _FULL_META.items() if k != "split"})


def test_a_null_report_key_is_missing_too():
    """``model``/``language`` reach a width-formatted summary line; a null there is a TypeError."""
    with pytest.raises(ValueError, match="model"):
        code_contests_regrade.validate_meta("f.jsonl", {**_FULL_META, "model": None})


def test_regrade_file_validates_before_it_grades_anything(tmp_path, monkeypatch):
    def unreachable(*args, **kwargs):
        raise AssertionError("the meta must be validated before the dataset load / environment build")

    monkeypatch.setattr(code_contests_regrade, "_load_payloads", unreachable)
    monkeypatch.setattr(code_contests_regrade, "build_environment", unreachable)
    path = _write_trajectories(tmp_path / "run_env.jsonl", _RUN_ENV_META)

    with pytest.raises(ValueError, match="trajectory meta is missing"):
        code_contests_regrade.regrade_file(path, workers=1)


def test_every_grading_knob_reaches_the_meta_block():
    """Derived from the dataclass: a hand-typed subset silently defaults the next field added."""
    env = CodeContestsEnvironment(language="python", sandbox_backend="local")
    meta = env.grading_spec.to_meta()

    assert set(meta) == {f.name for f in fields(GradingSpec)} - {"sandbox"}
    assert json.loads(json.dumps(meta)) == meta, "the block rides a JSONL meta line"


def test_the_regrader_rebuilds_the_run_s_contract_not_the_class_defaults():
    """The class defaults would TLE a solution the online run passed. The two offline overrides still win."""
    ran_at = CodeContestsEnvironment(
        language="cpp",
        sandbox_backend="local",
        output_comparison="tokens",
        timeout_per_test=9.0,
        max_time_limit=11.0,
        stop_on_first_failure=False,
        max_grading_seconds=150,
    ).grading_spec
    default_env = CodeContestsEnvironment(language="python", sandbox_backend="local")

    spec = default_env.grading_spec.with_meta(ran_at.to_meta(), stop_on_first_failure=True, max_grading_seconds=None)

    assert (spec.comparison, spec.language) == ("tokens", "cpp")
    assert (spec.default_timeout, spec.max_time_limit) == (9.0, 11.0)
    assert spec.sandbox is default_env.grading_spec.sandbox, "the executor comes from the rebuilt env"
    # s@k needs the all-pass verdict, and a wall-clock budget would score a slow-but-correct solve 0.
    assert (spec.stop_on_first_failure, spec.max_grading_seconds) == (True, None)


def test_a_retired_meta_spelling_is_refused_rather_than_defaulted():
    """Silently ignoring env-constructor spellings would re-grade at the class defaults and report a solve rate the
    run never produced."""
    spec = CodeContestsEnvironment(language="python", sandbox_backend="local").grading_spec
    with pytest.raises(ValueError, match="output_comparison"):
        spec.with_meta({"output_comparison": "tokens", "timeout_per_test": 5})


def test_an_unstamped_run_falls_back_to_the_rebuilt_environment():
    """A file with no block keeps the env's own spec."""
    spec = CodeContestsEnvironment(language="python", sandbox_backend="local", timeout_per_test=4.0).grading_spec
    assert spec.with_meta({}, stop_on_first_failure=True, max_grading_seconds=None).default_timeout == 4.0


def test_submitted_solutions_keep_only_the_calls_the_environment_admitted():
    """A refused call never ran and spent no budget, so it takes no slot in the re-graded prefix; admitted ones
    carry the language each named."""

    def call(arguments):
        return {"function": {"name": "submit_solution", "arguments": arguments}}

    episode = {
        "messages": [
            {
                "role": "assistant",
                "tool_calls": [
                    call(json.dumps({"code": "x", "language": "cpp"})),
                    call(json.dumps({"code": "y"})),
                    call(json.dumps({"code": "z", "language": "java"})),
                    call(json.dumps({"language": "python"})),
                    call("not json"),
                    call(json.dumps({"code": "w", "language": "python"})),
                    {"function": {"name": "run_code", "arguments": json.dumps({"code": "q", "language": "cpp"})}},
                ],
            }
        ]
    }
    choosing = CodeContestsEnvironment(language=["python", "cpp"], sandbox_backend="local")
    assert code_contests_regrade.submitted_solutions(episode, choosing) == [
        ("x", "cpp"),
        ("w", "python"),
    ]
    fixed = CodeContestsEnvironment(language="python", sandbox_backend="local")
    # A fixed-language run declares no language argument, so a call naming one is refused like any other
    # undeclared argument, and only the bare coded call binds.
    assert code_contests_regrade.submitted_solutions(episode, fixed) == [("y", None)]


def test_submitted_solutions_read_arguments_the_way_the_environment_admitted_them():
    """The environment admits a Python-literal ``arguments`` string, so a JSON-only re-grade would score a different
    prefix than the run graded."""
    tool_calls = [
        {"function": {"name": "submit_solution", "arguments": "{'code': 'print(1)', 'language': 'python'}"}},
        {"function": {"name": "submit_solution", "arguments": {"code": "print(2)", "language": "python"}}},
    ]
    episode = {"messages": [{"role": "assistant", "tool_calls": tool_calls}]}
    env = CodeContestsEnvironment(language=["python", "cpp"], sandbox_backend="local")
    assert code_contests_regrade.submitted_solutions(episode, env) == [
        ("print(1)", "python"),
        ("print(2)", "python"),
    ]


def test_submitted_solutions_skip_a_program_the_environment_refused_for_its_language():
    """Refused as written in another language, it was never graded; replayed, it would take the real first
    submission's slot and score s@1 on it."""
    cpp = "#include <bits/stdc++.h>\nusing namespace std;\nint main() { cout << 1; }"
    tool_calls = [
        {"function": {"name": "submit_solution", "arguments": json.dumps({"code": cpp, "language": "python"})}},
        {"function": {"name": "submit_solution", "arguments": json.dumps({"code": cpp, "language": "cpp"})}},
    ]
    episode = {"messages": [{"role": "assistant", "tool_calls": tool_calls}]}
    env = CodeContestsEnvironment(language=["python", "cpp"], sandbox_backend="local")
    assert code_contests_regrade.submitted_solutions(episode, env) == [(cpp, "cpp")]


def test_submitted_solutions_take_admission_from_each_call_s_recorded_reply():
    """A record re-grades under the rules it ran under: a call whose recorded reply starts ``Error:`` was refused
    unrun and takes no slot (an identical resubmission a run refused before such a resend counted), any other reply
    was a graded slot, a program today's tool would refuse to bind included."""

    def submit(call_id, code):
        return {"id": call_id, "function": {"name": "submit_solution", "arguments": json.dumps({"code": code})}}

    def reply(call_id, content):
        return {"role": "tool", "tool_call_id": call_id, "content": content}

    leaked = "print(2)\n</code>"
    episode = {
        "messages": [
            {"role": "assistant", "tool_calls": [submit("a", "print(1)")]},
            reply("a", "Passed 1/3 tests."),
            {"role": "assistant", "tool_calls": [submit("b", "print(1)")]},
            reply("b", "Error: Not graded: this program is identical to one already graded."),
            {"role": "assistant", "tool_calls": [submit("c", "print(1)"), submit("d", leaked)]},
            reply("c", "Identical to submission 1, already graded: its verdict is repeated.\nPassed 1/3 tests."),
            reply("d", "Passed 3/3 tests."),
        ]
    }
    env = CodeContestsEnvironment(language="python", sandbox_backend="local")
    assert code_contests_regrade.submitted_solutions(episode, env) == [
        ("print(1)", None),
        ("print(1)", None),
        (leaked, None),
    ]


def test_an_episode_s_statement_sets_the_tolerance_it_is_re_graded_under():
    stated = {
        "messages": [
            {"role": "system", "content": "error 10^-1"},
            {"role": "user", "content": "Absolute error up to 10^{-4}."},
        ]
    }
    assert code_contests_regrade.episode_stated_tolerance(stated) == FloatTolerance(absolute=1e-4, relative=None)
    assert (
        code_contests_regrade.episode_stated_tolerance({"messages": [{"role": "user", "content": "Print n."}]}) is None
    )


def test_display_language_joins_a_model_chosen_set():
    assert code_contests_regrade.display_language("python") == "python"
    assert code_contests_regrade.display_language(["python", "cpp"]) == "python,cpp"


def test_episode_submission_budget_reads_the_stamped_tool_budget():
    env = CodeContestsEnvironment(language="python", sandbox_backend="local", max_submissions=2)
    stamped = {"info": {"episode_tool_budgets": {"submit_solution": 3, "python_repl": 6}}}
    assert code_contests_regrade.episode_submission_budget(stamped, env) == 3
    assert code_contests_regrade.episode_submission_budget({"info": {}}, env) == 2


def test_the_regrader_rebuilds_the_run_s_protocol_from_the_meta_line_the_eval_writes():
    """A leaderboard run re-grades at one submission even where an episode stamped no budget; a meta line naming no
    protocol ran the harness."""
    profiles = {
        "high": {"reasoning_budget": 60000, "reasoning_price": 0.004, "max_submissions": 3, "max_test_calls": 6}
    }
    env = CodeContestsEnvironment(
        language="python", sandbox_backend="local", eval_protocol="leaderboard", reasoning_effort_profiles=profiles
    )
    written = contest_meta("livecodebench", ContestSelection(), env, "high", {"reasoning_effort_profiles": profiles})
    meta = json.loads(json.dumps({"env_type": "codeforces", **written}))

    rebuilt = code_contests_regrade.rebuild_environment(meta)
    assert (rebuilt.eval_protocol, rebuilt.max_submissions, rebuilt.max_test_calls) == ("leaderboard", 1, 0)
    assert code_contests_regrade.episode_submission_budget({"info": {}}, rebuilt) == 1
    unnamed = {key: value for key, value in meta.items() if key != "eval_protocol"}
    assert code_contests_regrade.rebuild_environment(unnamed).eval_protocol == "harness"


def test_a_meta_line_carrying_profile_keys_the_environment_no_longer_admits_still_re_grades(caplog):
    """Grading reads a profile's interaction budgets only: keys a later schema retired are dropped, said so, and the
    budgets they sat beside still bind."""
    meta = {
        "env_type": "codeforces",
        "language": "python",
        "reasoning_effort": "high",
        "env_kwargs": {
            "sandbox_backend": "local",
            "reasoning_effort_profiles": {
                "high": {"thinking_tokens": 16384, "enforce_thinking_cap": False, "max_submissions": 3}
            },
        },
    }
    with caplog.at_level(logging.INFO, logger=code_contests_regrade.logger.name):
        rebuilt = code_contests_regrade.rebuild_environment(meta)
    assert rebuilt.reasoning_effort_profiles["high"] == {"max_submissions": 3}
    assert "dropping ['enforce_thinking_cap', 'thinking_tokens']" in caplog.text
    with pytest.raises(ValueError, match="unknown keys"):
        CodeContestsEnvironment(language="python", **meta["env_kwargs"])


def _submission(code: str) -> dict:
    call = {
        "id": "c",
        "type": "function",
        "function": {"name": "submit_solution", "arguments": json.dumps({"code": code})},
    }
    return {"role": "assistant", "content": "", "tool_calls": [call]}


def test_an_episode_the_driver_lost_leaves_n_and_is_counted(tmp_path, monkeypatch):
    """Re-graded as unsolved it would pull ``s@1`` below the online number for an endpoint fault."""
    payload = {"tests": [{"input": "", "output": "X"}], "checker": None, "time_limit": None}
    monkeypatch.setattr(code_contests_regrade, "build_payloads", lambda meta: (payload,))
    episodes = [
        {"type": "episode", "index": 0, "messages": [_submission("print('X')")], "generation_error": None},
        {"type": "episode", "index": 0, "messages": [], "generation_error": "NotFoundError: gone"},
        {"type": "episode", "index": 0, "messages": [], "generation_error": None},
    ]
    path = tmp_path / "run.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [{"type": "meta", **_FULL_META}, *episodes]) + "\n")

    metrics = code_contests_regrade.regrade_file(str(path), workers=1)
    assert (metrics["n"], metrics["s@1"], metrics["generation_errors"]) == (2, 0.5, 1)


def test_an_episode_past_the_rebuilt_problems_refuses_the_file(tmp_path, monkeypatch):
    """The dataset or its selection changed since the run; re-grading on would score that episode unsolved with
    nothing saying why."""
    payload = {"tests": [{"input": "", "output": "X"}], "checker": None, "time_limit": None}
    monkeypatch.setattr(code_contests_regrade, "build_payloads", lambda meta: (payload,))
    episode = {"type": "episode", "index": 3, "messages": [_submission("print('X')")], "generation_error": None}
    path = tmp_path / "run.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [{"type": "meta", **_FULL_META}, episode]) + "\n")

    with pytest.raises(SystemExit, match=r"index\(es\) \[3\] address none of the 1 problems"):
        code_contests_regrade.regrade_file(str(path), workers=1)


def test_a_language_list_in_the_meta_rebuilds_the_choosing_environment():
    env = code_contests_regrade.build_environment("code_contests", {"language": ["python", "cpp"]})
    assert env.chooses_language and env.languages == ("python", "cpp")
    assert env.grading_spec.language == "python", "the contract's language is the set's first"


def test_each_regraded_file_is_reported_and_persisted_as_it_lands(monkeypatch, tmp_path, caplog):
    """The report line is the tool's output: it must reach the log under the WARNING root the package leaves, and
    each file's metrics land in the output file before the next file grades."""
    metrics = {
        "model": "m",
        "adapter": "cf",
        "language": "python",
        "eval_protocol": "single",
        "n": 2,
        "s@1": 0.5,
        "s@2": 1.0,
        "generation_errors": 0,
    }
    monkeypatch.setattr(code_contests_regrade, "regrade_file", lambda path, workers: {**metrics, "model": path})
    monkeypatch.setattr(logging.getLogger(), "level", logging.WARNING)
    output = tmp_path / "regraded.jsonl"
    code_contests_regrade.regrade_trajectory_files(["a.jsonl", "b.jsonl"], 1, str(output))
    reported = [r.getMessage() for r in caplog.records if r.name == code_contests_regrade.__name__]
    assert [line.split()[0] for line in reported] == ["a.jsonl", "b.jsonl"]
    assert "s@1=50% s@2=100%" in reported[0]
    assert [json.loads(line)["model"] for line in output.read_text().splitlines()] == ["a.jsonl", "b.jsonl"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
