#!/usr/bin/env python
"""The offline re-grader must validate a trajectory's meta record before it grades anything.

``regrade_trajectories.py`` consumes meta keys two different producers stamp: ``run_code_contests.py``
adds ``adapter``/``language`` on top of the generic eval meta, ``run_env.py`` does not. Pointed at a
``run_env.py`` file, an unvalidated selector surfaces as a bare ``KeyError`` and a report-only key as
a ``TypeError`` in the summary line — the latter only AFTER the whole file has been graded, which is
minutes of sandboxed execution per file.

Run: python tests/cpu/environments/test_regrade_trajectories.py  (or pytest)
"""

import json
from dataclasses import fields
from pathlib import Path

import pytest

from scripts.environments.inference import regrade_trajectories
from scripts.environments.inference.run_code_contests import contest_meta
from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment
from src.environments.envs.tasks.coding.datasets import ContestSelection
from src.environments.envs.tasks.coding.grading import GradingSpec

# What run_code_contests.py stamps (generic eval meta + its meta_extra).
_FULL_META = {
    "env_type": "code_contests",
    "adapter": "code_contests",
    "dataset": "deepmind/code_contests",
    "model": "org/model",
    "language": "python",
}
# What run_env.py stamps: the generic eval meta only.
_RUN_ENV_META = {k: v for k, v in _FULL_META.items() if k not in ("adapter", "language")}


def _write_trajectories(path: Path, meta: dict) -> str:
    lines = [json.dumps({"type": "meta", **meta})]
    lines.append(json.dumps({"type": "episode", "index": 0, "messages": []}))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


def test_full_meta_validates():
    assert regrade_trajectories.validate_meta("f.jsonl", dict(_FULL_META)) is None


def test_validate_meta_reports_every_missing_key_and_its_producer():
    with pytest.raises(ValueError) as excinfo:
        regrade_trajectories.validate_meta("f.jsonl", dict(_RUN_ENV_META))

    message = str(excinfo.value)
    assert "adapter" in message and "language" in message
    assert "run_code_contests.py" in message, "the error must name the producer that stamps the missing keys"


def test_a_null_report_key_is_missing_too():
    """``model``/``language`` reach a width-formatted summary line; a null there is a TypeError."""
    with pytest.raises(ValueError, match="model"):
        regrade_trajectories.validate_meta("f.jsonl", {**_FULL_META, "model": None})


def test_regrade_file_validates_before_it_grades_anything(tmp_path, monkeypatch):
    def unreachable(*args, **kwargs):
        raise AssertionError("the meta must be validated before the dataset load / environment build")

    monkeypatch.setattr(regrade_trajectories, "_load_payloads", unreachable)
    monkeypatch.setattr(regrade_trajectories, "resolve_environment", unreachable)
    path = _write_trajectories(tmp_path / "run_env.jsonl", _RUN_ENV_META)

    with pytest.raises(ValueError, match="trajectory meta is missing"):
        regrade_trajectories.regrade_file(path, workers=1)


# --- The grading contract carried across the dump ---


def test_every_grading_knob_reaches_the_meta_block():
    """The block ``run_code_contests.py`` stamps must cover the WHOLE contract, derived from the
    dataclass. A hand-typed subset silently defaults the ninth field added tomorrow — precisely the
    re-threading :class:`GradingSpec` exists to prevent."""
    env = CodeContestsEnvironment(language="python", sandbox_backend="local")
    meta = env.grading_spec.to_meta()

    assert set(meta) == {f.name for f in fields(GradingSpec)} - {"sandbox"}
    assert json.loads(json.dumps(meta)) == meta, "the block rides a JSONL meta line"


def test_the_regrader_rebuilds_the_run_s_contract_not_the_class_defaults():
    """A run that raised ``timeout_per_test``/``max_time_limit`` must be re-graded at ITS limits: the
    class defaults would TLE a solution the online run passed. The two offline overrides still win."""
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
    """The old block spelled two knobs after the ENV constructor (``output_comparison``,
    ``timeout_per_test``). Silently ignoring them would re-grade at the class defaults and report a
    solve rate the run never produced."""
    spec = CodeContestsEnvironment(language="python", sandbox_backend="local").grading_spec
    with pytest.raises(ValueError, match="output_comparison"):
        spec.with_meta({"output_comparison": "tokens", "timeout_per_test": 5})


def test_an_unstamped_run_falls_back_to_the_rebuilt_environment():
    """Older trajectory files carry no block at all; the env's own spec must stand."""
    spec = CodeContestsEnvironment(language="python", sandbox_backend="local", timeout_per_test=4.0).grading_spec
    assert spec.with_meta({}, stop_on_first_failure=True, max_grading_seconds=None).default_timeout == 4.0


# --- Per-submission language and the stamped submission budget ---


def test_submitted_solutions_keep_only_the_calls_the_environment_admitted():
    """A recorded ``submit_solution`` call the tool refused (no code, a missing or foreign language,
    unparseable arguments) never ran and spent no budget, so it takes no slot in the re-graded prefix;
    the admitted ones carry the language each call named."""

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
    assert regrade_trajectories.submitted_solutions(episode, choosing.registry.get("submit_solution")) == [
        ("x", "cpp"),
        ("w", "python"),
    ]
    fixed = CodeContestsEnvironment(language="python", sandbox_backend="local")
    # A fixed-language run has no language argument: the schema filter drops it and every coded call binds.
    assert regrade_trajectories.submitted_solutions(episode, fixed.registry.get("submit_solution")) == [
        ("x", None),
        ("y", None),
        ("z", None),
        ("w", None),
    ]


def test_display_language_joins_a_model_chosen_set():
    assert regrade_trajectories.display_language("python") == "python"
    assert regrade_trajectories.display_language(["python", "cpp"]) == "python,cpp"


def test_episode_submission_budget_reads_the_stamped_tool_budget():
    env = CodeContestsEnvironment(language="python", sandbox_backend="local", max_submissions=2)
    stamped = {"info": {"episode_tool_budgets": {"submit_solution": 3, "python_repl": 6}}}
    assert regrade_trajectories.episode_submission_budget(stamped, env) == 3
    assert regrade_trajectories.episode_submission_budget({"info": {}}, env) == 2


def test_the_regrader_rebuilds_the_run_s_protocol_from_the_meta_line_the_eval_writes():
    """Written by ``contest_meta``, read by ``rebuild_environment``: a leaderboard run re-grades at one
    submission even where an episode stamped no budget, over a trained ladder that bound three. A meta
    line naming no protocol ran the harness."""
    profiles = {"high": {"thinking_tokens": 16384, "max_submissions": 3, "max_test_calls": 6}}
    env = CodeContestsEnvironment(
        language="python", sandbox_backend="local", eval_protocol="leaderboard", reasoning_effort_profiles=profiles
    )
    written = contest_meta("livecodebench", ContestSelection(), env, "high", {"reasoning_effort_profiles": profiles})
    meta = json.loads(json.dumps({"env_type": "codeforces", **written}))

    rebuilt = regrade_trajectories.rebuild_environment(meta)
    assert (rebuilt.eval_protocol, rebuilt.max_submissions, rebuilt.max_test_calls) == ("leaderboard", 1, 0)
    assert regrade_trajectories.episode_submission_budget({"info": {}}, rebuilt) == 1
    unnamed = {key: value for key, value in meta.items() if key != "eval_protocol"}
    assert regrade_trajectories.rebuild_environment(unnamed).eval_protocol == "harness"


def _submission(code: str) -> dict:
    call = {
        "id": "c",
        "type": "function",
        "function": {"name": "submit_solution", "arguments": json.dumps({"code": code})},
    }
    return {"role": "assistant", "content": "", "tool_calls": [call]}


def test_an_episode_the_driver_lost_leaves_n_and_is_counted(tmp_path, monkeypatch):
    """The eval leaves a generation-error sample out of every score; re-graded as an unsolved episode it
    would pull ``s@1`` below the online number for an endpoint fault. The solved and the unsubmitted
    episodes are what ``n`` counts."""
    payload = {"tests": [{"input": "", "output": "X"}], "checker": None, "time_limit": None}
    monkeypatch.setattr(regrade_trajectories, "build_payloads", lambda meta: (payload,))
    episodes = [
        {"type": "episode", "index": 0, "messages": [_submission("print('X')")], "generation_error": None},
        {"type": "episode", "index": 0, "messages": [], "generation_error": "NotFoundError: gone"},
        {"type": "episode", "index": 0, "messages": [], "generation_error": None},
    ]
    path = tmp_path / "run.jsonl"
    path.write_text("\n".join(json.dumps(r) for r in [{"type": "meta", **_FULL_META}, *episodes]) + "\n")

    metrics = regrade_trajectories.regrade_file(str(path), workers=1)
    assert (metrics["n"], metrics["s@1"], metrics["generation_errors"]) == (2, 0.5, 1)


def test_a_language_list_in_the_meta_rebuilds_the_choosing_environment():
    env = regrade_trajectories.resolve_environment("code_contests", {"language": ["python", "cpp"]})
    assert env.chooses_language and env.languages == ("python", "cpp")
    assert env.grading_spec.language == "python", "the contract's language is the set's first"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
