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
from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
