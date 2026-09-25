#!/usr/bin/env python
"""CPU tests: a sandbox fault is booked by its class, never priced as the other one.

- An infrastructure fault (:class:`SandboxInfraError`) says nothing about the policy: the tool call is
  unpriced, the episode ends and is marked invalid, so the trainer's ``rollout_valid_mask`` drops it
  from the GRPO group baseline.
- A sandbox the program broke itself (:class:`SandboxAgentFault` — its working directory replaced by
  a link or file) is the policy's: the call is a failed one and the episode ends uncompleted, inside the
  baseline, where it is scored against its group like any other episode.
- Nothing the program controls may read as infra: a remote ``Failed`` run, a build past the compile
  limit, an output flood, a removed working directory (recreated) and a FIFO where the host stages or
  reads are verdicts or ordinary results, never an ``error`` that voids the episode or a host open that
  blocks.

    python tests/cpu/environments/test_sandbox_faults.py
"""

import asyncio
import errno
import json
import logging
import os
import pathlib
import resource
import shutil
import stat
import subprocess
import sys
import tempfile
import threading
import time

import pytest
import torch

from src.environments.base import (
    EPISODE_INVALID_REASON_KEY,
    REWARD_COMPONENTS_KEY,
    REWARD_PENDING_KEY,
    SANDBOX_FAULT_INFRA,
    SANDBOX_FAULT_KEY,
)
from src.environments.envs.protocols.native import AsyncNativeToolUseEnvironment, NativeToolUseEnvironment
from src.environments.envs.protocols.react import ReActEnvironment
from src.environments.envs.tasks.coding.code_contests import CodeContestsEnvironment
from src.environments.envs.tasks.coding.grading import GradingSpec, grade_solution, run_solution_against_tests
from src.environments.envs.tasks.coding.swe import SweEnvironment
from src.environments.episode import RolloutResult
from src.environments.sandbox.base import (
    LOCAL_FSIZE_LIMIT,
    SandboxAgentFault,
    SandboxInfraError,
    SandboxResult,
)
from src.environments.sandbox.bubblewrap import BubblewrapSandbox
from src.environments.sandbox.local import (
    TAMPERED_WORKDIR_RETURNCODE,
    LocalSubprocessSandbox,
    _grant_owner,
    _restore_owner_access,
)
from src.environments.sandbox.remote import RemoteSandbox
from src.environments.sandbox.repl import format_sandbox_repl_output
from src.environments.tools.definitions import NativeTool, NativeToolRegistry, ToolParameter
from src.environments.tools.factories import create_session_code_tools
from src.rewards import composer as composer_module
from src.rewards.scoring import Scorer, ScoreResult
from src.rewards.spec import JudgeTerm
from src.trainers.grpo.environmental import rollout_valid_mask
from src.trainers.grpo.objective.advantages import group_relative_advantages
from tests.common.code_contests import RecordingSandboxSession, StubSandbox
from tests.common.utils import REPO_ROOT

_TOOL_ERROR_PENALTY = 0.1
_REMOVE_OWN_WORKDIR = "import os, shutil\nprint('hello')\nshutil.rmtree(os.getcwd())\n"
_HOST_OPEN_TIMEOUT_S = 10.0


def _replace_own_workdir(target) -> str:
    """A program that replaces its working directory with a link to ``target``."""
    return (
        "import os, shutil\nworkdir = os.getcwd()\nos.chdir('/')\nshutil.rmtree(workdir)\n"
        f"os.symlink({str(target)!r}, workdir)\n"
    )


def _registry() -> NativeToolRegistry:
    """``run`` raises the fault its ``mode`` argument names, or answers ``ok``."""

    def run(mode: str) -> str:
        if mode == "infra":
            raise SandboxInfraError("sandbox backend failure: 503")
        if mode == "agent":
            raise SandboxAgentFault("the program removed its working directory")
        if mode == "crash":
            raise RuntimeError("tool bug")
        return "ok"

    registry = NativeToolRegistry()
    registry.register(
        NativeTool(name="run", description="Run.", parameters=[ToolParameter("mode", "string", "")], handler=run)
    )
    return registry


def _knobs() -> dict:
    return {"tool_success_reward": 0.0, "tool_error_penalty": _TOOL_ERROR_PENALTY}


def _call(mode: str, call_id: str = "c1") -> dict:
    return {"id": call_id, "function": {"name": "run", "arguments": json.dumps({"mode": mode})}}


def _native_episode(*modes: str, cls=NativeToolUseEnvironment):
    """One turn calling ``run`` once per mode; the trajectory after it."""
    env = cls(tool_registry=_registry(), **_knobs())
    ids, _ = env.reset(["task"], [{"answer": "4"}])
    context = {"answer": "4", "tool_calls": [_call(mode, f"c{i}") for i, mode in enumerate(modes)]}
    if cls is AsyncNativeToolUseEnvironment:
        step = asyncio.run(env.step_async(ids, ["calling"], [context]))[0]
    else:
        step = env.step(ids, ["calling"], [context])[0]
    return step, env


def _react_episode(mode: str):
    env = ReActEnvironment(tool_registry=_registry(), thought_reward=0.0, no_thought_penalty=0.0, **_knobs())
    ids, _ = env.reset(["task"], [{"answer": "4"}])
    return env.step(ids, [f'Thought: go\nAction: run(mode="{mode}")'], [{}])[0], env


# The protocols book each class by type


@pytest.mark.parametrize("cls", [NativeToolUseEnvironment, AsyncNativeToolUseEnvironment])
def test_infra_fault_ends_the_episode_unpriced_and_out_of_the_baseline(cls):
    step, env = _native_episode("infra", cls=cls)
    traj = step.trajectory
    assert step.done and not traj.truncated and not traj.info["completed"]
    assert traj.episode_invalid
    assert "503" in traj.info[EPISODE_INVALID_REASON_KEY]
    assert traj.total_reward == pytest.approx(0.0), "the backend's outage must not be priced as the policy's"
    assert traj.info["tool_results"][-1]["success"] is False
    metrics = env.rollout_metrics(traj)
    assert (metrics["episode/sandbox_infra_fault"], metrics["episode/sandbox_agent_fault"]) == (1.0, 0.0)


@pytest.mark.parametrize("cls", [NativeToolUseEnvironment, AsyncNativeToolUseEnvironment])
def test_agent_fault_ends_the_episode_failed_and_priced_in_the_baseline(cls):
    step, env = _native_episode("agent", cls=cls)
    traj = step.trajectory
    assert step.done and not traj.truncated and not traj.info["completed"]
    assert not traj.episode_invalid, "the policy's own fault must stay in the baseline"
    assert traj.total_reward == pytest.approx(-_TOOL_ERROR_PENALTY)
    metrics = env.rollout_metrics(traj)
    assert (metrics["episode/sandbox_infra_fault"], metrics["episode/sandbox_agent_fault"]) == (0.0, 1.0)


def test_each_agent_faulted_call_is_a_failed_call_and_infra_dominates_a_mixed_turn():
    agent_twice, _ = _native_episode("agent", "agent")
    assert agent_twice.done and agent_twice.trajectory.total_reward == pytest.approx(-2 * _TOOL_ERROR_PENALTY)
    for order in (("agent", "infra"), ("infra", "agent")):
        mixed, _ = _native_episode(*order)
        assert mixed.done and mixed.trajectory.episode_invalid
        assert mixed.trajectory.info[SANDBOX_FAULT_KEY] == SANDBOX_FAULT_INFRA, order


def test_an_ordinary_raising_tool_stays_a_priced_tool_error():
    step, env = _native_episode("crash")
    assert not step.done and not step.trajectory.episode_invalid
    assert step.trajectory.total_reward == pytest.approx(-_TOOL_ERROR_PENALTY)
    metrics = env.rollout_metrics(step.trajectory)
    assert (metrics["episode/sandbox_infra_fault"], metrics["episode/sandbox_agent_fault"]) == (0.0, 0.0)


@pytest.mark.parametrize("cls", [NativeToolUseEnvironment, AsyncNativeToolUseEnvironment])
def test_a_typed_sandbox_fault_logs_no_tool_traceback(cls, caplog):
    """A sandbox fault is booked by its class, not reported as a tool that broke: the traceback the
    protocol logs for any other raising tool stays off it."""
    caplog.set_level(logging.DEBUG)
    for mode in ("infra", "agent"):
        _native_episode(mode, cls=cls)
    assert not [record for record in caplog.records if record.exc_info]
    _native_episode("crash", cls=cls)
    assert [record for record in caplog.records if record.exc_info], "a tool that broke keeps its traceback"


def test_react_books_the_faults_the_same_way():
    infra, _ = _react_episode("infra")
    assert infra.done and infra.trajectory.episode_invalid and infra.trajectory.total_reward == pytest.approx(0.0)
    agent, _ = _react_episode("agent")
    assert agent.done and not agent.trajectory.episode_invalid
    assert agent.trajectory.total_reward == pytest.approx(-_TOOL_ERROR_PENALTY)


class _ScratchpadOutage(StubSandbox):
    """Grades every submission cleanly and loses each scratchpad run to the backend."""

    def run(self, code, *, stdin="", timeout=15.0, language="python", files=None):
        return SandboxResult(error="backend down") if "scratch" in code else super().run(code)


def test_a_turn_that_spends_the_last_submission_and_faults_ends_uncompleted():
    env = CodeContestsEnvironment(sandbox=_ScratchpadOutage(), max_submissions=1, **_knobs())
    context = {"answer": {"tests": [{"input": "", "output": "X"}]}}
    ids, _ = env.reset(["solve it"], [context])
    calls = [
        {"id": "s", "function": {"name": "submit_solution", "arguments": json.dumps({"code": "print('X')"})}},
        {"id": "r", "function": {"name": "python_repl", "arguments": json.dumps({"code": "# scratch"})}},
    ]
    step = env.step(ids, ["both"], [{**context, "tool_calls": calls}])[0]
    assert step.done and step.trajectory.info[SANDBOX_FAULT_KEY] == SANDBOX_FAULT_INFRA
    assert not step.trajectory.info["completed"], "a fault-ended episode is uncompleted"


@pytest.mark.parametrize("mode", ["infra", "agent"])
def test_a_fault_ended_episode_is_never_sent_to_a_scorer(mode, monkeypatch):
    """A judge verdict on a crashed fragment would be paid for and taught; on an outage, paid for a
    row the trainer drops anyway."""
    built = []

    class _Judge(Scorer):
        def __init__(self, term):
            super().__init__(term)
            built.append(self)

        async def score_one(self, sample):
            return ScoreResult(1.0)

        async def verify(self):
            pass

    monkeypatch.setitem(composer_module.SCORERS, JudgeTerm, _Judge)
    judge = {"source": "judge", "name": "quality", "requirements": [{"name": "done", "description": "Done."}]}
    env = NativeToolUseEnvironment(
        tool_registry=_registry(), reward_terms=[{"source": "environment"}, judge], **_knobs()
    )
    ids, _ = env.reset(["task"], [{"answer": "4"}])
    step = env.step(ids, ["calling"], [{"answer": "4", "tool_calls": [_call(mode)]}])[0]
    assert step.done and REWARD_PENDING_KEY not in step.trajectory.info
    env.settle(ids)
    assert built == [], "no scorer may be built for a fault-ended episode"
    assert step.trajectory.info[REWARD_COMPONENTS_KEY]["reward/quality"] == 0.0
    assert "episode/reward_scored" not in env.rollout_metrics(step.trajectory), "nothing was scored"


def test_the_trainer_trains_on_the_repercussion_and_drops_the_outage():
    """The classes land where the trainer already reads them: an infra-faulted row leaves the group
    baseline, an agent-faulted one stays in it, scored against its siblings."""
    rows = []
    for mode in ("ok", "ok", "agent", "infra"):
        step, env = _native_episode(mode)
        if not step.done:
            step = env.step([step.trajectory.info["episode_id"]], ["4"], [{"answer": "4"}])[0]
        assert step.done
        rows.append(
            RolloutResult(prompt="task", trajectory=step.trajectory, total_reward=step.trajectory.total_reward)
        )
    valid = rollout_valid_mask(rows, torch.device("cpu"))
    assert valid.tolist() == [True, True, True, False]
    rewards = torch.tensor([r.total_reward for r in rows])
    assert rewards.tolist() == pytest.approx([1.0, 1.0, -_TOOL_ERROR_PENALTY, 0.0])
    advantages = group_relative_advantages(rewards, 4, "none", valid_mask=valid)
    assert advantages[2] < 0, "below solving siblings, the agent-caused crash trains with a negative advantage"
    baseline = rewards[:3].mean()
    assert advantages[0].item() == pytest.approx((rewards[0] - baseline).item()), "the outage is out of the mean"


# The sandbox layer: nothing the program controls reads as infra


def test_the_repl_raises_each_fault_class_by_the_result():
    with pytest.raises(SandboxInfraError):
        format_sandbox_repl_output(SandboxResult(error="remote sandbox error: 503"), timeout=5)
    with pytest.raises(SandboxAgentFault):
        format_sandbox_repl_output(
            SandboxResult(stderr="x", returncode=TAMPERED_WORKDIR_RETURNCODE, agent_fault="x"), timeout=5
        )
    assert not issubclass(SandboxAgentFault, SandboxInfraError) and not issubclass(
        SandboxInfraError, SandboxAgentFault
    )


def test_remote_failed_run_is_the_programs_verdict():
    """SandboxFusion answers ``Failed`` for any run that exited non-zero: that is a runtime error of
    the program, not the service's failure. A ``Failed`` body that names no failing step, and a
    ``SandboxError``, stay the service's."""
    payload = {
        "status": "Failed",
        "message": "",
        "run_result": {"status": "Finished", "stdout": "", "stderr": "ZeroDivisionError", "return_code": 1},
    }
    result = RemoteSandbox("http://sandbox:8080", session=RecordingSandboxSession(payload)).run("1/0")
    assert result.error is None and result.returncode == 1 and not result.ok
    assert format_sandbox_repl_output(result, timeout=5) == "Error: ZeroDivisionError"
    contradictory = {"status": "Failed", "run_result": {"status": "Finished", "stdout": "1", "return_code": 0}}
    assert (
        RemoteSandbox("http://sandbox:8080", session=RecordingSandboxSession(contradictory)).run("print(1)").error
        == "Failed"
    )
    down = RemoteSandbox(
        "http://sandbox:8080", session=RecordingSandboxSession({"status": "SandboxError", "message": "oom"})
    )
    assert down.run("print(1)").error == "oom"


class _CompileTimesOut(LocalSubprocessSandbox):
    """Every launch reports its wall-clock limit hit, as an ``#include`` bomb's build does."""

    @staticmethod
    def _run_in_new_session(argv, *, stdin, timeout, cwd, env):
        return "", "", -9, True


def test_a_build_past_the_compile_limit_is_a_compile_verdict():
    with _CompileTimesOut(compile_timeout=1.0).open_session() as session:
        result = session.run("#include </dev/zero>", language="cpp")
    assert result.error is None and result.compile_failed and not result.ok
    assert format_sandbox_repl_output(result, timeout=5).startswith("Error: compilation exceeded")


def test_the_run_step_file_size_limit_is_the_documented_one():
    result = LocalSubprocessSandbox().run("import resource\nprint(resource.getrlimit(resource.RLIMIT_FSIZE)[0])")
    assert result.ok and int(result.stdout) == LOCAL_FSIZE_LIMIT


def test_an_output_flood_is_the_programs_failure_at_the_file_size_limit():
    """Output is captured in files under the child's ``RLIMIT_FSIZE``, so a flood ends as the program's
    own failure at the limit, never as gigabytes of host memory the grader runs out of (an infra error
    that would void the episode)."""
    chunks = LOCAL_FSIZE_LIMIT // (1 << 20) + 16
    flood = f"import sys\nchunk = 'x' * (1 << 20)\nfor _ in range({chunks}):\n    sys.stdout.write(chunk)\n"
    result = LocalSubprocessSandbox().run(flood, timeout=60)
    assert result.error is None and not result.timed_out
    assert result.returncode != 0 and "File too large" in result.stderr, "the flood must fail at the limit"
    assert len(result.stdout) == LOCAL_FSIZE_LIMIT, "the capture holds the output up to the limit"


def _executor(backend: str):
    if backend == "local":
        return LocalSubprocessSandbox()
    try:
        return BubblewrapSandbox()
    except RuntimeError as exc:
        pytest.skip(f"bubblewrap cannot sandbox here: {exc}")


@pytest.mark.parametrize("backend", ["local", "bubblewrap"])
def test_a_lone_surrogate_in_submitted_code_is_graded_not_voided(backend):
    """What UTF-8 cannot carry in the program's source is replaced, as in its stdin, so staging it can
    never fail into an infra error the policy controls."""
    tests = [{"input": "", "output": "ok"}]
    grade = run_solution_against_tests("print('ok')  # \ud83d\n", tests, sandbox=_executor(backend))
    assert grade.infra_errors == 0 and grade.passed == 1, grade.details


def test_a_remote_payload_carries_no_lone_surrogate():
    service = RecordingSandboxSession()
    RemoteSandbox("http://sandbox:8080", session=service).run(
        "print(1)  # \ud83d", stdin="a\ud83d", files={"h.py": "\ud83d"}
    )
    sent = service.posts[-1].payload
    assert (sent["code"], sent["stdin"], sent["files"]["h.py"]) == ("print(1)  # ?", "a?", "?")


_CHECKER_WANTS_EMPTY_INPUT = (
    "import sys\n"
    "given = open(sys.argv[1]).read()\n"
    "produced = open(sys.argv[3]).read().split()\n"
    "print(1 if given == '' and produced == ['ok'] else 0)\n"
)


def test_a_null_test_input_reaches_a_checker_as_an_empty_input():
    spec = GradingSpec(sandbox=LocalSubprocessSandbox())
    grade = grade_solution("print('ok')", [{"input": None, "output": "ok"}], spec, checker=_CHECKER_WANTS_EMPTY_INPUT)
    assert grade.infra_errors == 0 and grade.passed == 1, grade.details


@pytest.mark.parametrize(
    ("path", "reason"),
    [("a\ud83d.py", "not valid UTF-8"), ("", "unsafe"), (".", "unsafe"), ("..", "unsafe"), ("/abs", "unsafe")]
    + [("../up.py", "unsafe"), ("sub/../../up.py", "unsafe")],
)
@pytest.mark.parametrize("backend", ["local", "remote"])
def test_a_session_path_no_backend_can_take_is_refused_as_an_argument(backend, path, reason):
    """A priced tool error on every backend, never a name the backend fails on (an infra error)."""
    service = RecordingSandboxSession()
    if backend == "local":
        session = LocalSubprocessSandbox().open_session()
    else:
        session = RemoteSandbox("http://sandbox:8080", session=service).open_session()
    with session:
        for operation in (lambda: session.write_file(path, "x"), lambda: session.read_file(path)):
            with pytest.raises(ValueError, match=reason):
                operation()
        session.write_file("ok.py", "x")
        assert session.run("print(1)").error is None
    if backend == "remote":
        assert [list(post.payload["files"]) for post in service.posts] == [["ok.py"]], "no refused name is sent"


def test_a_null_test_input_is_no_input_not_an_infra_error():
    tests = [{"input": None, "output": "ok"}]
    grade = run_solution_against_tests("print('ok')", tests, sandbox=LocalSubprocessSandbox())
    assert grade.infra_errors == 0 and grade.passed == 1, grade.details


def test_a_removed_working_directory_is_recreated_not_a_fault():
    with LocalSubprocessSandbox().open_session() as session:
        session.write_file("notes.txt", "x")
        removed = session.run(_REMOVE_OWN_WORKDIR)
        assert removed.ok and removed.agent_fault is None and removed.stdout.strip() == "hello"
        session.reset_to_staged()
        again = session.run("print(1)")
        assert again.ok and again.stdout.strip() == "1"
        assert session.read_file("notes.txt") is None, "the program's own removal takes its files with it"


def test_a_replaced_working_directory_is_an_agent_fault_and_never_followed(tmp_path):
    host = tmp_path / "host"
    host.mkdir()
    (host / "keep.txt").write_text("host data")
    session = LocalSubprocessSandbox().open_session()
    result = session.run(_replace_own_workdir(host))
    assert result.agent_fault and "replaced" in result.agent_fault and result.error is None
    assert result.returncode == TAMPERED_WORKDIR_RETURNCODE and not result.ok
    assert session.run("print(1)").agent_fault
    assert sorted(os.listdir(host)) == ["keep.txt"], "a broken session never stages into the replaced path"
    session.reset_to_staged()
    assert (host / "keep.txt").read_text() == "host data", "reset must not delete through the link"
    for operation in (lambda: session.read_file("keep.txt"), session.list_files, lambda: session.write_file("a", "")):
        with pytest.raises(SandboxAgentFault):
            operation()
    session.close()
    assert not os.path.lexists(session.workdir), "close removes the link the program left in its place"
    assert (host / "keep.txt").read_text() == "host data"


def _completes_promptly(fn) -> bool:
    """Whether ``fn`` returns within the host-open bound (a daemon thread, so a hang cannot wedge pytest)."""
    worker = threading.Thread(target=fn, daemon=True)
    worker.start()
    worker.join(_HOST_OPEN_TIMEOUT_S)
    return not worker.is_alive()


# Takes back the owner's access to everything it wrote and to what the host staged.
_LOCK_THE_WORKDIR = (
    "import os\n"
    "os.makedirs('sub/deeper', exist_ok=True)\n"
    "open('sub/deeper/kept.txt', 'w').close()\n"
    "open('left.txt', 'w').close()\n"
    "os.chmod('sub/deeper/kept.txt', 0o000)\n"
    "os.chmod('sub/deeper', 0o500)\n"
    "os.chmod('sub', 0o000)\n"
    "os.chmod('main.py', 0o000)\n"
    "os.chmod('.', 0o500)\n"
    "print('wrong')\n"
)
# Grades the locking program, then drives one session through the two host steps that must give the
# owner its access back first: a reset (else the lock keeps the program's files) and the staging of the
# next run (else it cannot write its source).
_GRADE_AND_REPORT = f"""
import json, os, tempfile
from src.environments.envs.tasks.coding.grading import run_solution_against_tests
from src.environments.sandbox.local import LocalSubprocessSandbox
tempfile.tempdir = tempfile.mkdtemp()
tests = [{{"input": "", "output": "right"}}] * 3
sandbox = LocalSubprocessSandbox()
grade = run_solution_against_tests({_LOCK_THE_WORKDIR!r}, tests, sandbox=sandbox)
with sandbox.open_session() as session:
    session.run({_LOCK_THE_WORKDIR!r})
    session.reset_to_staged()
    after_reset = sorted(os.listdir(session.workdir))
    session.run({_LOCK_THE_WORKDIR!r})
    rerun = session.run("print(1)")
report = {{
    "uid": os.getuid(),
    "grade": grade._asdict(),
    "after_reset": after_reset,
    "rerun": [rerun.returncode, rerun.stdout.strip()],
    "left": os.listdir(tempfile.tempdir),
}}
print(json.dumps(report))
"""
# The uid a grader that is not root runs the program as, when the suite itself runs as root.
_UNPRIVILEGED_UID = "65534"


def _readable_by_others(root: pathlib.Path) -> bool:
    """Whether a uid that owns nothing on the host can import the toolkit from ``root``."""
    listable = stat.S_IROTH | stat.S_IXOTH
    return root.stat().st_mode & listable == listable and all(p.stat().st_mode & stat.S_IXOTH for p in root.parents)


def test_a_program_that_locks_its_workdir_gets_a_verdict_not_an_infra_error():
    """Graded by a user that is not root, which the program shares: every test is judged, its
    working directory is removed at the end, and nothing it locked reads as a grading outage."""
    command = [sys.executable, "-c", _GRADE_AND_REPORT]
    if os.geteuid() == 0:
        if shutil.which("setpriv") is None:
            pytest.skip("running as root without setpriv to drop to an unprivileged uid")
        if not _readable_by_others(REPO_ROOT):
            pytest.skip(f"uid {_UNPRIVILEGED_UID} cannot read the checkout at {REPO_ROOT}")
        command = [
            "setpriv",
            f"--reuid={_UNPRIVILEGED_UID}",
            f"--regid={_UNPRIVILEGED_UID}",
            "--clear-groups",
            *command,
        ]
    proc = subprocess.run(
        command, capture_output=True, text=True, cwd=REPO_ROOT, check=False, env={**os.environ, "HOME": "/tmp"}
    )
    assert proc.returncode == 0, proc.stderr
    report = json.loads(proc.stdout.splitlines()[-1])
    assert report["uid"] != 0
    grade = report["grade"]
    assert (grade["graded"], grade["infra_errors"], grade["passed"]) == (3, 0, 0), grade["details"]
    assert "Tests 1, 2, 3: FAIL" in grade["details"], "each test judges the program's own output"
    assert "tampered" not in grade["details"] and "ERROR --" not in grade["details"], grade["details"]
    assert report["after_reset"] == ["main.py"], "the reset removes what the program locked"
    assert report["rerun"] == [0, "1"], "the next run stages over the lock"
    assert report["left"] == [], "the session removes the working directory the program locked"


_NEST_PAST_THE_RECURSION_LIMIT = (
    "import os\nfor _ in range(1100):\n    os.mkdir('d')\n    os.chdir('d')\nprint('right')\n"
)


@pytest.mark.parametrize("backend", ["local", "bubblewrap"])
def test_a_tree_nested_past_the_recursion_limit_is_graded_and_removed(backend, tmp_path, monkeypatch):
    """The reset between tests and the removal after the grade walk the program's tree without
    recursion, so nesting it deep is judged like any program, never a host error that voids the grade."""
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))
    tests = [{"input": "", "output": "right"}, {"input": "", "output": "wrong"}]
    grade = run_solution_against_tests(_NEST_PAST_THE_RECURSION_LIMIT, tests, sandbox=_executor(backend))
    assert (grade.graded, grade.infra_errors, grade.passed) == (2, 0, 1), grade.details
    assert "Test 2: FAIL" in grade.details, grade.details
    assert os.listdir(tmp_path) == [], "the nested working directory is removed"


def test_closing_a_session_never_raises(tmp_path, monkeypatch):
    """What the removal cannot delete stays on disk; the episode's cleanup goes on."""

    def refused(path, *args, **kwargs):
        raise PermissionError(errno.EPERM, os.strerror(errno.EPERM), path)

    nested, replaced = LocalSubprocessSandbox().open_session(), LocalSubprocessSandbox().open_session()
    assert nested.run("import os\nos.makedirs('a/b')\nopen('a/b/f', 'w').close()").ok
    assert replaced.run(_replace_own_workdir(tmp_path)).agent_fault
    with monkeypatch.context() as patched:
        patched.setattr(os, "unlink", refused)
        patched.setattr(os, "rmdir", refused)
        nested.close()
        replaced.close()
    assert os.path.isfile(os.path.join(nested.workdir, "a", "b", "f")) and os.path.islink(replaced.workdir)
    nested.close()
    replaced.close()
    assert not os.path.lexists(nested.workdir) and not os.path.lexists(replaced.workdir)


def test_a_link_raced_in_after_the_type_check_is_left_alone(tmp_path, monkeypatch):
    """The walk checks an entry's type, then changes its mode without following a link: one swapped in
    between is refused and left alone, relative to a directory descriptor or not, never raised."""
    host = tmp_path / "host"
    host.mkdir()
    host.chmod(0o500)
    workdir = tmp_path / "work"
    workdir.mkdir()
    (workdir / "entry").symlink_to(host)
    real_stat = os.stat

    def a_directory_when_checked(path, *args, **kwargs):
        found = real_stat(path, *args, **kwargs)
        if os.path.basename(path) == "entry":
            return os.stat_result((stat.S_IFDIR | 0o500, *found[1:10]))
        return found

    dir_fd = os.open(workdir, os.O_RDONLY | os.O_DIRECTORY)
    try:
        with monkeypatch.context() as patched:
            patched.setattr(os, "stat", a_directory_when_checked)
            _grant_owner("entry", stat.S_IRWXU, dir_fd)
            _grant_owner(str(workdir / "entry"), stat.S_IRWXU)
    finally:
        os.close(dir_fd)
    assert host.stat().st_mode & 0o777 == 0o500


def test_a_staging_write_refused_after_the_restore_is_the_programs_runtime_error(monkeypatch):
    """A child that escaped the process group can lock a staged file again between the host's restore
    and its write: the refusal is the program's runtime error, never an infra error."""

    def locked(workdir, name, content):
        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), os.path.join(workdir, name))

    monkeypatch.setattr(LocalSubprocessSandbox, "_write_member", staticmethod(locked))
    tests = [{"input": "", "output": "1"}] * 2
    grade = run_solution_against_tests("print(1)", tests, sandbox=LocalSubprocessSandbox())
    assert (grade.graded, grade.infra_errors, grade.passed) == (2, 0, 0), grade.details
    assert "RUNTIME ERROR" in grade.details


def test_a_reset_refused_after_the_restore_leaves_the_entry(monkeypatch):
    def locked(path, *args, **kwargs):
        raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), path)

    with LocalSubprocessSandbox().open_session() as session:
        assert session.run("open('left.txt', 'w').close()").ok
        with monkeypatch.context() as patched:
            patched.setattr(os, "remove", locked)
            session.reset_to_staged()
        assert "left.txt" in session.list_files()


@pytest.mark.parametrize("denial", [errno.EPERM, errno.ENOSYS])
def test_a_host_that_refuses_pidfd_open_refuses_the_sandbox(denial, monkeypatch):
    """Every run waits on its program through a pidfd; refused, each would fail as a backend error and
    void its episode, so construction raises instead."""

    def refused(pid, flags=0):
        raise OSError(denial, os.strerror(denial))

    monkeypatch.setattr(os, "pidfd_open", refused)
    with pytest.raises(RuntimeError, match="pidfd_open"):
        LocalSubprocessSandbox()


@pytest.mark.parametrize("timeout", [0.0, -1.0, float("nan"), float("inf")])
def test_a_run_refuses_a_timeout_poll_cannot_bound(timeout):
    with pytest.raises(ValueError, match="timeout"):
        LocalSubprocessSandbox._run_in_new_session(["true"], stdin="", timeout=timeout, cwd="/", env={})


def test_the_group_is_killed_before_its_leader_is_reaped(monkeypatch):
    """Until it is reaped, the leader's zombie holds the process group's id; a kill after the reap
    could land on a group that reused it."""
    states = []
    real_killpg = os.killpg

    def recording(pgid, sig):
        with open(f"/proc/{pgid}/stat") as fh:
            record = fh.read()
        states.append(record[record.rindex(")") + 2])
        real_killpg(pgid, sig)

    monkeypatch.setattr(os, "killpg", recording)
    assert LocalSubprocessSandbox().run("print(1)").ok
    assert states == ["Z"]


def test_restoring_owner_access_reaches_nested_entries_and_never_follows_a_link(tmp_path):
    host = tmp_path / "host"
    host.mkdir()
    (host / "secret").write_text("host data")
    (host / "secret").chmod(0o400)
    workdir = tmp_path / "work"
    (workdir / "sub").mkdir(parents=True)
    (workdir / "sub" / "kept.txt").write_text("x")
    (workdir / "file_link").symlink_to(host / "secret")
    (workdir / "sub" / "dir_link").symlink_to(host)
    (workdir / "sub" / "kept.txt").chmod(0o000)
    (workdir / "sub").chmod(0o000)
    workdir.chmod(0o500)
    host.chmod(0o500)
    try:
        _restore_owner_access(str(workdir))
        assert workdir.stat().st_mode & 0o700 == 0o700
        assert (workdir / "sub").stat().st_mode & 0o700 == 0o700
        assert (workdir / "sub" / "kept.txt").stat().st_mode & 0o700 == 0o600
        assert host.stat().st_mode & 0o777 == 0o500, "a link to a directory is never followed"
        assert (host / "secret").stat().st_mode & 0o777 == 0o400, "a link to a file is never followed"
    finally:
        host.chmod(0o700)


def test_a_fifo_where_the_host_stages_or_reads_never_blocks_it():
    with LocalSubprocessSandbox().open_session() as session:
        assert session.run("import os\nos.remove('main.py')\nos.mkfifo('main.py')\nos.mkfifo('pipe')\n").ok
        results = {}
        assert _completes_promptly(lambda: results.update(staged=session.run("print(1)")))
        assert _completes_promptly(lambda: results.update(read=session.read_file("pipe")))
    staged = results["staged"]
    assert staged.error is None and staged.returncode == TAMPERED_WORKDIR_RETURNCODE, staged
    assert results["read"] is None


def test_a_lone_surrogate_in_stdin_is_replaced_not_a_host_hang():
    """A model-written stdin can carry a lone surrogate UTF-8 cannot encode: it reaches the program as
    ``?`` and never raises once the child has started, where the host would wait on it unbounded."""
    sandbox = LocalSubprocessSandbox()
    results = {}
    echo = "import sys\nprint(repr(sys.stdin.read()))"
    assert _completes_promptly(lambda: results.update(echo=sandbox.run(echo, stdin="a\ud83db")))
    sleeper = "import time\ntime.sleep(60)"
    assert _completes_promptly(lambda: results.update(slept=sandbox.run(sleeper, stdin="\ud83d", timeout=1)))
    assert results["echo"].stdout.strip() == "'a?b'"
    assert results["slept"].timed_out


def _live_group_members(pgid: int) -> list[int]:
    """Processes of group ``pgid`` still alive (a zombie awaiting its reaper is not)."""
    members = []
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as fh:
                state, _ppid, pgrp = fh.read().rsplit(")", 1)[1].split()[:3]
        except OSError:
            continue
        if int(pgrp) == pgid and state != "Z":
            members.append(int(entry))
    return members


def test_a_forked_child_the_program_leaves_behind_dies_with_the_run():
    """The run is judged on the leader's exit and output, and no process of its group outlives it."""
    program = "import os, time\nif os.fork() == 0:\n    time.sleep(60)\n    os._exit(0)\nprint(os.getpgrp())\n"
    start = time.monotonic()
    result = LocalSubprocessSandbox().run(program, timeout=30)
    assert result.ok and not result.timed_out
    assert time.monotonic() - start < 10, "judged on the leader's exit, not on the child it left behind"
    pgid = int(result.stdout)
    deadline = time.monotonic() + 5
    while _live_group_members(pgid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not _live_group_members(pgid), "a forked child outlived the run"


def test_a_run_waits_on_its_child_past_select_s_descriptor_limit():
    """A busy host (many sandbox slots, Ray) hands the run descriptors past ``FD_SETSIZE``."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    wanted = 2048
    if hard != resource.RLIM_INFINITY and hard < wanted:
        pytest.skip(f"RLIMIT_NOFILE hard limit {hard} is below {wanted}")
    resource.setrlimit(resource.RLIMIT_NOFILE, (max(soft, wanted), hard))
    held = [os.open(os.devnull, os.O_RDONLY) for _ in range(1100)]
    try:
        assert max(held) >= 1024
        result = LocalSubprocessSandbox().run("print(1)")
    finally:
        for fd in held:
            os.close(fd)
        resource.setrlimit(resource.RLIMIT_NOFILE, (soft, hard))
    assert result.ok and result.stdout.strip() == "1"


@pytest.mark.parametrize(
    ("expected", "produced"), [("1\n2\n", "1\r\n2\r\n"), ("1\r\n2\r\n", "1\n2\n"), ("1\r2", "1\n2")]
)
def test_crlf_and_lf_output_compare_equal_on_every_backend(expected, produced):
    """An exact comparison judges lines, not line endings: raw CRLF output (as a remote service returns
    it, or a CRLF test file) grades like LF."""
    raw = StubSandbox(SandboxResult(stdout=produced, returncode=0))
    assert run_solution_against_tests("code", [{"input": "", "output": expected}], sandbox=raw).passed == 1
    assert not run_solution_against_tests("code", [{"input": "", "output": "1 2"}], sandbox=raw).passed


def test_a_local_program_writing_crlf_passes_an_lf_test():
    program = "import sys\nsys.stdout.buffer.write(b'1\\r\\n2\\r\\n')\n"
    grade = run_solution_against_tests(program, [{"input": "", "output": "1\n2\n"}], sandbox=LocalSubprocessSandbox())
    assert grade.passed == 1, grade.details


def test_grading_judges_a_workdir_removing_program_instead_of_voiding_it():
    tests = [{"input": "", "output": "hello"}] * 3
    grade = run_solution_against_tests(_REMOVE_OWN_WORKDIR, tests, sandbox=LocalSubprocessSandbox())
    assert grade.infra_errors == 0, "the program's own act must never read as a grading outage"
    assert (grade.graded, grade.passed) == (3, 3), grade.details


def test_grading_rebuilds_a_compiled_program_whose_workdir_it_removed():
    if shutil.which("g++") is None:
        pytest.skip("g++ not installed")
    program = (
        "#include <filesystem>\n#include <iostream>\n"
        'int main() { std::cout << "hello"; std::filesystem::remove_all(std::filesystem::current_path()); }\n'
    )
    grade = run_solution_against_tests(
        program, [{"input": "", "output": "hello"}] * 3, sandbox=LocalSubprocessSandbox(), language="cpp"
    )
    assert grade.infra_errors == 0 and grade.passed == 3, grade.details


def test_repl_path_raises_the_agent_fault_for_a_workdir_replacing_program(tmp_path):
    with LocalSubprocessSandbox().open_session() as session, pytest.raises(SandboxAgentFault):
        format_sandbox_repl_output(session.run(_replace_own_workdir(tmp_path)), timeout=5)


def _bash(command: str) -> dict:
    return {"id": "c1", "function": {"name": "run_bash_command", "arguments": json.dumps({"command": command})}}


def test_a_swe_command_that_replaces_its_workspace_fails_the_episode(tmp_path):
    env = SweEnvironment(sandbox_backend="local", **_knobs())
    replace = f'd="$PWD"; cd /; rm -rf "$d"; ln -s {tmp_path} "$d"'
    try:
        ids, _ = env.reset(["fix the bug"], [{"answer": "done"}])
        step = env.step(ids, ["breaking"], [{"answer": "done", "tool_calls": [_bash(replace)]}])[0]
    finally:
        env.close()
    traj = step.trajectory
    assert step.done and not traj.info["completed"] and not traj.episode_invalid
    assert "replaced its working directory" in traj.info["tool_results"][-1]["content"]
    assert traj.total_reward == pytest.approx(-_TOOL_ERROR_PENALTY)


def test_a_swe_command_that_removes_its_workspace_just_loses_its_files():
    env = SweEnvironment(sandbox_backend="local", **_knobs())
    note = {"id": "c0", "function": {"name": "write_file", "arguments": json.dumps({"path": "n.txt", "content": "x"})}}
    try:
        ids, _ = env.reset(["fix the bug"], [{"answer": "done"}])
        env.step(ids, ["noting"], [{"answer": "done", "tool_calls": [note]}])
        step = env.step(ids, ["cleaning"], [{"answer": "done", "tool_calls": [_bash('rm -rf "$PWD"')]}])[0]
        assert not step.done and SANDBOX_FAULT_KEY not in step.trajectory.info
        check = _bash("test -e n.txt && echo kept || echo gone")
        step = env.step(ids, ["checking"], [{"answer": "done", "tool_calls": [check]}])[0]
    finally:
        env.close()
    assert not step.done and step.trajectory.info["tool_results"][-1]["content"].strip() == "gone"


def test_a_missing_session_is_an_infra_fault():
    handler = create_session_code_tools(lambda: None).get("run_python").handler
    with pytest.raises(SandboxInfraError):
        handler(code="print(1)")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
