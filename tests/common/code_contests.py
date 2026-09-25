"""Code-contests episode drivers for CPU tests: a canned-result sandbox, a recording stand-in for the
SandboxFusion service, and the reset / tool-call steps, so a test runs episodes through the native
protocol's own dispatch without a subprocess or a network."""

from typing import Any, NamedTuple

from src.environments.base import Trajectory
from src.environments.sandbox.base import SandboxExecutor, SandboxResult
from src.environments.tools.definitions import NativeToolCall

# One hidden test that the default stub run's ``X`` passes, as the env's ``answer`` context.
SINGLE_TEST_ANSWER = {"answer": {"tests": [{"input": "", "output": "X"}]}}
# The body SandboxFusion answers for a program that ran to a clean exit.
FINISHED_RUN = {"status": "Success", "run_result": {"status": "Finished", "stdout": "", "return_code": 0}}


class StubSandbox(SandboxExecutor):
    """Always returns the same canned result; by default a clean run printing ``X``."""

    # It runs nothing, so nothing escapes it.
    isolated = True

    def __init__(self, result: SandboxResult | None = None):
        self._result = result or SandboxResult(stdout="X\n", returncode=0)

    def open_session(self):  # pragma: no cover
        raise NotImplementedError

    def run(self, code, *, stdin="", timeout=15.0, language="python", files=None):
        return self._result


class SandboxPost(NamedTuple):
    """One ``/run_code`` request a :class:`RecordingSandboxSession` received."""

    url: str
    payload: dict[str, Any]
    timeout: float | None


class RecordingSandboxSession:
    """A ``requests`` session standing in for a SandboxFusion service (``RemoteSandbox(session=...)``):
    it records every POST and answers each with one canned body, a clean run by default, or raises
    ``exc`` as a failed transport does."""

    def __init__(self, body: dict[str, Any] | None = None, *, exc: Exception | None = None):
        self.body = FINISHED_RUN if body is None else body
        self.exc = exc
        self.posts: list[SandboxPost] = []

    def post(self, url: str, json: dict[str, Any] | None = None, timeout: float | None = None):
        self.posts.append(SandboxPost(url, json, timeout))
        if self.exc is not None:
            raise self.exc
        return self

    def raise_for_status(self) -> None:
        """The canned answer is always a 200."""

    def json(self) -> dict[str, Any]:
        return self.body


def reset_episode(env: Any, context: dict[str, Any]) -> Trajectory:
    """Reset one episode on ``env`` with ``context`` and return its trajectory."""
    ids, _ = env.reset(["solve it"], [context])
    return env.get_trajectories(ids)[0]


def call_tool(env: Any, trajectory: Trajectory, name: str, code: str = "print('X')") -> str:
    """Dispatch one ``name`` tool call through the protocol; returns the observation text."""
    results, _ = env._execute_tool_calls([NativeToolCall(id="c", name=name, arguments={"code": code})], trajectory)
    return results[0].content
