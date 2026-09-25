"""Code-contests episode drivers for CPU tests: a canned-result sandbox and the reset / tool-call steps,
so a test runs episodes through the native protocol's own dispatch without a subprocess."""

from typing import Any

from src.environments.base import Trajectory
from src.environments.sandbox.base import SandboxExecutor, SandboxResult
from src.environments.tools.definitions import NativeToolCall

# One hidden test that the default stub run's ``X`` passes, as the env's ``answer`` context.
SINGLE_TEST_ANSWER = {"answer": {"tests": [{"input": "", "output": "X"}]}}


class StubSandbox(SandboxExecutor):
    """Always returns the same canned result; by default a clean run printing ``X``."""

    def __init__(self, result: SandboxResult | None = None):
        self._result = result or SandboxResult(stdout="X\n", returncode=0)

    def open_session(self):  # pragma: no cover
        raise NotImplementedError

    def run(self, code, *, stdin="", timeout=15.0, language="python", files=None):
        return self._result


def reset_episode(env: Any, context: dict[str, Any]) -> Trajectory:
    """Reset one episode on ``env`` with ``context`` and return its trajectory."""
    ids, _ = env.reset(["solve it"], [context])
    return env.get_trajectories(ids)[0]


def call_tool(env: Any, trajectory: Trajectory, name: str, code: str = "print('X')") -> str:
    """Dispatch one ``name`` tool call through the protocol; returns the observation text."""
    results, _ = env._execute_tool_calls([NativeToolCall(id="c", name=name, arguments={"code": code})], trajectory)
    return results[0].content
