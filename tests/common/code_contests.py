"""Code-contests episode drivers for CPU tests, running episodes through the native protocol's dispatch with no
subprocess or network."""

import re
from typing import Any, NamedTuple

from halo.environments.episodes.messages import Trajectory
from halo.environments.sandbox.base import SandboxExecutor
from halo.environments.sandbox.results import SandboxResult
from halo.environments.tools.definitions import NativeToolCall

# One hidden test the default stub run's ``X`` passes, as the env's ``answer`` context.
SINGLE_TEST_ANSWER = {"answer": {"tests": [{"input": "", "output": "X"}]}}
# What SandboxFusion answers for a program that ran to a clean exit.
FINISHED_RUN = {"status": "Success", "run_result": {"status": "Finished", "stdout": "", "return_code": 0}}
# Wording no task message, tool description or tool reply may carry: budgets are enforced, never stated to the model.
RETIRED_BUDGET_PHRASES = (
    "Budgets for this task",
    "runs left",
    "submissions left",
    "limit reached",
    "stated in the task message",
    "You get up to",
    "only graded submission",
    "disabled for this task",
)
# The window around each "submission"/"run" word, where a stated budget would put its count.
_BUDGET_WORD_WINDOW = re.compile(r"[^.\n]{0,24}\b(?:graded\s+)?(?:submissions?|runs?)\b[^.\n]{0,24}", re.IGNORECASE)


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
    url: str
    payload: dict[str, Any]
    timeout: float | None


class RecordingSandboxSession:
    """A ``requests`` session standing in for SandboxFusion: records every POST and answers one canned body, or raises
    ``exc``."""

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


def retired_budget_phrases(text: str) -> list[str]:
    return [phrase for phrase in RETIRED_BUDGET_PHRASES if phrase in text]


def counts_beside_budget_words(text: str) -> list[str]:
    """Windows of ``text`` with a count beside "submission" or "run"; not for a task message, whose time-limit note is
    one."""
    return [window.group(0) for window in _BUDGET_WORD_WINDOW.finditer(text) if re.search(r"\d", window.group(0))]


def reset_episode(env: Any, context: dict[str, Any]) -> Trajectory:
    ids, _ = env.reset(["solve it"], [context])
    return env.get_trajectories(ids)[0]


def call_tool(env: Any, trajectory: Trajectory, name: str, code: str = "print('X')") -> str:
    """Dispatch one ``name`` tool call through the protocol; returns the observation text."""
    results, _ = env._execute_tool_calls([NativeToolCall(id="c", name=name, arguments={"code": code})], trajectory)
    return results[0].content
