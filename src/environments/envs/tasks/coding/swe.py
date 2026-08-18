"""SWE-agent style environment for software engineering tasks.

Multi-turn and stateful: each episode gets its own persistent :class:`SandboxSession`, so files written
on one turn are visible on the next. Tools bind to the active episode's session via a ``ContextVar``,
so one env instance can serve concurrent rollouts.
"""

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from src.environments.base import EPISODE_INVALID_KEY, Trajectory
from src.environments.envs.protocols.native import NativeToolUseEnvironment
from src.environments.sandbox.base import SANDBOX_DEFAULT_TIMEOUT, SandboxExecutor, SandboxSession
from src.environments.sandbox.resolve import resolve_sandbox
from src.environments.tools.definitions import NativeToolRegistry
from src.environments.tools.factories import create_session_code_tools, create_session_file_tools

logger = logging.getLogger(__name__)

# Per-episode, so concurrent episodes on one env instance each operate on their own workspace.
_ACTIVE_SESSION: ContextVar = ContextVar("swe_env_active_session", default=None)


class SweEnvironment(NativeToolUseEnvironment):
    """Code-execution / software-engineering environment.

    Persistent workspace tools (read/write/list files) and a code-execution tool, all backed by a
    per-episode :class:`SandboxSession` so state carries across turns.
    """

    # An edit/run/test loop needs more turns than the protocol's generic budget allows.
    DEFAULT_MAX_TURNS = 20

    SWE_SYSTEM_PROMPT = """You are a skilled software engineer. You have access to tools for reading, writing, and executing code.

Use the available tools to solve the task. When you're done, provide your final answer.

Tips:
- Read files before modifying them
- Test your code with run_code before submitting
- Break complex tasks into smaller steps"""

    def __init__(
        self,
        extra_tools: NativeToolRegistry | None = None,
        system_prompt: str | None = None,
        test_function: Callable[..., bool] | None = None,
        language: str = "python",
        code_timeout: float = SANDBOX_DEFAULT_TIMEOUT,
        sandbox: SandboxExecutor | None = None,
        sandbox_backend: str | None = None,
        sandbox_url: str | None = None,
        **kwargs,
    ):
        self.sandbox = sandbox or resolve_sandbox(backend=sandbox_backend, url=sandbox_url)
        self._sessions: dict[int, SandboxSession] = {}

        # Session tools resolve the active episode's session per call, hence the getter not a session.
        registry = NativeToolRegistry.combine(
            create_session_file_tools(_ACTIVE_SESSION.get),
            create_session_code_tools(
                _ACTIVE_SESSION.get, language=language, timeout=code_timeout, tool_name="run_code"
            ),
        )
        if extra_tools:
            registry.merge(extra_tools)

        super().__init__(
            tool_registry=registry,
            system_prompt=system_prompt or self.SWE_SYSTEM_PROMPT,
            **kwargs,
        )
        self.test_function = test_function

    def _session_for(self, trajectory: Trajectory) -> SandboxSession:
        """Return (creating on first use) the persistent session for this episode."""
        episode_id = trajectory.info["episode_id"]
        session = self._sessions.get(episode_id)
        if session is None:
            session = self.sandbox.open_session()
            self._sessions[episode_id] = session
        return session

    @contextmanager
    def _episode_binding(self, trajectory: Trajectory) -> Iterator[None]:
        """Bind the episode's persistent session so file/code tools operate on that workspace."""
        token = _ACTIVE_SESSION.set(self._session_for(trajectory))
        try:
            yield
        finally:
            _ACTIVE_SESSION.reset(token)

    def _release_episode(self, episode_id: int) -> None:
        """Close the episode's persistent session, freeing its temp dir."""
        session = self._sessions.pop(episode_id, None)
        if session is not None:
            session.close()

    def close(self) -> None:
        """Close every outstanding session on environment teardown."""
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()
        super().close()

    def _compute_reward(
        self,
        trajectory: Trajectory,
        context: dict[str, Any] | None = None,
    ) -> float:
        """Compute reward with optional test function validation."""
        base_reward = self._shaped_base_reward(trajectory)

        if not trajectory.info.get("completed"):
            return base_reward + self.failure_reward

        if self.test_function:
            try:
                if self.test_function(trajectory):
                    return base_reward + self.success_reward
                else:
                    return base_reward + self.failure_reward
            except Exception:
                # A grader that itself errors scores every solution 0, so log it.
                logger.warning("SWE test_function raised an exception; scoring as failure", exc_info=True)
                # The forced failure says nothing about the completion, so the row is dropped from the
                # GRPO group baseline rather than biasing every sibling's advantage.
                trajectory.info[EPISODE_INVALID_KEY] = True
                return base_reward + self.failure_reward

        ctx = context or trajectory.info.get("context") or {}
        if ctx.get("validator") or ctx.get("answer") is not None:
            return super()._compute_reward(trajectory, context)

        # Ungraded: a completion with no successful tool call takes the full failure_reward.
        if trajectory.info.get("successful_tool_calls", 0) > 0:
            return base_reward + self.success_reward
        return base_reward + self.failure_reward
