"""SWE-agent style environment: each episode gets its own persistent :class:`SandboxSession`, bound through a
``ContextVar`` so one env instance serves concurrent rollouts. Completing an episode alone earns nothing."""

import logging
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from halo.config_validation import present, require_callable
from halo.environments.episode_keys import ANSWER_KEY
from halo.environments.episode_scoring import EpisodeGrade
from halo.environments.messages import Trajectory
from halo.environments.protocols.native_tool_use import NativeToolUseEnvironment
from halo.environments.sandbox.base import SANDBOX_DEFAULT_TIMEOUT, SandboxExecutor, SandboxSession
from halo.environments.sandbox.factory import build_sandbox, warn_if_unisolated
from halo.environments.tools.definitions import NativeToolRegistry
from halo.environments.tools.toolsets import (
    build_session_bash_tools,
    build_session_code_tools,
    build_session_file_tools,
)

logger = logging.getLogger(__name__)

# Per-episode, so concurrent episodes on one env instance each operate on their own workspace.
_ACTIVE_SESSION: ContextVar = ContextVar("swe_env_active_session", default=None)


class SweEnvironment(NativeToolUseEnvironment):
    """Graded by ``test_function`` when given, else against the row's ``validator`` or ``answer`` (``requires_answer``
    then defaults on); ``requires_answer=False`` without a ``test_function`` is refused while the reward prices the
    environment's grade."""

    # An agentic edit-run-test loop needs more turns than the protocol's generic budget.
    DEFAULT_MAX_TURNS = 20

    SWE_SYSTEM_PROMPT = """You are a skilled software engineer. You have access to tools for reading, writing, and executing code, plus a bash shell in the same workspace.

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
        require_callable(type(self).__name__, **present(test_function=test_function))
        self.sandbox = sandbox or build_sandbox(backend=sandbox_backend, url=sandbox_url)
        warn_if_unisolated(self.sandbox, type(self).__name__)
        self._sessions: dict[int, SandboxSession] = {}

        # Session tools resolve the active episode's session per call, hence the getter not a session.
        registry = NativeToolRegistry.combine(
            build_session_file_tools(_ACTIVE_SESSION.get),
            build_session_code_tools(
                _ACTIVE_SESSION.get, language=language, timeout=code_timeout, tool_name="run_code"
            ),
            build_session_bash_tools(_ACTIVE_SESSION.get, timeout=code_timeout),
        )
        if extra_tools:
            registry.merge(extra_tools)

        super().__init__(
            tool_registry=registry,
            system_prompt=system_prompt or self.SWE_SYSTEM_PROMPT,
            **kwargs,
        )
        self.test_function = test_function
        if test_function is None and self.objective_term is not None:
            if kwargs.get("requires_answer") is False:
                raise ValueError(
                    f"{type(self).__name__} has nothing to grade against: requires_answer is false and no "
                    f"test_function is set. Grade against the dataset's answer column (requires_answer), "
                    f"pass a test_function, or price the episode with an external reward term only."
                )
            self.requires_answer = True

    @contextmanager
    def _episode_binding(self, trajectory: Trajectory) -> Iterator[None]:
        with super()._episode_binding(trajectory):
            token = _ACTIVE_SESSION.set(self._session_for(trajectory))
            try:
                yield
            finally:
                _ACTIVE_SESSION.reset(token)

    def _session_for(self, trajectory: Trajectory) -> SandboxSession:
        episode_id = trajectory.info["episode_id"]
        session = self._sessions.get(episode_id)
        if session is None:
            session = self.sandbox.open_session()
            self._sessions[episode_id] = session
        return session

    def _release_episode(self, episode_id: int) -> None:
        session = self._sessions.pop(episode_id, None)
        if session is not None:
            session.close()

    def close(self) -> None:
        for session in self._sessions.values():
            session.close()
        self._sessions.clear()
        super().close()

    def _grade_episode(self, trajectory: Trajectory, context: dict[str, Any] | None = None) -> EpisodeGrade:
        """A row with no grader raises, since running a command proves nothing. A reward without an ``environment``
        term takes no grade, which could only void the episode for a verdict no term reads."""
        if self.objective_term is None or not trajectory.info.get("completed"):
            return EpisodeGrade(0.0)

        if self.test_function:
            try:
                return EpisodeGrade(1.0 if self.test_function(trajectory) else 0.0)
            except Exception as exc:
                logger.warning("SWE test_function raised; scoring the episode invalid", exc_info=True)
                return self._invalid_grade(trajectory, f"test_function raised {type(exc).__name__}: {exc}")

        ctx = context or trajectory.info.get("context") or {}
        # Key presence, not value: the protocol marks a null ``answer`` cell invalid instead of paying it.
        if callable(ctx.get("validator")) or ANSWER_KEY in ctx:
            return super()._grade_episode(trajectory, context)
        raise ValueError(
            f"{type(self).__name__} has nothing to grade this episode against: no test_function, and the "
            f"row carries no {ANSWER_KEY!r} or 'validator' (the trainer refuses such a dataset under "
            f"requires_answer; this driver passed one)."
        )
