"""Tool-call accounting: per-episode budgets and counters, a call's admission before it runs, and its price after,
shared by every protocol that dispatches registry tools."""

import logging
from typing import Any

from halo.config_validation import require_finite, require_int
from halo.environments.episode_keys import (
    EPISODE_INVALID_KEY,
    EPISODE_INVALID_REASON_KEY,
    SANDBOX_FAULT_AGENT,
    SANDBOX_FAULT_INFRA,
    SANDBOX_FAULT_KEY,
)
from halo.environments.messages import Trajectory, truncate_text
from halo.environments.sandbox.results import SandboxFault, SandboxInfraError
from halo.environments.tools.definitions import (
    TOOL_CALL_REFUSALS,
    NativeTool,
    NativeToolRegistry,
    ToolBudgetExhausted,
    ToolDisabled,
)

logger = logging.getLogger(__name__)

# The episode's per-tool call caps (stamped at reset; an unlisted tool is uncapped) and its admitted calls per tool.
EPISODE_TOOL_BUDGETS_KEY = "episode_tool_budgets"
TOOL_CALL_COUNTS_KEY = "tool_call_counts"
# The episode's executed and successful calls, and what its successful ones were paid against ``tool_reward_cap``.
TOTAL_TOOL_CALLS_KEY = "total_tool_calls"
_SUCCESSFUL_TOOL_CALLS_KEY = "successful_tool_calls"
TOOL_REWARD_PAID_KEY = "tool_reward_paid"
# Calls a handler returned to the budget that the accounting has yet to book (:meth:`_refund_tool_call`).
_REFUNDED_TOOL_CALLS_KEY = "_refunded_tool_calls"
# The minus sign is applied where it is charged, so a negative value would farm a penalty as a bonus.
MAGNITUDE_WHY = " (a magnitude: the sign is applied where it is charged)"


def validate_tool_budgets(budgets: dict[str, int] | None, registry: NativeToolRegistry) -> dict[str, int]:
    """``0`` disables the tool for the episode."""
    validated: dict[str, int] = {}
    for name, cap in (budgets or {}).items():
        if registry.get(name) is None:
            raise ValueError(
                f"tool_budgets names {name!r}, which is not a registered tool: {sorted(registry.names())}"
            )
        require_int("tool_budgets", ge=0, **{repr(name): cap})
        validated[name] = cap
    return validated


def initial_tool_counters(budgets: dict[str, int]) -> dict[str, Any]:
    return {
        TOTAL_TOOL_CALLS_KEY: 0,
        _SUCCESSFUL_TOOL_CALLS_KEY: 0,
        TOOL_CALL_COUNTS_KEY: {},
        EPISODE_TOOL_BUDGETS_KEY: dict(budgets),
    }


def tool_call_outcome(name: str, outcome: str | Exception) -> tuple[str, bool, SandboxFault | None]:
    """Any exception but a refusal or a sandbox fault is a broken tool, logged with its traceback."""
    if not isinstance(outcome, Exception):
        return str(outcome), True, None
    fault = outcome if isinstance(outcome, SandboxFault) else None
    if isinstance(outcome, TOOL_CALL_REFUSALS):
        logger.debug("Tool %r refused the call: %s", name, outcome)
    elif fault is None:
        logger.warning("Tool %r raised during execution", name, exc_info=outcome)
    return f"Error: {outcome}", False, fault


class ToolCallAccountingMixin:
    """The counters live on each episode's trajectory, so one instance serves concurrent episodes."""

    # Per-call pricing when the config names none; a task env that departs declares its own (code contests: 0).
    DEFAULT_TOOL_SUCCESS_REWARD: float = 0.05
    DEFAULT_TOOL_ERROR_PENALTY: float = 0.1

    # The calls a turn may make; ``_add_action_message`` trims the stored ``tool_calls`` to it. ``None`` = no cap.
    max_tool_calls_per_turn: int | None = None

    def _init_tool_pricing(
        self, tool_success_reward: float | None, tool_error_penalty: float | None, tool_reward_cap: float | None
    ) -> None:
        """``tool_reward_cap`` bounds an episode's total pay; ``None`` means one paid call per turn of ``max_turns``,
        so call spam cannot out-earn the objective."""
        if tool_success_reward is None:
            tool_success_reward = self.DEFAULT_TOOL_SUCCESS_REWARD
        if tool_error_penalty is None:
            tool_error_penalty = self.DEFAULT_TOOL_ERROR_PENALTY
        if tool_reward_cap is None:
            tool_reward_cap = tool_success_reward * self.max_turns
        require_finite(
            type(self).__name__,
            ge=0,
            why=MAGNITUDE_WHY,
            tool_success_reward=tool_success_reward,
            tool_error_penalty=tool_error_penalty,
            tool_reward_cap=tool_reward_cap,
        )
        self.tool_success_reward = tool_success_reward
        self.tool_error_penalty = tool_error_penalty
        self.tool_reward_cap = tool_reward_cap

    def _admit_tool_call(
        self, tool: NativeTool, arguments: dict[str, Any], trajectory: Trajectory, *, for_async: bool = False
    ) -> dict[str, Any]:
        """Bind the arguments, then spend one of the episode's calls. A refusal counts nothing; it runs before any
        await, so concurrent calls in one turn cannot both pass a one-call cap."""
        bound = tool.bind(arguments, for_async=for_async)
        cap = self._exhausted_tool_cap(trajectory, tool.name)
        if cap is not None:
            raise (ToolDisabled if cap == 0 else ToolBudgetExhausted)(tool.budget_exhausted_message())
        self._count_tool_call(trajectory, tool.name)
        return bound

    @staticmethod
    def _tool_calls_made(trajectory: Trajectory, name: str) -> int:
        return trajectory.info.get(TOOL_CALL_COUNTS_KEY, {}).get(name, 0)

    @classmethod
    def _exhausted_tool_cap(cls, trajectory: Trajectory, name: str) -> int | None:
        """The episode cap another call would exceed, or ``None`` while within budget."""
        cap = trajectory.info.get(EPISODE_TOOL_BUDGETS_KEY, {}).get(name)
        if cap is not None and cls._tool_calls_made(trajectory, name) >= cap:
            return cap
        return None

    @staticmethod
    def _count_tool_call(trajectory: Trajectory, name: str) -> int:
        counts = trajectory.info.setdefault(TOOL_CALL_COUNTS_KEY, {})
        counts[name] = counts.get(name, 0) + 1
        return counts[name]

    @staticmethod
    def _uncount_tool_call(trajectory: Trajectory, name: str) -> None:
        """For a handler that refuses the call as a tool error after admission."""
        counts = trajectory.info.setdefault(TOOL_CALL_COUNTS_KEY, {})
        counts[name] = max(0, counts.get(name, 0) - 1)

    @classmethod
    def _refund_tool_call(cls, trajectory: Trajectory, name: str) -> None:
        """For a handler that found the call spent nothing and replies as a success; the accounting then books it as
        neither paid nor successful."""
        cls._uncount_tool_call(trajectory, name)
        trajectory.info[_REFUNDED_TOOL_CALLS_KEY] = trajectory.info.get(_REFUNDED_TOOL_CALLS_KEY, 0) + 1

    def _book_tool_call(
        self,
        trajectory: Trajectory,
        tool: str,
        success: bool,
        fault: SandboxFault | None = None,
    ) -> float:
        """The one entry every protocol books a call through."""
        if fault is not None:
            return self._book_sandbox_fault(trajectory, tool, fault)
        return self._credit_tool_call(trajectory, success)

    def _credit_tool_call(self, trajectory: Trajectory, success: bool) -> float:
        trajectory.info[TOTAL_TOOL_CALLS_KEY] += 1
        if not success:
            return -self.tool_error_penalty
        refunded = trajectory.info.get(_REFUNDED_TOOL_CALLS_KEY, 0)
        if refunded:
            # A refunded call returns its reply as a success; within one turn's batch every one does, so
            # booking the next successes unpaid books exactly the refunded ones.
            trajectory.info[_REFUNDED_TOOL_CALLS_KEY] = refunded - 1
            return 0.0
        trajectory.info[_SUCCESSFUL_TOOL_CALLS_KEY] += 1
        paid = trajectory.info.get(TOOL_REWARD_PAID_KEY, 0.0)
        credit = max(0.0, min(self.tool_success_reward, self.tool_reward_cap - paid))
        trajectory.info[TOOL_REWARD_PAID_KEY] = paid + credit
        return credit

    def _book_sandbox_fault(self, trajectory: Trajectory, tool: str, fault: SandboxFault) -> float:
        """The step then ends the episode, uncompleted. An infrastructure fault says nothing about the policy:
        unpriced, and the episode leaves the baseline. An agent-caused one is a failed call; the episode stays."""
        if isinstance(fault, SandboxInfraError):
            logger.warning("Tool %r lost to a sandbox infrastructure fault; the episode is dropped: %s", tool, fault)
            trajectory.info[TOTAL_TOOL_CALLS_KEY] += 1
            trajectory.info[SANDBOX_FAULT_KEY] = SANDBOX_FAULT_INFRA
            trajectory.info[EPISODE_INVALID_KEY] = True
            trajectory.info.setdefault(
                EPISODE_INVALID_REASON_KEY, f"sandbox infrastructure fault in tool {tool!r}: {fault}"
            )
            return 0.0
        logger.debug("Tool %r ended the episode on an agent-caused sandbox fault: %s", tool, fault)
        trajectory.info.setdefault(SANDBOX_FAULT_KEY, SANDBOX_FAULT_AGENT)
        return self._credit_tool_call(trajectory, False)

    def _truncate_observation(self, content: str, limit: int | None = None) -> str:
        """``0`` caps nothing. Capping at the source keeps the trajectory small and rollout and recompute identical."""
        limit = self.max_observation_chars if limit is None else limit
        return truncate_text(content, limit) if limit else content
