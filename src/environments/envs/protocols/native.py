"""Native tool-use environments over OpenAI-format tool calling, which both rollout engines parse (sync + async)."""

import asyncio
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from src.environments.base import (
    ANSWER_KEY,
    CUT_IN_TOOL_CALL_KEY,
    EPISODE_ERROR_KEY,
    EPISODE_TOOL_BUDGETS_KEY,
    TOOL_CALL_COUNTS_KEY,
    AsyncBaseEnvironment,
    BaseEnvironment,
    EpisodeGrade,
    Trajectory,
    require_magnitudes,
)
from src.environments.sandbox.base import SANDBOX_FAULTS, SandboxAgentFault, SandboxInfraError
from src.environments.tools.definitions import (
    NativeTool,
    NativeToolCall,
    NativeToolRegistry,
    NativeToolResult,
    ToolArgumentError,
    ToolBudgetExhausted,
)
from src.inference.response import ENGINE_CUT_FINISH_REASONS
from src.rewards.matching import validate_answer

logger = logging.getLogger(__name__)

# The episode whose tool batch is executing, per asyncio Task / thread: one env instance serves
# concurrent rollouts, so a handler reaching for per-episode state (the tests it grades against, a
# workspace) reads it from here, never from an instance attribute.
_ACTIVE_TRAJECTORY: ContextVar[Trajectory | None] = ContextVar("native_tool_use_active_trajectory", default=None)


def validate_tool_budgets(budgets: dict[str, int] | None, registry: NativeToolRegistry) -> dict[str, int]:
    """Per-tool episode caps checked against the registry: every capped tool must exist and a cap is a
    non-negative int (``0`` disables the tool for the episode)."""
    validated: dict[str, int] = {}
    for name, cap in (budgets or {}).items():
        if registry.get(name) is None:
            raise ValueError(
                f"tool_budgets names {name!r}, which is not a registered tool: {sorted(registry.names())}"
            )
        if isinstance(cap, bool) or not isinstance(cap, int) or cap < 0:
            raise ValueError(f"tool_budgets[{name!r}] must be an int >= 0, got {cap!r}")
        validated[name] = cap
    return validated


def admit_tool_call(
    env: BaseEnvironment,
    tool: NativeTool,
    arguments: dict[str, Any],
    trajectory: Trajectory,
    *,
    for_async: bool = False,
) -> dict[str, Any]:
    """Admit one call before it runs, under either protocol: bind its arguments (against the handler
    that will run), then spend one of the episode's calls on the tool. Refuses
    (:class:`ToolArgumentError`, :class:`ToolBudgetExhausted`) without counting, so a call the handler
    could never run does not consume the budget; runs synchronously before any await so concurrent
    calls in one turn cannot both pass a one-call cap."""
    bound = tool.bind(arguments, for_async=for_async)
    cap = env._tool_budget_exhausted(trajectory, tool.name)
    if cap is not None:
        raise ToolBudgetExhausted(tool.budget_exhausted_message(cap, env._tool_budgets_left(trajectory)))
    env._count_tool_call(trajectory, tool.name)
    return bound


def tool_call_outcome(
    name: str, outcome: str | Exception
) -> tuple[str, bool, SandboxInfraError | SandboxAgentFault | None]:
    """A call's observation, whether it succeeded, and the sandbox fault it ended on, from what tool
    ``name`` returned or raised (admission included), under either protocol.

    A refusal (:class:`ToolBudgetExhausted`, :class:`ToolArgumentError`) is expected control flow,
    logged without a traceback: an env with a 2-submission cap in a 15-turn episode refuses by design.
    A sandbox fault is booked and logged by type in the accounting. Any other exception is a broken
    tool, logged with its traceback: the graded tools run here too, and a submit handler that dies on
    a malformed payload would otherwise grade 0 with nothing anywhere saying why.
    """
    if not isinstance(outcome, Exception):
        return outcome, True, None
    fault = outcome if isinstance(outcome, SANDBOX_FAULTS) else None
    if isinstance(outcome, ToolBudgetExhausted | ToolArgumentError):
        logger.debug("Tool %r refused the call: %s", name, outcome)
    elif fault is None:
        logger.warning("Tool %r raised during execution", name, exc_info=outcome)
    return f"Error: {outcome}", False, fault


class NativeToolUseEnvironment(BaseEnvironment):
    """Environment using native OpenAI-format tool calling."""

    SHAPING_COMPONENTS = ("tool_shaping",)

    # State the fact and ask for the action — never for shorter reasoning. These texts are trained on
    # wherever a recovery succeeds, so any instruction here becomes a GLOBAL lesson, learned far
    # outside the situation it was written for.
    LENGTH_CUTOFF_NUDGE = (
        "Your previous turn was cut off before you made a tool call, so nothing was recorded. Make "
        "your tool call now with the best solution you have."
    )
    # Sent only on a token-cap cut inside a call, so it may name the length limit; the action it asks for
    # is where the reasoning goes, not less of it.
    LENGTH_CUTOFF_IN_CALL_NUDGE = (
        "Your previous turn reached its length limit while writing a tool call, so the call was not run and "
        "nothing was recorded. Make the call again, keeping your reasoning out of its arguments."
    )
    EMPTY_TURN_NUDGE = (
        "Your previous turn ended without a tool call or an answer, so nothing was recorded. Make "
        "your tool call now, or give your final answer, with the best solution you have."
    )

    def __init__(
        self,
        tool_registry: NativeToolRegistry,
        system_prompt: str | None = None,
        max_tool_calls_per_turn: int = 5,
        no_tool_use_penalty: float = 0.0,
        turn_overflow_penalty: float = 0.0,
        length_cutoff_penalty: float = 0.0,
        tool_budgets: dict[str, int] | None = None,
        **kwargs,
    ):
        """``tool_budgets`` caps calls per tool per episode (``{tool_name: cap}``; an unlisted tool is
        uncapped): a call past its cap is refused as a tool error with the tool's ``budget_message``
        and never runs. Stamped into every episode at reset; a subclass may tighten the stamp per
        episode (:meth:`~src.environments.base.BaseEnvironment._apply_effort_profile`). The per-call
        knobs (``tool_success_reward``, ``tool_error_penalty``, ``tool_reward_cap``) are the base's."""
        super().__init__(**kwargs)

        require_magnitudes(
            no_tool_use_penalty=no_tool_use_penalty,
            turn_overflow_penalty=turn_overflow_penalty,
            length_cutoff_penalty=length_cutoff_penalty,
        )

        self.registry = tool_registry
        self.tool_budgets = validate_tool_budgets(tool_budgets, tool_registry)
        self.system_prompt = system_prompt
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self.no_tool_use_penalty = no_tool_use_penalty
        self.turn_overflow_penalty = turn_overflow_penalty
        # Per recovered unproductive turn (engine-cut, or ended on nothing). With carried reasoning a cut
        # costs the policy only the turn, and the retry thinks on from where it stopped, so the per-turn
        # budget binds nothing until it is priced.
        self.length_cutoff_penalty = length_cutoff_penalty

    def get_tools_schema(self) -> list[dict[str, Any]]:
        """Get tools in OpenAI format for the rollout engine's generation request."""
        return self.registry.to_openai_tools()

    def _reset_single(self, prompt: str | list[dict[str, str]], context: dict[str, Any] | None = None) -> Trajectory:
        """Initialize episode with task prompt."""
        return self._init_trajectory(
            prompt,
            context,
            system_prompt=self.system_prompt,
            extra_info={
                "total_tool_calls": 0,
                "successful_tool_calls": 0,
                TOOL_CALL_COUNTS_KEY: {},
                EPISODE_TOOL_BUDGETS_KEY: dict(self.tool_budgets),
            },
        )

    @staticmethod
    def active_trajectory() -> Trajectory | None:
        """The episode whose tool call is executing — for a handler that grades or acts on per-episode
        state — or ``None`` outside an episode binding (a handler called directly)."""
        return _ACTIVE_TRAJECTORY.get()

    @staticmethod
    def _coerce_tool_calls(tool_calls_data: list[Any]) -> list[NativeToolCall]:
        """Normalize a context's raw tool-call payload (OpenAI dicts and/or already-parsed calls)."""
        return [NativeToolCall.from_openai_format(tc) if isinstance(tc, dict) else tc for tc in tool_calls_data]

    def _unknown_tool_result(self, tc: NativeToolCall) -> NativeToolResult:
        """Build the error result for a tool call naming a tool not in the registry.

        The ``unknown_tool`` marker is what tokenization reads; the observation text is the
        registry's one wording (:meth:`NativeToolRegistry.unknown_tool_message`).
        """
        return NativeToolResult(
            tool_call_id=tc.id,
            name=tc.name,
            content=self.registry.unknown_tool_message(tc.name),
            success=False,
            unknown_tool=True,
        )

    def _result_from_call(self, tc: NativeToolCall, outcome: str | Exception) -> NativeToolResult:
        """Build a NativeToolResult from a success payload or caught exception (:func:`tool_call_outcome`,
        observation truncated). A sandbox fault rides on the result, so the accounting books it by type."""
        content, success, fault = tool_call_outcome(tc.name, outcome)
        return NativeToolResult(
            tool_call_id=tc.id,
            name=tc.name,
            content=self._truncate_observation(content),
            success=success,
            sandbox_fault=fault,
        )

    def _account_tool_result(self, result: NativeToolResult, trajectory: Trajectory) -> float:
        """Book one result on the episode's counters and return its reward delta (the base's accounting)."""
        return self._book_tool_call(trajectory, result.name, result.success, result.sandbox_fault)

    def _finalize_text_response(
        self, trajectory: Trajectory, action: str
    ) -> tuple[Trajectory, float, bool, bool, dict[str, Any]]:
        """Terminal step for a plain-text (no tool call) model response, shared sync/async.

        The step itself pays nothing: a zero-tool-call finish is priced by the episode-level
        ``no_tool_use_penalty``, charged once by :meth:`_tool_use_shaping` (the single owner), and a
        per-call charge here as well would double-bill the same condition.
        """
        trajectory.info["completed"] = True
        trajectory.info["final_response"] = action
        return trajectory, 0.0, True, False, {}

    @contextmanager
    def _episode_binding(self, trajectory: Trajectory) -> Iterator[None]:
        """Bind the executing episode for the ``with`` block, so a tool handler reaching for per-episode
        state gets THIS episode's (:meth:`active_trajectory`). Entered around every tool batch, sync
        and async, and reset on the way out — one env instance serves concurrent rollouts. A subclass
        binding more (a workspace session) nests its own ContextVar inside ``super()``'s block."""
        token = _ACTIVE_TRAJECTORY.set(trajectory)
        try:
            yield
        finally:
            _ACTIVE_TRAJECTORY.reset(token)

    def _execute_tool_calls(
        self,
        tool_calls: list[NativeToolCall],
        trajectory: Trajectory,
    ) -> tuple[list[NativeToolResult], float]:
        """Execute a batch of tool calls. Returns (results, reward_delta)."""
        results = []
        reward = 0.0

        with self._episode_binding(trajectory):
            for tc in tool_calls[: self.max_tool_calls_per_turn]:
                tool = self.registry.get(tc.name)
                if not tool:
                    result = self._unknown_tool_result(tc)
                else:
                    try:
                        bound = admit_tool_call(self, tool, tc.arguments, trajectory)
                        outcome = tool.execute(**bound)
                    except Exception as e:  # a tool fault is an observation, not an episode kill
                        outcome = e
                    result = self._result_from_call(tc, outcome)

                reward += self._account_tool_result(result, trajectory)
                results.append(result)

        return results, reward

    def _record_tool_interaction(self, results: list[NativeToolResult], trajectory: Trajectory) -> dict[str, Any]:
        """Record the results as tool messages in the trajectory, return step info."""
        for result in results:
            trajectory.add_message(result.to_message())

        # Nothing this turn could execute: mark the assistant message so the trainer never rewards it
        # (an episode that recovers must not reinforce the invented call). Read off ``unknown_tool``,
        # never the error text — a tool whose backend answers "Tool not found: x" failed for real, and
        # dropping that turn would hide a broken tool as a model mistake.
        if results and all(r.unknown_tool for r in results):
            self._flag_calls_rejected(trajectory)

        # Executed calls (post per-turn cap), so this cannot disagree with total_tool_calls.
        return {"step_tool_calls": len(results)}

    def _step_single(
        self, trajectory: Trajectory, action: str, context: dict[str, Any] | None = None
    ) -> tuple[Trajectory, float, bool, bool, dict[str, Any]]:
        """Process model response with potential tool calls."""
        ctx = context or {}
        tool_calls_data = ctx.get("tool_calls", [])
        if not tool_calls_data:
            return self._step_without_tool_calls(trajectory, action, ctx)

        tool_calls = self._coerce_tool_calls(tool_calls_data)
        results, reward = self._execute_tool_calls(tool_calls, trajectory)
        info = self._record_tool_interaction(results, trajectory)
        return trajectory, reward, False, False, info

    def _step_without_tool_calls(
        self, trajectory: Trajectory, action: str, ctx: dict[str, Any]
    ) -> tuple[Trajectory, float, bool, bool, dict[str, Any]]:
        """Handle a turn that called no tool, shared by the sync and async steps: an engine-cut turn
        and a turn that ended on nothing recover, anything else is the model's final text answer."""
        if ctx.get("finish_reason") in ENGINE_CUT_FINISH_REASONS:
            return self._handle_length_cutoff(trajectory, in_tool_call=bool(ctx.get(CUT_IN_TOOL_CALL_KEY)))
        if not action.strip():
            return self._handle_empty_turn(trajectory)
        return self._finalize_text_response(trajectory, action)

    def _tool_use_shaping(self, trajectory: Trajectory) -> float:
        """Per-episode agentic-loop shaping: penalize 0 tool calls and a ``max_turns`` overflow
        (``trajectory.truncated``, set by ``_finalize_step`` before the reward runs) — an episode that
        burns the turn budget without terminating pays ``turn_overflow_penalty`` regardless of what it
        did earn. An episode its driver lost (:data:`EPISODE_ERROR_KEY`) is truncated too but pays no
        overflow: the fault is not the policy's. Each unproductive turn the episode recovered from —
        cut by the engine, or ended by the model on nothing — pays ``length_cutoff_penalty``; the one
        that exhausted the recovery cap pays the overflow price instead, never both. All magnitudes
        default to 0 (no-op). Distinct from the per-call knobs."""
        shaping = -self.no_tool_use_penalty if trajectory.info.get("total_tool_calls", 0) == 0 else 0.0
        if trajectory.truncated and EPISODE_ERROR_KEY not in trajectory.info:
            shaping -= self.turn_overflow_penalty
        unproductive = self._unproductive_turns(trajectory)
        recovered = unproductive - 1 if trajectory.info.get("unrecovered_turn") else unproductive
        shaping -= self.length_cutoff_penalty * recovered
        return shaping

    def _episode_shaping(self, trajectory: Trajectory) -> dict[str, float]:
        """The protocol's episode-level term, :meth:`_tool_use_shaping`, logged as ``reward/tool_shaping``."""
        return {"tool_shaping": self._tool_use_shaping(trajectory)}

    def _grade_episode(self, trajectory: Trajectory, context: dict[str, Any] | None = None) -> EpisodeGrade:
        """Grade the final answer: 1 for a completed episode whose answer validates, else 0.

        A completed episode with nothing to grade against (no validator, no ``answer`` key) grades 1 —
        completing IS the objective there. A row that carries an ``answer`` key holding ``None`` is a
        data fault, not such an episode, and takes the invalid path instead.
        """
        if not trajectory.info.get("completed"):
            return EpisodeGrade(0.0)

        ctx = context or trajectory.info.get("context") or {}

        validator = ctx.get("validator")
        if validator and callable(validator):
            return EpisodeGrade(1.0 if validator(trajectory) else 0.0)

        expected = ctx.get(ANSWER_KEY)
        if expected is not None:
            return EpisodeGrade(1.0 if validate_answer(trajectory.info.get("final_response", ""), expected) else 0.0)

        if ANSWER_KEY in ctx:
            return self._null_answer_grade(trajectory)

        return EpisodeGrade(1.0)


class AsyncNativeToolUseEnvironment(AsyncBaseEnvironment, NativeToolUseEnvironment):
    """Async NativeToolUseEnvironment with concurrent tool execution."""

    async def _execute_tool_calls_async(
        self,
        tool_calls: list[NativeToolCall],
        trajectory: Trajectory,
    ) -> tuple[list[NativeToolResult], float]:
        """Execute tool calls concurrently."""

        async def execute_one(tc: NativeToolCall) -> NativeToolResult:
            tool = self.registry.get(tc.name)
            if not tool:
                return self._unknown_tool_result(tc)
            try:
                bound = admit_tool_call(self, tool, tc.arguments, trajectory, for_async=True)
                outcome = await tool.execute_async(**bound)
            except Exception as e:  # same contract as the sync path above
                outcome = e
            return self._result_from_call(tc, outcome)

        with self._episode_binding(trajectory):
            # gather's child tasks copy the context at creation, so the binding reaches every handler.
            results = list(
                await asyncio.gather(*[execute_one(tc) for tc in tool_calls[: self.max_tool_calls_per_turn]])
            )
            reward = sum(self._account_tool_result(result, trajectory) for result in results)
        return results, reward

    async def _step_single_async(
        self, trajectory: Trajectory, action: str, context: dict[str, Any] | None = None
    ) -> tuple[Trajectory, float, bool, bool, dict[str, Any]]:
        """Process model response with async tool execution."""
        ctx = context or {}
        tool_calls_data = ctx.get("tool_calls", [])
        if not tool_calls_data:
            return self._step_without_tool_calls(trajectory, action, ctx)

        tool_calls = self._coerce_tool_calls(tool_calls_data)
        results, reward = await self._execute_tool_calls_async(tool_calls, trajectory)
        info = self._record_tool_interaction(results, trajectory)
        return trajectory, reward, False, False, info
