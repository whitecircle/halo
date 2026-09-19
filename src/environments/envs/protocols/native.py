"""Native tool-use environments using vLLM/OpenAI tool calling (sync + async)."""

import asyncio
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from src.environments.base import (
    EMPTY_FINAL_ANSWER_KEY,
    EPISODE_ERROR_KEY,
    EPISODE_INVALID_KEY,
    EPISODE_TOOL_BUDGETS_KEY,
    TOOL_CALL_COUNTS_KEY,
    AsyncBaseEnvironment,
    BaseEnvironment,
    EpisodeGrade,
    Trajectory,
    require_magnitudes,
)
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


class NativeToolUseEnvironment(BaseEnvironment):
    """Environment using native vLLM/OpenAI tool calling.

    With DistributedAsyncEnvironmentalGRPOTrainer, pass ``tools=env.get_tools_schema()`` to the generation config.
    """

    SHAPING_COMPONENTS = ("tool_shaping",)

    # States the fact and asks for the action — never for shorter reasoning. This text is trained on
    # wherever a recovery succeeds, so any instruction here becomes a GLOBAL lesson, learned far
    # outside the situation it was written for.
    LENGTH_CUTOFF_NUDGE = (
        "Your previous turn was cut off before you made a tool call, so nothing was recorded. Make "
        "your tool call now with the best solution you have."
    )

    def __init__(
        self,
        tool_registry: NativeToolRegistry,
        system_prompt: str | None = None,
        max_tool_calls_per_turn: int = 5,
        require_tool_use: bool = False,
        no_tool_use_penalty: float = 0.0,
        multi_turn_reward: float = 0.0,
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
            multi_turn_reward=multi_turn_reward,
            turn_overflow_penalty=turn_overflow_penalty,
            length_cutoff_penalty=length_cutoff_penalty,
        )

        self.registry = tool_registry
        self.tool_budgets = validate_tool_budgets(tool_budgets, tool_registry)
        self.system_prompt = system_prompt
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self.require_tool_use = require_tool_use
        self.no_tool_use_penalty = no_tool_use_penalty
        self.multi_turn_reward = multi_turn_reward
        self.turn_overflow_penalty = turn_overflow_penalty
        # Per recovered engine-cut turn. With carried reasoning a cut costs the policy only the turn, and
        # the retry thinks on from where it stopped, so the per-turn budget binds nothing until it is priced.
        self.length_cutoff_penalty = length_cutoff_penalty

    def get_tools_schema(self) -> list[dict[str, Any]]:
        """Get tools in OpenAI format for vLLM generation."""
        return self.registry.to_openai_tools()

    def _reset_single(self, prompt: str | list[dict[str, str]], context: dict[str, Any] | None = None) -> Trajectory:
        """Initialize episode with task prompt."""
        return self._init_trajectory(
            prompt,
            context,
            system_prompt=self.system_prompt,
            extra_info={
                "tool_calls": [],
                "tool_results": [],
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
        """Build a NativeToolResult from a success payload or caught exception (observation truncated)."""
        if isinstance(outcome, Exception):
            return NativeToolResult(
                tool_call_id=tc.id,
                name=tc.name,
                content=self._truncate_observation(f"Error: {outcome}"),
                success=False,
            )
        return NativeToolResult(
            tool_call_id=tc.id,
            name=tc.name,
            content=self._truncate_observation(outcome),
            success=True,
        )

    def _refused_call_result(
        self, tc: NativeToolCall, exc: ToolBudgetExhausted | ToolArgumentError
    ) -> NativeToolResult:
        """A call the tool refused — over its per-episode budget, or arguments its handler cannot bind:
        a tool error like any other, but expected control flow.

        Logged without a traceback — an env with a 2-submission cap in a 15-turn episode refuses by
        design, a ``submit_solution`` with no ``code`` is a model slip, and a stack trace per refusal
        buries the faults that ``_execute_tool_calls`` logs.
        """
        logger.debug("Tool %r refused the call: %s", tc.name, exc)
        return self._result_from_call(tc, exc)

    def _account_tool_result(self, result: NativeToolResult, trajectory: Trajectory) -> float:
        """Book one result on the episode's counters and return its reward delta (the base's accounting)."""
        return self._credit_tool_call(trajectory, result.success)

    def _finalize_text_response(
        self, trajectory: Trajectory, action: str
    ) -> tuple[Trajectory, float, bool, bool, dict[str, Any]]:
        """Terminal step for a plain-text (no tool call) model response, shared sync/async.

        ``require_tool_use`` only FLAGS a zero-tool-call finish here; its price is the episode-level
        ``no_tool_use_penalty``, charged once by :meth:`_tool_use_shaping` (the single owner). Charging
        a per-call knob here as well would double-bill the same condition.
        """
        trajectory.info["completed"] = True
        trajectory.info["final_response"] = action
        # A turn that stopped inside its reasoning arrives as a final answer with nothing visible.
        trajectory.info[EMPTY_FINAL_ANSWER_KEY] = not action.strip()

        info: dict[str, Any] = {}
        if self.require_tool_use and trajectory.info["total_tool_calls"] == 0:
            info["no_tool_use"] = True

        return trajectory, 0.0, True, False, info

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

    def _admit_call(
        self, tool: NativeTool, tc: NativeToolCall, trajectory: Trajectory, *, for_async: bool = False
    ) -> dict[str, Any]:
        """Admit one call before it runs: bind its arguments (against the handler that will run), then
        spend one of the episode's calls on the tool. Refuses (:class:`ToolArgumentError`,
        :class:`ToolBudgetExhausted`) without counting, so a call the handler could never run does not
        consume the budget; runs synchronously before any await so concurrent calls in one turn cannot
        both pass a one-call cap."""
        bound = tool.bind(tc.arguments, for_async=for_async)
        cap = self._tool_budget_exhausted(trajectory, tc.name)
        if cap is not None:
            raise ToolBudgetExhausted(tool.budget_exhausted_message(cap))
        self._count_tool_call(trajectory, tc.name)
        return bound

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
                        bound = self._admit_call(tool, tc, trajectory)
                        result = self._result_from_call(tc, tool.execute(**bound))
                    except (ToolBudgetExhausted, ToolArgumentError) as e:
                        result = self._refused_call_result(tc, e)
                    except Exception as e:  # a tool fault is an observation, not an episode kill
                        # Logged because the graded tools run here too: a submit handler that dies on a
                        # malformed payload becomes an ordinary tool error, and without this line the
                        # episode just grades 0 with nothing anywhere saying why.
                        logger.warning("Tool %r raised during execution", tc.name, exc_info=True)
                        result = self._result_from_call(tc, e)

                reward += self._account_tool_result(result, trajectory)
                results.append(result)

        return results, reward

    def _record_tool_interaction(
        self,
        tool_calls: list[NativeToolCall],
        results: list[NativeToolResult],
        trajectory: Trajectory,
    ) -> dict[str, Any]:
        """Record tool calls and results in trajectory, return step info."""
        trajectory.info["tool_calls"].extend(
            [
                {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
                for tc in tool_calls[: self.max_tool_calls_per_turn]
            ]
        )
        trajectory.info["tool_results"].extend(
            [{"id": r.tool_call_id, "name": r.name, "content": r.content, "success": r.success} for r in results]
        )
        for result in results:
            trajectory.add_message(result.to_message())

        # Nothing this turn could execute: mark the assistant message so the trainer skips it (an
        # episode that recovers must not reinforce the invented call). Read off ``unknown_tool``,
        # never the error text — a tool whose backend answers "Tool not found: x" failed for real, and
        # dropping that turn would hide a broken tool as a model mistake.
        if results and all(r.unknown_tool for r in results):
            self._flag_calls_rejected(trajectory)

        return {
            # Executed calls (post per-turn cap), so this cannot disagree with total_tool_calls.
            "step_tool_calls": len(results),
            "step_successful": sum(1 for r in results if r.success),
        }

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
        info = self._record_tool_interaction(tool_calls, results, trajectory)
        return trajectory, reward, False, False, info

    def _step_without_tool_calls(
        self, trajectory: Trajectory, action: str, ctx: dict[str, Any]
    ) -> tuple[Trajectory, float, bool, bool, dict[str, Any]]:
        """Handle a turn that called no tool, shared by the sync and async steps: an engine-cut turn
        recovers, anything else is the model's final text answer."""
        if ctx.get("finish_reason") in ENGINE_CUT_FINISH_REASONS:
            return self._handle_length_cutoff(trajectory)
        return self._finalize_text_response(trajectory, action)

    def _tool_use_engaged(self, trajectory: Trajectory) -> bool:
        """Gate for ``multi_turn_reward``: whether >1 tool call counts as genuine engagement.

        Subclasses tighten it (e.g. CodeContests requires an actual submission, not test-tool spam).
        """
        return True

    def _tool_use_shaping(self, trajectory: Trajectory) -> float:
        """Per-episode agentic-loop shaping: penalize 0 tool calls, reward >1 (gated by
        ``_tool_use_engaged``), and penalize a ``max_turns`` overflow (``trajectory.truncated``, set by
        ``_finalize_step`` before the reward runs) — an episode that burns the turn budget without
        terminating pays ``turn_overflow_penalty`` regardless of what it did earn. An episode its
        driver lost (:data:`EPISODE_ERROR_KEY`) is truncated too but pays no overflow: the fault is
        not the policy's. Each engine-cut turn the episode recovered from pays ``length_cutoff_penalty``;
        the cut that exhausted the recovery cap pays the overflow price instead, never both. All
        magnitudes default to 0 (no-op). Distinct from the per-call knobs."""
        calls = trajectory.info.get("total_tool_calls", 0)
        if calls == 0:
            shaping = -self.no_tool_use_penalty
        elif calls > 1 and self._tool_use_engaged(trajectory):
            shaping = self.multi_turn_reward
        else:
            shaping = 0.0
        if trajectory.truncated and EPISODE_ERROR_KEY not in trajectory.info:
            shaping -= self.turn_overflow_penalty
        cuts = trajectory.info.get("length_cutoff_turns", 0)
        recovered = cuts - 1 if trajectory.info.get("length_cutoff_recoveries_exhausted") else cuts
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

        expected = ctx.get("answer")
        if expected is not None:
            return EpisodeGrade(1.0 if validate_answer(trajectory.info.get("final_response", ""), expected) else 0.0)

        if "answer" in ctx:
            # The dataset row is answer-graded and its cell is null: nothing was verified, so paying
            # the completion fallback would hand the full objective to ANY episode that finished —
            # and to its whole GRPO group, since every sibling row completes just as easily. Drop it
            # from the baseline instead (same contract as a grading-infra outage).
            logger.warning("Episode context carries a null 'answer'; scoring it invalid, not a success")
            trajectory.info[EPISODE_INVALID_KEY] = True
            return EpisodeGrade(0.0)

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
                bound = self._admit_call(tool, tc, trajectory, for_async=True)
                return self._result_from_call(tc, await tool.execute_async(**bound))
            except (ToolBudgetExhausted, ToolArgumentError) as e:
                return self._refused_call_result(tc, e)
            except Exception as e:  # same contract as the sync path above
                logger.warning("Tool %r raised during async execution", tc.name, exc_info=True)
                return self._result_from_call(tc, e)

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
        info = self._record_tool_interaction(tool_calls, results, trajectory)
        return trajectory, reward, False, False, info
