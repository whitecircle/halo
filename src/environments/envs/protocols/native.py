"""Native tool-use environments using vLLM/OpenAI tool calling (sync + async)."""

import asyncio
import logging
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from src.environments.base import (
    EPISODE_INVALID_KEY,
    AsyncBaseEnvironment,
    BaseEnvironment,
    Trajectory,
    require_magnitudes,
)
from src.environments.rewards import compute_answer_reward
from src.environments.tools.definitions import (
    NativeToolCall,
    NativeToolRegistry,
    NativeToolResult,
    ToolBudgetExhausted,
)
from src.inference.response import FINISH_REASON_LENGTH

logger = logging.getLogger(__name__)


class NativeToolUseEnvironment(BaseEnvironment):
    """Environment using native vLLM/OpenAI tool calling.

    With DistributedAsyncEnvironmentalGRPOTrainer, pass ``tools=env.get_tools_schema()`` to the generation config.
    """

    # States the fact and asks for the action — never for shorter reasoning. This text is trained on
    # wherever a recovery succeeds, so any instruction here becomes a GLOBAL lesson, learned far
    # outside the situation it was written for.
    LENGTH_CUTOFF_NUDGE = (
        "Your previous turn was cut off at its length limit before you made a tool call, so nothing "
        "was recorded. Make your tool call now with the best solution you have."
    )

    # Per-tool-call shaping a class gets when the config names none, declared like
    # :data:`~src.environments.base.BaseEnvironment.DEFAULT_MAX_TURNS` so a task env that departs
    # states its own value once instead of re-defaulting its constructor.
    DEFAULT_TOOL_SUCCESS_REWARD: float = 0.05
    DEFAULT_TOOL_ERROR_PENALTY: float = 0.1

    def __init__(
        self,
        tool_registry: NativeToolRegistry,
        system_prompt: str | None = None,
        max_tool_calls_per_turn: int = 5,
        success_reward: float = 1.0,
        failure_reward: float = 0.0,
        tool_success_reward: float | None = None,
        tool_error_penalty: float | None = None,
        require_tool_use: bool = False,
        no_tool_use_penalty: float = 0.0,
        multi_turn_reward: float = 0.0,
        turn_overflow_penalty: float = 0.0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        tool_success_reward = self.DEFAULT_TOOL_SUCCESS_REWARD if tool_success_reward is None else tool_success_reward
        tool_error_penalty = self.DEFAULT_TOOL_ERROR_PENALTY if tool_error_penalty is None else tool_error_penalty

        require_magnitudes(
            no_tool_use_penalty=no_tool_use_penalty,
            multi_turn_reward=multi_turn_reward,
            turn_overflow_penalty=turn_overflow_penalty,
            tool_success_reward=tool_success_reward,
            tool_error_penalty=tool_error_penalty,
        )

        self.registry = tool_registry
        self.system_prompt = system_prompt
        self.max_tool_calls_per_turn = max_tool_calls_per_turn
        self.success_reward = success_reward
        self.failure_reward = failure_reward
        self.tool_success_reward = tool_success_reward
        self.tool_error_penalty = tool_error_penalty
        self.require_tool_use = require_tool_use
        self.no_tool_use_penalty = no_tool_use_penalty
        self.multi_turn_reward = multi_turn_reward
        self.turn_overflow_penalty = turn_overflow_penalty

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
            },
        )

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

    def _budget_exhausted_result(self, tc: NativeToolCall, exc: ToolBudgetExhausted) -> NativeToolResult:
        """A refused over-budget call: a tool error like any other, but expected control flow.

        Logged without a traceback — an env with a 2-submission cap in a 15-turn episode refuses by
        design, and a stack trace per refusal buries the faults that ``_execute_tool_calls`` logs.
        """
        logger.debug("Tool %r refused the call: %s", tc.name, exc)
        return self._result_from_call(tc, exc)

    def _account_tool_result(self, result: NativeToolResult, trajectory: Trajectory) -> float:
        """Update per-trajectory tool counters for one result and return its reward delta."""
        trajectory.info["total_tool_calls"] += 1
        if result.success:
            trajectory.info["successful_tool_calls"] += 1
            return self.tool_success_reward
        return -self.tool_error_penalty

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

        info: dict[str, Any] = {}
        if self.require_tool_use and trajectory.info["total_tool_calls"] == 0:
            info["no_tool_use"] = True

        return trajectory, 0.0, True, False, info

    @contextmanager
    def _episode_binding(self, trajectory: Trajectory) -> Iterator[None]:
        """Bind the executing episode for the ``with`` block, so a tool handler reaching for per-episode
        state gets THIS episode's. Stateful subclasses set their ContextVars here; a base episode has
        none. Entered around every tool batch, sync and async, and reset on the way out — one env
        instance serves concurrent rollouts."""
        yield

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
                        result = self._result_from_call(tc, tool.execute(**tc.arguments))
                    except ToolBudgetExhausted as e:
                        result = self._budget_exhausted_result(tc, e)
                    except Exception as e:  # a tool fault is an observation, not an episode kill
                        # Logged because the graded tools run here too: a submit handler that dies on a
                        # malformed payload becomes an ordinary tool error, and without this line the
                        # episode just scores failure_reward with nothing anywhere saying why.
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
        # episode that recovers must not reinforce the invented call). The assistant message precedes
        # the tool results just appended. Read off ``unknown_tool``, never the error text — a tool
        # whose backend answers "Tool not found: x" failed for real, and dropping that turn would hide
        # a broken tool as a model mistake.
        if results and all(r.unknown_tool for r in results):
            for message in reversed(trajectory.messages):
                if message.role == "assistant":
                    message.calls_rejected = True
                    break

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
        """The two outcomes of a turn that called no tool, shared by the sync and async steps: an
        engine-cut turn recovers, anything else is the model's final text answer."""
        if ctx.get("finish_reason") == FINISH_REASON_LENGTH:
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
        terminating pays ``turn_overflow_penalty`` regardless of what it did earn. All magnitudes
        default to 0 (no-op). Distinct from the per-call knobs."""
        calls = trajectory.info.get("total_tool_calls", 0)
        if calls == 0:
            shaping = -self.no_tool_use_penalty
        elif calls > 1 and self._tool_use_engaged(trajectory):
            shaping = self.multi_turn_reward
        else:
            shaping = 0.0
        if trajectory.truncated:
            shaping -= self.turn_overflow_penalty
        return shaping

    def _shaped_base_reward(self, trajectory: Trajectory) -> float:
        """Accrued per-turn rewards plus the protocol-level tool-use shaping.

        The single base every ``_compute_reward`` (base class and subclass overrides alike) builds
        on — an override composing from ``trajectory.total_reward`` directly would silently drop
        the shaping knobs (``turn_overflow_penalty`` et al.), the exact drift this seam prevents.
        """
        return trajectory.total_reward + self._tool_use_shaping(trajectory)

    def _compute_reward(self, trajectory: Trajectory, context: dict[str, Any] | None = None) -> float:
        """Compute final reward for the trajectory (answer validation + generic tool-use shaping).

        The terminal ``success_reward`` is for episodes with nothing to grade against (no validator,
        no ``answer`` key) — completing IS the objective there. A row that carries an ``answer`` key
        holding ``None`` is a data fault, not such an episode, and takes the invalid path instead.
        """
        base_reward = self._shaped_base_reward(trajectory)

        if not trajectory.info.get("completed"):
            return base_reward + self.failure_reward

        ctx = context or trajectory.info.get("context") or {}

        validator = ctx.get("validator")
        if validator and callable(validator):
            is_success = validator(trajectory)
            return base_reward + (self.success_reward if is_success else self.failure_reward)

        expected = ctx.get("answer")
        if expected is not None:
            final_response = trajectory.info.get("final_response", "")
            return base_reward + compute_answer_reward(
                predicted=final_response,
                expected=expected,
                success_reward=self.success_reward,
                failure_reward=self.failure_reward,
            )

        if "answer" in ctx:
            # The dataset row is answer-graded and its cell is null: nothing was verified, so paying
            # the completion fallback would hand full success_reward to ANY episode that finished —
            # and to its whole GRPO group, since every sibling row completes just as easily. Drop it
            # from the baseline instead (same contract as a grading-infra outage).
            logger.warning("Episode context carries a null 'answer'; scoring it invalid, not a success")
            trajectory.info[EPISODE_INVALID_KEY] = True
            return base_reward + self.failure_reward

        return base_reward + self.success_reward


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
                return self._result_from_call(tc, await tool.execute_async(**tc.arguments))
            except ToolBudgetExhausted as e:
                return self._budget_exhausted_result(tc, e)
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
