"""Native tool-use environments over OpenAI-format tool calling, which both rollout engines parse (sync + async)."""

import asyncio
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

from halo.config_validation import require_finite
from halo.environments.environment_base import AsyncEnvironmentBase, EnvironmentBase
from halo.environments.episode_keys import ANSWER_KEY, COMPLETED_KEY, CONTEXT_KEY, EPISODE_ERROR_KEY, LAST_TURN_KEY
from halo.environments.episode_scoring import EpisodeGrade
from halo.environments.messages import StepOutcome, Trajectory
from halo.environments.tool_accounting import (
    MAGNITUDE_WHY,
    TOTAL_TOOL_CALLS_KEY,
    initial_tool_counters,
    tool_call_outcome,
    validate_tool_budgets,
)
from halo.environments.tools.definitions import NativeToolCall, NativeToolRegistry, NativeToolResult, is_refused_unrun
from halo.inference.response import ENGINE_CUT_FINISH_REASONS
from halo.rewards.graders.matching import is_correct_answer
from halo.rewards.samples import CUT_TOOL_CALLS_KEY

# Per asyncio Task / thread: one env instance serves concurrent rollouts, so a handler reads per-episode state here.
_ACTIVE_TRAJECTORY: ContextVar[Trajectory | None] = ContextVar("native_tool_use_active_trajectory", default=None)


class NativeToolUseEnvironment(EnvironmentBase):
    SHAPING_COMPONENTS = ("tool_shaping",)

    # State the fact and ask for the action, never for shorter reasoning: these texts are trained on wherever a
    # recovery succeeds, so any instruction here becomes a global lesson.
    LENGTH_CUTOFF_NUDGE = (
        "Your previous turn was cut off before you made a tool call, so nothing was recorded. Make "
        "your tool call now with the best solution you have."
    )
    # Sent only on a cut inside a call, so it may name the limit; it asks where the reasoning goes, not for less of it.
    LENGTH_CUTOFF_IN_CALL_NUDGE = (
        "Your previous turn reached its length limit while writing a tool call, so the call was not run and "
        "nothing was recorded. Make the call again, keeping your reasoning out of its arguments."
    )
    EMPTY_TURN_NUDGE = (
        "Your previous turn ended without a tool call or an answer (a call written inside your reasoning "
        "is not run), so nothing was recorded. Make your tool call now, or give your final answer, with the "
        "best solution you have."
    )
    # The same facts for an instance whose registry holds no tool: a nudge asking for a tool call there
    # is an instruction the model cannot follow, and the turn that answers it is trained.
    TOOLLESS_LENGTH_CUTOFF_NUDGE = (
        "Your previous turn was cut off before you gave your answer, so nothing was recorded. Give your "
        "final answer now with the best solution you have."
    )
    TOOLLESS_EMPTY_TURN_NUDGE = (
        "Your previous turn ended without an answer (text written inside your reasoning is not read), so "
        "nothing was recorded. Give your final answer now with the best solution you have."
    )
    # A cut the driver flags as inside a call is told the same: there is no call to make again.
    TOOLLESS_LENGTH_CUTOFF_IN_CALL_NUDGE = TOOLLESS_LENGTH_CUTOFF_NUDGE

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
        """``tool_budgets`` caps calls per tool per episode (``{tool_name: cap}``; unlisted is uncapped); a call past
        its cap is refused unrun."""
        super().__init__(**kwargs)

        require_finite(
            type(self).__name__,
            ge=0,
            why=MAGNITUDE_WHY,
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
        # Per recovered unproductive turn: with carried reasoning a cut costs the policy only the turn, so the
        # per-turn budget binds nothing until it is priced.
        self.length_cutoff_penalty = length_cutoff_penalty

    def get_tools_schema(self) -> list[dict[str, Any]]:
        return self.registry.to_openai_tools()

    def _unproductive_turn_nudge(self, nudge_attr: str) -> str | None:
        """A tool-less environment takes the ``TOOLLESS_`` nudge, so it is never told to make a tool call."""
        if not self.registry.names():
            nudge_attr = f"TOOLLESS_{nudge_attr}"
        return super()._unproductive_turn_nudge(nudge_attr)

    def _final_answer(self, trajectory: Trajectory) -> str | None:
        return trajectory.info.get("final_response") if trajectory.info.get(COMPLETED_KEY) else None

    def _reset_single(self, prompt: str | list[dict[str, str]], context: dict[str, Any] | None = None) -> Trajectory:
        return self._init_trajectory(
            prompt,
            context,
            system_prompt=self.system_prompt,
            extra_info=initial_tool_counters(self.tool_budgets),
        )

    @staticmethod
    def active_trajectory() -> Trajectory | None:
        """``None`` outside an episode binding (a handler called directly)."""
        return _ACTIVE_TRAJECTORY.get()

    @staticmethod
    def _coerce_tool_calls(tool_calls_data: list[dict[str, Any]]) -> list[NativeToolCall]:
        return [NativeToolCall.from_openai_format(tc) for tc in tool_calls_data]

    def _unknown_tool_result(self, tc: NativeToolCall) -> NativeToolResult:
        """Marked ``unknown_tool`` for tokenization."""
        return NativeToolResult(
            tool_call_id=tc.id,
            name=tc.name,
            content=self.registry.unknown_tool_message(tc.name),
            success=False,
            unknown_tool=True,
        )

    def _result_from_call(self, tc: NativeToolCall, outcome: str | Exception) -> NativeToolResult:
        content, success, fault = tool_call_outcome(tc.name, outcome)
        return NativeToolResult(
            tool_call_id=tc.id,
            name=tc.name,
            content=self._truncate_observation(content),
            success=success,
            refused=is_refused_unrun(outcome),
            sandbox_fault=fault,
        )

    def _account_tool_result(self, result: NativeToolResult, trajectory: Trajectory) -> float:
        return self._book_tool_call(trajectory, result.name, result.success, result.sandbox_fault)

    def _finalize_text_response(self, trajectory: Trajectory, action: str) -> StepOutcome:
        """Pays nothing: a finish without tool calls is priced once, by :meth:`_tool_use_shaping`."""
        trajectory.info[COMPLETED_KEY] = True
        trajectory.info["final_response"] = action
        return StepOutcome(trajectory=trajectory, reward=0.0, done=True, truncated=False, info={})

    @contextmanager
    def _episode_binding(self, trajectory: Trajectory) -> Iterator[None]:
        """A subclass binding more (a workspace session) nests its own ContextVar inside ``super()``'s block."""
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
        results = []
        reward = 0.0

        with self._episode_binding(trajectory):
            for tc in tool_calls[: self.max_tool_calls_per_turn]:
                tool = self.registry.get(tc.name)
                if not tool:
                    result = self._unknown_tool_result(tc)
                else:
                    try:
                        bound = self._admit_tool_call(tool, tc.arguments, trajectory)
                        outcome = tool.execute(**bound)
                    except Exception as e:  # a tool fault is an observation, not an episode kill
                        outcome = e
                    result = self._result_from_call(tc, outcome)

                reward += self._account_tool_result(result, trajectory)
                results.append(result)

        return results, reward

    def _record_tool_interaction(self, results: list[NativeToolResult], trajectory: Trajectory) -> dict[str, Any]:
        for result in results:
            trajectory.add_message(result.to_message())

        # Nothing this turn could execute: the turn is never rewarded. Read off the flags, never the reply text, which
        # a broken tool backend may word the same way.
        if results and all(r.unknown_tool or r.refused for r in results):
            trajectory.last_assistant_message().calls_rejected = True
        refused = tuple(r.tool_call_id for r in results if (r.unknown_tool or r.refused) and r.tool_call_id)
        if refused:
            trajectory.last_assistant_message().refused_call_ids = refused

        # Executed calls (post per-turn cap), so this cannot disagree with total_tool_calls.
        return {"step_tool_calls": len(results)}

    def _step_single(self, trajectory: Trajectory, action: str, context: dict[str, Any] | None = None) -> StepOutcome:
        ctx = context or {}
        tool_calls_data = ctx.get("tool_calls", [])
        if not tool_calls_data:
            return self._step_without_tool_calls(trajectory, action, ctx)

        tool_calls = self._coerce_tool_calls(tool_calls_data)
        results, reward = self._execute_tool_calls(tool_calls, trajectory)
        info = self._record_tool_interaction(results, trajectory)
        return StepOutcome(trajectory=trajectory, reward=reward, done=False, truncated=False, info=info)

    def _step_without_tool_calls(self, trajectory: Trajectory, action: str, ctx: dict[str, Any]) -> StepOutcome:
        """An engine-cut or empty turn recovers; anything else is the final answer."""
        last_turn = bool(ctx.get(LAST_TURN_KEY))
        if ctx.get("finish_reason") in ENGINE_CUT_FINISH_REASONS:
            in_call = bool(ctx.get(CUT_TOOL_CALLS_KEY))
            return self._recover_length_cutoff(trajectory, in_tool_call=in_call, last_turn=last_turn)
        if not action.strip():
            return self._recover_empty_turn(trajectory, last_turn=last_turn)
        return self._finalize_text_response(trajectory, action)

    def _tool_use_shaping(self, trajectory: Trajectory) -> float:
        """``no_tool_use_penalty`` for no tool call, ``turn_overflow_penalty`` for a ``max_turns`` overflow (never for
        an episode its driver lost), ``length_cutoff_penalty`` per recovered unproductive turn (the one that exhausted
        the recovery cap pays the overflow instead)."""
        shaping = -self.no_tool_use_penalty if trajectory.info.get(TOTAL_TOOL_CALLS_KEY, 0) == 0 else 0.0
        if trajectory.truncated and EPISODE_ERROR_KEY not in trajectory.info:
            shaping -= self.turn_overflow_penalty
        unproductive = self._unproductive_turns(trajectory)
        recovered = unproductive - 1 if trajectory.info.get("unrecovered_turn") else unproductive
        shaping -= self.length_cutoff_penalty * recovered
        return shaping

    def _episode_shaping(self, trajectory: Trajectory) -> dict[str, float]:
        return {"tool_shaping": self._tool_use_shaping(trajectory)}

    def _grade_episode(self, trajectory: Trajectory, context: dict[str, Any] | None = None) -> EpisodeGrade:
        """1 for a completed episode whose answer validates, or with nothing to grade against (completing is the
        objective there), else 0; a null ``answer`` cell grades invalid."""
        if not trajectory.info.get(COMPLETED_KEY):
            return EpisodeGrade(0.0)

        ctx = context or trajectory.info.get(CONTEXT_KEY) or {}

        validator = ctx.get("validator")
        if callable(validator):
            return EpisodeGrade(1.0 if validator(trajectory) else 0.0)

        expected = ctx.get(ANSWER_KEY)
        if expected is not None:
            return EpisodeGrade(1.0 if is_correct_answer(trajectory.info.get("final_response", ""), expected) else 0.0)

        if ANSWER_KEY in ctx:
            return self._null_answer_grade(trajectory)

        return EpisodeGrade(1.0)


class AsyncNativeToolUseEnvironment(AsyncEnvironmentBase, NativeToolUseEnvironment):
    async def _execute_tool_calls_async(
        self,
        tool_calls: list[NativeToolCall],
        trajectory: Trajectory,
    ) -> tuple[list[NativeToolResult], float]:
        async def execute_one(tc: NativeToolCall) -> NativeToolResult:
            tool = self.registry.get(tc.name)
            if not tool:
                return self._unknown_tool_result(tc)
            try:
                bound = self._admit_tool_call(tool, tc.arguments, trajectory, for_async=True)
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
    ) -> StepOutcome:
        ctx = context or {}
        tool_calls_data = ctx.get("tool_calls", [])
        if not tool_calls_data:
            return self._step_without_tool_calls(trajectory, action, ctx)

        tool_calls = self._coerce_tool_calls(tool_calls_data)
        results, reward = await self._execute_tool_calls_async(tool_calls, trajectory)
        info = self._record_tool_interaction(results, trajectory)
        return StepOutcome(trajectory=trajectory, reward=reward, done=False, truncated=False, info=info)
