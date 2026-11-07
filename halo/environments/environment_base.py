"""Environment lifecycle: :class:`EnvironmentBase` opens, steps, closes and records episodes over its mixins;
:class:`AsyncEnvironmentBase` runs a turn batch through ``asyncio.gather``."""

import asyncio
import itertools
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any

from halo.config_validation import present, require_bool, require_int
from halo.environments.episode_keys import (
    COMPLETED_KEY,
    CONTEXT_KEY,
    OUTPUT_BUDGET_EXHAUSTED_KEY,
    SANDBOX_FAULT_AGENT,
    SANDBOX_FAULT_INFRA,
    SANDBOX_FAULT_KEY,
)
from halo.environments.episode_scoring import REWARD_PENDING_KEY, EpisodeGrade, EpisodeScoringMixin
from halo.environments.messages import EnvStep, Message, StepOutcome, Trajectory, engine_view
from halo.environments.reasoning_effort import EffortProfilesMixin
from halo.environments.tool_accounting import TOTAL_TOOL_CALLS_KEY, ToolCallAccountingMixin
from halo.environments.turn_recovery import (
    EMPTY_TURNS_KEY,
    LENGTH_CUTOFF_IN_CALL_TURNS_KEY,
    LENGTH_CUTOFF_TURNS_KEY,
    TurnRecoveryMixin,
)
from halo.inference.response import ENGINE_CUT_FINISH_REASONS
from halo.rewards.samples import CUT_TOOL_CALLS_KEY
from halo.rewards.terms import RewardTerm

logger = logging.getLogger(__name__)

# CJK ideographs, kana and hangul: what a Latin-script task's CoT drifts into under RL.
_CJK_SCRIPT = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")


class EnvironmentBase(ToolCallAccountingMixin, TurnRecoveryMixin, EffortProfilesMixin, EpisodeScoringMixin, ABC):
    """A multi-turn GRPO environment; subclasses implement ``_reset_single``, ``_step_single`` and ``_grade_episode``.
    One instance serves concurrent episodes, so per-episode state lives on the trajectory, never on the instance."""

    DEFAULT_MAX_TURNS: int = 10

    # Whether the reward grades only against ``context["answer"]``, so the trainer refuses an answerless dataset.
    requires_answer: bool = False

    # Whether the tools arrive on the first reset (an MCP server's, when it connects).
    TOOLS_ARRIVE_ON_RESET: bool = False

    # Set by the protocol layers; the eval records it. ``None`` = no system turn.
    system_prompt: str | None = None

    def __init__(
        self,
        max_turns: int | None = None,
        max_observation_chars: int = 16384,
        max_length_cutoff_recoveries: int | None = None,
        reasoning_effort: str | None = None,
        reasoning_effort_profiles: dict[str, dict[str, int | float | bool]] | None = None,
        carry_reasoning: bool = False,
        requires_answer: bool | None = None,
        tool_success_reward: float | None = None,
        tool_error_penalty: float | None = None,
        tool_reward_cap: float | None = None,
        reward_terms: Sequence[RewardTerm | Mapping[str, Any]] | None = None,
        **kwargs,
    ):
        """Every knob is a YAML ``environment_kwargs`` key. ``max_length_cutoff_recoveries``: unproductive turns an
        episode may recover from (``None`` = all in budget). ``reasoning_effort``: ``low``/``medium``/``high``,
        ``"random"`` (one draw per episode) or ``None``. ``carry_reasoning``: send the last assistant turn's reasoning
        back to the engine. ``reward_terms``: ``None`` is the env's own grade at weight 1.

        Any other keyword raises: the registry forwards the whole ``env_config``, so an unbound key is a typo or a knob
        of another ``env_type``.
        """
        max_turns = self.DEFAULT_MAX_TURNS if max_turns is None else max_turns
        # 0 turns makes the rollout loop a no-op, producing an all-zero batch.
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")
        self.max_turns = max_turns
        self._init_turn_recovery(max_length_cutoff_recoveries)
        require_int(type(self).__name__, ge=0, why=" (0 = no cap)", max_observation_chars=max_observation_chars)
        self.max_observation_chars = max_observation_chars
        require_bool(type(self).__name__, carry_reasoning=carry_reasoning, **present(requires_answer=requires_answer))
        self.carry_reasoning = carry_reasoning
        if requires_answer is not None:
            self.requires_answer = requires_answer
        self._init_tool_pricing(tool_success_reward, tool_error_penalty, tool_reward_cap)
        self._init_effort_profiles(reasoning_effort, reasoning_effort_profiles)
        self._init_rewards(reward_terms)
        if kwargs:
            raise TypeError(
                f"{type(self).__name__} got unexpected environment option(s) {sorted(kwargs)}. Every "
                f"environment_kwargs key must be a constructor parameter of the resolved environment "
                f"(check the env_type it belongs to and the spelling)."
            )
        self._background_tasks: set[asyncio.Task] = set()
        self._trajectories: dict[int, Trajectory] = {}
        self._episode_ids = itertools.count()

    def reset(
        self, prompts: list[str | list[dict[str, str]]], contexts: list[dict[str, Any]] | None = None
    ) -> tuple[list[int], list[EnvStep]]:
        episode_ids, steps = [], []
        for prompt, context in zip(prompts, self._contexts(contexts, prompts), strict=True):
            episode_id = next(self._episode_ids)
            episode_ids.append(episode_id)
            steps.append(self._register_episode(episode_id, self._reset_single(prompt, context), context))
        return episode_ids, steps

    def step(
        self, episode_ids: list[int], actions: list[str], contexts: list[dict[str, Any]] | None = None
    ) -> list[EnvStep]:
        steps = []
        for episode_id, action, context in zip(
            episode_ids, actions, self._contexts(contexts, episode_ids), strict=True
        ):
            trajectory, standing = self._begin_step(episode_id, action, context)
            if standing is None:
                result = self._step_single(trajectory, action, context)
                standing = self._finalize_step(episode_id, *result, context)
            steps.append(standing)
        return steps

    def finalize_truncated(self, episode_ids: list[int]) -> list[EnvStep]:
        """Close open episodes as truncated without a synthetic model turn, which would mark them ``completed`` and pay
        completion-graded envs. A driver that lost the episode stamps :data:`EPISODE_ERROR_KEY` first, keeping the
        turn-overflow price off it."""
        steps = []
        for episode_id in episode_ids:
            trajectory = self._episode(episode_id)
            if trajectory.done:
                steps.append(self._env_step(trajectory))
                continue
            steps.append(self._finalize_step(episode_id, trajectory, 0.0, True, True, {}, None))
        return steps

    def get_trajectories(self, episode_ids: list[int]) -> list[Trajectory | None]:
        return [self._trajectories.get(eid) for eid in episode_ids]

    def cleanup(self, episode_ids: list[int]) -> None:
        for eid in episode_ids:
            self._release_episode(eid)
            self._trajectories.pop(eid, None)

    def close(self) -> None:
        """Subclasses extend and call ``super()``."""
        if self._rewards.external_terms:
            self._run_or_schedule(self._rewards.aclose())

    def get_tools_schema(self) -> list[dict[str, Any]] | None:
        """Passed to the rollout engine as ``tools=``; overridden by the tool-use envs."""
        return None

    def rollout_metrics(self, trajectory: Trajectory) -> dict[str, float]:
        """Per-episode metrics the trainer means, by full path: ``outcome/*`` (added by task envs), ``episode/*``,
        ``reward/*`` (sum to the reward) and the scorers' keys."""
        if REWARD_PENDING_KEY in trajectory.info:
            raise RuntimeError(
                "the episode's reward still owes its externally scored terms: the driver must await "
                "settle_async (EpisodeDispatcher does) before reading the episode"
            )
        metrics: dict[str, float] = {}
        if TOTAL_TOOL_CALLS_KEY in trajectory.info:
            metrics["episode/tool_calls"] = float(trajectory.info[TOTAL_TOOL_CALLS_KEY])
            fault = trajectory.info.get(SANDBOX_FAULT_KEY)
            metrics["episode/sandbox_infra_fault"] = 1.0 if fault == SANDBOX_FAULT_INFRA else 0.0
            metrics["episode/sandbox_agent_fault"] = 1.0 if fault == SANDBOX_FAULT_AGENT else 0.0
        # Apart: a termination rate cannot tell a cut-off turn, or one stopped inside its reasoning, from an answer.
        for key in (LENGTH_CUTOFF_TURNS_KEY, LENGTH_CUTOFF_IN_CALL_TURNS_KEY, EMPTY_TURNS_KEY):
            metrics[f"episode/{key}"] = float(trajectory.info.get(key, 0))
        counted = [m for m in trajectory.messages if m.role == "assistant" and m.reasoning_tokens is not None]
        if counted:
            metrics["episode/thinking_cap_turns"] = float(sum(m.reasoning_capped for m in counted))
        if OUTPUT_BUDGET_EXHAUSTED_KEY in trajectory.info:
            metrics["episode/output_budget_exhausted"] = 1.0 if trajectory.info[OUTPUT_BUDGET_EXHAUSTED_KEY] else 0.0
        metrics["episode/reasoning_cjk_rate"] = (
            1.0
            if any(
                m.role == "assistant" and m.thinking and _CJK_SCRIPT.search(m.thinking) for m in trajectory.messages
            )
            else 0.0
        )
        metrics.update(self._reward_metrics(trajectory))
        return metrics

    @abstractmethod
    def _reset_single(self, prompt: str | list[dict[str, str]], context: dict[str, Any] | None = None) -> Trajectory:
        """Initialize a single episode."""

    @abstractmethod
    def _step_single(self, trajectory: Trajectory, action: str, context: dict[str, Any] | None = None) -> StepOutcome:
        """Process a single action in the environment."""

    @abstractmethod
    def _grade_episode(self, trajectory: Trajectory, context: dict[str, Any] | None = None) -> EpisodeGrade:
        """Grade a finished episode: the objective in ``[0, 1]`` and the environment's own shaping terms."""

    @staticmethod
    def _contexts(contexts: list[dict[str, Any]] | None, batch: list) -> list[dict[str, Any] | None]:
        return [None] * len(batch) if contexts is None else contexts

    @staticmethod
    def _init_trajectory(
        prompt: str | list[dict[str, str]],
        context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        extra_info: dict[str, Any] | None = None,
    ) -> Trajectory:
        traj = Trajectory()
        context = context or {}

        if system_prompt:
            traj.add_message(Message.system(system_prompt))

        if isinstance(prompt, str):
            traj.add_message(Message.user(prompt))
        elif isinstance(prompt, list):
            for msg in prompt:
                traj.add_message(Message.from_dict(msg))

        traj.info.update({CONTEXT_KEY: context, COMPLETED_KEY: False})
        if extra_info:
            traj.info.update(extra_info)
        return traj

    def _register_episode(self, episode_id: int, trajectory: Trajectory, context: dict[str, Any] | None) -> EnvStep:
        """Shared by the sync and async reset paths."""
        self._bind_effort_profile(trajectory, context)
        trajectory.info["episode_id"] = episode_id
        self._trajectories[episode_id] = trajectory
        return self._env_step(trajectory)

    def _begin_step(
        self, episode_id: int, action: str, context: dict[str, Any] | None
    ) -> tuple[Trajectory, EnvStep | None]:
        """The trajectory with the action recorded, or its standing step when the episode is already done."""
        trajectory = self._episode(episode_id)
        if trajectory.done:
            return trajectory, self._env_step(trajectory)
        self._add_action_message(trajectory, action, context)
        return trajectory, None

    def _episode(self, episode_id: int) -> Trajectory:
        trajectory = self._trajectories.get(episode_id)
        if trajectory is None:
            raise ValueError(f"Episode {episode_id} not found")
        return trajectory

    def _episodes(self, episode_ids: list[int]) -> list[Trajectory]:
        return [self._episode(episode_id) for episode_id in episode_ids]

    def _add_action_message(self, trajectory: Trajectory, action: str, context: dict[str, Any] | None) -> None:
        ctx = context or {}
        tool_calls = ctx.get("tool_calls")
        # Uncapped, the message advertises more calls than tool-result messages, which the re-tokenizer rejects.
        cap = self.max_tool_calls_per_turn
        if cap is not None and tool_calls is not None and len(tool_calls) > cap:
            tool_calls = tool_calls[:cap]
        trajectory.add_message(
            Message.assistant(
                action,
                tool_calls=tool_calls,
                thinking=ctx.get("reasoning"),
                token_ids=ctx.get("token_ids"),
                token_logprobs=ctx.get("token_logprobs"),
                candidates=ctx.get("candidates"),
                routing_mask=ctx.get("routing_mask"),
                routing_prompt_tokens=ctx.get("routing_prompt_tokens"),
                prompt_token_ids=ctx.get("prompt_token_ids"),
                thinking_cap=ctx.get("thinking_cap"),
                reasoning_tokens=ctx.get("reasoning_tokens"),
                # An engine abort is a cut turn too: the fragment must never train as a natural stop.
                truncated=ctx.get("finish_reason") in ENGINE_CUT_FINISH_REASONS,
                cut_tool_calls=ctx.get(CUT_TOOL_CALLS_KEY),
            )
        )

    def _finalize_step(
        self,
        episode_id: int,
        trajectory: Trajectory,
        reward: float,
        done: bool,
        truncated: bool,
        info: dict[str, Any],
        context: dict[str, Any] | None,
    ) -> EnvStep:
        """End the episode on a booked sandbox fault (uncompleted) or past ``max_turns`` (truncated)."""
        done = done or SANDBOX_FAULT_KEY in trajectory.info
        if trajectory.num_turns >= self.max_turns and not done:
            truncated = True
            done = True

        trajectory.done = done
        trajectory.truncated = truncated
        trajectory.total_reward += reward
        trajectory.info.update(info)

        if done:
            self._settle_grade(trajectory, context)
            if self._rewards.external_terms and not self._cut_short(trajectory):
                trajectory.info[REWARD_PENDING_KEY] = True
            else:
                self._drop_grading_payload(trajectory)

        self._trajectories[episode_id] = trajectory
        return self._env_step(trajectory, reward)

    def _env_step(self, trajectory: Trajectory, reward: float = 0.0) -> EnvStep:
        """``reward`` is this step's delta; the rest mirrors ``trajectory`` as it stands."""
        return EnvStep(
            trajectory=trajectory,
            observation=[
                m.to_dict(include_thinking=True) for m in engine_view(trajectory.messages, self.carry_reasoning)
            ],
            reward=reward,
            done=trajectory.done,
            truncated=trajectory.truncated,
            info=trajectory.info,
        )

    def _release_episode(self, episode_id: int) -> None:  # optional hook; subclasses override
        """Release one episode's external resources (a sandbox session, a connection) as :meth:`cleanup` drops it."""

    def _run_or_schedule(self, coroutine) -> None:
        """A close hook is called from an actor's loop and from plain teardown alike."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None
        if loop is not None:
            # Held strongly until done: the loop keeps only weak references to its tasks.
            task = loop.create_task(coroutine)
            self._background_tasks.add(task)
            task.add_done_callback(self._background_tasks.discard)
            return
        try:
            asyncio.run(coroutine)
        except Exception:
            logger.debug("%s: the client's owning loop is already gone", type(self).__name__, exc_info=True)


class AsyncEnvironmentBase(EnvironmentBase):
    async def reset_async(
        self, prompts: list[str | list[dict[str, str]]], contexts: list[dict[str, Any]] | None = None
    ) -> tuple[list[int], list[EnvStep]]:
        async def reset_one(prompt, context):
            episode_id = next(self._episode_ids)
            return episode_id, self._register_episode(
                episode_id, await self._reset_single_async(prompt, context), context
            )

        results = await asyncio.gather(
            *[reset_one(p, c) for p, c in zip(prompts, self._contexts(contexts, prompts), strict=True)]
        )
        return [r[0] for r in results], [r[1] for r in results]

    async def step_async(
        self, episode_ids: list[int], actions: list[str], contexts: list[dict[str, Any]] | None = None
    ) -> list[EnvStep]:
        async def step_one(episode_id, action, context):
            trajectory, standing = self._begin_step(episode_id, action, context)
            if standing is not None:
                return standing
            result = await self._step_single_async(trajectory, action, context)
            return self._finalize_step(episode_id, *result, context)

        return await asyncio.gather(
            *[
                step_one(eid, act, ctx)
                for eid, act, ctx in zip(episode_ids, actions, self._contexts(contexts, episode_ids), strict=True)
            ]
        )

    async def _reset_single_async(
        self, prompt: str | list[dict[str, str]], context: dict[str, Any] | None = None
    ) -> Trajectory:
        """Override for async I/O."""
        return self._reset_single(prompt, context)

    async def _step_single_async(
        self, trajectory: Trajectory, action: str, context: dict[str, Any] | None = None
    ) -> StepOutcome:
        """Override for async I/O."""
        return self._step_single(trajectory, action, context)
