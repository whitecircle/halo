"""The rollout-driver half of the environment contract: what a driver binds, generates and hands back.

The Ray actor and the eval runner both drive an episode through here, so an episode is generated and
graded the same way whichever one collects it.
"""

import asyncio
import contextvars
from dataclasses import dataclass, field
from functools import partial
from typing import Any

from src.environments.base import (
    AsyncBaseEnvironment,
    BaseEnvironment,
    EnvStep,
    Trajectory,
    resolve_reasoning_effort,
)
from src.inference.response import ENGINE_CUT_FINISH_REASONS

# Reasoning-budget calibration band, as fractions of the episode's applied CoT budget, and the two
# penalty weights outside it (over-use weighs more than under-use). See
# :func:`reasoning_calibration_penalty`. The under-use weight is a run knob, so it is also the default
# ``AsyncTrainingConfig.reasoning_compliance_under_use_weight`` reads: one constant keeps the pair in step.
_CALIBRATION_BAND_LO = 0.3
_CALIBRATION_BAND_HI = 0.9
DEFAULT_UNDER_USE_WEIGHT = 0.3
_OVER_USE_WEIGHT = 1.0


@dataclass(frozen=True)
class EpisodeEffort:
    """The generation contract one episode runs under: its resolved reasoning-effort level, the CoT
    budget bound to that level, and the turn's total token cap."""

    level: str | None
    thinking_budget: int | None
    max_tokens: int

    def stamp(self, trajectory: Trajectory | None) -> None:
        """Record this contract on the episode's trajectory (no-op when the episode produced none).

        Every rollout driver stamps through here: re-tokenization must render the same steer the model
        generated under, and the calibration reward prices CoT against the applied budget."""
        if trajectory is None:
            return
        trajectory.reasoning_effort = self.level
        trajectory.reasoning_budget = self.thinking_budget


def resolve_episode_effort(context: dict[str, Any] | None, env: BaseEnvironment) -> str | None:
    """The episode's concrete reasoning-effort level: context-supplied first, else the env setting.

    A context-supplied level takes precedence because the trainer stamps a single group-level draw into
    every group member's context; a ``'random'`` env setting drawn independently per episode would be
    intra-group conditioning noise, and GRPO's group baseline assumes the members of a group share
    identical conditioning. Either source resolves through :func:`resolve_reasoning_effort`
    (``'random'`` draws a concrete level).
    """
    return resolve_reasoning_effort((context or {}).get("reasoning_effort") or env.reasoning_effort)


def bind_episode_effort(
    context: dict[str, Any] | None,
    env: BaseEnvironment,
    *,
    max_tokens: int,
    max_thinking_tokens: int | None = None,
) -> EpisodeEffort:
    """Resolve one episode's effort level and bind the env's per-level CoT budget into its token caps.

    Every rollout driver (the Ray actor, the eval runner) binds through here, so an episode runs under
    the same contract whichever one collects it. Call once per episode: the level may be a ``'random'``
    draw and every turn must share it.

    The thinking budget caps only the reasoning channel; the visible channel would otherwise run to the
    global ``max_tokens`` and crowd out the tool call the turn exists to make. The per-effort total is
    therefore the level's budget plus the global answer headroom (``max_tokens - max_thinking_tokens``),
    so an effort level bounds the whole turn rather than its reasoning alone.
    """
    level = resolve_episode_effort(context, env)
    budget = env.thinking_budget_for_effort(level) if level is not None else None
    if budget is None:
        # No per-level budget (or no level at all): the global caps stand.
        return EpisodeEffort(level=level, thinking_budget=max_thinking_tokens, max_tokens=max_tokens)
    if max_thinking_tokens is not None:
        budget = min(budget, max_thinking_tokens)
        headroom = max(0, max_tokens - max_thinking_tokens)
    else:
        headroom = max_tokens
    return EpisodeEffort(level=level, thinking_budget=budget, max_tokens=min(max_tokens, budget + headroom))


def reasoning_calibration_penalty(
    reasoning_tokens: list[int], budget: int, under_use_weight: float = DEFAULT_UNDER_USE_WEIGHT
) -> float:
    """Asymmetric reasoning-budget calibration penalty (0 = compliant), averaged over turns. Per turn,
    given reasoning tokens ``r`` and budget ``B``: inside the compliant band → 0; below →
    ``-under_use_weight`` scaled by the shortfall (``-under_use_weight`` at ``r = 0``); above → strong
    penalty, ``-1`` once ``r >= B``. Over-use is punished harder than under-use.

    The band and the over-use weight are the term's definition, not run knobs — the config help and
    ``agent-docs/training-methods/grpo/async-grpo/README.md`` state them. ``under_use_weight`` is the
    run knob ``reasoning_compliance_under_use_weight`` (0 disables the below-band side), and
    ``reasoning_compliance_weight`` scales the whole term.
    """
    if not reasoning_tokens or budget <= 0:
        return 0.0
    lo, hi = _CALIBRATION_BAND_LO * budget, _CALIBRATION_BAND_HI * budget
    penalties = []
    for r in reasoning_tokens:
        if r < lo:
            penalties.append(-under_use_weight * (lo - r) / lo if lo > 0 else 0.0)
        elif r > hi:
            over = (r - hi) / max(budget - hi, 1.0)
            penalties.append(-_OVER_USE_WEIGHT * min(over, 1.0))
        else:
            penalties.append(0.0)
    return sum(penalties) / len(penalties)


@dataclass
class RolloutResult:
    """One collected episode, as the rollout drivers hand it to the trainer.

    Every field is pickled through the Ray object store and again through the TP broadcast, once per
    episode, so each one needs a consumer (checked by ``tests/cpu/environments/test_ray_actors.py``).
    """

    prompt: str
    trajectory: Trajectory | None = None
    episode_length: int = 0
    total_reward: float = 0.0
    success: bool = False
    latency: float = 0.0
    error: str | None = None
    generation_tokens: int = 0
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class TurnGeneration:
    """One assistant turn's generation from the rollout engine — everything a trainer may capture.

    The optional fields are engine-side captures (exact sampled ids, behavior-policy logprobs, MoE
    routing, the engine-rendered prompt ids); each is ``None`` unless its ``RolloutConfig`` capture
    flag requested it and the server returned it.
    """

    text: str
    tool_calls: list
    reasoning: str
    tokens: int
    finish_reason: str | None = None
    """Why generation stopped. ``"length"`` (token cap) and ``"abort"`` (the engine dropped the
    request, e.g. a pause in abort mode) both mean the text is a fragment rather than a completed
    answer."""
    token_ids: list[int] | None = None
    token_logprobs: list[float] | None = None
    routing_mask: str | None = None
    routing_prompt_tokens: int | None = None
    prompt_token_ids: list[int] | None = None


def step_context_from_generation(context: dict[str, Any] | None, gen: TurnGeneration) -> dict[str, Any]:
    """The per-turn context an ``env.step`` receives, stamped from one turn's generation.

    Shared by every rollout driver (the Ray actor's raw aiohttp transport, the eval runner's OpenAI
    client): a driver that omits ``finish_reason`` grades an engine-cut fragment as a deliberate final
    answer. The capture keys are forwarded only where they are id-aligned, since an unaligned logprob
    or routing vector would produce incorrect training data rather than a missing field.
    """
    step_ctx = dict(context) if context else {}
    step_ctx["finish_reason"] = gen.finish_reason
    # A cut turn is a fragment whatever the parser salvaged from it: the call it holds was never
    # finished, and executing it books a malformed call and trains the fragment as a normal row.
    if gen.tool_calls and gen.finish_reason not in ENGINE_CUT_FINISH_REASONS:
        step_ctx["tool_calls"] = gen.tool_calls
    if gen.reasoning:
        step_ctx["reasoning"] = gen.reasoning
    # ``is None``, not truthiness: an empty capture is a zero-token turn the engine did return ids
    # for, and it must reach the trainer as one rather than as a capture failure.
    captured = gen.token_ids is not None
    if captured:
        step_ctx["token_ids"] = gen.token_ids
    # Behavior-policy logprobs for the IS trust region; keep only when id-aligned.
    if captured and gen.token_logprobs is not None and len(gen.token_logprobs) == len(gen.token_ids):
        step_ctx["token_logprobs"] = gen.token_logprobs
    # Engine routing for R3 replay; only meaningful beside the ids it aligns with.
    if captured and gen.routing_mask:
        step_ctx["routing_mask"] = gen.routing_mask
        step_ctx["routing_prompt_tokens"] = gen.routing_prompt_tokens
    if captured and gen.prompt_token_ids:
        step_ctx["prompt_token_ids"] = gen.prompt_token_ids
    return step_ctx


class EpisodeDispatcher:
    """Routes one episode's ``reset``/``step``/``finalize_truncated`` onto the env's own execution path.

    Every rollout driver (the Ray actor, the eval runner) goes through this, so an episode is driven
    the same way whichever one collects it. An :class:`AsyncBaseEnvironment` runs inline on the event
    loop; a sync env is offloaded to a worker thread, since its tool and grading handlers block on
    sandboxed execution for minutes and would stall every episode sharing the loop.

    The offload reuses a single context copied per episode rather than the fresh per-call copy
    ``asyncio.to_thread`` makes, so a tool's ContextVar writes (the simulated file store) stay visible
    on the next turn.
    """

    def __init__(self, env: BaseEnvironment):
        self.env = env
        self._is_async = isinstance(env, AsyncBaseEnvironment)
        self._context = None if self._is_async else contextvars.copy_context()

    async def _offload(self, fn, *args):
        """Run a sync env call in a worker thread, inside this episode's copied context."""
        return await asyncio.get_running_loop().run_in_executor(None, partial(self._context.run, fn, *args))

    async def reset(
        self, prompts: list[str | list[dict[str, str]]], contexts: list[dict[str, Any] | None]
    ) -> tuple[list[int], list[EnvStep]]:
        if self._is_async:
            return await self.env.reset_async(prompts, contexts)
        return await self._offload(self.env.reset, prompts, contexts)

    async def step(
        self, episode_ids: list[int], actions: list[str], contexts: list[dict[str, Any] | None]
    ) -> list[EnvStep]:
        if self._is_async:
            steps = await self.env.step_async(episode_ids, actions, contexts)
        else:
            steps = await self._offload(self.env.step, episode_ids, actions, contexts)
        return await self._settled(episode_ids, steps)

    async def finalize_truncated(self, episode_ids: list[int]) -> list[EnvStep]:
        """Close still-open episodes as truncated, keeping the reward they already earned."""
        if self._is_async:
            steps = self.env.finalize_truncated(episode_ids)
        else:
            steps = await self._offload(self.env.finalize_truncated, episode_ids)
        return await self._settled(episode_ids, steps)

    async def _settled(self, episode_ids: list[int], steps: list[EnvStep]) -> list[EnvStep]:
        """Score the externally rewarded terms of every episode these steps closed. Pure I/O over the
        finished trajectory, so it runs on the loop for sync and async envs alike; an episode whose
        reward has no external term is already settled and costs nothing here."""
        done = [eid for eid, step in zip(episode_ids, steps, strict=True) if step.done]
        if done:
            await self.env.settle_async(done)
        return steps
