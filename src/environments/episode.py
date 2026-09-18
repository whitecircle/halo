"""The rollout-driver half of the environment contract: what a driver binds, generates and hands back.

The Ray actor and the eval runner both drive an episode through here, so an episode is generated and
graded the same way whichever one collects it.
"""

import asyncio
import contextvars
import math
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


@dataclass(frozen=True)
class EpisodeEffort:
    """The generation contract one episode runs under: its resolved reasoning-effort level, the CoT
    budget bound to that level, and the turn's total token cap."""

    level: str | None
    thinking_budget: int | None
    max_tokens: int

    def stamp(self, trajectory: Trajectory | None) -> None:
        """Record this contract on the episode's trajectory (no-op when the episode produced none).

        Every rollout driver stamps through here: re-tokenization has to render the level and budget
        the model generated under, and the effort length terms price the episode by its level."""
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


def effort_length_penalty(
    reasoning_tokens: list[int], effort: float, effort_min: float, k0: float, tau: float, c_max: float, l_norm: float
) -> float:
    """Capped, effort-conditioned reasoning-length price in ``[-c_max, 0]``:
    ``-min(c_max, k(effort) * sum(reasoning_tokens) / l_norm)`` with ``k(effort) = k0 * exp(-(effort - effort_min) / tau)``.

    The coefficient falls by ``e`` per ``tau`` effort units above the lowest level, so the same trace
    costs most at the lowest effort; the cap keeps a long trace from outweighing the task reward, which
    an uncapped per-token price does. Prices reasoning tokens only, summed over the trajectory's turns."""
    tokens = sum(reasoning_tokens)
    if tokens <= 0:
        return 0.0
    k = k0 * math.exp(-(effort - effort_min) / tau)
    return -min(c_max, k * tokens / l_norm)


def effort_length_floor(reasoning_tokens: list[int], min_tokens: int, weight: float) -> float:
    """Under-use floor in ``[-weight, 0]``: ``-weight * (min_tokens - total) / min_tokens`` while the
    trajectory's reasoning tokens fall short of ``min_tokens``, a multiple of its per-turn budget.

    The price only ever pays for less reasoning; this is the term that resists reasoning shrinking
    toward nothing. Summed over the episode, never averaged per turn: a short repair turn after a
    verdict is not under-use, and an extra tool turn cannot lower the score. An episode with no
    assistant turn is a lost one, not under-use, and pays nothing; turns that carry no reasoning at
    all pay the whole weight."""
    shortfall = min_tokens - sum(reasoning_tokens)
    if not reasoning_tokens or min_tokens <= 0 or weight <= 0 or shortfall <= 0:
        return 0.0
    return -weight * shortfall / min_tokens


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
