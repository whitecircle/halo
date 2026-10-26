"""The rollout-driver contract: the one episode loop (:func:`play_episode`) the Ray actor and the eval runner
both play, the turn generation and retry policy it runs, and the row an episode hands the trainer."""

import asyncio
import contextvars
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field, replace
from functools import partial
from typing import Any

import backoff

from src.environments.environment_base import AsyncEnvironmentBase, EnvironmentBase
from src.environments.episode_keys import EPISODE_ERROR_KEY, LAST_TURN_KEY
from src.environments.messages import EnvStep, Trajectory
from src.environments.reasoning_effort import EpisodeEffort
from src.environments.rollout_config import RolloutConfig
from src.inference.openai_client import TRANSIENT_CLIENT_ERROR_CODES
from src.inference.response import ENGINE_CUT_FINISH_REASONS, FINISH_REASON_ABORT, FINISH_REASON_LENGTH
from src.rewards.samples import CUT_TOOL_CALLS_KEY

logger = logging.getLogger(__name__)

# What an engine's client error says, lowercased, when the conversation outgrew the served context (vLLM, SGLang,
# and the OpenAI API's error code).
CONTEXT_OVERFLOW_MARKERS = ("context length", "maximum model length", "context_length_exceeded")


@dataclass
class RolloutResult:
    """One collected episode as the rollout drivers hand it to the trainer; every field is pickled through Ray and
    the TP broadcast per episode, so each needs a consumer (checked by ``test_ray_actors.py``)."""

    prompt: str
    trajectory: Trajectory | None = None
    episode_length: int = 0
    total_reward: float = 0.0
    success: bool = False
    latency: float = 0.0
    error: str | None = None
    generation_tokens: int = 0
    requests_expired_in_sync: int = 0
    """Request deadlines that expired on a request in flight across a weight-sync pause, pause credited."""
    metrics: dict[str, float] = field(default_factory=dict)

    @property
    def counts_toward_baseline(self) -> bool:
        """Whether the episode counts toward the GRPO group baseline: not when the rollout errored or the
        environment marked its reward as carrying no signal, which would bias every sibling's advantage."""
        return not self.error and not (self.trajectory is not None and self.trajectory.episode_invalid)

    @classmethod
    def errored(cls, prompt: str, reason: str, **fields: Any) -> "RolloutResult":
        """The row of an episode its driver lost: no reward, outside the group baseline (``error``), its
        trajectory carrying only the cause."""
        return cls(
            prompt=prompt, trajectory=Trajectory(done=True, info={EPISODE_ERROR_KEY: reason}), error=reason, **fields
        )


@dataclass(frozen=True)
class TurnGeneration:
    """One assistant turn's generation from the rollout engine; the capture fields (sampled ids, their logprobs,
    MoE routing, the engine's prompt ids) are ``None`` unless the ``RolloutConfig`` requested and the server sent them."""

    text: str
    tool_calls: list
    reasoning: str
    tokens: int
    finish_reason: str | None = None
    """Why generation stopped; ``"length"`` and ``"abort"`` both mean the text is a fragment."""
    token_ids: list[int] | None = None
    token_logprobs: list[float] | None = None
    routing_mask: str | None = None
    routing_prompt_tokens: int | None = None
    prompt_token_ids: list[int] | None = None


def describe_exception(exc: BaseException) -> str:
    """``Type: message``, keeping the type when the message is empty (a stalled engine's ``TimeoutError``)."""
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def is_engine_fault(status: int, body: str) -> bool:
    """A client-error status the engine returned for its own fault, which a fresh request clears: vLLM answers
    400 when a NaN log-prob keeps it from serialising its own response to a valid request."""
    return 400 <= status < 500 and "not JSON compliant" in body


def is_terminal_client_status(status: int, body: str) -> bool:
    """A client error the request itself caused (the conversation outgrew the served context, a
    malformed request): retrying the same request cannot succeed."""
    return 400 <= status < 500 and status not in TRANSIENT_CLIENT_ERROR_CODES and not is_engine_fault(status, body)


def is_context_overflow(status: int, body: str) -> bool:
    """A terminal client error reporting that the conversation outgrew the served context
    (:data:`CONTEXT_OVERFLOW_MARKERS`), the one terminal client error an episode's own length causes."""
    lowered = body.lower()
    return is_terminal_client_status(status, body) and any(marker in lowered for marker in CONTEXT_OVERFLOW_MARKERS)


async def generate_turn(
    request: Callable[[], Awaitable[TurnGeneration]],
    config: RolloutConfig,
    *,
    retry_on: tuple[type[BaseException], ...],
    giveup: Callable[[BaseException], bool],
    log_prefix: str,
    on_failure: Callable[[BaseException], None] | None = None,
) -> TurnGeneration:
    """One turn under the retry policy every driver shares: a failed ``retry_on`` request is retried with backoff
    up to ``config.max_retries`` times unless ``giveup`` names it terminal, an engine abort re-issued as often;
    ``on_failure`` sees every failed request."""

    def _failed(details: dict) -> None:
        if on_failure is not None:
            on_failure(details["exception"])

    def _exhausted(details: dict) -> None:
        # A terminal failure is the caller's to report; only a spent retry budget is logged here.
        if not giveup(details["exception"]):
            logger.warning(
                f"{log_prefix}: gave up after {details['tries']} tries — {describe_exception(details['exception'])}"
            )

    @backoff.on_exception(
        backoff.expo,
        retry_on,
        # backoff counts total attempts: max_retries=0 would never match and retry forever.
        max_tries=config.max_retries + 1,
        factor=config.retry_base_wait,
        giveup=giveup,
        logger=None,
        on_backoff=[
            _failed,
            lambda d: logger.warning(
                f"{log_prefix}: rollout retry {d['tries']}/{config.max_retries} after {d['wait']:.1f}s — "
                f"{describe_exception(d['exception'])}"
            ),
        ],
        on_giveup=[_failed, _exhausted],
    )
    async def _attempt() -> TurnGeneration:
        return await request()

    for _ in range(config.max_retries + 1):
        gen = await _attempt()
        if gen.finish_reason != FINISH_REASON_ABORT:
            return gen
        logger.warning(f"{log_prefix}: the {config.backend} engine aborted the turn; re-issuing it")
    raise RuntimeError(
        f"the {config.backend} engine aborted the same turn {config.max_retries + 1} times in a row "
        f"(finish_reason={FINISH_REASON_ABORT!r}); the turn is not stepped with a fragment"
    )


def sampled_reasoning_tokens(token_ids: list[int] | None, reasoning_end_token_id: int | None) -> int | None:
    """The reasoning tokens in a turn's sampled ids: up to and including the reasoning-end token, or every id when
    the turn was cut before its reasoning closed; ``None`` without the ids or the end token's id. On a turn vLLM
    force-closed this is the budget it enforced, one past where the prompt ends on a bare ``<think>``."""
    if token_ids is None or reasoning_end_token_id is None:
        return None
    try:
        return token_ids.index(reasoning_end_token_id) + 1
    except ValueError:
        return len(token_ids)


def step_context_from_generation(
    context: dict[str, Any] | None,
    gen: TurnGeneration,
    *,
    thinking_cap: int | None = None,
    reasoning_end_token_id: int | None = None,
    last_turn: bool = False,
) -> dict[str, Any]:
    """The per-turn context an ``env.step`` receives, stamped from one turn's generation: ``finish_reason``, the
    calls, the reasoning, the captures where id-aligned, the reasoning cap with the reasoning sampled under it,
    and ``last_turn`` when the output budget affords no further turn."""
    step_ctx = dict(context) if context else {}
    step_ctx["finish_reason"] = gen.finish_reason
    if last_turn:
        step_ctx[LAST_TURN_KEY] = True
    # A cut turn's calls never run: a token-cap cut carries them apart, for the recovery's in-call nudge and the
    # judge; an abort names no cause to pass on.
    if gen.tool_calls:
        if gen.finish_reason not in ENGINE_CUT_FINISH_REASONS:
            step_ctx["tool_calls"] = gen.tool_calls
        elif gen.finish_reason == FINISH_REASON_LENGTH:
            step_ctx[CUT_TOOL_CALLS_KEY] = gen.tool_calls
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
    if thinking_cap is not None:
        step_ctx["thinking_cap"] = thinking_cap
    reasoning_tokens = sampled_reasoning_tokens(gen.token_ids, reasoning_end_token_id)
    if reasoning_tokens is not None:
        step_ctx["reasoning_tokens"] = reasoning_tokens
    return step_ctx


class EpisodeDispatcher:
    """Routes one episode's env calls inline on the loop for an :class:`AsyncEnvironmentBase`, else to a worker
    thread in one context copied per episode, so a tool's ContextVar writes survive to the next turn."""

    def __init__(self, env: EnvironmentBase):
        self.env = env
        self._is_async = isinstance(env, AsyncEnvironmentBase)
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
        """Score the external terms of every episode these steps closed, on the loop for any env."""
        done = [eid for eid, step in zip(episode_ids, steps, strict=True) if step.done]
        if done:
            await self.env.settle_async(done)
        return steps


@dataclass(frozen=True)
class PlayedEpisode:
    """One episode as :func:`play_episode` played it: its closed trajectory, every turn generation it stepped,
    the tokens they sampled, and the generation failure that ended it early, where one did."""

    trajectory: Trajectory | None
    turns: list[TurnGeneration]
    generated: int
    lost_turn: Exception | None = None


async def play_episode(
    env: EnvironmentBase,
    prompt: str | list[dict[str, Any]],
    context: dict[str, Any] | None,
    effort: EpisodeEffort,
    request_turn: Callable[[list[dict[str, Any]], RolloutConfig, EpisodeEffort], Awaitable[TurnGeneration]],
    rollout: RolloutConfig,
    *,
    log_prefix: str,
    keep_lost_episode: bool = False,
) -> PlayedEpisode:
    """Play one episode, the loop every driver shares: reset, request (``request_turn``, the driver's transport and
    retries) and step each turn under ``effort``'s caps, close it truncated where still open, stamp the effort,
    and release it on every path. A lost turn raises, unless ``keep_lost_episode``: then the episode closes on
    what it earned, stamped :data:`EPISODE_ERROR_KEY`, the failure returned."""
    episode = EpisodeDispatcher(env)
    episode_ids, steps = await episode.reset([prompt], [context])
    eid, step = episode_ids[0], steps[0]
    turns: list[TurnGeneration] = []
    generated = 0
    lost: Exception | None = None
    try:
        for _ in range(env.max_turns):
            # The engine caps this turn: the level's reasoning cap and the turn total, both narrowed to what the
            # output budget has left; none once it holds no turn.
            caps = effort.turn_caps(generated)
            if step.done or caps is None:
                break
            try:
                gen = await request_turn(step.observation, replace(rollout, **caps), effort)
            except Exception as exc:
                if not keep_lost_episode:
                    raise
                lost = exc
                break
            turns.append(gen)
            generated += gen.tokens
            step_ctx = step_context_from_generation(
                context,
                gen,
                thinking_cap=effort.thinking_cap,
                reasoning_end_token_id=rollout.reasoning_end_token_id,
                last_turn=effort.turn_caps(generated) is None,
            )
            step = (await episode.step([eid], [gen.text], [step_ctx]))[0]
        trajectory = env.get_trajectories([eid])[0]
        if trajectory is not None and not trajectory.done:
            # Closed by explicit truncation, never a synthetic empty turn, which would mark it ``completed`` and pay
            # completion-graded envs; what it earned still counts, priced like a max_turns overflow unless lost.
            if lost is not None:
                trajectory.info[EPISODE_ERROR_KEY] = describe_exception(lost)
            trajectory = (await episode.finalize_truncated([eid]))[0].trajectory
        effort.stamp(trajectory, generated)
        return PlayedEpisode(trajectory, turns, generated, lost)
    finally:
        # On every path: a stateful env (SweEnvironment) leaks a sandbox working dir per episode.
        try:
            env.cleanup([eid])
        except Exception:  # cleanup must never mask the episode result
            logger.debug("%s: cleanup failed for episode %s", log_prefix, eid, exc_info=True)
