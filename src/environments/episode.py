"""The rollout-driver half of the environment contract: what a driver binds, generates and hands back.

The Ray actor and the eval runner both drive an episode through here, so an episode is generated and
graded the same way whichever one collects it.
"""

import asyncio
import contextvars
import logging
import math
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import backoff

from src.configs.rollout_config import (
    DEFAULT_THINKING_TURN_RESERVE,
    THINKING_SCOPE_EPISODE,
    THINKING_SCOPE_TURN,
    RolloutConfig,
)
from src.environments.base import (
    THINKING_BUDGET_EXHAUSTED_KEY,
    VALID_REASONING_EFFORTS,
    AsyncBaseEnvironment,
    BaseEnvironment,
    EnvStep,
    Trajectory,
    resolve_reasoning_effort,
)
from src.inference.response import ENGINE_CUT_FINISH_REASONS, FINISH_REASON_ABORT

logger = logging.getLogger(__name__)

# vLLM answers 400 when a NaN log-prob keeps it from serialising its OWN response: the request was
# valid and a fresh one succeeds, so both drivers retry it rather than lose the turn.
ENGINE_SERIALIZATION_FAULT = "not JSON compliant"
# Client-error statuses that report a transient server condition, not a bad request.
RETRYABLE_4XX = frozenset({408, 429})
# What an engine's client error says when the conversation outgrew the served context, lowercased:
# vLLM's and SGLang's "maximum context length" / "model's context length", vLLM's "maximum model
# length", and the OpenAI API's error code.
CONTEXT_OVERFLOW_MARKERS = ("context length", "maximum model length", "context_length_exceeded")


@dataclass(frozen=True)
class EpisodeEffort:
    """The generation contract one episode runs under: its resolved reasoning-effort level, the CoT
    budget bound to that level, what that budget covers, and the turn's total token cap."""

    level: str | None
    thinking_budget: int | None
    max_tokens: int
    scope: str = THINKING_SCOPE_TURN
    turn_reserve: int = DEFAULT_THINKING_TURN_RESERVE
    """The reasoning a turn keeps once the episode's budget is spent; :func:`validate_thinking_budget_scope`
    refuses one above a level's budget."""
    turn_ceiling: int | None = None
    """The run's per-turn reasoning ceiling (``rollout_max_thinking_tokens``), read under the episode scope."""

    def turn_thinking_cap(self, reasoning_spent: int) -> int | None:
        """The engine's reasoning cap for the turn about to be generated.

        Per-turn scope: the bound budget, every turn. Episode scope: what the budget has left after
        the reasoning the earlier turns spent, never below the reserve (a turn always gets enough to
        close its reasoning and act) and never above the run's per-turn ceiling."""
        if self.thinking_budget is None or self.scope == THINKING_SCOPE_TURN:
            return self.thinking_budget
        remaining = max(self.thinking_budget - reasoning_spent, self.turn_reserve)
        return remaining if self.turn_ceiling is None else min(remaining, self.turn_ceiling)

    def spend_of(self, gen: "TurnGeneration", reasoning_end_token_id: int | None) -> int:
        """The reasoning a generated turn charges against the episode's budget: nothing under the
        per-turn scope, else :func:`reasoning_tokens_of`."""
        if self.scope == THINKING_SCOPE_TURN:
            return 0
        return reasoning_tokens_of(gen, reasoning_end_token_id)

    def budget_exhausted(self, reasoning_spent: int) -> bool:
        """Whether the episode's budget has run down to the reserve: a further turn reasons only that."""
        return (
            self.scope == THINKING_SCOPE_EPISODE
            and self.thinking_budget is not None
            and self.thinking_budget - reasoning_spent <= self.turn_reserve
        )

    def stamp(self, trajectory: Trajectory | None, reasoning_spent: int | None = None) -> None:
        """Record this contract on the episode's trajectory (no-op when the episode produced none).

        Every rollout driver stamps through here: re-tokenization has to render the level and budget
        the model generated under, and the effort length terms price the episode by its level. Under
        the episode scope the driver also hands over the reasoning the episode spent, recorded as
        whether the budget ran out."""
        if trajectory is None:
            return
        trajectory.reasoning_effort = self.level
        trajectory.reasoning_budget = self.thinking_budget
        if self.scope == THINKING_SCOPE_EPISODE and reasoning_spent is not None:
            trajectory.info[THINKING_BUDGET_EXHAUSTED_KEY] = self.budget_exhausted(reasoning_spent)


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
    scope: str = THINKING_SCOPE_TURN,
    turn_reserve: int = DEFAULT_THINKING_TURN_RESERVE,
) -> EpisodeEffort:
    """Resolve one episode's effort level and bind the env's per-level CoT budget into its token caps.

    Every rollout driver (the Ray actor, the eval runner) binds through here, so an episode runs under
    the same contract whichever one collects it. Call once per episode: the level may be a ``'random'``
    draw and every turn must share it.

    The thinking budget caps only the reasoning channel; the visible channel would otherwise run to the
    global ``max_tokens`` and crowd out the tool call the turn exists to make. The per-effort total is
    therefore the first turn's reasoning cap plus the global answer headroom
    (``max_tokens - max_thinking_tokens``), so an effort level bounds the whole turn rather than its
    reasoning alone.

    ``max_thinking_tokens`` is the run's per-turn ceiling. Under the ``turn`` scope it clamps the level's
    budget; under the ``episode`` scope the level's budget is the episode's total and the ceiling bounds
    only how much of it one turn may take (:meth:`EpisodeEffort.turn_thinking_cap`).
    """
    level = resolve_episode_effort(context, env)
    budget = env.thinking_budget_for_effort(level) if level is not None else None
    if budget is None:
        if scope == THINKING_SCOPE_EPISODE and max_thinking_tokens is None:
            raise ValueError(
                "rollout_thinking_budget_scope='episode' with nothing to share: the episode's level sets no "
                "thinking_tokens and rollout_max_thinking_tokens is unset, so no turn would be capped"
            )
        # No per-level budget (or no level at all): the global caps stand.
        return EpisodeEffort(
            level=level,
            thinking_budget=max_thinking_tokens,
            max_tokens=max_tokens,
            scope=scope,
            turn_reserve=turn_reserve,
            turn_ceiling=max_thinking_tokens,
        )
    if scope == THINKING_SCOPE_TURN and max_thinking_tokens is not None:
        budget = min(budget, max_thinking_tokens)
    if max_thinking_tokens is not None:
        first_turn_cap = min(budget, max_thinking_tokens)
        headroom = max(0, max_tokens - max_thinking_tokens)
    else:
        first_turn_cap = budget
        headroom = max_tokens
    return EpisodeEffort(
        level=level,
        thinking_budget=budget,
        max_tokens=min(max_tokens, first_turn_cap + headroom),
        scope=scope,
        turn_reserve=turn_reserve,
        turn_ceiling=max_thinking_tokens,
    )


def validate_thinking_budget_scope(
    env: BaseEnvironment, *, scope: str, max_thinking_tokens: int | None, turn_reserve: int
) -> None:
    """Refuse an episode thinking scope that some episode could not bind through :func:`bind_episode_effort`.

    With ``max_thinking_tokens`` unset an episode's budget is its level's ``thinking_tokens`` alone, so the
    env must resolve a level for every episode (``reasoning_effort`` set) and every level must carry a
    budget: otherwise each episode that lands on the gap fails at its first turn, a masked row in
    training and a zero-reward error sample in an eval. A level's budget must also hold ``turn_reserve``,
    the reasoning a spent turn keeps: above the episode's whole budget the first turn would already take
    more than the total the template states. The per-turn scope shares nothing and passes.

    The trainer runs it at construction and the eval runner before its first episode, so a gap is
    refused before any episode is generated.
    """
    if scope != THINKING_SCOPE_EPISODE:
        return
    budgets = {level: env.thinking_budget_for_effort(level) for level in VALID_REASONING_EFFORTS}
    if max_thinking_tokens is None:
        unbudgeted = [level for level, budget in budgets.items() if budget is None]
        if env.reasoning_effort is None or unbudgeted:
            raise ValueError(
                "rollout_thinking_budget_scope='episode' would leave episodes with nothing to share: with "
                "rollout_max_thinking_tokens unset, every episode needs a level (reasoning_effort, got "
                f"{env.reasoning_effort!r}) whose profile sets thinking_tokens (unset for {unbudgeted}). Set both, "
                "or set rollout_max_thinking_tokens, the budget of an episode its level leaves unbudgeted."
            )
    short = {level: budget for level, budget in budgets.items() if budget is not None and budget < turn_reserve}
    if short:
        raise ValueError(
            f"rollout_thinking_turn_reserve ({turn_reserve}) exceeds the thinking_tokens of {short}: a turn's "
            "reserve cannot be more than the episode's whole budget."
        )


def resolve_reasoning_end_token_id(tokenizer, token: str) -> int:
    """The id of ``token`` under the tokenizer, required to resolve: a marker the tokenizer does not
    know would count every turn's whole generation as reasoning and starve the episode of its budget
    after the first turn."""
    tid = tokenizer.convert_tokens_to_ids(token)
    if tid is None or tid == getattr(tokenizer, "unk_token_id", None):
        raise ValueError(
            f"rollout_reasoning_end_token {token!r} is not a token of this tokenizer; the episode thinking scope "
            "counts a turn's reasoning as the sampled ids up to and including that token, so name the model's "
            "own marker"
        )
    return tid


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
    trajectory's reasoning tokens fall short of ``min_tokens``, a multiple of the budget it ran under.

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
    requests_expired_in_sync: int = 0
    """Request deadlines that expired on a request in flight across a weight-sync pause, pause credited."""
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


def describe_exception(exc: BaseException) -> str:
    """``Type: message``, keeping the type when the message is empty.

    ``asyncio.TimeoutError``, the usual signature of a stalled engine, stringifies to ``""``, so
    interpolating the exception alone carries no information.
    """
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


def is_engine_fault(status: int, body: str) -> bool:
    """A client-error status the engine returned for its own fault, which a fresh request clears."""
    return 400 <= status < 500 and ENGINE_SERIALIZATION_FAULT in body


def is_terminal_client_status(status: int, body: str) -> bool:
    """A client error the request itself caused (the conversation outgrew the served context, a
    malformed request): retrying the same request cannot succeed."""
    return 400 <= status < 500 and status not in RETRYABLE_4XX and not is_engine_fault(status, body)


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
    """One turn under the rollout retry policy every driver shares.

    A failed ``request`` of a ``retry_on`` type is retried with exponential backoff from
    ``config.retry_base_wait``, up to ``config.max_retries`` times, unless ``giveup`` names it terminal.
    A generation the engine aborted is re-issued for the same observation, up to ``config.max_retries``
    times: the fragment is the engine's doing (SGLang's sync pause drops every in-flight request), so
    stepping it would spend the episode's length-cutoff recovery cap on a cut the policy did not make.
    Past either bound the turn raises. ``on_failure`` sees every failed ``retry_on`` request, retried or not.
    """

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


def reasoning_tokens_of(gen: TurnGeneration, reasoning_end_token_id: int | None) -> int:
    """The reasoning tokens one turn spent, read off the engine's sampled ids: the ids up to and
    including the reasoning-end token (the engine's budget counts the close it forces), or all of them
    when the turn was cut before its reasoning closed.

    The engine's usage block carries no reasoning count and the drivers hold no tokenizer, so the ids
    are the one exact source; a driver without them cannot run the episode scope and says so."""
    if gen.token_ids is None:
        raise ValueError(
            "the episode thinking scope needs the turn's sampled token ids to count its reasoning, and none "
            "were captured: the training rollout reads them off the logprobs (train_on_sampled_tokens with the "
            "server flag --return-tokens-as-token-ids), the eval client off the choice's return_token_ids"
        )
    if reasoning_end_token_id is None:
        raise ValueError(
            "the episode thinking scope needs reasoning_end_token_id (rollout_reasoning_end_token resolved "
            "through the tokenizer) to tell a turn's reasoning from its answer"
        )
    try:
        return gen.token_ids.index(reasoning_end_token_id) + 1
    except ValueError:
        return len(gen.token_ids)


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
