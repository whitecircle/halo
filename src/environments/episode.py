"""The rollout-driver half of the environment contract: what a driver binds, generates and hands back.

The Ray actor and the eval runner both drive an episode through here, so an episode is generated and
graded the same way whichever one collects it.
"""

import asyncio
import contextvars
import logging
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from functools import partial
from typing import Any

import backoff

from src.configs.rollout_config import REASONING_END_TOKEN_EXAMPLES, RolloutConfig
from src.environments.base import (
    CUT_TOOL_CALLS_KEY,
    LAST_TURN_KEY,
    OUTPUT_BUDGET_EXHAUSTED_KEY,
    RANDOM_REASONING_EFFORT,
    VALID_REASONING_EFFORTS,
    AsyncBaseEnvironment,
    BaseEnvironment,
    EnvStep,
    Trajectory,
    resolve_reasoning_effort,
)
from src.inference.openai_client import TRANSIENT_CLIENT_ERROR_CODES
from src.inference.response import ENGINE_CUT_FINISH_REASONS, FINISH_REASON_ABORT, FINISH_REASON_LENGTH

logger = logging.getLogger(__name__)

# vLLM answers 400 when a NaN log-prob keeps it from serialising its OWN response: the request was
# valid and a fresh one succeeds, so both drivers retry it rather than lose the turn.
ENGINE_SERIALIZATION_FAULT = "not JSON compliant"
# What an engine's client error says when the conversation outgrew the served context, lowercased:
# vLLM's and SGLang's "maximum context length" / "model's context length", vLLM's "maximum model
# length", and the OpenAI API's error code.
CONTEXT_OVERFLOW_MARKERS = ("context length", "maximum model length", "context_length_exceeded")
# The share of its level's reasoning cap a turn retrying an unproductive one gets: room to read the
# nudge or the refusal, fix and act, never a second budget a cut could buy.
RECOVERY_THINKING_SHARE = 0.25


@dataclass(frozen=True)
class EpisodeEffort:
    """The generation contract one episode runs under: its resolved reasoning-effort level, the per-turn
    reasoning cap bound to that level (``None`` = uncapped), the turn's total token cap, the episode's
    output budget (``rollout_max_episode_tokens``: the most its turns may sample together, reasoning and
    visible output alike; ``None`` leaves only the per-turn caps), and the answer-room bound
    (``rollout_max_answer_tokens``: the most a turn may generate past its reasoning cap; ``None`` leaves
    the turn total at ``max_tokens``)."""

    level: str | None
    thinking_budget: int | None
    max_tokens: int
    episode_tokens: int | None = None
    answer_tokens: int | None = None

    def turn_thinking_cap(self, recovery: bool = False) -> int | None:
        """The reasoning cap the turn about to start runs under: the level's, or on a turn retrying an
        unproductive one (``recovery``, :func:`recovering_turn`) :data:`RECOVERY_THINKING_SHARE` of it."""
        if self.thinking_budget is None or not recovery:
            return self.thinking_budget
        return max(1, round(self.thinking_budget * RECOVERY_THINKING_SHARE))

    def turn_total(self, thinking_cap: int | None) -> int:
        """The total of a turn reasoning under ``thinking_cap``: ``max_tokens``, or the cap plus
        ``answer_tokens`` where that is smaller."""
        if thinking_cap is None or self.answer_tokens is None:
            return self.max_tokens
        return min(self.max_tokens, thinking_cap + self.answer_tokens)

    @property
    def answer_room(self) -> int:
        """What a turn may generate past its level's reasoning cap; the whole turn without one."""
        return self.turn_total(self.thinking_budget) - (self.thinking_budget or 0)

    def turn_caps(self, generated: int, *, recovery: bool = False) -> dict[str, int | None] | None:
        """The engine caps of the turn about to start, as the request fields they set: the level's total
        (:meth:`turn_total`) narrowed to what the output budget has left after ``generated`` tokens, and the
        reasoning cap with it, so the turn keeps its :attr:`answer_room`; a retry (``recovery``) clamps
        both to its reserve (:meth:`turn_thinking_cap`). ``None`` once the budget no longer holds that
        room, whatever the turn: no further turn starts, and the driver closes the episode truncated."""
        level_total = self.turn_total(self.thinking_budget)
        total = level_total if self.episode_tokens is None else min(level_total, self.episode_tokens - generated)
        if total < self.answer_room:
            return None
        thinking = self.thinking_budget
        if thinking is not None:
            # The cap gives up what the total gave up, never down to 0: a cap of 0 closes the reasoning
            # before it opened. A retry's reserve clamps what is left, never shrinks with it.
            thinking = max(thinking - (level_total - total), 1)
            if recovery:
                reserve = self.turn_thinking_cap(recovery=True)
                thinking = min(thinking, reserve)
                total = min(total, self.turn_total(reserve))
        return {"max_tokens": total, "max_thinking_tokens": thinking}

    def stamp(self, trajectory: Trajectory | None, generated: int) -> None:
        """Record this contract on the episode's trajectory (no-op when the episode produced none):
        re-tokenization has to render the level and budget the model generated under, the reasoning
        terms price the episode by its level, and an output budget records whether it ran out."""
        if trajectory is None:
            return
        trajectory.reasoning_effort = self.level
        trajectory.reasoning_budget = self.thinking_budget
        if self.episode_tokens is not None:
            trajectory.info[OUTPUT_BUDGET_EXHAUSTED_KEY] = self.turn_caps(generated) is None


def bind_episode_effort(
    context: dict[str, Any] | None,
    env: BaseEnvironment,
    *,
    max_tokens: int,
    max_thinking_tokens: int | None = None,
    max_episode_tokens: int | None = None,
    max_answer_tokens: int | None = None,
) -> EpisodeEffort:
    """Resolve one episode's effort level and bind the env's per-level CoT budget into its token caps.

    Every rollout driver (the Ray actor, the eval runner) binds through here, so an episode runs under
    the same contract whichever one collects it. Call once per episode: the level may be a ``'random'``
    draw and every turn must share it. A context-supplied level wins over the env setting: the trainer
    stamps one group-level draw into every group member's context, and GRPO's group baseline assumes
    the members of a group share their conditioning.

    The level's ``thinking_tokens`` caps the reasoning channel of every turn, clamped by the run's
    ``max_thinking_tokens``, which alone caps an episode whose level sets none; ``max_tokens`` bounds
    the whole turn, ``max_answer_tokens`` what it generates past its reasoning cap, and
    ``max_episode_tokens`` the episode (:meth:`EpisodeEffort.turn_caps`).
    """
    level = resolve_reasoning_effort((context or {}).get("reasoning_effort") or env.reasoning_effort)
    level_budget = env.thinking_budget_for_effort(level) if level is not None else None
    caps = [cap for cap in (level_budget, max_thinking_tokens) if cap is not None]
    budget = min(caps) if caps else None
    if budget is not None and budget >= max_tokens:
        source = f"the {level!r} level's thinking_tokens" if budget == level_budget else "rollout_max_thinking_tokens"
        raise ValueError(
            f"{source} ({budget}) must sit below rollout_max_tokens ({max_tokens}), which bounds the whole turn: "
            "at or above it the turn has no answer room and is cut mid-reasoning."
        )
    if max_episode_tokens is not None and max_episode_tokens < max_tokens:
        raise ValueError(
            f"rollout_max_episode_tokens ({max_episode_tokens}) must be at least rollout_max_tokens ({max_tokens}), "
            "so one whole turn fits the episode: below it no turn could start."
        )
    # Bounding past a cap the turn lacks would bound nothing: rollout_max_tokens already is its whole room.
    if max_answer_tokens is not None and budget is None:
        raise ValueError(
            f"rollout_max_answer_tokens ({max_answer_tokens}) bounds what a turn generates past its reasoning cap, "
            f"and an episode at reasoning_effort={level!r} has none: the level sets no thinking_tokens and "
            "rollout_max_thinking_tokens is unset. Give every drawable level a cap, or unset "
            "rollout_max_answer_tokens."
        )
    return EpisodeEffort(
        level=level,
        thinking_budget=budget,
        max_tokens=max_tokens,
        episode_tokens=max_episode_tokens,
        answer_tokens=max_answer_tokens,
    )


def recovering_turn(trajectory: Trajectory | None) -> bool:
    """Whether the turn about to start retries an unproductive one: the episode's last assistant turn
    is untrainable — cut by the engine, ended on nothing, or every call unknown or refused — and the
    environment has answered it (the nudge, or the refusals' tool replies)."""
    messages = trajectory.messages if trajectory is not None else []
    last = next((m for m in reversed(messages) if m.role == "assistant"), None)
    return last is not None and last is not messages[-1] and last.untrainable


def thinking_caps_by_level(
    env: BaseEnvironment,
    *,
    max_tokens: int,
    max_thinking_tokens: int | None,
    max_episode_tokens: int | None = None,
    max_answer_tokens: int | None = None,
) -> dict[str | None, int | None]:
    """The per-turn reasoning cap an episode of ``env`` binds at each level it can draw — every level
    under ``random``, the one it sets, or ``None`` with the run's own cap under no setting — bound as an
    episode binds it. Both drivers read it before their first request, so a level whose cap fills the
    turn, a level without one under an answer-room bound, or an output budget under one turn refuses
    the run with one loud error rather than failing every episode at its first turn."""
    levels = VALID_REASONING_EFFORTS if env.reasoning_effort == RANDOM_REASONING_EFFORT else (env.reasoning_effort,)
    return {
        level: bind_episode_effort(
            {"reasoning_effort": level},
            env,
            max_tokens=max_tokens,
            max_thinking_tokens=max_thinking_tokens,
            max_episode_tokens=max_episode_tokens,
            max_answer_tokens=max_answer_tokens,
        ).thinking_budget
        for level in levels
    }


def resolve_rollout_stop_token_ids(tokenizer, names: list[str]) -> list[int] | None:
    """The ids ``rollout_stop_tokens`` names under ``tokenizer``, sent as the engine's ``stop_token_ids``;
    ``None`` when none is configured.

    Every name must resolve. A dropped terminator runs the turn past the call it ends (a gpt-oss episode
    whose ``<|call|>`` never stops the turn plays out in one generation), and the trainer and the eval
    runner both resolve through here, so a recipe that trains is the recipe its eval samples under."""
    if not names:
        return None
    unk = getattr(tokenizer, "unk_token_id", None)
    ids = {name: tokenizer.convert_tokens_to_ids(name) for name in names}
    unresolved = [name for name, tid in ids.items() if tid is None or tid == unk]
    if unresolved:
        raise ValueError(
            f"rollout_stop_tokens {unresolved} are not tokens of "
            f"{getattr(tokenizer, 'name_or_path', 'the tokenizer')!r}, so no turn would stop where the config "
            "says. Check the spellings against its special tokens, or drop them."
        )
    return list(ids.values())


def resolve_reasoning_end_ids(tokenizer, marker: str) -> tuple[int, ...]:
    """``marker``'s ids as the engine forces them: encoded without special tokens, the way vLLM encodes its
    reasoning parser's end string (one id for ``</think>`` or Gemma 4's ``<channel|>``, five for gpt-oss's
    final-channel opener). A reasoning close sits on the tokenizer's added tokens, so an encoding holding none
    of them is a marker this model does not write (``</think>`` where the vocabulary lacks it splits into
    plain text) and raises."""
    ids = tuple(tokenizer.encode(marker, add_special_tokens=False))
    if not set(ids) & set(tokenizer.added_tokens_decoder):
        raise ValueError(
            f"rollout_reasoning_end_token {marker!r} is not a reasoning marker of this tokenizer: it encodes to "
            f"{list(ids)}, none of them one of its added tokens. Name the end string the server's reasoning "
            f"parser forces ({REASONING_END_TOKEN_EXAMPLES})."
        )
    return ids


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

    @property
    def counts_toward_baseline(self) -> bool:
        """Whether the episode counts toward the GRPO group baseline.

        Not when the rollout infrastructure errored (``error``) or the environment marked the reward as
        carrying no learning signal (``Trajectory.episode_invalid`` — e.g. a grading-infrastructure
        outage forced the failure reward). Either way the reward says nothing about the policy, so
        averaging it into the baseline would bias every sibling's advantage.
        """
        return not self.error and not (self.trajectory is not None and self.trajectory.episode_invalid)


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


def sampled_reasoning_tokens(token_ids: list[int] | None, reasoning_end_token_id: int | None) -> int | None:
    """The reasoning tokens in a turn's sampled ids: up to and including the reasoning-end token, or every
    id when the turn was cut before its reasoning closed. ``None`` without the ids or the end token's id.

    On a turn vLLM force-closed this is exactly the budget it enforced. Its counter starts after the last
    ``<think>`` the request holds, so it counts the generation prompt's tokens after that (the ``\n`` the
    Qwen3.6 template ends on) and the reasoning sampled, but not the close it forces: the prompt's one token
    and the close cancel. A prompt ending on a bare ``<think>`` reads one past the budget, and a ``<think>``
    the model emits itself two past."""
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
    """The per-turn context an ``env.step`` receives, stamped from one turn's generation.

    Shared by every rollout driver (the Ray actor's raw aiohttp transport, the eval runner's OpenAI
    client): a driver that omits ``finish_reason`` grades an engine-cut fragment as a deliberate final
    answer. The capture keys are forwarded only where they are id-aligned, since an unaligned logprob
    or routing vector would produce incorrect training data rather than a missing field. ``thinking_cap``
    is the reasoning cap the turn ran under — its level's, or a retry's reserve (the request's own cap
    may sit below it under an output budget, and SGLang ignores the field); with it goes the reasoning the turn sampled
    (:func:`sampled_reasoning_tokens`), the pair ``episode/thinking_cap_turns`` reads.
    ``last_turn`` says the output budget affords no turn after this one, so an unproductive turn is
    closed as an overflow rather than nudged into a retry that cannot run.
    """
    step_ctx = dict(context) if context else {}
    step_ctx["finish_reason"] = gen.finish_reason
    if last_turn:
        step_ctx[LAST_TURN_KEY] = True
    # A cut turn is a fragment whatever the parser salvaged from it: the call it holds was never
    # finished, and executing it books a malformed call and trains the fragment as a normal row. A turn
    # that hit its token cap inside a call carries the salvaged calls apart, never run: the recovery tells
    # it so, the one case the generic cut nudge misreads, and a judge reads what it wrote. An abort names
    # no cause the recovery could pass on.
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
