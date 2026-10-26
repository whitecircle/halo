"""Reasoning effort: the per-episode level draw, an environment's per-level effort profiles, and the generation
contract one episode binds from them (:class:`EpisodeEffort`), the same whichever driver collects it."""

import hashlib
import math
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from src.environments.episode_keys import OUTPUT_BUDGET_EXHAUSTED_KEY
from src.environments.messages import Trajectory, task_prompt
from src.environments.rollout_config import RANDOM_REASONING_EFFORT, VALID_REASONING_EFFORTS


def resolve_reasoning_effort(effort: str | None) -> str | None:
    """A setting's concrete level for one episode: ``"random"`` draws uniformly from
    :data:`VALID_REASONING_EFFORTS` (call once per episode), any other value passes through."""
    if effort == RANDOM_REASONING_EFFORT:
        return random.choice(VALID_REASONING_EFFORTS)
    return effort


def stable_reasoning_effort(task: str | list[dict[str, Any]]) -> str:
    """A level drawn from the task alone, the same in every process and every run: an evaluation's draw for
    a ``"random"`` setting, so each problem is scored at one level from checkpoint to checkpoint. A
    conversation is keyed by its :func:`task_prompt`, so it draws what its task text alone would."""
    text = str(task_prompt(task) or "")
    digest = hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=8).digest()
    return VALID_REASONING_EFFORTS[int.from_bytes(digest, "big") % len(VALID_REASONING_EFFORTS)]


class EffortProfilesMixin:
    """An environment's reasoning-effort setting and per-level profiles: the thinking budget each level binds and
    the task keys a level prices."""

    # Effort level -> profile defaults; the base binds no budget to any level, an env that prices effort declares
    # its own table, and ``reasoning_effort_profiles`` merges over it per level.
    REASONING_EFFORT_PROFILES: dict[str, dict[str, int | float | bool]] = {
        level: {} for level in VALID_REASONING_EFFORTS
    }
    # The profile keys a class admits with their minima (an int minimum admits only ints), unioned over the MRO;
    # ``thinking_tokens`` is the level's per-turn CoT budget.
    EFFORT_PROFILE_KEY_MINIMA: dict[str, int | float] = {"thinking_tokens": 1}
    # The true-or-false profile keys, unioned over the MRO. ``enforce_thinking_cap: false`` makes the level's budget
    # soft: stated and priced, but the engine closes reasoning only at the run's ceiling.
    EFFORT_PROFILE_FLAGS: frozenset[str] = frozenset({"enforce_thinking_cap"})

    def _init_effort_profiles(
        self,
        reasoning_effort: str | None,
        reasoning_effort_profiles: dict[str, dict[str, int | float | bool]] | None,
    ) -> None:
        settings = (*VALID_REASONING_EFFORTS, RANDOM_REASONING_EFFORT)
        if reasoning_effort is not None and reasoning_effort not in settings:
            raise ValueError(f"reasoning_effort must be one of {settings} or None, got {reasoning_effort!r}")
        self.reasoning_effort = reasoning_effort
        self.reasoning_effort_profiles = self._merge_effort_profiles(reasoning_effort_profiles)

    def thinking_budget_for_effort(self, effort: str) -> int | None:
        """The per-turn thinking-token budget for a resolved effort level (the level's profile
        ``thinking_tokens``), or ``None`` to use the global one."""
        return self.reasoning_effort_profiles.get(effort, {}).get("thinking_tokens")

    def thinking_budget_enforced(self, effort: str) -> bool:
        """Whether the engine closes a turn's reasoning at the level's budget (the profile's
        ``enforce_thinking_cap``, true unless a profile sets it false)."""
        return self.reasoning_effort_profiles.get(effort, {}).get("enforce_thinking_cap", True)

    def reset_effort_level(self, context: dict[str, Any] | None) -> str | None:
        """The episode's level when concrete at reset (a context stamp or a non-``random`` setting), else ``None``:
        a ``random`` draw resolves once, actor-side, after reset."""
        level = (context or {}).get("reasoning_effort") or self.reasoning_effort
        return level if level in VALID_REASONING_EFFORTS else None

    def _merge_effort_profiles(
        self, overrides: dict[str, dict[str, int | float | bool]] | None
    ) -> dict[str, dict[str, int | float | bool]]:
        """Validate ``reasoning_effort_profiles`` overrides and merge them per level over the class defaults; the
        admitted keys are every ``EFFORT_PROFILE_KEY_MINIMA`` and ``EFFORT_PROFILE_FLAGS`` declared along the MRO,
        a subclass's minimum winning over its bases'."""
        minima: dict[str, int | float] = {}
        for klass in reversed(type(self).__mro__):
            minima.update(vars(klass).get("EFFORT_PROFILE_KEY_MINIMA", {}))
        flags = frozenset(flag for klass in type(self).__mro__ for flag in vars(klass).get("EFFORT_PROFILE_FLAGS", ()))
        profiles = {level: dict(entry) for level, entry in self.REASONING_EFFORT_PROFILES.items()}
        for level, entry in (overrides or {}).items():
            if level not in VALID_REASONING_EFFORTS:
                raise ValueError(
                    f"reasoning_effort_profiles level must be one of {VALID_REASONING_EFFORTS}, got {level!r}"
                )
            unknown = set(entry) - set(minima) - flags
            if unknown:
                raise ValueError(
                    f"reasoning_effort_profiles[{level!r}] has unknown keys {sorted(unknown)}; "
                    f"allowed: {sorted(set(minima) | flags)}"
                )
            for flag in flags & set(entry):
                if not isinstance(entry[flag], bool):
                    raise ValueError(f"{flag} for effort {level!r} must be true or false, got {entry[flag]!r}")
            for key, minimum in minima.items():
                if key not in entry:
                    continue
                value = entry[key]
                # NaN passes a plain minimum check and would poison every reward it enters.
                if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                    raise ValueError(f"{key} for effort {level!r} must be a finite number, got {value!r}")
                if isinstance(minimum, int) and not isinstance(value, int):
                    raise ValueError(f"{key} for effort {level!r} is a count and must be an int, got {value!r}")
                if value < minimum:
                    raise ValueError(f"{key} for effort {level!r} must be >= {minimum}, got {value}")
            profiles.setdefault(level, {}).update(entry)
        for level, profile in profiles.items():
            if profile.get("enforce_thinking_cap") is False and profile.get("thinking_tokens") is None:
                raise ValueError(
                    f"reasoning_effort_profiles[{level!r}] sets enforce_thinking_cap: false without thinking_tokens: "
                    "a soft budget is one the model is told and the reasoning terms read, so the level needs one."
                )
        return profiles

    def _bind_effort_profile(self, trajectory: Trajectory, context: dict[str, Any] | None) -> None:
        """Bind the episode's effort profile through :meth:`_apply_effort_profile`, an empty one where the level is
        undetermined at reset, so the hook still runs and can state the class caps."""
        level = self.reset_effort_level(context)
        profile = self.reasoning_effort_profiles.get(level, {}) if level is not None else {}
        self._apply_effort_profile(trajectory, level, profile)

    def _apply_effort_profile(  # noqa: B027  optional hook; task envs override
        self, trajectory: Trajectory, level: str | None, profile: dict[str, int | float | bool]
    ) -> None:
        """Bind a task's own profile keys (interaction budgets, bonuses) for one episode. Runs after
        ``_reset_single`` on every episode, with ``level`` ``None`` and an empty ``profile`` when the
        effort is undetermined at reset."""


@dataclass(frozen=True)
class EpisodeEffort:
    """The generation contract one episode runs under: its level, the per-turn reasoning budget the template
    states and the reasoning terms read, the reasoning cap the engine enforces per turn (the budget, or a soft
    level's ceiling), the turn's token cap, the episode's output budget (all turns together) and the answer room
    a turn may generate past its reasoning cap; ``None`` leaves a bound unset."""

    level: str | None
    thinking_budget: int | None
    thinking_cap: int | None
    max_tokens: int
    episode_tokens: int | None = None
    answer_tokens: int | None = None

    @property
    def budget_soft(self) -> bool:
        """Whether the budget is stated without the engine enforcing it: the cap is not the budget."""
        return self.thinking_budget is not None and self.thinking_cap != self.thinking_budget

    @property
    def turn_total(self) -> int:
        """The total of a turn: ``max_tokens``, or the reasoning cap plus ``answer_tokens`` where that is
        smaller."""
        if self.thinking_cap is None or self.answer_tokens is None:
            return self.max_tokens
        return min(self.max_tokens, self.thinking_cap + self.answer_tokens)

    @property
    def answer_room(self) -> int:
        """What a turn may generate past the reasoning cap the engine enforces; the whole turn without one."""
        return self.turn_total - (self.thinking_cap or 0)

    def turn_caps(self, generated: int) -> dict[str, int | None] | None:
        """The next turn's engine caps as request fields: the level's total narrowed to what the output budget has
        left after ``generated`` tokens, the reasoning cap with it; ``None`` once that leaves no :attr:`answer_room`."""
        level_total = self.turn_total
        total = level_total if self.episode_tokens is None else min(level_total, self.episode_tokens - generated)
        if total < self.answer_room:
            return None
        thinking = self.thinking_cap
        if thinking is not None:
            # The cap gives up what the total gave up, never down to 0: a cap of 0 closes the reasoning
            # before it opened.
            thinking = max(thinking - (level_total - total), 1)
        return {"max_tokens": total, "max_thinking_tokens": thinking}

    def stamp(self, trajectory: Trajectory | None, generated: int) -> None:
        """Record this contract on the episode's trajectory (no-op when the episode produced none):
        re-tokenization has to render the level and budget the model generated under, the reasoning
        terms price the episode by its level, and an output budget records whether it ran out."""
        if trajectory is None:
            return
        trajectory.reasoning_effort = self.level
        trajectory.reasoning_budget = self.thinking_budget
        trajectory.reasoning_budget_soft = self.budget_soft
        if self.episode_tokens is not None:
            trajectory.info[OUTPUT_BUDGET_EXHAUSTED_KEY] = self.turn_caps(generated) is None


def _soft_ceiling(
    level: str, level_budget: int, max_tokens: int, max_thinking_tokens: int | None, max_answer_tokens: int | None
) -> int | None:
    """Where the engine closes a soft level's reasoning: ``max_thinking_tokens``, else where only
    ``max_answer_tokens`` is left of the turn (``None`` without either). The soft budget must sit below it."""
    ceiling = max_thinking_tokens
    if ceiling is None and max_answer_tokens is not None:
        ceiling = max_tokens - max_answer_tokens
    if ceiling is not None and level_budget >= ceiling:
        raise ValueError(
            f"the {level!r} level's soft thinking_tokens ({level_budget}) must sit below its ceiling ({ceiling}: "
            "rollout_max_thinking_tokens, else rollout_max_tokens less rollout_max_answer_tokens): at or past it the "
            "engine enforces the budget, so it is not soft. Lower the budget or enforce it."
        )
    return ceiling


def _validate_effort_caps(
    level: str | None,
    level_budget: int | None,
    budget: int | None,
    thinking_cap: int | None,
    max_tokens: int,
    max_episode_tokens: int | None,
    max_answer_tokens: int | None,
) -> None:
    """Refuse a bound episode the turn caps cannot hold: a budget or reasoning cap that leaves the turn no answer
    room, an output budget below one turn, or an answer-room bound on a turn with no reasoning cap."""
    # The budget, and under a soft one the ceiling the engine closes at: each has to leave the turn room.
    for bound in (budget, thinking_cap):
        if bound is not None and bound >= max_tokens:
            source = (
                f"the {level!r} level's thinking_tokens" if bound == level_budget else "rollout_max_thinking_tokens"
            )
            raise ValueError(
                f"{source} ({bound}) must sit below rollout_max_tokens ({max_tokens}), which bounds the whole turn: "
                "at or above it the turn has no answer room and is cut mid-reasoning."
            )
    if max_episode_tokens is not None and max_episode_tokens < max_tokens:
        raise ValueError(
            f"rollout_max_episode_tokens ({max_episode_tokens}) must be at least rollout_max_tokens ({max_tokens}), "
            "so one whole turn fits the episode: below it no turn could start."
        )
    # Bounding past a cap the turn lacks would bound nothing: rollout_max_tokens already is its whole room.
    if max_answer_tokens is not None and thinking_cap is None:
        raise ValueError(
            f"rollout_max_answer_tokens ({max_answer_tokens}) bounds what a turn generates past its reasoning cap, "
            f"and an episode at reasoning_effort={level!r} has none: the level sets no thinking_tokens and "
            "rollout_max_thinking_tokens is unset. Give every drawable level a cap, or unset "
            "rollout_max_answer_tokens."
        )


def bind_episode_effort(
    context: dict[str, Any] | None,
    env: EffortProfilesMixin,
    *,
    max_tokens: int,
    max_thinking_tokens: int | None = None,
    max_episode_tokens: int | None = None,
    max_answer_tokens: int | None = None,
) -> EpisodeEffort:
    """Resolve one episode's level (a context stamp over the env setting; call once per episode) and bind its
    ``thinking_tokens`` into the turn caps, clamped by ``max_thinking_tokens``, which alone budgets a level that
    sets none. A soft level is closed only at the run's ceiling (:func:`_soft_ceiling`)."""
    level = resolve_reasoning_effort((context or {}).get("reasoning_effort") or env.reasoning_effort)
    level_budget = env.thinking_budget_for_effort(level) if level is not None else None
    soft = level is not None and level_budget is not None and not env.thinking_budget_enforced(level)
    ceiling = max_thinking_tokens
    if soft:
        ceiling = _soft_ceiling(level, level_budget, max_tokens, max_thinking_tokens, max_answer_tokens)
    caps = [cap for cap in (level_budget, None if soft else ceiling) if cap is not None]
    budget = min(caps) if caps else None
    thinking_cap = ceiling if soft else budget
    _validate_effort_caps(level, level_budget, budget, thinking_cap, max_tokens, max_episode_tokens, max_answer_tokens)
    return EpisodeEffort(
        level=level,
        thinking_budget=budget,
        thinking_cap=thinking_cap,
        max_tokens=max_tokens,
        episode_tokens=max_episode_tokens,
        answer_tokens=max_answer_tokens,
    )


def efforts_by_level(
    env: EffortProfilesMixin,
    *,
    max_tokens: int,
    max_thinking_tokens: int | None,
    max_episode_tokens: int | None = None,
    max_answer_tokens: int | None = None,
) -> dict[str | None, EpisodeEffort]:
    """The contract an episode of ``env`` binds at each level it can draw (every level under ``random``), bound
    before the first request so a bad level refuses the run rather than failing every episode."""
    levels = VALID_REASONING_EFFORTS if env.reasoning_effort == RANDOM_REASONING_EFFORT else (env.reasoning_effort,)
    return {
        level: bind_episode_effort(
            {"reasoning_effort": level},
            env,
            max_tokens=max_tokens,
            max_thinking_tokens=max_thinking_tokens,
            max_episode_tokens=max_episode_tokens,
            max_answer_tokens=max_answer_tokens,
        )
        for level in levels
    }


def has_thinking_budget(efforts: Mapping[str | None, EpisodeEffort]) -> bool:
    """Whether some drawable level (:func:`efforts_by_level`) carries a per-turn thinking budget."""
    return any(effort.thinking_budget is not None for effort in efforts.values())
