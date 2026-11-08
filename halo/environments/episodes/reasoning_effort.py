"""Reasoning effort: the per-episode level draw, per-level effort profiles (a reasoning budget and price the reward
reads), and the generation contract one episode binds (:class:`EpisodeEffort`)."""

import hashlib
import random
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from halo.config_validation import require_finite, require_int
from halo.environments.episodes.episode_keys import OUTPUT_BUDGET_EXHAUSTED_KEY
from halo.environments.episodes.messages import Trajectory, task_prompt
from halo.environments.rollout_config import RANDOM_REASONING_EFFORT, VALID_REASONING_EFFORTS

# The effort-profile keys the trainer's reasoning terms read off an episode's level.
REASONING_BUDGET_KEY = "reasoning_budget"
REASONING_PRICE_KEY = "reasoning_price"


def resolve_reasoning_effort(effort: str | None) -> str | None:
    """``"random"`` draws uniformly (call once per episode); any other value passes through."""
    if effort == RANDOM_REASONING_EFFORT:
        return random.choice(VALID_REASONING_EFFORTS)
    return effort


def stable_reasoning_effort(task: str | list[dict[str, Any]]) -> str:
    """A level drawn from the task alone, the same in every process and run, so an evaluation scores each problem at
    one level from checkpoint to checkpoint."""
    text = str(task_prompt(task) or "")
    digest = hashlib.blake2b(text.encode("utf-8", "surrogatepass"), digest_size=8).digest()
    return VALID_REASONING_EFFORTS[int.from_bytes(digest, "big") % len(VALID_REASONING_EFFORTS)]


class EffortProfilesMixin:
    # Admitted profile keys with their minima (an int minimum admits only ints), unioned over the MRO.
    # ``reasoning_budget`` is a level's reasoning tokens for a whole episode, summed over its turns, and
    # ``reasoning_price`` what the trainer charges per 1,000 of them past it: the reasoning terms read both, the
    # model is told neither.
    EFFORT_PROFILE_KEY_MINIMA: dict[str, int | float] = {REASONING_BUDGET_KEY: 0, REASONING_PRICE_KEY: 0.0}

    def _init_effort_profiles(
        self,
        reasoning_effort: str | None,
        reasoning_effort_profiles: dict[str, dict[str, int | float]] | None,
    ) -> None:
        settings = (*VALID_REASONING_EFFORTS, RANDOM_REASONING_EFFORT)
        if reasoning_effort is not None and reasoning_effort not in settings:
            raise ValueError(f"reasoning_effort must be one of {settings} or None, got {reasoning_effort!r}")
        self.reasoning_effort = reasoning_effort
        self.reasoning_effort_profiles = self._merge_effort_profiles(reasoning_effort_profiles)

    def reasoning_terms_for_effort(self, effort: str | None) -> tuple[int | None, float]:
        """The level's reasoning budget for an episode (``None`` = the level sets none) and its price per 1,000
        reasoning tokens past it (0 = unpriced), as the trainer's reasoning terms read them."""
        profile = self.reasoning_effort_profiles.get(effort, {}) if effort is not None else {}
        return profile.get(REASONING_BUDGET_KEY), float(profile.get(REASONING_PRICE_KEY, 0.0))

    @classmethod
    def effort_profile_key_minima(cls) -> dict[str, int | float]:
        """The profile keys this class admits with their minima: every ``EFFORT_PROFILE_KEY_MINIMA`` along the MRO, a
        subclass's minimum winning over its bases'."""
        minima: dict[str, int | float] = {}
        for klass in reversed(cls.__mro__):
            minima.update(vars(klass).get("EFFORT_PROFILE_KEY_MINIMA", {}))
        return minima

    def effort_level_at_reset(self, context: dict[str, Any] | None) -> str | None:
        """The level when concrete at reset, else ``None``: a ``random`` draw resolves once, actor-side, later."""
        level = (context or {}).get("reasoning_effort") or self.reasoning_effort
        return level if level in VALID_REASONING_EFFORTS else None

    def _merge_effort_profiles(
        self, overrides: dict[str, dict[str, int | float]] | None
    ) -> dict[str, dict[str, int | float]]:
        """One profile per level, empty where it sets none."""
        minima = self.effort_profile_key_minima()
        profiles: dict[str, dict[str, int | float]] = {level: {} for level in VALID_REASONING_EFFORTS}
        for level, entry in (overrides or {}).items():
            if level not in VALID_REASONING_EFFORTS:
                raise ValueError(
                    f"reasoning_effort_profiles level must be one of {VALID_REASONING_EFFORTS}, got {level!r}"
                )
            unknown = set(entry) - set(minima)
            if unknown:
                raise ValueError(
                    f"reasoning_effort_profiles[{level!r}] has unknown keys {sorted(unknown)}; allowed: {sorted(minima)}"
                )
            owner = f"reasoning_effort_profiles[{level!r}]"
            for key, minimum in minima.items():
                if key in entry:
                    # An int minimum makes the key a count, which a float would stand in for.
                    (require_int if isinstance(minimum, int) else require_finite)(
                        owner, ge=minimum, **{key: entry[key]}
                    )
            profiles[level].update(entry)
        for level, profile in profiles.items():
            if REASONING_PRICE_KEY in profile and REASONING_BUDGET_KEY not in profile:
                raise ValueError(
                    f"reasoning_effort_profiles[{level!r}] sets {REASONING_PRICE_KEY} without {REASONING_BUDGET_KEY}: "
                    f"the price charges what an episode reasons past the budget ({REASONING_BUDGET_KEY}: 0 prices "
                    "every reasoning token)."
                )
        return profiles

    def _bind_effort_profile(self, trajectory: Trajectory, context: dict[str, Any] | None) -> None:
        """Empty where the level is undetermined at reset, so the hook still runs and can state the class caps."""
        level = self.effort_level_at_reset(context)
        profile = self.reasoning_effort_profiles.get(level, {}) if level is not None else {}
        self._apply_effort_profile(trajectory, profile)

    def _apply_effort_profile(  # optional hook; task envs override
        self, trajectory: Trajectory, profile: dict[str, int | float]
    ) -> None:
        """Bind a task's own profile keys for one episode; ``profile`` is empty when the effort is undetermined at
        reset."""


@dataclass(frozen=True)
class EpisodeEffort:
    """The generation contract one episode runs under: its resolved level (the one thing the template states),
    ``thinking_cap`` the reasoning cap the engine enforces per turn (the run's ``rollout_max_thinking_tokens``, else the
    turn less its answer room; ``None`` = uncapped), and ``answer_tokens`` the room a turn may generate past that cap.
    Every level binds the same caps: a level differs only in what the template states and what its profile prices."""

    level: str | None
    thinking_cap: int | None
    max_tokens: int
    episode_tokens: int | None = None
    answer_tokens: int | None = None

    @property
    def turn_total(self) -> int:
        if self.thinking_cap is None or self.answer_tokens is None:
            return self.max_tokens
        return min(self.max_tokens, self.thinking_cap + self.answer_tokens)

    @property
    def answer_room(self) -> int:
        return self.turn_total - (self.thinking_cap or 0)

    def turn_caps(self, generated: int) -> dict[str, int | None] | None:
        """The next turn's engine caps as request fields, narrowed to what the output budget leaves after
        ``generated``; ``None`` once that leaves no :attr:`answer_room`."""
        full = self.turn_total
        total = full if self.episode_tokens is None else min(full, self.episode_tokens - generated)
        if total < self.answer_room:
            return None
        thinking = self.thinking_cap
        if thinking is not None:
            # Never down to 0: a cap of 0 closes the reasoning before it opened.
            thinking = max(thinking - (full - total), 1)
        return {"max_tokens": total, "max_thinking_tokens": thinking}

    def stamp(self, trajectory: Trajectory, generated: int) -> None:
        """Re-tokenization has to render the level the model generated under, and the reasoning terms read the
        level's budget and price."""
        trajectory.reasoning_effort = self.level
        if self.episode_tokens is not None:
            trajectory.info[OUTPUT_BUDGET_EXHAUSTED_KEY] = self.turn_caps(generated) is None


def bind_episode_effort(
    context: dict[str, Any] | None,
    env: EffortProfilesMixin,
    *,
    max_tokens: int,
    max_thinking_tokens: int | None = None,
    max_episode_tokens: int | None = None,
    max_answer_tokens: int | None = None,
) -> EpisodeEffort:
    """Resolve one episode's level (a context stamp over the env setting; call once per episode) and bind the run's
    token caps. The engine closes reasoning at ``max_thinking_tokens``, else where only ``max_answer_tokens`` is left
    of the turn; a level's profile budget is the reward's reference, never an engine cap."""
    level = resolve_reasoning_effort((context or {}).get("reasoning_effort") or env.reasoning_effort)
    thinking_cap = max_thinking_tokens
    if thinking_cap is None and max_answer_tokens is not None:
        thinking_cap = max_tokens - max_answer_tokens
    if thinking_cap is not None and not 1 <= thinking_cap < max_tokens:
        raise ValueError(
            f"the turn's reasoning cap ({thinking_cap}: rollout_max_thinking_tokens, else rollout_max_tokens less "
            f"rollout_max_answer_tokens) must sit in [1, rollout_max_tokens={max_tokens}): at or above it the turn "
            "has no answer room and is cut mid-reasoning."
        )
    if max_episode_tokens is not None and max_episode_tokens < max_tokens:
        raise ValueError(
            f"rollout_max_episode_tokens ({max_episode_tokens}) must be at least rollout_max_tokens ({max_tokens}), "
            "so one whole turn fits the episode: below it no turn could start."
        )
    return EpisodeEffort(
        level=level,
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
    """Each drawable level's contract, bound before the first request so a bad level refuses the run rather than
    failing every episode."""
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


def has_thinking_cap(efforts: Mapping[str | None, EpisodeEffort]) -> bool:
    return any(effort.thinking_cap is not None for effort in efforts.values())
