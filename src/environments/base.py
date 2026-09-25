"""Environment protocol for multi-turn GRPO tasks: the conversation record, the step result, and the
base classes an environment subclasses. ``AsyncBaseEnvironment`` runs its turn batch through
``asyncio.gather`` for I/O-bound tool/API calls. The episode driver is :mod:`src.environments.episode`."""

import asyncio
import itertools
import logging
import math
import random
import re
from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from src.environments.sandbox.base import SandboxAgentFault, SandboxInfraError
from src.inference.response import ENGINE_CUT_FINISH_REASONS
from src.rewards.composer import RewardComposer
from src.rewards.samples import ScoringSample
from src.rewards.scoring import ScoreResult
from src.rewards.spec import (
    ENVIRONMENT_REWARD_SOURCES,
    OBJECTIVE_TERM_NAME,
    EnvironmentTerm,
    RewardTerm,
    component_key,
    parse_reward_terms,
)

logger = logging.getLogger(__name__)

# Reasoning-effort levels for the chat template ("Reasoning: <level>"). "random" resolves per episode.
VALID_REASONING_EFFORTS = ("low", "medium", "high")

# Set in ``info`` when an episode's reward carries no learning signal — a grading or sandbox backend
# failed, a scorer returned nothing, a null ``answer`` cell; the trainer excludes it from the GRPO group
# baseline.
EPISODE_INVALID_KEY = "episode_invalid"
# Why an episode is invalid (a sandbox or scorer fault, a failed chat-template re-render); read by the
# trainer's all-invalid step halt so its message names the cause.
EPISODE_INVALID_REASON_KEY = "episode_invalid_reason"
# The cause on an episode its driver lost (a generation that raised), the spelling the Ray actor
# stamps on the row it hands back for one. A driver stamps it before closing the episode through
# ``finalize_truncated``, and the turn-overflow price then stays off it: the fault is not the policy's.
EPISODE_ERROR_KEY = "error"
# Stamped by the rollout driver under the episode thinking scope: whether the episode's reasoning budget
# ran down to the per-turn reserve (a later turn would have reasoned only its reserve).
THINKING_BUDGET_EXHAUSTED_KEY = "thinking_budget_exhausted"
# Set in ``info`` when a sandbox fault ended the episode, naming its class: ``SANDBOX_FAULT_INFRA`` for
# a backend/transport failure (the episode is also marked invalid and leaves the GRPO group baseline),
# ``SANDBOX_FAULT_AGENT`` for a sandbox the program's own action broke (the episode stays in the
# baseline, ended uncompleted, the call priced as a failed one).
SANDBOX_FAULT_KEY = "sandbox_fault"
SANDBOX_FAULT_INFRA = "infra"
SANDBOX_FAULT_AGENT = "agent"

# The environment's own grade, priced by the reward's environment term, in ``reward_components``: the
# term advantage shaping gates on (it falls back to the total reward when absent).
OBJECTIVE_REWARD_KEY = component_key(OBJECTIVE_TERM_NAME)
# Every term of the episode reward, ``reward/<name>`` → contribution; the values sum to the reward.
REWARD_COMPONENTS_KEY = "reward_components"
# Set in ``info`` while an episode's externally scored terms (a judge, a reward model) are still owed;
# ``settle_async`` clears it. A driver reading the reward before then reads a partial one.
REWARD_PENDING_KEY = "reward_pending"
# What the external scorers report per episode: their diagnostics (``judge/<term>/<requirement>``,
# merged into the rollout metrics), their accounts (a judge's rationale) and their failures, by term.
REWARD_METRICS_KEY = "reward_metrics"
REWARD_DETAILS_KEY = "reward_details"
REWARD_ERRORS_KEY = "reward_errors"
# The component holding the accrued per-turn deltas; the base owns it, an environment may not reuse it.
TURN_SHAPING_COMPONENT = "turn_shaping"

# The per-episode solve flag the rollout metrics average into the group solve rate; a misspelled key
# drops the metric rather than raising.
SOLVE_RATE_KEY = "outcome/solve_rate"

# Categorical per-episode facts an env stamps into ``info`` (``{"language": "cpp"}``); the trainer
# slices its rollout metrics by each one as ``<slice>/<value>/*``, the way it slices by effort level.
EPISODE_SLICES_KEY = "slices"

# Per-tool call caps for one episode (``{tool_name: cap}``), stamped at reset; a tool absent from the
# mapping is uncapped. The protocols that dispatch registry tools enforce them and count admitted
# calls per tool under ``TOOL_CALL_COUNTS_KEY``.
EPISODE_TOOL_BUDGETS_KEY = "episode_tool_budgets"
TOOL_CALL_COUNTS_KEY = "tool_call_counts"
# What the episode has been paid for successful tool calls so far, against ``tool_reward_cap``.
TOOL_REWARD_PAID_KEY = "tool_reward_paid"

# CJK ideographs, kana and hangul: the scripts a Latin-script task's CoT drifts into under RL.
# ``episode/reasoning_cjk_rate`` counts the episodes whose reasoning carries any of them.
_CJK_SCRIPT = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")


# Message keys a chat template may read assistant CoT from: harmony (gpt-oss) reads ``thinking``,
# other reasoning families ``reasoning_content``. Templates ignore a spelling they do not know, so
# emitting both renders the CoT once everywhere; emitting only the wrong one renders an empty block.
REASONING_KEYS = ("thinking", "reasoning_content")


def resolve_reasoning_effort(effort: str | None) -> str | None:
    """Resolve a reasoning-effort setting to a concrete level for one episode.

    ``"random"`` picks uniformly from :data:`VALID_REASONING_EFFORTS`; call once per episode so every
    turn shares the level. Other values pass through.
    """
    if effort == "random":
        return random.choice(VALID_REASONING_EFFORTS)
    return effort


def solve_verdict(metrics: Mapping[str, float]) -> bool | None:
    """Whether an episode solved its task, read off its rollout metrics: the environment's own verdict
    (:data:`SOLVE_RATE_KEY`), or ``None`` where the environment reports none."""
    verdict = metrics.get(SOLVE_RATE_KEY)
    return None if verdict is None else verdict >= 1.0


def require_magnitudes(**knobs: float) -> None:
    """Reject a negative or non-finite value for any reward/penalty magnitude knob.

    The minus sign is applied at the use site, so a negative config value would farm a penalty as a
    bonus; NaN or infinity would pass a sign check and poison every reward the knob enters, even at a
    zero multiplier.
    """
    for name, value in knobs.items():
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"{name} must be a finite value >= 0 (a magnitude), got {value}")


@dataclass(frozen=True)
class EpisodeGrade:
    """What an environment says about a finished episode: ``objective``, its own grade of the task in
    ``[0, 1]`` (priced by the reward's environment term), and ``shaping``, the environment's own
    episode-level terms by bare name (``{"submission": 0.25}``), each added as ``reward/<name>``."""

    objective: float
    shaping: Mapping[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        objective = self.objective
        if isinstance(objective, bool) or not isinstance(objective, int | float) or not (0.0 <= objective <= 1.0):
            raise ValueError(f"an episode's objective grade must lie in [0, 1], got {objective!r}")
        for name, value in self.shaping.items():
            if (
                not isinstance(name, str)
                or not name
                or "/" in name
                or name in (OBJECTIVE_TERM_NAME, TURN_SHAPING_COMPONENT)
            ):
                raise ValueError(f"shaping term name {name!r} is malformed or reserved")
            if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
                raise ValueError(f"shaping term {name!r} must be a finite number, got {value!r}")


@dataclass(slots=True)
class Message:
    """A single conversation message (``__slots__`` for memory efficiency at scale)."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    # Emitted by to_dict only under include_thinking: the training render, and the engine request when
    # the env carries reasoning across turns (a template that reads no reasoning key renders nothing).
    thinking: str | None = None
    # Engine-side captures, all dropped by to_dict. ``routing_mask`` stays raw base64: decoding here
    # would pickle a large array through Ray. The prompt ids are the engine's; a re-render can differ.
    token_ids: list[int] | None = None
    token_logprobs: list[float] | None = None
    routing_mask: str | None = None
    routing_prompt_tokens: int | None = None
    prompt_token_ids: list[int] | None = None
    # Engine cut the turn off at its token cap: the text is a fragment, so the trainer skips it.
    truncated: bool = False
    # Every tool call named a tool that does not exist, so the turn accomplished nothing; skipped
    # like a fragment to avoid reinforcing the invented call.
    calls_rejected: bool = False
    # The model ended the turn with neither visible content nor a tool call — skipped for the same
    # reason: a recovering episode would reinforce stopping on nothing.
    empty: bool = False

    def to_dict(self, include_thinking: bool = False) -> dict[str, Any]:
        """Convert to dict for tokenizer/API. ``include_thinking`` is opt-in: the training render, and the
        engine request when the env carries reasoning; the default keeps assistant CoT out of it."""
        d = {"role": self.role, "content": self.content}
        if self.name:
            d["name"] = self.name
        if self.tool_calls:
            d["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            d["tool_call_id"] = self.tool_call_id
        # Only when truthy: the harmony template concats a present-but-None CoT and raises TypeError.
        if include_thinking and self.thinking:
            d.update(dict.fromkeys(REASONING_KEYS, self.thinking))
        return d

    @property
    def untrainable(self) -> bool:
        """An assistant turn no tokenization path may weight: an engine-cut fragment (``truncated``),
        a turn whose every tool call named a nonexistent tool (``calls_rejected``) or one that ended
        on nothing (``empty``). It stays in the render later turns condition on, but reinforcing it
        would reward the runaway, the invented call or the empty stop whenever the episode recovers."""
        return self.truncated or self.calls_rejected or self.empty

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Message":
        return cls(
            role=d["role"],
            content=d.get("content", ""),
            name=d.get("name"),
            tool_calls=d.get("tool_calls"),
            tool_call_id=d.get("tool_call_id"),
            thinking=next((d[key] for key in REASONING_KEYS if d.get(key)), None),
        )

    @classmethod
    def user(cls, content: str) -> "Message":
        return cls(role="user", content=content)

    @classmethod
    def assistant(cls, content: str, **fields) -> "Message":
        """An assistant turn. ``fields`` are this class's own optional fields (CoT, engine captures,
        the turn flags); an unknown name raises TypeError rather than being dropped."""
        return cls(role="assistant", content=content, **fields)

    @classmethod
    def system(cls, content: str) -> "Message":
        return cls(role="system", content=content)

    @classmethod
    def tool(cls, content: str, tool_call_id: str, name: str) -> "Message":
        return cls(role="tool", content=content, tool_call_id=tool_call_id, name=name)


def engine_view(messages: Sequence[Message], carry_reasoning: bool) -> list[Message]:
    """The conversation as the engine is told it: every message's visible text and, with
    ``carry_reasoning``, the LAST assistant turn's reasoning — the thought the next turn continues from.
    Earlier turns' reasoning is withheld, so a request grows by at most one reasoning budget over the
    visible history. The one owner of what the engine sees: the rollout observation and the trainer's
    context re-render both come through here, and where a carried thought renders is the chat template's
    decision, not this function's."""
    last = max((i for i, m in enumerate(messages) if m.role == "assistant"), default=None)
    return [
        m if m.thinking is None or (carry_reasoning and i == last) else replace(m, thinking=None)
        for i, m in enumerate(messages)
    ]


@dataclass
class Trajectory:
    """A complete or partial trajectory through the environment (messages + rewards + state)."""

    messages: list[Message] = field(default_factory=list)
    total_reward: float = 0.0
    done: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)
    # Set by the rollout so re-tokenization renders the same steer the model generated under.
    reasoning_effort: str | None = None
    # The CoT budget the episode ran under — per turn, or for the whole episode under the episode
    # scope: the template states it, so a re-render needs it, and the trainer's under-use floor is a
    # multiple of it.
    reasoning_budget: int | None = None

    _assistant_count: int = field(default=0, repr=False)

    def append_to_last_user(self, text: str) -> None:
        """Append ``text`` to the most recent user message: how an env states per-episode facts (a task's
        budgets, a question's choices) inside the prompt the model already reads."""
        for message in reversed(self.messages):
            if message.role == "user":
                message.content += text
                return
        raise ValueError("the trajectory has no user message to append to")

    def add_message(self, message: Message) -> None:
        """Add a message to the trajectory."""
        self.messages.append(message)
        if message.role == "assistant":
            self._assistant_count += 1

    def add_reward(self, reward: float) -> None:
        """Add a step reward to the running total."""
        self.total_reward += reward

    @property
    def num_turns(self) -> int:
        """Number of assistant turns (cached)."""
        return self._assistant_count

    @property
    def episode_invalid(self) -> bool:
        """True when the environment marked this episode's reward as carrying no learning signal
        (see ``EPISODE_INVALID_KEY``); the trainer excludes it from the GRPO group baseline."""
        return bool(self.info.get(EPISODE_INVALID_KEY, False))

    def get_conversation(self, include_thinking: bool = False) -> list[dict[str, Any]]:
        """The conversation as message dicts for the engine (:func:`engine_view`); ``include_thinking``
        carries the last assistant turn's reasoning along under :data:`REASONING_KEYS`."""
        return [m.to_dict(include_thinking=True) for m in engine_view(self.messages, include_thinking)]


@dataclass(slots=True)
class EnvStep:
    """Result of an environment step (``__slots__`` for memory efficiency).

    Gym-shaped: ``reward`` is this step's delta (the trajectory keeps only the running total), while
    ``done``/``truncated``/``info`` mirror the trajectory's state at the moment the step returned.
    ``info`` is shared by identity, so a caller mutating it mutates the trajectory.
    """

    trajectory: Trajectory
    observation: list[dict[str, Any]]
    reward: float = 0.0
    done: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class BaseEnvironment(ABC):
    """Abstract base class for multi-turn GRPO environments.

    Subclasses implement ``_reset_single``, ``_step_single``, ``_grade_episode``. Parallel rollout
    collection runs one episode per Ray actor instance (see ray_actors.py).
    """

    # Turn budget used when the config names none. Per class, since an agentic edit-run-test loop and
    # a one-shot exam need very different budgets.
    DEFAULT_MAX_TURNS: int = 10

    # Per-tool-call shaping a class gets when the config names none; a task env that departs states
    # its own value once (code-contests turns both off) instead of re-defaulting its constructor.
    DEFAULT_TOOL_SUCCESS_REWARD: float = 0.05
    DEFAULT_TOOL_ERROR_PENALTY: float = 0.1

    # Whether this class's reward grades ONLY against ``context["answer"]``: the trainer refuses a
    # dataset carrying no ``answer`` column when it is set, since nothing would be graded. The
    # dual-mode classes — the native protocol and its presets — leave it False and grade against the
    # answer wherever a row carries one, paying for completing the task where no row does.
    requires_answer: bool = False

    # Declared here because ``_add_action_message`` trims stored ``tool_calls`` to it; ``None`` = no cap.
    max_tool_calls_per_turn: int | None = None

    # The prompt trajectories open with, assigned by the protocol layers that build one; eval records
    # it in the trajectory meta. ``None`` = no system turn.
    system_prompt: str | None = None

    # What :meth:`_handle_length_cutoff` feeds back after the engine cuts a turn short, and what
    # :meth:`_handle_empty_turn` feeds back after the model ends one on nothing. Per protocol, since
    # the text must ask for that protocol's next move; ``None`` means the protocol has no recovery
    # path for that kind of turn and does not route it there.
    LENGTH_CUTOFF_NUDGE: str | None = None
    EMPTY_TURN_NUDGE: str | None = None

    # Names of the episode-level shaping components this class adds to the reward (``reward/<name>``),
    # the union over the MRO being what an episode may carry: a protocol declares its own
    # (``tool_shaping``), a task env its rungs. Declared so a reward term cannot take a shaping
    # component's name, and an undeclared shaping name fails at the first settled episode.
    SHAPING_COMPONENTS: tuple[str, ...] = ()

    # Effort level -> profile defaults. The base binds no budget to any level (the global rollout caps
    # stand); an env that prices effort declares its own table. ``reasoning_effort_profiles`` merges
    # per level over it.
    REASONING_EFFORT_PROFILES: dict[str, dict[str, int | float]] = {level: {} for level in VALID_REASONING_EFFORTS}

    # Profile keys this class admits and the minimum each takes; a subclass declares only the keys it
    # adds, and the union over the MRO is what a profile may carry. ``thinking_tokens`` is the
    # level's CoT budget (per turn, or the episode's total under the episode thinking scope),
    # ``max_length_cutoff_recoveries`` tightens the env's recovery cap for the level. An int minimum
    # declares a count (only ints admitted); a float minimum admits any finite number.
    EFFORT_PROFILE_KEY_MINIMA: dict[str, int | float] = {
        "thinking_tokens": 1,
        "max_length_cutoff_recoveries": 0,
    }

    def __init__(
        self,
        max_turns: int | None = None,
        max_observation_chars: int = 16384,
        max_length_cutoff_recoveries: int | None = None,
        reasoning_effort_profiles: dict[str, dict[str, int | float]] | None = None,
        carry_reasoning: bool = False,
        requires_answer: bool | None = None,
        tool_success_reward: float | None = None,
        tool_error_penalty: float | None = None,
        tool_reward_cap: float | None = None,
        reward_terms: Sequence[RewardTerm | Mapping[str, Any]] | None = None,
        **kwargs,
    ):
        """``max_turns`` caps turns before truncation, defaulting to this class's
        :data:`DEFAULT_MAX_TURNS`; ``max_observation_chars`` caps a tool observation's length.
        ``tool_success_reward`` / ``tool_error_penalty`` price every executed tool call
        (:meth:`_credit_tool_call`), defaulting to :data:`DEFAULT_TOOL_SUCCESS_REWARD` /
        :data:`DEFAULT_TOOL_ERROR_PENALTY`; ``tool_reward_cap`` bounds what one episode earns from
        successful calls in total, ``None`` resolving to ``tool_success_reward * max_turns`` — at most
        one paid call per turn of the budget, so per-call pay cannot out-earn the objective through
        call spam (five paid calls a turn over ten turns would otherwise pay 2.5 against a 1.0 solve).
        ``max_length_cutoff_recoveries`` caps how many unproductive turns (engine-cut, or ended on nothing) an episode may recover from
        (``None`` = every one within ``max_turns``): such a turn spends a turn but no tool budget, so
        without a cap an episode whose thoughts overrun their budget re-thinks until ``max_turns``.
        ``carry_reasoning`` sends the previous assistant turn's reasoning back to the engine with the
        conversation, so the next turn (a tool round, the retry after a cut) conditions on the thought
        that produced it instead of restarting; earlier turns stay as visible text, which bounds a
        request's growth to one reasoning budget. Off, the engine sees only visible text. Where the
        carried thought renders is the chat template's call (Qwen3.x: turns after the last user
        message, or all of them under ``preserve_thinking``).
        ``requires_answer`` overrides the class's :attr:`requires_answer` declaration that its reward
        grades against ``context["answer"]``, which makes the trainer refuse a dataset carrying no
        ``answer`` column rather than score the whole run on nothing. ``None`` keeps the declaration,
        so read the resolved verdict off an INSTANCE: the class attribute is only the default, which a
        class deriving its own (ReAct, from ``answer_validator``) overrides per instance.
        ``reward_terms`` are the episode reward's terms (:mod:`src.rewards.spec`, ``source`` in
        ``environment`` / ``judge`` / ``reward_model``), as typed terms or config mappings; ``None`` is
        the environment's own grade at weight 1. The accrued per-turn deltas and the protocol's and
        environment's shaping add on top of them.
        ``reasoning_effort`` (popped from kwargs) steers the chat template's CoT depth:
        ``low``/``medium``/``high``, ``"random"``, or ``None``. ``reasoning_effort_profiles`` overrides
        the class's per-level profiles (:data:`REASONING_EFFORT_PROFILES`), merged per level; the
        admitted keys are the union of :data:`EFFORT_PROFILE_KEY_MINIMA` over the class hierarchy.

        Any remaining keyword raises: the registry factories forward the whole ``env_config``, so a key
        no constructor in the chain binds is a typo or a knob meant for another ``env_type``.
        """
        max_turns = self.DEFAULT_MAX_TURNS if max_turns is None else max_turns
        # 0 turns makes the rollout loop a no-op, producing an all-zero batch.
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")
        self.max_turns = max_turns
        if max_length_cutoff_recoveries is not None and max_length_cutoff_recoveries < 0:
            raise ValueError(f"max_length_cutoff_recoveries must be >= 0 or None, got {max_length_cutoff_recoveries}")
        self.max_length_cutoff_recoveries = max_length_cutoff_recoveries
        self.max_observation_chars = max_observation_chars
        self.carry_reasoning = carry_reasoning
        if requires_answer is not None:
            self.requires_answer = requires_answer
        if tool_success_reward is None:
            tool_success_reward = self.DEFAULT_TOOL_SUCCESS_REWARD
        if tool_error_penalty is None:
            tool_error_penalty = self.DEFAULT_TOOL_ERROR_PENALTY
        if tool_reward_cap is None:
            tool_reward_cap = tool_success_reward * max_turns
        require_magnitudes(
            tool_success_reward=tool_success_reward,
            tool_error_penalty=tool_error_penalty,
            tool_reward_cap=tool_reward_cap,
        )
        self.tool_success_reward = tool_success_reward
        self.tool_error_penalty = tool_error_penalty
        self.tool_reward_cap = tool_reward_cap
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        if reasoning_effort is not None and reasoning_effort not in (*VALID_REASONING_EFFORTS, "random"):
            raise ValueError(
                f"reasoning_effort must be one of {(*VALID_REASONING_EFFORTS, 'random')} or None, "
                f"got {reasoning_effort!r}"
            )
        self.reasoning_effort = reasoning_effort
        self.reasoning_effort_profiles = self._merge_effort_profiles(reasoning_effort_profiles)
        terms = (
            (EnvironmentTerm(),)
            if reward_terms is None
            else parse_reward_terms(reward_terms, ENVIRONMENT_REWARD_SOURCES)
        )
        self._rewards = RewardComposer(terms)
        reserved = {TURN_SHAPING_COMPONENT, *self.shaping_components()}
        taken = sorted(term.name for term in self._rewards.external_terms if term.name in reserved)
        if taken:
            raise ValueError(
                f"reward term name(s) {taken} are shaping components of {type(self).__name__}; "
                f"pick another name for the term"
            )
        self._background_tasks: set[asyncio.Task] = set()
        if hasattr(type(self), "_compute_reward"):
            raise TypeError(
                f"{type(self).__name__} defines _compute_reward, which nothing calls: an environment grades "
                f"through _grade_episode, returning an EpisodeGrade."
            )
        if kwargs:
            raise TypeError(
                f"{type(self).__name__} got unexpected environment option(s) {sorted(kwargs)}. Every "
                f"environment_kwargs key must be a constructor parameter of the resolved environment "
                f"(check the env_type it belongs to and the spelling)."
            )

        self._trajectories: dict[int, Trajectory] = {}
        self._episode_id_generator = itertools.count()

    @property
    def reward_terms(self) -> tuple[RewardTerm, ...]:
        """The episode reward's terms, as configured."""
        return self._rewards.terms

    @classmethod
    def shaping_components(cls) -> frozenset[str]:
        """The shaping component names this class may add: every ``SHAPING_COMPONENTS`` along the MRO."""
        return frozenset(name for klass in cls.__mro__ for name in getattr(klass, "SHAPING_COMPONENTS", ()))

    @classmethod
    def effort_profile_key_minima(cls) -> dict[str, int | float]:
        """The profile keys this class admits with their minima: every ``EFFORT_PROFILE_KEY_MINIMA``
        declared along the MRO, a subclass's entry winning over its bases'."""
        minima: dict[str, int | float] = {}
        for klass in reversed(cls.__mro__):
            minima.update(vars(klass).get("EFFORT_PROFILE_KEY_MINIMA", {}))
        return minima

    def _merge_effort_profiles(
        self, overrides: dict[str, dict[str, int | float]] | None
    ) -> dict[str, dict[str, int | float]]:
        """Validate ``reasoning_effort_profiles`` overrides and merge them per level over the class defaults.

        A level's recovery cap may tighten the env's ``max_length_cutoff_recoveries``, never exceed it.
        """
        minima = self.effort_profile_key_minima()
        profiles = {level: dict(entry) for level, entry in self.REASONING_EFFORT_PROFILES.items()}
        for level, entry in (overrides or {}).items():
            if level not in VALID_REASONING_EFFORTS:
                raise ValueError(
                    f"reasoning_effort_profiles level must be one of {VALID_REASONING_EFFORTS}, got {level!r}"
                )
            unknown = set(entry) - set(minima)
            if unknown:
                raise ValueError(
                    f"reasoning_effort_profiles[{level!r}] has unknown keys {sorted(unknown)}; "
                    f"allowed: {sorted(minima)}"
                )
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
        env_cap = self.max_length_cutoff_recoveries
        for level, profile in profiles.items():
            if env_cap is not None and profile.get("max_length_cutoff_recoveries", 0) > env_cap:
                raise ValueError(
                    f"reasoning_effort_profiles[{level!r}].max_length_cutoff_recoveries "
                    f"({profile['max_length_cutoff_recoveries']}) exceeds the env's max_length_cutoff_recoveries ({env_cap})"
                )
        return profiles

    def _bind_effort_profile(self, trajectory: Trajectory, context: dict[str, Any] | None) -> None:
        """Stamp the episode's effort profile once its level is concrete at reset.

        The generic keys land as the ``info`` stamps their consumers read (``_handle_length_cutoff``);
        task-specific keys go through :meth:`_apply_effort_profile`.
        An undetermined level (:meth:`reset_effort_level` returns ``None``) binds an empty profile, so
        the hook still runs and can state the class caps.
        """
        level = self.reset_effort_level(context)
        profile = self.reasoning_effort_profiles.get(level, {}) if level is not None else {}
        if "max_length_cutoff_recoveries" in profile:
            trajectory.info["episode_max_length_cutoff_recoveries"] = profile["max_length_cutoff_recoveries"]
        self._apply_effort_profile(trajectory, level, profile)

    def _apply_effort_profile(  # noqa: B027  optional hook; task envs override
        self, trajectory: Trajectory, level: str | None, profile: dict[str, int | float]
    ) -> None:
        """Bind a task's own profile keys (interaction budgets, bonuses) for one episode. Runs after
        ``_reset_single`` on every episode, with ``level`` ``None`` and an empty ``profile`` when the
        effort is undetermined at reset."""

    @staticmethod
    def _tool_calls_made(trajectory: Trajectory, name: str) -> int:
        """Admitted calls of tool ``name`` so far in the episode (a call is counted before its handler runs)."""
        return trajectory.info.get(TOOL_CALL_COUNTS_KEY, {}).get(name, 0)

    @classmethod
    def _tool_budget_exhausted(cls, trajectory: Trajectory, name: str) -> int | None:
        """The episode cap another call of tool ``name`` would exceed, or ``None`` while within budget."""
        cap = trajectory.info.get(EPISODE_TOOL_BUDGETS_KEY, {}).get(name)
        if cap is not None and cls._tool_calls_made(trajectory, name) >= cap:
            return cap
        return None

    @staticmethod
    def _count_tool_call(trajectory: Trajectory, name: str) -> int:
        """Count one admitted call of tool ``name`` for the episode; returns the new count."""
        counts = trajectory.info.setdefault(TOOL_CALL_COUNTS_KEY, {})
        counts[name] = counts.get(name, 0) + 1
        return counts[name]

    def _credit_tool_call(self, trajectory: Trajectory, success: bool) -> float:
        """Book one executed tool call on the episode's counters and return its reward delta:
        ``-tool_error_penalty`` for a failed call; for a successful one ``tool_success_reward``, less
        whatever would take the episode's paid total (``TOOL_REWARD_PAID_KEY``) past
        ``tool_reward_cap``. The one accounting every protocol pays through."""
        trajectory.info["total_tool_calls"] += 1
        if not success:
            return -self.tool_error_penalty
        trajectory.info["successful_tool_calls"] += 1
        paid = trajectory.info.get(TOOL_REWARD_PAID_KEY, 0.0)
        credit = max(0.0, min(self.tool_success_reward, self.tool_reward_cap - paid))
        trajectory.info[TOOL_REWARD_PAID_KEY] = paid + credit
        return credit

    def _book_tool_call(
        self,
        trajectory: Trajectory,
        tool: str,
        success: bool,
        fault: SandboxInfraError | SandboxAgentFault | None = None,
    ) -> float:
        """Book one executed tool call and return its reward delta: by the class of the sandbox fault
        that ended it (:meth:`_book_sandbox_fault`), else as a success or a failure
        (:meth:`_credit_tool_call`). The one entry every protocol books a call through."""
        if fault is not None:
            return self._book_sandbox_fault(trajectory, tool, fault)
        return self._credit_tool_call(trajectory, success)

    def _book_sandbox_fault(
        self, trajectory: Trajectory, tool: str, fault: SandboxInfraError | SandboxAgentFault
    ) -> float:
        """Book one tool call a sandbox fault ended and return its reward delta; the step then ends the
        episode on :data:`SANDBOX_FAULT_KEY`, uncompleted (:meth:`_finalize_step`).

        An infrastructure fault says nothing about the policy: the call goes unpriced and the episode
        leaves the GRPO group baseline, the fault stamped as the reason the trainer's all-invalid halt
        names. An agent-caused one is the policy's: a failed call, and the episode, ended uncompleted,
        stays in the baseline.
        """
        if isinstance(fault, SandboxInfraError):
            logger.warning("Tool %r lost to a sandbox infrastructure fault; the episode is dropped: %s", tool, fault)
            trajectory.info["total_tool_calls"] += 1
            trajectory.info[SANDBOX_FAULT_KEY] = SANDBOX_FAULT_INFRA
            trajectory.info[EPISODE_INVALID_KEY] = True
            trajectory.info.setdefault(
                EPISODE_INVALID_REASON_KEY, f"sandbox infrastructure fault in tool {tool!r}: {fault}"
            )
            return 0.0
        logger.debug("Tool %r ended the episode on an agent-caused sandbox fault: %s", tool, fault)
        trajectory.info.setdefault(SANDBOX_FAULT_KEY, SANDBOX_FAULT_AGENT)
        return self._credit_tool_call(trajectory, False)

    @staticmethod
    def _last_assistant_message(trajectory: Trajectory) -> Message:
        """The turn just taken: every step appends the model's message before it is handled."""
        message = next((m for m in reversed(trajectory.messages) if m.role == "assistant"), None)
        if message is None:
            raise ValueError(
                "the trajectory holds no assistant turn to flag; a step records the model's message first"
            )
        return message

    def _flag_calls_rejected(self, trajectory: Trajectory) -> None:
        """Mark the turn just taken as one whose every call named a nonexistent tool
        (:attr:`Message.calls_rejected`), so no tokenization path weights it."""
        self._last_assistant_message(trajectory).calls_rejected = True

    def _truncate_observation(self, content: str) -> str:
        """Cap a tool observation's length. An unbounded output bloats the trajectory and makes the
        per-turn re-render slow; capping at the source keeps rollout and recompute identical."""
        limit = self.max_observation_chars
        if limit and len(content) > limit:
            return content[:limit] + f"\n…[truncated {len(content) - limit} chars]"
        return content

    def get_tools_schema(self) -> list[dict[str, Any]] | None:
        """OpenAI-format tool schema passed to vLLM as ``tools=``. Default ``None``; overridden by the
        tool-use envs."""
        return None

    def thinking_budget_for_effort(self, effort: str) -> int | None:
        """The thinking-token budget for a resolved effort level (the level's profile ``thinking_tokens``;
        per turn, or the episode's total under the episode scope), or ``None`` to use the global one."""
        return self.reasoning_effort_profiles.get(effort, {}).get("thinking_tokens")

    def reset_effort_level(self, context: dict[str, Any] | None) -> str | None:
        """The episode's effort level when it is already concrete at reset, else ``None``.

        Concrete means a context-supplied level (training stamps one per GRPO group) or a
        non-``random`` env setting. A ``random`` draw without a context stamp resolves actor-side after
        reset; resolving it here as well would draw twice and diverge, so per-episode
        effort-conditioned setup (interaction budgets, prompts) keys off this and skips on ``None``."""
        level = (context or {}).get("reasoning_effort") or self.reasoning_effort
        return level if level in VALID_REASONING_EFFORTS else None

    def rollout_metrics(self, trajectory: "Trajectory") -> dict[str, float]:
        """Per-episode diagnostic metrics (mean-aggregated by the trainer), keyed by full metric path:
        ``outcome/*`` (task success), ``episode/*`` (agent behavior), ``reward/*`` (the reward's
        components, which sum to it) and the external scorers' own keys. Base emits the tool-use
        signal and the reward decomposition; task envs override to add outcome metrics."""
        if REWARD_PENDING_KEY in trajectory.info:
            raise RuntimeError(
                "the episode's reward still owes its externally scored terms: the driver must await "
                "settle_async (EpisodeDispatcher does) before reading the episode"
            )
        metrics: dict[str, float] = {}
        if "total_tool_calls" in trajectory.info:
            metrics["episode/tool_calls"] = float(trajectory.info["total_tool_calls"])
            fault = trajectory.info.get(SANDBOX_FAULT_KEY)
            metrics["episode/sandbox_infra_fault"] = 1.0 if fault == SANDBOX_FAULT_INFRA else 0.0
            metrics["episode/sandbox_agent_fault"] = 1.0 if fault == SANDBOX_FAULT_AGENT else 0.0
        # Tracked separately: a termination-rate metric cannot tell a cut-off turn, or one that
        # stopped inside its reasoning, from an answer.
        metrics["episode/length_cutoff_turns"] = float(trajectory.info.get("length_cutoff_turns", 0))
        metrics["episode/empty_turns"] = float(trajectory.info.get("empty_turns", 0))
        if THINKING_BUDGET_EXHAUSTED_KEY in trajectory.info:
            metrics["episode/thinking_budget_exhausted"] = (
                1.0 if trajectory.info[THINKING_BUDGET_EXHAUSTED_KEY] else 0.0
            )
        metrics["episode/reasoning_cjk_rate"] = (
            1.0
            if any(
                m.role == "assistant" and m.thinking and _CJK_SCRIPT.search(m.thinking) for m in trajectory.messages
            )
            else 0.0
        )
        metrics.update(trajectory.info.get(REWARD_COMPONENTS_KEY, {}))
        metrics.update(trajectory.info.get(REWARD_METRICS_KEY, {}))
        if (
            self._rewards.external_terms
            and REWARD_COMPONENTS_KEY in trajectory.info
            and not self._cut_short(trajectory)
        ):
            metrics["episode/reward_scored"] = 0.0 if trajectory.info.get(REWARD_ERRORS_KEY) else 1.0
        return metrics

    def _get_next_episode_id(self) -> int:
        """Get a unique episode ID (thread-safe via itertools.count())."""
        return next(self._episode_id_generator)

    @staticmethod
    def _init_trajectory(
        prompt: str | list[dict[str, str]],
        context: dict[str, Any] | None = None,
        system_prompt: str | None = None,
        extra_info: dict[str, Any] | None = None,
    ) -> Trajectory:
        """Build a fresh trajectory: optional system prompt, the user prompt, and base info. Shared
        by every ``_reset_single``; subclasses pass tool/state keys via ``extra_info``."""
        traj = Trajectory()
        context = context or {}

        if system_prompt:
            traj.add_message(Message.system(system_prompt))

        if isinstance(prompt, str):
            traj.add_message(Message.user(prompt))
        elif isinstance(prompt, list):
            for msg in prompt:
                traj.add_message(Message.from_dict(msg))

        traj.info.update(
            {
                "task": prompt if isinstance(prompt, str) else str(prompt),
                "context": context,
                "completed": False,
            }
        )
        if extra_info:
            traj.info.update(extra_info)
        return traj

    def _add_action_message(self, trajectory: Trajectory, action: str, context: dict[str, Any] | None) -> None:
        """Append the model's response as an assistant message with tool_calls and reasoning from
        context."""
        ctx = context or {}
        tool_calls = ctx.get("tool_calls")
        # Uncapped, the message advertises more calls than tool-result messages, which is rejected
        # by the re-tokenizer.
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
                routing_mask=ctx.get("routing_mask"),
                routing_prompt_tokens=ctx.get("routing_prompt_tokens"),
                prompt_token_ids=ctx.get("prompt_token_ids"),
                # An engine abort is a cut turn too: the fragment must never train as a natural stop.
                truncated=ctx.get("finish_reason") in ENGINE_CUT_FINISH_REASONS,
            )
        )

    def _handle_length_cutoff(self, trajectory: Trajectory) -> tuple[Trajectory, float, bool, bool, dict[str, Any]]:
        """Handle a turn the engine cut short before it produced anything — at its token cap, or by
        aborting it (:data:`~src.inference.response.ENGINE_CUT_FINISH_REASONS`).

        Never graded — the fragment would end the episode on a mid-sentence string that reads as a
        *natural* termination. Owned by the base so ``episode/length_cutoff_turns`` means the same
        thing for every protocol that can recover; the wording is each protocol's
        (:data:`LENGTH_CUTOFF_NUDGE`).
        """
        return self._recover_unproductive_turn(trajectory, "length_cutoff", "LENGTH_CUTOFF_NUDGE")

    def _handle_empty_turn(self, trajectory: Trajectory) -> tuple[Trajectory, float, bool, bool, dict[str, Any]]:
        """Handle a turn the model ended with neither visible content nor a tool call: a stop inside
        its reasoning, below the cap.

        Graded, it would end the episode on an empty final answer; the turn is flagged
        (:attr:`Message.empty`) so no tokenization path weights it, and the episode recovers like it
        does from a cut. The wording is each protocol's (:data:`EMPTY_TURN_NUDGE`).
        """
        self._last_assistant_message(trajectory).empty = True
        return self._recover_unproductive_turn(trajectory, "empty", "EMPTY_TURN_NUDGE")

    @staticmethod
    def _unproductive_turns(trajectory: Trajectory) -> int:
        """Turns that produced nothing — cut by the engine or ended by the model on nothing."""
        return trajectory.info.get("length_cutoff_turns", 0) + trajectory.info.get("empty_turns", 0)

    def _recover_unproductive_turn(
        self, trajectory: Trajectory, kind: str, nudge_attr: str
    ) -> tuple[Trajectory, float, bool, bool, dict[str, Any]]:
        """Nudge and retry a turn that produced nothing, within ``max_turns`` and within the episode's
        recovery cap (``episode_max_length_cutoff_recoveries`` when an env stamped one, else
        ``max_length_cutoff_recoveries``), which the two kinds of unproductive turn share. The turn is
        counted under ``<kind>_turns`` and stamped ``<kind>`` in the step info. A turn past the cap, or
        on the episode's last turn, cannot be retried: it ends the episode truncated, priced like a
        ``max_turns`` overflow and never as a recovered turn (``unrecovered_turn``).
        A recovered one is priced by the protocol where it configures ``length_cutoff_penalty``,
        never here."""
        nudge = getattr(self, nudge_attr)
        if nudge is None:
            raise NotImplementedError(
                f"{type(self).__name__} routed an unproductive turn to recovery without declaring "
                f"{nudge_attr} — the episode would continue with no message telling the model what happened."
            )
        counter = f"{kind}_turns"
        trajectory.info[counter] = trajectory.info.get(counter, 0) + 1
        cap = trajectory.info.get("episode_max_length_cutoff_recoveries", self.max_length_cutoff_recoveries)
        past_cap = cap is not None and self._unproductive_turns(trajectory) > cap
        if past_cap or trajectory.num_turns >= self.max_turns:
            return trajectory, 0.0, True, True, {kind: True, "unrecovered_turn": True}
        trajectory.add_message(Message.user(nudge))
        return trajectory, 0.0, False, False, {kind: True}

    def _first_step(self, trajectory: Trajectory) -> EnvStep:
        """Opening :class:`EnvStep` for a freshly reset episode (sync + async reset paths)."""
        return EnvStep(
            trajectory=trajectory,
            observation=trajectory.get_conversation(include_thinking=self.carry_reasoning),
            reward=0.0,
            done=False,
            truncated=False,
            info=trajectory.info,
        )

    def _done_step(self, trajectory: Trajectory) -> EnvStep:
        """Terminal :class:`EnvStep` for an episode that is already complete (a no-op step)."""
        return EnvStep(
            trajectory=trajectory,
            observation=trajectory.get_conversation(include_thinking=self.carry_reasoning),
            reward=0.0,
            done=True,
            truncated=trajectory.truncated,
            info=trajectory.info,
        )

    @staticmethod
    def _drop_grading_payload(trajectory: Trajectory) -> None:
        """Strip what a settled episode no longer needs but every hop after it would carry — the Ray
        object store, the TP broadcast: the env's private ``_``-prefixed stamps (hidden tests, a
        checker source) and the context's ``answer`` payload. Nothing downstream of the reward reads
        them; a driver's own stamps land after this runs."""
        for key in [key for key in trajectory.info if key.startswith("_")]:
            del trajectory.info[key]
        context = trajectory.info.get("context")
        if context and "answer" in context:
            trajectory.info["context"] = {key: value for key, value in context.items() if key != "answer"}

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
        """Post-``_step_single`` bookkeeping (sync + async paths): end an episode a sandbox fault was
        booked on (uncompleted, not truncated), enforce ``max_turns`` truncation, record reward/state,
        price the reward on termination, persist, return the step."""
        done = done or SANDBOX_FAULT_KEY in trajectory.info
        if trajectory.num_turns >= self.max_turns and not done:
            truncated = True
            done = True

        trajectory.done = done
        trajectory.truncated = truncated
        trajectory.add_reward(reward)
        trajectory.info.update(info)

        if done:
            self._settle_grade(trajectory, context)
            if self._rewards.external_terms and not self._cut_short(trajectory):
                trajectory.info[REWARD_PENDING_KEY] = True
            else:
                self._drop_grading_payload(trajectory)

        self._trajectories[episode_id] = trajectory

        return EnvStep(
            trajectory=trajectory,
            observation=trajectory.get_conversation(include_thinking=self.carry_reasoning),
            reward=reward,
            done=done,
            truncated=truncated,
            info=trajectory.info,
        )

    @abstractmethod
    def _reset_single(self, prompt: str | list[dict[str, str]], context: dict[str, Any] | None = None) -> Trajectory:
        """Initialize a single episode."""

    @abstractmethod
    def _step_single(
        self, trajectory: Trajectory, action: str, context: dict[str, Any] | None = None
    ) -> tuple[Trajectory, float, bool, bool, dict[str, Any]]:
        """Process a single action in the environment."""

    @abstractmethod
    def _grade_episode(self, trajectory: Trajectory, context: dict[str, Any] | None = None) -> EpisodeGrade:
        """Grade a finished episode: the objective in ``[0, 1]`` and the environment's own shaping terms."""

    @staticmethod
    def _cut_short(trajectory: Trajectory) -> bool:
        """An episode its driver lost, or one a sandbox fault ended: graded on what it earned and never
        sent to an external scorer, since a verdict on the fragment would be paid for and taught."""
        return EPISODE_ERROR_KEY in trajectory.info or SANDBOX_FAULT_KEY in trajectory.info

    def _episode_shaping(self, trajectory: Trajectory) -> dict[str, float]:
        """The protocol's episode-level shaping terms by bare name; the base has none."""
        return {}

    def _settle_grade(self, trajectory: Trajectory, context: dict[str, Any] | None) -> None:
        """Price the environment's side of the reward into ``reward_components``: the accrued per-turn
        deltas (``reward/turn_shaping``), the protocol's and the environment's shaping, and the grade
        through the reward's environment term. The externally scored terms follow in :meth:`settle_async`."""
        grade = self._grade_episode(trajectory, context)
        if not isinstance(grade, EpisodeGrade):
            raise TypeError(
                f"{type(self).__name__}._grade_episode must return an EpisodeGrade, got {type(grade).__name__}"
            )
        components = {component_key(TURN_SHAPING_COMPONENT): trajectory.total_reward}
        declared = self.shaping_components()
        for shaping in (self._episode_shaping(trajectory), grade.shaping):
            for name, value in shaping.items():
                if name not in declared:
                    raise ValueError(
                        f"{type(self).__name__} produced the shaping component {name!r} without declaring it "
                        f"in SHAPING_COMPONENTS ({sorted(declared)})"
                    )
                key = component_key(name)
                if key in components:
                    raise ValueError(f"reward component {key!r} is declared twice")
                components[key] = float(value)
        term = self._rewards.environment_term
        if term is not None:
            components[term.key] = term.price(grade.objective)
        # Every external term contributes 0 until it is scored, so the record holds the whole key set.
        for external in self._rewards.external_terms:
            components[external.key] = 0.0
        trajectory.info[REWARD_COMPONENTS_KEY] = components
        trajectory.total_reward = sum(components.values())

    def _scoring_sample(self, trajectory: Trajectory) -> ScoringSample:
        """What an external scorer reads of a finished episode: the prompt turns (everything before the
        first assistant turn), the policy's turns after them, and the row's reference answer. A task
        environment overrides this to hand the scorer its graded artifact (a submitted program)
        instead of the last visible message."""
        messages = [message.to_dict() for message in trajectory.messages]
        first = next(
            (i for i, message in enumerate(trajectory.messages) if message.role == "assistant"), len(messages)
        )
        context = trajectory.info.get("context") or {}
        return ScoringSample(prompt=messages[:first], completion=messages[first:], reference=context.get("answer"))

    def _apply_external_scores(self, trajectory: Trajectory, verdict: Mapping[str, ScoreResult]) -> None:
        """Price the external terms' verdicts into the components, record their diagnostics, and close
        the episode's reward: the pending mark goes, the grading payload with it."""
        components = trajectory.info[REWARD_COMPONENTS_KEY]
        for term in self._rewards.external_terms:
            result = verdict[term.name]
            if result.metrics:
                trajectory.info.setdefault(REWARD_METRICS_KEY, {}).update(result.metrics)
            if result.detail is not None:
                trajectory.info.setdefault(REWARD_DETAILS_KEY, {})[term.name] = result.detail
            if result.score is None:
                # The scorer, not the policy, failed: the term contributes nothing and the episode
                # leaves the group baseline rather than teaching a forced verdict. The reason is
                # stamped where the trainer's all-invalid halt reads it, so a dead judge names itself.
                error = result.error or "no score"
                trajectory.info.setdefault(REWARD_ERRORS_KEY, {})[term.name] = error
                trajectory.info[EPISODE_INVALID_KEY] = True
                trajectory.info[EPISODE_INVALID_REASON_KEY] = f"reward term {term.name!r} scored nothing: {error}"
            else:
                components[term.key] = term.price(result.score)
        trajectory.total_reward = sum(components.values())
        del trajectory.info[REWARD_PENDING_KEY]
        self._drop_grading_payload(trajectory)

    async def settle_async(self, episode_ids: list[int]) -> None:
        """Score the externally rewarded terms of the named finished episodes and complete their rewards.

        Every term's scorer runs concurrently over the batch. Pure I/O over finished trajectories, so
        the dispatcher runs it on its loop for sync and async environments alike; an episode without
        a pending reward is a no-op.
        """
        pending = [trajectory for trajectory in self._episodes(episode_ids) if REWARD_PENDING_KEY in trajectory.info]
        if not pending:
            return
        verdicts = await self._rewards.score([self._scoring_sample(trajectory) for trajectory in pending])
        for trajectory, verdict in zip(pending, verdicts, strict=True):
            self._apply_external_scores(trajectory, verdict)

    def settle(self, episode_ids: list[int]) -> None:
        """:meth:`settle_async` for a caller with no event loop running (a sync driver). Each call runs
        on a loop of its own and releases the scorers' clients with it, so the next call builds fresh
        ones instead of reusing clients bound to a closed loop."""
        if not any(REWARD_PENDING_KEY in trajectory.info for trajectory in self._episodes(episode_ids)):
            return

        async def settle_and_release() -> None:
            try:
                await self.settle_async(episode_ids)
            finally:
                await self._rewards.aclose()

        asyncio.run(settle_and_release())

    def _episodes(self, episode_ids: list[int]) -> list[Trajectory]:
        trajectories = []
        for episode_id in episode_ids:
            trajectory = self._trajectories.get(episode_id)
            if trajectory is None:
                raise ValueError(f"Episode {episode_id} not found")
            trajectories.append(trajectory)
        return trajectories

    def _run_or_schedule(self, coroutine) -> None:
        """Run a teardown coroutine to completion when no loop runs here, else schedule it on the
        running one (a close hook is called from an actor's loop and from plain teardown alike)."""
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

    def reset(
        self, prompts: list[str | list[dict[str, str]]], contexts: list[dict[str, Any]] | None = None
    ) -> tuple[list[int], list[EnvStep]]:
        """Reset the environment with multiple prompts."""
        if contexts is None:
            contexts = [None] * len(prompts)

        results = []
        for prompt, context in zip(prompts, contexts, strict=False):
            episode_id = self._get_next_episode_id()
            trajectory = self._reset_single(prompt, context)
            self._bind_effort_profile(trajectory, context)
            trajectory.info["episode_id"] = episode_id

            self._trajectories[episode_id] = trajectory

            results.append((episode_id, self._first_step(trajectory)))

        episode_ids = [r[0] for r in results]
        steps = [r[1] for r in results]
        return episode_ids, steps

    def step(
        self, episode_ids: list[int], actions: list[str], contexts: list[dict[str, Any]] | None = None
    ) -> list[EnvStep]:
        """Take a step in multiple episodes."""
        if contexts is None:
            contexts = [None] * len(episode_ids)

        steps = []
        for episode_id, action, context in zip(episode_ids, actions, contexts, strict=False):
            trajectory = self._trajectories.get(episode_id)
            if trajectory is None:
                raise ValueError(f"Episode {episode_id} not found")

            if trajectory.done:
                steps.append(self._done_step(trajectory))
                continue

            self._add_action_message(trajectory, action, context)
            trajectory, reward, done, truncated, info = self._step_single(trajectory, action, context)
            steps.append(self._finalize_step(episode_id, trajectory, reward, done, truncated, info, context))

        return steps

    def finalize_truncated(self, episode_ids: list[int]) -> list[EnvStep]:
        """Finalize still-open episodes as truncated, without a synthetic model turn.

        For drivers whose episode ended mid-flight (generation failure, external abort). An empty-text
        step would take the plain-text terminal path and mark the episode ``completed``, paying
        completion-graded envs the full objective for an episode that never finished. Here
        ``info["completed"]`` stays False while reward already earned (tool rewards, a graded
        ``submit_solution``) is preserved by the grade. A driver that lost the episode
        stamps :data:`EPISODE_ERROR_KEY` first, which keeps the turn-overflow price off it. No-op for
        already-done episodes.
        """
        steps = []
        for episode_id in episode_ids:
            trajectory = self._trajectories.get(episode_id)
            if trajectory is None:
                raise ValueError(f"Episode {episode_id} not found")
            if trajectory.done:
                steps.append(self._done_step(trajectory))
                continue
            steps.append(self._finalize_step(episode_id, trajectory, 0.0, True, True, {}, None))
        return steps

    def get_trajectories(self, episode_ids: list[int]) -> list[Trajectory | None]:
        """Get trajectories for the given episode IDs."""
        return [self._trajectories.get(eid) for eid in episode_ids]

    def cleanup(self, episode_ids: list[int]) -> None:
        """Drop the named episodes, releasing what each holds."""
        for eid in episode_ids:
            self._release_episode(eid)
            self._trajectories.pop(eid, None)

    def _release_episode(self, episode_id: int) -> None:  # noqa: B027  optional hook; subclasses override
        """Release one episode's external resources (a sandbox session, a connection) as
        :meth:`cleanup` drops it. A base episode holds none beyond its trajectory."""

    def close(self) -> None:
        """Clean up resources: the reward scorers' clients here; subclasses extend and call ``super()``."""
        if self._rewards.external_terms:
            self._run_or_schedule(self._rewards.aclose())

    def verify_backend(self) -> None:
        """Reachability check for the environment's external scoring/tool backends; raise to abort.

        The launch script calls this once on global rank 0 before training starts (the verdict is
        broadcast so all ranks raise together). The base probes every externally scored reward term
        (a judge, a reward model) once, in the run's exact request shape; an environment with a
        backend of its own extends this and calls ``super()`` — a misconfigured backend must fail
        the launch, not invalidate every episode of a running multi-GPU job."""
        if self._rewards.external_terms:
            asyncio.run(self._rewards.verify())


class AsyncBaseEnvironment(BaseEnvironment):
    """BaseEnvironment with async reset/step for I/O-bound operations."""

    async def _reset_single_async(
        self, prompt: str | list[dict[str, str]], context: dict[str, Any] | None = None
    ) -> Trajectory:
        """Async version of reset. Override for async I/O."""
        return self._reset_single(prompt, context)

    async def _step_single_async(
        self, trajectory: Trajectory, action: str, context: dict[str, Any] | None = None
    ) -> tuple[Trajectory, float, bool, bool, dict[str, Any]]:
        """Async version of step. Override for async I/O."""
        return self._step_single(trajectory, action, context)

    async def reset_async(
        self, prompts: list[str | list[dict[str, str]]], contexts: list[dict[str, Any]] | None = None
    ) -> tuple[list[int], list[EnvStep]]:
        """Async reset for multiple prompts."""
        if contexts is None:
            contexts = [None] * len(prompts)

        async def reset_one(prompt, context):
            episode_id = self._get_next_episode_id()
            trajectory = await self._reset_single_async(prompt, context)
            self._bind_effort_profile(trajectory, context)
            trajectory.info["episode_id"] = episode_id
            self._trajectories[episode_id] = trajectory

            return episode_id, self._first_step(trajectory)

        results = await asyncio.gather(*[reset_one(p, c) for p, c in zip(prompts, contexts, strict=False)])

        return [r[0] for r in results], [r[1] for r in results]

    async def step_async(
        self, episode_ids: list[int], actions: list[str], contexts: list[dict[str, Any]] | None = None
    ) -> list[EnvStep]:
        """Async step for multiple episodes."""
        if contexts is None:
            contexts = [None] * len(episode_ids)

        async def step_one(episode_id, action, context):
            trajectory = self._trajectories.get(episode_id)
            if trajectory is None:
                raise ValueError(f"Episode {episode_id} not found")

            if trajectory.done:
                return self._done_step(trajectory)

            self._add_action_message(trajectory, action, context)
            trajectory, reward, done, truncated, info = await self._step_single_async(trajectory, action, context)
            return self._finalize_step(episode_id, trajectory, reward, done, truncated, info, context)

        return await asyncio.gather(
            *[step_one(eid, act, ctx) for eid, act, ctx in zip(episode_ids, actions, contexts, strict=False)]
        )
