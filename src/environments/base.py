"""Environment protocol for multi-turn GRPO tasks: the conversation record, the step result, and the
base classes an environment subclasses. ``AsyncBaseEnvironment`` runs its turn batch through
``asyncio.gather`` for I/O-bound tool/API calls. The episode driver is :mod:`src.environments.episode`."""

import asyncio
import itertools
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from src.inference.response import FINISH_REASON_LENGTH

# Reasoning-effort levels for the chat template ("Reasoning: <level>"). "random" resolves per episode.
VALID_REASONING_EFFORTS = ("low", "medium", "high")

# Set in ``info`` when an episode completed but its reward carries no learning signal (a failed grading
# backend forced the failure reward); the trainer excludes it from the GRPO group baseline.
EPISODE_INVALID_KEY = "episode_invalid"

# The task-outcome component of an env's ``reward_components``, the term advantage shaping gates on.
# Shaping falls back to the total reward when the component is absent, so a misspelled key shapes on
# the total instead.
OBJECTIVE_REWARD_KEY = "reward/objective"

# The per-episode solve flag the rollout metrics average into the group solve rate; a misspelled key
# drops the metric rather than raising.
SOLVE_RATE_KEY = "outcome/solve_rate"


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


def require_magnitudes(**knobs: float) -> None:
    """Reject a negative value for any reward/penalty magnitude knob.

    The minus sign is applied at the use site, so a negative config value would turn a penalty into a
    bonus.
    """
    for name, value in knobs.items():
        if value < 0:
            raise ValueError(f"{name} must be >= 0 (a magnitude), got {value}")


@dataclass(slots=True)
class Message:
    """A single conversation message (``__slots__`` for memory efficiency at scale)."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    # Emitted by to_dict only for training (include_thinking) — an unknown field can 400 a vLLM request.
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

    def to_dict(self, include_thinking: bool = False) -> dict[str, Any]:
        """Convert to dict for tokenizer/API. ``include_thinking`` is opt-in (training tokenization
        only); default keeps assistant CoT out of vLLM generation requests."""
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
    # Applied per-episode CoT budget (min(level budget, global cap)), for the calibration reward.
    reasoning_budget: int | None = None

    _assistant_count: int = field(default=0, repr=False)

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

    def get_conversation(self) -> list[dict[str, Any]]:
        """Get conversation as list of dicts for tokenizer."""
        return [m.to_dict() for m in self.messages]


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

    Subclasses implement ``_reset_single``, ``_step_single``, ``_compute_reward``. Parallel rollout
    collection runs one episode per Ray actor instance (see ray_actors.py).
    """

    # Turn budget used when the config names none. Per class, since an agentic edit-run-test loop and
    # a one-shot exam need very different budgets.
    DEFAULT_MAX_TURNS: int = 10

    # Declared here because ``_add_action_message`` trims stored ``tool_calls`` to it; ``None`` = no cap.
    max_tool_calls_per_turn: int | None = None

    # The prompt trajectories open with, assigned by the protocol layers that build one; eval records
    # it in the trajectory meta. ``None`` = no system turn.
    system_prompt: str | None = None

    # What :meth:`_handle_length_cutoff` feeds back after the engine cuts a turn off at its token cap.
    # Per protocol, since the text must ask for that protocol's next move; ``None`` means the protocol
    # has no recovery path and does not route cut-off turns there.
    LENGTH_CUTOFF_NUDGE: str | None = None

    def __init__(self, max_turns: int | None = None, max_observation_chars: int = 16384, **kwargs):
        """``max_turns`` caps turns before truncation, defaulting to this class's
        :data:`DEFAULT_MAX_TURNS`; ``max_observation_chars`` caps a tool observation's length.
        ``reasoning_effort`` (popped from kwargs) steers the chat template's CoT depth:
        ``low``/``medium``/``high``, ``"random"``, or ``None``.

        Any remaining keyword raises: the registry factories forward the whole ``env_config``, so a key
        no constructor in the chain binds is a typo or a knob meant for another ``env_type``.
        """
        max_turns = self.DEFAULT_MAX_TURNS if max_turns is None else max_turns
        # 0 turns makes the rollout loop a no-op, producing an all-zero batch.
        if max_turns < 1:
            raise ValueError(f"max_turns must be >= 1, got {max_turns}")
        self.max_turns = max_turns
        self.max_observation_chars = max_observation_chars
        reasoning_effort = kwargs.pop("reasoning_effort", None)
        if reasoning_effort is not None and reasoning_effort not in (*VALID_REASONING_EFFORTS, "random"):
            raise ValueError(
                f"reasoning_effort must be one of {(*VALID_REASONING_EFFORTS, 'random')} or None, "
                f"got {reasoning_effort!r}"
            )
        self.reasoning_effort = reasoning_effort
        if kwargs:
            raise TypeError(
                f"{type(self).__name__} got unexpected environment option(s) {sorted(kwargs)}. Every "
                f"environment_kwargs key must be a constructor parameter of the resolved environment "
                f"(check the env_type it belongs to and the spelling)."
            )

        self._trajectories: dict[int, Trajectory] = {}
        self._episode_id_generator = itertools.count()

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
        """Hard per-turn thinking-token budget for a resolved effort level, or ``None`` to use the
        global rollout budget. Override in envs that bind a level to a token budget."""
        return None

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
        ``outcome/*`` (task success), ``episode/*`` (agent behavior), ``reward/*`` (decomposition). Base
        emits the tool-use signal; task envs override to add outcome + reward-component metrics."""
        metrics: dict[str, float] = {}
        if "total_tool_calls" in trajectory.info:
            metrics["episode/tool_calls"] = float(trajectory.info["total_tool_calls"])
        # Tracked separately: a termination-rate metric cannot tell a cut-off turn from an answer.
        metrics["episode/length_cutoff_turns"] = float(trajectory.info.get("length_cutoff_turns", 0))
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
                truncated=ctx.get("finish_reason") == FINISH_REASON_LENGTH,
            )
        )

    def _handle_length_cutoff(self, trajectory: Trajectory) -> tuple[Trajectory, float, bool, bool, dict[str, Any]]:
        """Handle a turn the engine cut off at its token cap before it produced anything.

        The turn is nudged and retried within ``max_turns`` rather than graded: the fragment would end
        the episode on a mid-sentence string that reads as a natural termination. It carries no reward
        penalty of its own (rationale: ``agent-docs/training-methods/grpo/environmental-grpo.md``).
        Implemented on the base so ``episode/length_cutoff_turns`` counts the same event for every
        protocol; the nudge wording is per protocol (:data:`LENGTH_CUTOFF_NUDGE`).
        """
        if self.LENGTH_CUTOFF_NUDGE is None:
            raise NotImplementedError(
                f"{type(self).__name__} routed a length-cut turn to _handle_length_cutoff without "
                f"declaring LENGTH_CUTOFF_NUDGE — the episode would continue with no message telling "
                f"the model what happened."
            )
        trajectory.info["length_cutoff_turns"] = trajectory.info.get("length_cutoff_turns", 0) + 1
        trajectory.add_message(Message.user(self.LENGTH_CUTOFF_NUDGE))
        return trajectory, 0.0, False, False, {"length_cutoff": True}

    def _first_step(self, trajectory: Trajectory) -> EnvStep:
        """Opening :class:`EnvStep` for a freshly reset episode (sync + async reset paths)."""
        return EnvStep(
            trajectory=trajectory,
            observation=trajectory.get_conversation(),
            reward=0.0,
            done=False,
            truncated=False,
            info=trajectory.info,
        )

    def _done_step(self, trajectory: Trajectory) -> EnvStep:
        """Terminal :class:`EnvStep` for an episode that is already complete (a no-op step)."""
        return EnvStep(
            trajectory=trajectory,
            observation=trajectory.get_conversation(),
            reward=0.0,
            done=True,
            truncated=trajectory.truncated,
            info=trajectory.info,
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
        """Post-``_step_single`` bookkeeping (sync + async paths): enforce ``max_turns`` truncation,
        record reward/state, compute the final reward on termination, persist, return the step."""
        if trajectory.num_turns >= self.max_turns and not done:
            truncated = True
            done = True

        trajectory.done = done
        trajectory.truncated = truncated
        trajectory.add_reward(reward)
        trajectory.info.update(info)

        if done:
            trajectory.total_reward = self._compute_reward(trajectory, context)

        self._trajectories[episode_id] = trajectory

        return EnvStep(
            trajectory=trajectory,
            observation=trajectory.get_conversation(),
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
    def _compute_reward(self, trajectory: Trajectory, context: dict[str, Any] | None = None) -> float:
        """Compute the final reward for a complete trajectory."""

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
        completion-rewarded envs ``success_reward`` for an episode that never finished. Here
        ``info["completed"]`` stays False while reward already earned (tool rewards, a graded
        ``submit_solution``) is preserved by ``_compute_reward``. No-op for already-done episodes.
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

    def close(self) -> None:  # noqa: B027  optional no-op lifecycle hook; subclasses override
        """Clean up resources."""

    def verify_backend(self) -> None:  # noqa: B027  optional no-op lifecycle hook; subclasses override
        """Reachability check for the environment's external scoring/tool backend; raise to abort.

        The launch script calls this once on global rank 0 before training starts (the verdict is
        broadcast so all ranks raise together). Environments whose reward depends on an external
        service (an LLM judge, a remote sandbox) override it, so a misconfigured backend fails the
        launch instead of invalidating every episode of a running multi-GPU job."""


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
