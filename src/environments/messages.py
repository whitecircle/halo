"""The conversation an episode carries: its messages, the trajectory holding them, the step results an
environment returns, and the view of them the engine is shown."""

from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from typing import Any, NamedTuple

from src.environments.episode_keys import EPISODE_INVALID_KEY

# What follows a tool observation cut at its length cap, counting what it dropped.
TRUNCATION_MARKER = "\n…[truncated {dropped} chars]"

# Message keys a chat template may read assistant CoT from: harmony (gpt-oss) reads ``thinking``,
# other reasoning families ``reasoning_content``. Templates ignore a spelling they do not know, so
# emitting both renders the CoT once everywhere; emitting only the wrong one renders an empty block.
REASONING_KEYS = ("thinking", "reasoning_content")


def _content_text(content: Any) -> Any:
    """A user turn's ``content`` as the text an environment carries: a string as is, a list of content
    parts as its ``text`` parts concatenated, the way a chat template renders them. Any other part raises
    ``ValueError``: an environment's conversation is text, so an image part would be dropped unseen."""
    if not isinstance(content, list):
        return content
    other = [
        part
        for part in content
        if not (isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str))
    ]
    if other:
        kinds = sorted({str(part.get("type")) if isinstance(part, dict) else type(part).__name__ for part in other})
        raise ValueError(
            f"the task's user turn carries content parts other than text ({kinds}); an environment's "
            "conversation is text, so only {'type': 'text', 'text': ...} parts can be handed to it"
        )
    return "".join(part["text"] for part in content)


def task_prompt(prompt: str | list[dict[str, Any]]) -> Any:
    """The task an environment is handed for a dataset prompt, the same from the trainer and the eval: a string as
    is, or a conversation's last user turn (the rest is framing) as text; a conversation without one raises."""
    if not isinstance(prompt, list):
        return prompt
    for message in reversed(prompt):
        if message.get("role") == "user":
            return _content_text(message["content"])
    raise ValueError(
        f"the conversation has no 'user' message (roles: {[m.get('role') for m in prompt]}); an environment "
        "is handed the last user turn as the task, so there is nothing to send it"
    )


def truncate_text(text: str, max_chars: int) -> str:
    """``text`` cut to ``max_chars`` with the cut marked (:data:`TRUNCATION_MARKER`), so a reader knows
    it read a prefix."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + TRUNCATION_MARKER.format(dropped=len(text) - max_chars)


@dataclass(slots=True)
class Message:
    """A single conversation message."""

    role: str  # "user", "assistant", "system", "tool"
    content: str
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None
    # Rendered by to_dict only under include_thinking: the training render, and the request of a carrying env.
    thinking: str | None = None
    # Engine captures, dropped by to_dict; ``routing_mask`` stays raw base64, cheap to pickle through Ray.
    token_ids: list[int] | None = None
    token_logprobs: list[float] | None = None
    routing_mask: str | None = None
    routing_prompt_tokens: int | None = None
    prompt_token_ids: list[int] | None = None
    # The reasoning cap the turn ran under and the reasoning tokens it sampled: what ``reasoning_capped`` compares.
    thinking_cap: int | None = None
    reasoning_tokens: int | None = None
    # The turn flags behind ``untrainable``: cut by the engine, every call refused or unknown, ended on nothing.
    truncated: bool = False
    # The calls a cut turn was writing, never run: dropped by to_dict, shown marked to a scorer and the record.
    cut_tool_calls: list[dict[str, Any]] | None = None
    calls_rejected: bool = False
    # The calls the environment refused unrun or that named no tool: a scorer's view marks them as never run.
    refused_call_ids: tuple[str, ...] = ()
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
        """An assistant turn no tokenization path may reward (engine-cut, every call rejected, or empty): it stays
        in the render, its ids training only under a negative advantage."""
        return self.truncated or self.calls_rejected or self.empty

    @property
    def reasoning_capped(self) -> bool:
        """The turn's counted reasoning reached the cap it ran under, so the engine closed it and what the turn
        wrote after it was written past the cap."""
        cap, sampled = self.thinking_cap, self.reasoning_tokens
        return cap is not None and sampled is not None and sampled >= cap

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
    """The conversation as the engine is told it: every message's visible text and, with ``carry_reasoning``,
    the LAST assistant turn's reasoning, so a request grows by at most one reasoning budget. The rollout
    observation and the trainer's context re-render both come through here."""
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
    # The generation contract the rollout stamps, so a re-render states the level and budget the model ran under
    # (soft: stated, never enforced by the engine); the trainer's under-use floor is a share of the budget.
    reasoning_effort: str | None = None
    reasoning_budget: int | None = None
    reasoning_budget_soft: bool = False

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

    def last_assistant_message(self) -> Message:
        """The turn just taken: every step appends the model's message before it is handled."""
        message = next((m for m in reversed(self.messages) if m.role == "assistant"), None)
        if message is None:
            raise ValueError(
                "the trajectory holds no assistant turn to flag; a step records the model's message first"
            )
        return message

    @property
    def num_turns(self) -> int:
        """Number of assistant turns (cached)."""
        return self._assistant_count

    @property
    def episode_invalid(self) -> bool:
        """True when the environment marked this episode's reward as carrying no learning signal
        (see ``EPISODE_INVALID_KEY``); the trainer excludes it from the GRPO group baseline."""
        return bool(self.info.get(EPISODE_INVALID_KEY, False))


@dataclass(slots=True)
class EnvStep:
    """Result of an environment step, Gym-shaped: ``reward`` is this step's delta, ``done``/``truncated``/``info``
    mirror the trajectory as the step returned it (``info`` by identity, so mutating it mutates the trajectory)."""

    trajectory: Trajectory
    observation: list[dict[str, Any]]
    reward: float = 0.0
    done: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=dict)


class StepOutcome(NamedTuple):
    """A protocol step's result: an :class:`EnvStep` before ``_finalize_step``, no observation, reward unrecorded."""

    trajectory: Trajectory
    reward: float
    done: bool
    truncated: bool
    info: dict[str, Any]
