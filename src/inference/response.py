"""The normalized response record every OpenAI-compatible call returns, and the finish-reason contract.

A leaf: the client, its JSONL resume store and the aiohttp rollout path all read it, so it can live
in none of them.
"""

from collections.abc import Mapping
from typing import Any

from openai.types.chat import ChatCompletionMessageToolCall
from pydantic import BaseModel

# The OpenAI-wire finish_reason for a generation cut off at its token cap. One spelling: the
# length-cutoff recovery and the truncated-turn flag both compare against it, and a drifted copy
# silently grades fragments as deliberate answers.
FINISH_REASON_LENGTH = "length"


class OpenAIResponse(BaseModel):
    answer: BaseModel | dict[str, object] | str | None
    reasoning: str | None
    finish_reason: str
    tool_calls: list[ChatCompletionMessageToolCall] | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def get_finish_reason(choice: Mapping[str, Any] | Any) -> str | None:
    """The finish reason of one completion choice, whether it arrives as raw JSON or an SDK object.

    SGLang spells a length cut-off ``stop_reason``; a reader of ``finish_reason`` alone grades the
    fragment as a deliberate answer. vLLM's ``stop_reason`` can be a stop-token id, which is no reason.
    """
    if isinstance(choice, Mapping):
        reason = choice.get("finish_reason") or choice.get("stop_reason")
    else:
        reason = getattr(choice, "finish_reason", None) or getattr(choice, "stop_reason", None)
    return reason if isinstance(reason, str) else None


def get_reasoning_text(message: Mapping[str, Any] | Any) -> str | None:
    """The CoT of one assistant message, whether it arrives as raw JSON or an SDK object.

    The engines disagree on the spelling — vLLM answers ``reasoning``, SGLang ``reasoning_content`` —
    and a reader of one alone silently sees an empty CoT on the other, which grades as "the model
    did not think" rather than as a missing field.
    """
    if isinstance(message, Mapping):
        text = message.get("reasoning") or message.get("reasoning_content")
    else:
        text = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
    return text if isinstance(text, str) else None
