"""The normalized response record every OpenAI-compatible call returns, plus the finish-reason and
reasoning-text accessors shared by the client, the JSONL resume store and the aiohttp rollout path.
"""

from collections.abc import Mapping
from typing import Any

from openai.types.chat import ChatCompletionMessageToolCall
from pydantic import BaseModel

# The OpenAI-wire finish_reason for a generation cut off at its token cap. Both the length-cutoff
# recovery and the truncated-turn flag compare against this constant.
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

    SGLang reports a length cut-off in ``stop_reason``, so reading ``finish_reason`` alone would
    grade a truncated fragment as a completed answer. vLLM's ``stop_reason`` can instead hold a
    stop-token id, which is not a reason.
    """
    if isinstance(choice, Mapping):
        reason = choice.get("finish_reason") or choice.get("stop_reason")
    else:
        reason = getattr(choice, "finish_reason", None) or getattr(choice, "stop_reason", None)
    return reason if isinstance(reason, str) else None


def get_reasoning_text(message: Mapping[str, Any] | Any) -> str | None:
    """The CoT of one assistant message, whether it arrives as raw JSON or an SDK object.

    The engines use different field names: vLLM answers ``reasoning``, SGLang
    ``reasoning_content``. Reading only one of them yields an empty CoT against the other engine.
    """
    if isinstance(message, Mapping):
        text = message.get("reasoning") or message.get("reasoning_content")
    else:
        text = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
    return text if isinstance(text, str) else None
