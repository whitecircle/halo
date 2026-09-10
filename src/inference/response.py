"""The normalized response record every OpenAI-compatible call returns, plus the finish-reason and
reasoning-text accessors shared by the client, the JSONL resume store and the aiohttp rollout path.
"""

from collections.abc import Mapping
from typing import Any

from openai.types.chat import ChatCompletionMessageToolCall
from pydantic import BaseModel

# The OpenAI-wire finish_reason for a generation cut off at its token cap.
FINISH_REASON_LENGTH = "length"
# vLLM's finish reason for a generation the engine aborted (a pause in abort mode, an engine restart).
FINISH_REASON_ABORT = "abort"
# Finish reasons that ended a turn without the model choosing to stop. The fragment is never a
# natural termination, so it is neither trained as one (the truncated-turn flag) nor graded as the
# model's answer (the cut-turn recovery each protocol routes to).
ENGINE_CUT_FINISH_REASONS = (FINISH_REASON_LENGTH, FINISH_REASON_ABORT)


class OpenAIResponse(BaseModel):
    answer: BaseModel | dict[str, object] | str | None
    reasoning: str | None
    finish_reason: str
    tool_calls: list[ChatCompletionMessageToolCall] | None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def get_finish_reason(
    choice: Mapping[str, Any] | Any,
    *,
    completion_tokens: int | None = None,
    max_tokens: int | None = None,
) -> str | None:
    """The finish reason of one completion choice, whether it arrives as raw JSON or an SDK object.

    SGLang reports a length cut-off in ``stop_reason``, so reading ``finish_reason`` alone would
    grade a truncated fragment as a completed answer. vLLM's ``stop_reason`` can instead hold a
    stop-token id, which is not a reason.

    A completion that consumed its whole ``max_tokens`` is a cut whatever the engine labelled it:
    vLLM reports ``tool_calls`` whenever its parser salvaged a call from the text, so a turn cut
    inside its call arrives labelled complete, with the call's name and empty arguments. Pass
    ``completion_tokens`` and ``max_tokens`` to read the cap off the count as well.
    """
    if isinstance(choice, Mapping):
        reason = choice.get("finish_reason") or choice.get("stop_reason")
    else:
        reason = getattr(choice, "finish_reason", None) or getattr(choice, "stop_reason", None)
    reason = reason if isinstance(reason, str) else None
    if reason in ENGINE_CUT_FINISH_REASONS:
        return reason
    if completion_tokens is not None and max_tokens is not None and completion_tokens >= max_tokens:
        return FINISH_REASON_LENGTH
    return reason


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
