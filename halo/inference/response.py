"""The normalized response record of an OpenAI-compatible call, with the finish-reason and reasoning-text
accessors that read either engine's spelling."""

from collections.abc import Mapping
from typing import Any

from openai.types.chat import ChatCompletionMessageToolCall
from pydantic import BaseModel

FINISH_REASON_LENGTH = "length"
# An engine aborted the generation (a weight-sync pause dropping in-flight requests, an engine restart).
FINISH_REASON_ABORT = "abort"
# The fragment is never a natural termination: neither trained as one (the truncated-turn flag) nor graded as the
# model's answer (the cut-turn recovery).
ENGINE_CUT_FINISH_REASONS = (FINISH_REASON_LENGTH, FINISH_REASON_ABORT)


class OpenAIResponse(BaseModel):
    answer: str | None
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
    """Reads raw JSON or an SDK object.

    SGLang reports a length cut-off in ``stop_reason``; vLLM's ``stop_reason`` can hold a stop-token id, which is not a
    reason. A completion that consumed its whole ``max_tokens`` is a cut whatever the label: vLLM reports
    ``tool_calls`` whenever its parser salvaged a call, so a turn cut inside its call arrives labelled complete with
    empty arguments. Pass ``completion_tokens`` and ``max_tokens`` to read the cap off the count.
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
    """vLLM answers ``reasoning``, SGLang ``reasoning_content``; reading one yields an empty CoT against the other."""
    if isinstance(message, Mapping):
        text = message.get("reasoning") or message.get("reasoning_content")
    else:
        text = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
    return text if isinstance(text, str) else None
