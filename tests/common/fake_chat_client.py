"""``AsyncOpenAI`` stand-ins for the chat path: a minimal completion, the client wiring every fake shares, a
client that records what it was asked, and scripted replies in the shape the SDK's ``model_dump()`` hands
:func:`~src.environments.engine_wire.turn_from_chat_completion`."""

import copy
import types
from typing import Any


def fake_completion(content: str = "ok", **choice_extras):
    """Minimal stand-in for a chat-completion response (choices[0].message + usage); ``choice_extras``
    are attributes outside the OpenAI schema that an engine attaches to the choice."""
    message = types.SimpleNamespace(content=content, tool_calls=None, reasoning=None, reasoning_content=None)
    choice = types.SimpleNamespace(message=message, finish_reason="stop", stop_reason=None, **choice_extras)
    usage = types.SimpleNamespace(prompt_tokens=1, completion_tokens=2, total_tokens=3)
    return types.SimpleNamespace(choices=[choice], usage=usage)


class FakeChatClient:
    """An ``AsyncOpenAI`` stand-in whose ``chat.completions.create`` is the subclass's ``_create``."""

    def __init__(self):
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kwargs):
        raise NotImplementedError


class RecordingChatClient(FakeChatClient):
    """Records the create() kwargs it is called with (the last call in ``captured``, every call in
    ``calls``) and answers with ``completion`` (a plain :func:`fake_completion` when omitted)."""

    def __init__(self, completion=None):
        super().__init__()
        self.captured: dict = {}
        self.calls: list[dict] = []
        self._completion = completion

    async def _create(self, **kwargs):
        self.captured = kwargs
        self.calls.append(kwargs)
        return self._completion if self._completion is not None else fake_completion()


def chat_reply(
    answer: str | None = "done",
    *,
    finish_reason: str | None = "stop",
    completion_tokens: int | None = 1,
    tool_calls: list[Any] | None = None,
    reasoning: str | None = None,
    token_ids: list[int] | None = None,
) -> types.SimpleNamespace:
    """A stand-in for an SDK ``ChatCompletion``. A tool call is given as an object with ``id`` and
    ``function.name``/``function.arguments``, as the SDK types it; ``completion_tokens=None`` reports no usage;
    ``token_ids`` arrive as vLLM's ``token_id:<N>`` logprob entries."""
    calls = [
        {
            "id": call.id,
            "function": {"arguments": call.function.arguments, "name": call.function.name},
            "type": "function",
        }
        for call in tool_calls or []
    ]
    message = {"content": answer, "role": "assistant", "tool_calls": calls or None, "reasoning": reasoning}
    usage = None if completion_tokens is None else {"completion_tokens": completion_tokens, "prompt_tokens": 0}
    choice = {"finish_reason": finish_reason, "index": 0, "message": message}
    if token_ids is not None:
        choice["logprobs"] = {"content": [{"token": f"token_id:{i}", "logprob": -0.5} for i in token_ids]}
    data = {"choices": [choice], "usage": usage}
    return types.SimpleNamespace(model_dump=lambda: copy.deepcopy(data))


def scripted_chat(script: list[Any]):
    """An async stand-in for ``chat_completion(client, **request)`` driven by a list of replies/exceptions."""
    queue = list(script)

    async def chat(client: Any, **request: Any) -> Any:
        item = queue.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    return chat
