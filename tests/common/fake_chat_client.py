"""``AsyncOpenAI`` stand-ins for the chat path: a minimal completion, the client wiring every fake shares,
and a client that records what it was asked."""

import types


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
