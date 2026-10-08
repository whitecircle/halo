#!/usr/bin/env python
"""The reward-model scripts write each generated turn as ``{"role", "content"}`` plus the reply's
non-null extras, never its null optional fields.

A served reply carries the SDK's optional fields (``annotations``, ``refusal``, ``audio``,
``function_call``) and the engine's own (``reasoning``, ``reasoning_content``), nearly all null; a
dump of the reply object writes them into every ``full_conversation``, ``chosen`` / ``rejected`` and
``completions`` turn. A non-null ``reasoning_content`` is the chain of thought the chat template
renders, so it stays.

Run: python tests/cpu/inference/test_rm_record_shape.py
"""

import asyncio
import copy
import types

import pytest
from openai.types.chat import ChatCompletion

from scripts.inference.reward_model import _common as rm_common

# A chat completion as vLLM serializes it: every optional field present, null or empty.
_SERVED_REPLY = {
    "id": "chatcmpl-1",
    "object": "chat.completion",
    "created": 0,
    "model": "gen-model",
    "choices": [
        {
            "index": 0,
            "finish_reason": "stop",
            "stop_reason": None,
            "logprobs": None,
            "message": {
                "role": "assistant",
                "content": "The answer is 4.",
                "refusal": None,
                "annotations": None,
                "audio": None,
                "function_call": None,
                "tool_calls": [],
                "reasoning": None,
                "reasoning_content": None,
            },
        }
    ],
}


class _ServedClient:
    def __init__(self, reply=_SERVED_REPLY):
        async def _create(**_kwargs):
            return ChatCompletion.model_validate(reply)

        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=_create))


_ARGS = types.SimpleNamespace(model="gen-model", temperature=0.0, max_gen_tokens=8)


def _generate(client):
    return asyncio.run(
        rm_common.generate_chat_message(client, [{"role": "user", "content": "q"}], _ARGS, {"type": "text"})
    )


def test_a_generated_turn_drops_the_null_reply_fields():
    turn, finish_reason = _generate(_ServedClient())

    assert finish_reason == "stop"
    assert turn == {"role": "assistant", "content": "The answer is 4."}


def test_a_generated_turn_keeps_the_engine_reasoning():
    reply = copy.deepcopy(_SERVED_REPLY)
    reply["choices"][0]["message"]["reasoning_content"] = "2 + 2 = 4."
    turn, _ = _generate(_ServedClient(reply))

    assert turn == {"role": "assistant", "content": "The answer is 4.", "reasoning_content": "2 + 2 = 4."}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
