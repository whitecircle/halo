#!/usr/bin/env python
"""The engine-cut finish reasons are recognised in the spelling both rollout engines put on the OpenAI wire (a plain
``"abort"`` string and ``"length"``), while a normal stop stays a natural termination."""

import asyncio
from types import SimpleNamespace

import pytest

from halo.environments.engine_wire import turn_from_chat_completion
from halo.environments.rollout_config import RolloutConfig
from halo.inference.openai_client import generate_openai_response
from halo.inference.response import (
    ENGINE_CUT_FINISH_REASONS,
    FINISH_REASON_ABORT,
    FINISH_REASON_LENGTH,
    get_finish_reason,
)
from tests.common.fake_chat_client import RecordingChatClient, fake_completion


@pytest.mark.parametrize("reason", [FINISH_REASON_ABORT, FINISH_REASON_LENGTH])
def test_cut_reasons_are_read_off_raw_json_and_sdk_objects(reason):
    assert get_finish_reason({"finish_reason": reason}) == reason
    assert get_finish_reason(SimpleNamespace(finish_reason=reason, stop_reason=None)) == reason
    assert reason in ENGINE_CUT_FINISH_REASONS


def test_sglang_stop_reason_length_is_a_cut():
    """SGLang reports the token cap in ``stop_reason`` with ``finish_reason`` absent."""
    assert get_finish_reason({"finish_reason": None, "stop_reason": "length"}) in ENGINE_CUT_FINISH_REASONS


def test_a_natural_stop_and_a_stop_token_id_are_not_cuts():
    assert get_finish_reason({"finish_reason": "stop"}) not in ENGINE_CUT_FINISH_REASONS
    # vLLM puts the stop-token id in stop_reason; an int is not a reason.
    assert get_finish_reason({"finish_reason": None, "stop_reason": 200002}) is None


def test_a_completion_that_consumed_its_cap_is_a_cut_whatever_the_engine_labelled_it():
    """vLLM reports ``tool_calls`` whenever its parser salvaged a call, so a turn cut inside its tool call arrives
    labelled complete. A completion that used its whole ``max_tokens`` did not stop on its own."""
    assert (
        get_finish_reason({"finish_reason": "tool_calls"}, completion_tokens=4300, max_tokens=4300)
        == FINISH_REASON_LENGTH
    )
    assert get_finish_reason({"finish_reason": "tool_calls"}, completion_tokens=4299, max_tokens=4300) == "tool_calls"
    assert get_finish_reason({"finish_reason": "stop"}, completion_tokens=None, max_tokens=4300) == "stop"
    assert get_finish_reason({"finish_reason": "stop"}, completion_tokens=4300, max_tokens=None) == "stop"
    # An abort stays an abort: both are cuts, and the label says which.
    assert (
        get_finish_reason({"finish_reason": FINISH_REASON_ABORT}, completion_tokens=4300, max_tokens=4300)
        == FINISH_REASON_ABORT
    )


@pytest.mark.parametrize(("completion_tokens", "label"), [(16, FINISH_REASON_LENGTH), (15, "tool_calls")])
def test_batch_generation_reads_a_consumed_cap_as_the_rollout_path_does(completion_tokens, label):
    """Batch generation writes this label into its ``finish_reason`` column, so a row with a truncated call must not
    read as finished."""
    completion = fake_completion(None)
    completion.choices[0].finish_reason = "tool_calls"
    completion.usage.completion_tokens = completion_tokens
    client = RecordingChatClient(completion=completion)
    assert asyncio.run(generate_openai_response("m", "hi", client=client, max_tokens=16)).finish_reason == label

    wire = {
        "choices": [{"finish_reason": "tool_calls", "message": {"role": "assistant", "content": None}}],
        "usage": {"completion_tokens": completion_tokens},
    }
    assert turn_from_chat_completion(wire, RolloutConfig(max_tokens=16)).finish_reason == label


def test_batch_generation_keeps_the_label_of_a_reply_that_reports_no_usage():
    completion = fake_completion()
    completion.usage = None
    client = RecordingChatClient(completion=completion)
    assert asyncio.run(generate_openai_response("m", "hi", client=client, max_tokens=1)).finish_reason == "stop"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
