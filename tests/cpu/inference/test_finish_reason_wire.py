#!/usr/bin/env python
"""The engine-cut finish reasons must be recognised in the spelling both rollout engines put on the
OpenAI wire: a plain ``"abort"`` string (vLLM's abort-mode pause, SGLang's abort-mode quiesce) and
``"length"``, while a normal stop stays a natural termination.

Run: python tests/cpu/inference/test_finish_reason_wire.py
"""

from types import SimpleNamespace

import pytest

from src.inference.response import (
    ENGINE_CUT_FINISH_REASONS,
    FINISH_REASON_ABORT,
    FINISH_REASON_LENGTH,
    get_finish_reason,
)


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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
