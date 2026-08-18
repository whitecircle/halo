#!/usr/bin/env python
"""``VLLM_DISABLE_PYNCCL`` is a vLLM variable and must be read the way vLLM reads it.

The trainer refuses to build its weight-sync communicator when this is set, because a server-aimed
``--env-file`` reaching the trainer would silently disable the broadcast and leave RL training
against a frozen generator. That refusal is correct — but only for values vLLM itself acts on.

The toolkit's ``env_flag`` additionally accepts ``yes``/``on``. Reading a third-party variable
through it makes the trainer refuse a launch the server happily ignores: pynccl stays ENABLED on the
server while the trainer dies claiming it is disabled. ``src/env.py``'s rule is that third-party
vars match their owner's parse, and this is the only place that rule is enforceable in a test.

Run: pytest tests/cpu/grpo/test_vllm_pynccl_parse.py
"""

import sys

import pytest

from src.distributed.nccl.transport.pynccl import vllm_pynccl_disabled


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "True", " true "])
def test_vllm_truthy_spellings_disable(monkeypatch, value):
    monkeypatch.setenv("VLLM_DISABLE_PYNCCL", value)
    assert vllm_pynccl_disabled() is True


@pytest.mark.parametrize("value", ["yes", "on", "y", "enabled"])
def test_spellings_vllm_ignores_do_not_refuse_the_trainer(monkeypatch, value):
    """These are truthy to ``env_flag`` and meaningless to vLLM — refusing on them fails a launch
    whose server has pynccl fully enabled."""
    monkeypatch.setenv("VLLM_DISABLE_PYNCCL", value)
    assert vllm_pynccl_disabled() is False


@pytest.mark.parametrize("value", ["0", "false", ""])
def test_falsey_and_empty_values_are_not_set(monkeypatch, value):
    monkeypatch.setenv("VLLM_DISABLE_PYNCCL", value)
    assert vllm_pynccl_disabled() is False


def test_absent_is_not_set(monkeypatch):
    monkeypatch.delenv("VLLM_DISABLE_PYNCCL", raising=False)
    assert vllm_pynccl_disabled() is False


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
