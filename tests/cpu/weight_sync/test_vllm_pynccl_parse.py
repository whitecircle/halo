#!/usr/bin/env python
"""``VLLM_DISABLE_PYNCCL`` must be read the way vLLM reads it, since third-party variables match their owner's parse.
``env_flag`` also accepts ``yes``/``on``, which would refuse a launch whose server has pynccl enabled."""

import pytest

from halo.weight_sync.transport.pynccl import vllm_pynccl_disabled


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "True", " true "])
def test_vllm_truthy_spellings_disable(monkeypatch, value):
    monkeypatch.setenv("VLLM_DISABLE_PYNCCL", value)
    assert vllm_pynccl_disabled() is True


@pytest.mark.parametrize("value", ["yes", "on", "y", "enabled"])
def test_spellings_vllm_ignores_do_not_refuse_the_trainer(monkeypatch, value):
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
    raise SystemExit(pytest.main([__file__, "-v"]))
