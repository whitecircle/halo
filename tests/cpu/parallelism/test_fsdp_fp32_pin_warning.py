"""CPU tests for the FSDP2 fp32-pin cast warning (``src/distributed/fsdp.py``).

FSDP2's ``MixedPrecisionPolicy`` casts per fully_shard group and has no per-parameter dtype
mechanism, so params pinned fp32 via transformers' ``_keep_in_fp32_modules(_strict)`` class
attributes (e.g. DeepSeek-V4's HC mixers / ``sinks`` / norms) compute at ``param_dtype`` under
multi-GPU while a single-GPU run honors fp32. The wrap-time warning must fire exactly when
fp32-stored pinned params meet a low-precision policy — and stay silent for uniform-bf16 loads
(the EP loader path) and for families without pins.

    python tests/cpu/parallelism/test_fsdp_fp32_pin_warning.py
"""

import logging
import sys

import pytest
import torch
import torch.nn as nn
from torch.distributed.fsdp import MixedPrecisionPolicy

from src.distributed.fsdp import _warn_fp32_pins_cast_by_policy


class _PinnedModel(nn.Module):
    """Minimal stand-in for a transformers class carrying fp32 pins (module names AND bare params)."""

    _keep_in_fp32_modules = ["compressor.kv_proj"]
    _keep_in_fp32_modules_strict = ["sinks", "input_layernorm"]

    def __init__(self, pin_dtype=torch.float32):
        super().__init__()
        self.q_proj = nn.Linear(4, 4).to(torch.bfloat16)
        self.input_layernorm = nn.LayerNorm(4).to(pin_dtype)
        self.sinks = nn.Parameter(torch.zeros(4, dtype=pin_dtype))
        self.compressor = nn.Module()
        self.compressor.kv_proj = nn.Linear(4, 4).to(pin_dtype)


class _PinlessModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4)  # fp32, but no pin declaration → not a divergence


def _bf16_policy():
    return MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16)


def _warning_text(caplog, model, mp_policy) -> str:
    """The wrap-time warning the policy produces for ``model``, or ``""`` when it stays silent."""
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="src.distributed.fsdp"):
        _warn_fp32_pins_cast_by_policy(model, mp_policy)
    return " ".join(record.message for record in caplog.records)


def test_fp32_pins_under_bf16_policy_are_named(caplog):
    """The warning must name the affected modules — a count alone cannot be acted on."""
    message = _warning_text(caplog, _PinnedModel(), _bf16_policy())
    assert "fp32-pinned" in message
    assert "5 fp32-pinned" in message  # sinks + LayerNorm weight/bias + kv_proj weight/bias
    for pinned_module in ("sinks", "input_layernorm", "compressor.kv_proj"):
        assert pinned_module in message
    assert "q_proj" not in message  # unpinned bf16 param never reported


def test_uniform_bf16_load_is_silent(caplog):
    """The EP loader materializes uniform bf16 — no fp32 storage, no warning."""
    assert _warning_text(caplog, _PinnedModel(pin_dtype=torch.bfloat16), _bf16_policy()) == ""


def test_fp32_policy_and_pinless_families_are_silent(caplog):
    fp32_policy = MixedPrecisionPolicy(param_dtype=torch.float32, reduce_dtype=torch.float32)
    assert _warning_text(caplog, _PinnedModel(), fp32_policy) == ""
    assert _warning_text(caplog, _PinlessModel(), _bf16_policy()) == ""
    assert _warning_text(caplog, _PinnedModel(), None) == ""


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
