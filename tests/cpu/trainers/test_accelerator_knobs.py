#!/usr/bin/env python
"""Test the accelerator-level parallelism knobs the mixin applies on BOTH launch branches.

``fp32_output_conversion`` (default False, i.e. off) clears accelerate's
``native_amp`` so ``prepare`` does not wrap the forward in ``convert_outputs_to_fp32`` — the full
``[B, S, V]`` logits upcast. It must apply outside ``_create_plain_accelerator``, which an
``accelerate launch`` that manages FSDP/DDP never reaches — a knob confined there leaves the upcast
in place on that branch. The knobs that genuinely cannot apply there must say so instead of
vanishing.

Run: python tests/cpu/trainers/test_accelerator_knobs.py
"""

import types

import pytest
from accelerate import PartialState

import src.trainers.mixins.base as mixin_module
from src.trainers.mixins.base import DistributedTrainerMixin

PartialState()  # the mixin's accelerate logger requires an initialized state

_configure = DistributedTrainerMixin._configure_fp32_output_conversion
_warn_unapplied = DistributedTrainerMixin._warn_knobs_unapplied_under_accelerate


def _fake_self(*, fp32_output_conversion=False, fp16=False, **knobs):
    config = types.SimpleNamespace(
        fp32_output_conversion=fp32_output_conversion,
        use_hsdp=False,
        fsdp_reshard_after_forward=False,
        fsdp_reshard_after_backward=True,
        fp32_grad_reduce=False,
    )
    for name, value in knobs.items():
        setattr(config, name, value)
    return types.SimpleNamespace(
        parallelism_config=config,
        args=types.SimpleNamespace(fp16=fp16, bf16=not fp16),
        accelerator=types.SimpleNamespace(native_amp=True),
    )


def _capture_warnings(fn) -> list[str]:
    calls: list[str] = []
    original = mixin_module.logger.warning
    mixin_module.logger.warning = lambda msg, *a, **k: calls.append(str(msg))
    try:
        fn()
    finally:
        mixin_module.logger.warning = original
    return calls


# ── fp32_output_conversion ───────────────────────────────────────────────


def test_native_amp_disabled_when_requested():
    me = _fake_self()
    _configure(me)
    assert me.accelerator.native_amp is False


def test_native_amp_kept_when_knob_on():
    me = _fake_self(fp32_output_conversion=True)
    _configure(me)
    assert me.accelerator.native_amp is True


def test_native_amp_kept_under_fp16():
    # native_amp also gates GradScaler unscaling — clearing it would clip on scaled grads.
    me = _fake_self(fp16=True)
    warnings = _capture_warnings(lambda: _configure(me))
    assert me.accelerator.native_amp is True
    assert any("fp16" in w for w in warnings), warnings


# ── knobs an accelerate-managed launch cannot honour ─────────────────────


def test_silent_when_all_knobs_are_default():
    assert _capture_warnings(lambda: _warn_unapplied(_fake_self())) == []


def test_names_every_unapplicable_knob_that_is_set():
    me = _fake_self(use_hsdp=True, fp32_grad_reduce=True)
    warnings = _capture_warnings(lambda: _warn_unapplied(me))
    assert len(warnings) == 1, warnings
    assert "use_hsdp" in warnings[0] and "fp32_grad_reduce" in warnings[0]
    # Not listed: it IS applied on this branch (moved out of the plain-accelerator path).
    assert "fp32_output_conversion" not in warnings[0]


def test_reshard_after_forward_is_reported():
    warnings = _capture_warnings(lambda: _warn_unapplied(_fake_self(fsdp_reshard_after_forward=True)))
    assert len(warnings) == 1 and "fsdp_reshard_after_forward" in warnings[0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
