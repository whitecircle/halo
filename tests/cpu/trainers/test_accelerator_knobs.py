#!/usr/bin/env python
"""Accelerator knobs on both launch branches; knobs an accelerate-managed launch cannot apply warn. The fp32 output
upcast is cleared outside ``build_plain_accelerator``, which ``accelerate launch`` never reaches."""

import types

import pytest

import halo.trainers.construction.accelerator as mixin_module
from halo.configs.parallelism_arguments import ParallelismArguments
from halo.configs.parser import HaloArgumentParser
from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.distributed_trainer import DistributedTrainerMixin

_disable_upcast = DistributedTrainerMixin._disable_native_amp_logits_upcast
_warn_unapplied = DistributedTrainerMixin._warn_knobs_unapplied_under_accelerate


def _fake_self(*, fp16=False, **knobs):
    # Typed but unvalidated: the warning reads each knob's default off the dataclass that declares it.
    config = object.__new__(ParallelismConfig)
    vars(config).update(
        use_hsdp=False,
        fsdp_reshard_after_forward=False,
        fsdp_reshard_after_backward=True,
        fsdp_defer_grad_sync=False,
        fp32_grad_reduce=False,
    )
    vars(config).update(knobs)
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


def test_native_amp_disabled_outside_fp16():
    me = _fake_self()
    _disable_upcast(me)
    assert me.accelerator.native_amp is False


def test_native_amp_kept_under_fp16():
    # native_amp also gates GradScaler unscaling; clearing it would clip on scaled grads.
    me = _fake_self(fp16=True)
    _disable_upcast(me)
    assert me.accelerator.native_amp is True


def test_the_upcast_has_no_knob():
    with pytest.raises(ValueError, match="fp32_output_conversion"):
        HaloArgumentParser((ParallelismArguments,)).parse_dict({"fp32_output_conversion": True})


def test_silent_when_all_knobs_are_default():
    assert _capture_warnings(lambda: _warn_unapplied(_fake_self())) == []


def test_names_every_unapplicable_knob_that_is_set():
    me = _fake_self(use_hsdp=True, fp32_grad_reduce=True)
    warnings = _capture_warnings(lambda: _warn_unapplied(me))
    assert len(warnings) == 1, warnings
    assert "use_hsdp" in warnings[0] and "fp32_grad_reduce" in warnings[0]


def test_reshard_after_forward_is_reported():
    warnings = _capture_warnings(lambda: _warn_unapplied(_fake_self(fsdp_reshard_after_forward=True)))
    assert len(warnings) == 1 and "fsdp_reshard_after_forward" in warnings[0]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
