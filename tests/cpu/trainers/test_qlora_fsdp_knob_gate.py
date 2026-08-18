#!/usr/bin/env python
"""QLoRA rejects FSDP-shaping knobs it can never honor.

QLoRA skips FSDP2 entirely (``fully_shard`` cannot wrap bnb's non-float Params4bit), so ``use_hsdp`` /
``fsdp_reshard_after_forward`` / ``fsdp_reshard_after_backward`` have nothing to act on — a
multi-node run asking for use_hsdp would otherwise do a flat replicated all-reduce with only an info
line about FSDP2 being skipped.

    python tests/cpu/trainers/test_qlora_fsdp_knob_gate.py
"""

import types

import pytest

from src.trainers.mixins.base import DistributedTrainerMixin

_reject = DistributedTrainerMixin._reject_fsdp_knobs_under_qlora


def _me(**knobs):
    config = types.SimpleNamespace(use_hsdp=False, fsdp_reshard_after_forward=False, fsdp_reshard_after_backward=True)
    for name, value in knobs.items():
        setattr(config, name, value)
    return types.SimpleNamespace(parallelism_config=config)


def test_default_knobs_pass():
    _reject(_me())


@pytest.mark.parametrize(
    "knob, value",
    [("use_hsdp", True), ("fsdp_reshard_after_forward", True), ("fsdp_reshard_after_backward", False)],
)
def test_non_default_knob_raises_naming_it(knob, value):
    with pytest.raises(ValueError, match=knob):
        _reject(_me(**{knob: value}))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
