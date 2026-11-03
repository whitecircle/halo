#!/usr/bin/env python
"""QLoRA rejects FSDP-shaping knobs it can never honor.
It skips FSDP2 (``fully_shard`` cannot wrap bnb's non-float Params4bit), so ``use_hsdp`` runs a flat all-reduce."""

import pytest

from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.trainer_rules import reject_fsdp_knobs_under_qlora


def _config(**knobs):
    # Typed but unvalidated: the gate reads each knob's default off the dataclass that declares it.
    config = object.__new__(ParallelismConfig)
    vars(config).update(
        use_hsdp=False,
        fsdp_reshard_after_forward=False,
        fsdp_reshard_after_backward=True,
        fsdp_defer_grad_sync=False,
    )
    vars(config).update(knobs)
    return config


def test_default_knobs_pass():
    reject_fsdp_knobs_under_qlora(_config())


@pytest.mark.parametrize(
    "knob, value",
    [
        ("use_hsdp", True),
        ("fsdp_reshard_after_forward", True),
        ("fsdp_reshard_after_backward", False),
        ("fsdp_defer_grad_sync", True),
    ],
)
def test_non_default_knob_raises_naming_it(knob, value):
    with pytest.raises(ValueError, match=knob):
        reject_fsdp_knobs_under_qlora(_config(**{knob: value}))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
