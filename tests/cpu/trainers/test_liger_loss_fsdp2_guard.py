#!/usr/bin/env python
"""TRL's fused Liger loss is switched off once FSDP2 has sharded the model, whatever TRL calls it.

The loss matmuls ``lm_head.weight`` outside FSDP2's forward hooks, where that weight is a sharded
DTensor. TRL caches ``use_liger_kernel`` on the trainer and keeps the loss under a per-trainer
attribute, which the guard probes by name; a name it does not know (a TRL release that renamed it)
must raise rather than leave the fused loss running on a shard.

    python tests/cpu/trainers/test_liger_loss_fsdp2_guard.py
"""

from types import SimpleNamespace

import pytest

from src.trainers.mixins.base import DistributedTrainerMixin


def _host(*, fsdp_wrapped=True, use_liger_kernel=True, **loss_attrs):
    return SimpleNamespace(
        _fsdp_wrapped=fsdp_wrapped,
        use_liger_kernel=use_liger_kernel,
        args=SimpleNamespace(use_liger_kernel=use_liger_kernel),
        **loss_attrs,
    )


@pytest.mark.parametrize("attr", ["liger_loss_fn", "liger_grpo_loss", "liger_loss"])
def test_every_known_trl_liger_loss_is_disabled_under_fsdp2(attr):
    """DPO/KTO, GRPO, and the single name later TRL releases use for all of them."""
    host = _host(**{attr: object()})
    DistributedTrainerMixin._disable_trl_liger_loss_under_fsdp2(host)
    assert host.use_liger_kernel is False and host.args.use_liger_kernel is False


def test_an_unknown_liger_loss_name_raises_under_fsdp2():
    with pytest.raises(RuntimeError, match="none of TRL's known fused Liger loss attributes"):
        DistributedTrainerMixin._disable_trl_liger_loss_under_fsdp2(_host())


@pytest.mark.parametrize("host_kwargs", [{"fsdp_wrapped": False}, {"use_liger_kernel": False}])
def test_nothing_is_touched_without_fsdp2_or_without_the_flag(host_kwargs):
    """Unsharded, the fused loss reads a whole weight; with the flag off TRL built no loss at all."""
    host = _host(**host_kwargs)
    before = host.use_liger_kernel
    DistributedTrainerMixin._disable_trl_liger_loss_under_fsdp2(host)
    assert host.use_liger_kernel is before


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
