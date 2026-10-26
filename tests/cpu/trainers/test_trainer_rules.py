#!/usr/bin/env python
"""The trainer spine's optimizer, launcher and reference-sync refusals: each fires on its own condition, named by
its mechanism, and the supported run passes it.

Run: python tests/cpu/trainers/test_trainer_rules.py
"""

from types import SimpleNamespace

import pytest

from halo.optimizers.registry import NAMED_OPTIMIZER_BUILDERS
from halo.trainers.trainer_rules import (
    reject_conflicting_optimizer_choice,
    reject_ep_layers_under_accelerate,
    reject_sync_ref_model_on_sharded_policy,
)

_EP_LAYER = ("model.layers.0.mlp", object())


@pytest.mark.parametrize("optim", sorted(NAMED_OPTIMIZER_BUILDERS))
def test_bf16_optimizer_beside_a_named_optimizer_is_refused(optim):
    with pytest.raises(ValueError, match=rf"bf16_optimizer=True and optim='{optim}' both select an optimizer"):
        reject_conflicting_optimizer_choice(True, SimpleNamespace(optim=optim))


def test_either_optimizer_choice_alone_passes():
    """The refusal is the pair: the bf16 path beside a stock AdamW, or a named optimizer without it, is a run."""
    name = next(iter(NAMED_OPTIMIZER_BUILDERS))
    reject_conflicting_optimizer_choice(True, SimpleNamespace(optim="adamw_torch_fused"))
    reject_conflicting_optimizer_choice(False, SimpleNamespace(optim=name))
    reject_conflicting_optimizer_choice(True, None)


@pytest.mark.parametrize(
    ("refusal", "match"),
    [
        (
            lambda: reject_ep_layers_under_accelerate([_EP_LAYER]),
            r"EP layers but this is an 'accelerate launch' run",
        ),
        (
            lambda: reject_sync_ref_model_on_sharded_policy(SimpleNamespace(sync_ref_model=True), sharded=True),
            r"sync_ref_model is not supported under FSDP2/EP/TP: TRL's SyncRefModelCallback assumes an unwrapped",
        ),
    ],
    ids=["ep-layers-under-accelerate", "sync-ref-model-on-sharded-policy"],
)
def test_each_launcher_and_reference_refusal_names_its_mechanism(refusal, match):
    with pytest.raises(ValueError, match=match):
        refusal()


def test_the_launcher_and_reference_refusals_pass_where_their_mechanism_is_absent():
    reject_ep_layers_under_accelerate([])
    reject_sync_ref_model_on_sharded_policy(SimpleNamespace(sync_ref_model=True), sharded=False)
    reject_sync_ref_model_on_sharded_policy(SimpleNamespace(sync_ref_model=False), sharded=True)
    reject_sync_ref_model_on_sharded_policy(SimpleNamespace(), sharded=True)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
