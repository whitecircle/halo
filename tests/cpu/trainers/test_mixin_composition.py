#!/usr/bin/env python
"""Composition contract of ``DistributedTrainerMixin``: which sub-mixin owns what, and where its
zero-arg ``super()`` calls land.

Two invariants the split into ``src/trainers/mixins/`` depends on, both silent when broken:

* every method the trainers and tests reach as ``DistributedTrainerMixin.<name>`` still resolves
  through the MRO after moving to a sub-mixin (a typo in the bases tuple drops it);
* the checkpoint methods' ``super()`` calls reach the **base Trainer**, not a sibling mixin. They
  are spelled ``super(CheckpointingMixin, self)``, so a base declaring ``save_model`` or
  ``_save_checkpoint`` between the two would silently intercept the save.

Run: python tests/cpu/trainers/test_mixin_composition.py
"""

from abc import ABC

import pytest

from src.trainers.mixins.base import DistributedTrainerMixin
from src.trainers.mixins.checkpointing import CheckpointingMixin
from src.trainers.mixins.grad_sync import GradientSyncMixin

# name -> the sub-mixin that must own it. Each is reached off ``DistributedTrainerMixin`` by a
# trainer, a test or a strategy, so an owner change must stay MRO-visible from the composed class.
_OWNERSHIP = {
    CheckpointingMixin: (
        "_checkpoint_context",
        "_checkpoint_load_context",
        "_checkpoint_loader",
        "_load_best_model",
        "_load_from_checkpoint",
        "_load_optimizer_and_scheduler",
        "_optimizer_store",
        "_persist_lr_scheduler_for_resume",
        "_persist_router_balancing_biases",
        "_restore_router_balancing_biases",
        "_rotate_checkpoints_after_sidecars",
        "_save_checkpoint",
        "save_model",
    ),
    GradientSyncMixin: (
        "_compute_global_grad_norm",
        "_compute_tp_grad_norm",
        "_patch_gradient_clipping_for_ep",
        "_patch_gradient_clipping_for_qlora",
        "_patch_gradient_clipping_for_tp",
        "_register_deferred_ep_grad_sync_hook",
        "_register_qlora_grad_sync_hook",
        "_register_tp_replicated_grad_sync_hook",
        "_setup_cp_gradient_sync",
        "_setup_ep_gradient_sync",
        "_setup_ep_tp_gradient_sync",
        "_setup_qlora_gradient_sync",
        "_sync_deferred_expert_grads",
        "_sync_qlora_grads",
        "_sync_tp_replicated_grads",
        "_tp_per_head_norm_param_ids",
        "_tp_sharded_plain_param_ids",
    ),
}

# The checkpoint methods that call ``super()`` — the HF Trainer implementations they extend.
_SUPER_DELEGATED = ("_load_from_checkpoint", "_load_optimizer_and_scheduler", "_save_checkpoint", "save_model")


class _BaseTrainer:
    """Stands in for ``transformers.Trainer`` — the intended target of every ``super()`` above."""

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        return "base"

    def _load_optimizer_and_scheduler(self, checkpoint):
        return "base"

    def _save_checkpoint(self, model, trial):
        return "base"

    def save_model(self, output_dir=None, _internal_call=False):
        return "base"


class _ComposedTrainer(DistributedTrainerMixin, _BaseTrainer):
    """The shape every concrete trainer has: the mixin first, a Trainer base behind it."""


def _owner(cls: type, name: str) -> type:
    return next(base for base in cls.__mro__ if name in base.__dict__)


def test_every_moved_method_is_owned_by_its_sub_mixin_and_reachable_from_the_composed_class():
    for mixin, names in _OWNERSHIP.items():
        assert mixin in DistributedTrainerMixin.__mro__, f"{mixin.__name__} is not a base of the trainer mixin"
        for name in names:
            assert _owner(mixin, name) is mixin, f"{name} is not defined by {mixin.__name__}"
            assert _owner(DistributedTrainerMixin, name) is mixin, (
                f"DistributedTrainerMixin.{name} resolves to {_owner(DistributedTrainerMixin, name).__name__}, "
                f"not {mixin.__name__} — the bases tuple or an override shadows the sub-mixin."
            )


def test_checkpoint_super_calls_reach_the_base_trainer():
    """No class between ``CheckpointingMixin`` and the Trainer base may define the delegated names."""
    mro = _ComposedTrainer.__mro__
    behind = mro[mro.index(CheckpointingMixin) + 1 :]
    for name in _SUPER_DELEGATED:
        owner = next(base for base in behind if name in base.__dict__)
        assert owner is _BaseTrainer, (
            f"super().{name}() from CheckpointingMixin resolves to {owner.__name__}, not the base Trainer — "
            f"a sibling mixin now intercepts it."
        )


def test_sibling_mixins_declare_no_overlapping_methods():
    """One concern each: two bases defining the same method make resolution depend on base order.

    Read off the composed class rather than listed here, so a sub-mixin added to the bases tuple is
    covered on arrival — a hand-maintained list silently stops checking the one it omits.
    """
    bases = tuple(base for base in DistributedTrainerMixin.__bases__ if base is not ABC)
    assert len(bases) >= 6, f"only {len(bases)} sub-mixins discovered — the derivation lost the bases tuple"
    seen: dict[str, str] = {}
    for base in bases:
        for name, value in vars(base).items():
            if name.startswith("__") or not callable(value):
                continue
            assert name not in seen, f"{name} is defined by both {seen[name]} and {base.__name__}"
            seen[name] = base.__name__


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
