#!/usr/bin/env python
"""``DistributedTrainerMixin``'s composition: which sub-mixin owns what, and where its zero-arg ``super()`` calls land.
Both are silent when broken: a typo in the bases tuple drops a method, and a base in between intercepts ``super()``."""

import pytest

from halo.trainers.accelerator import AcceleratorMixin
from halo.trainers.checkpointing import CheckpointingMixin
from halo.trainers.distributed_trainer import DistributedTrainerMixin
from halo.trainers.evaluation import EvaluationMixin
from halo.trainers.parallelism.grad_clipping import GradientClippingMixin
from halo.trainers.parallelism.grad_sync import GradientSyncMixin
from halo.trainers.parallelism.mode_setup import ModeSetupMixin
from halo.trainers.parallelism.tp_leader import TPLeaderBroadcastMixin
from halo.trainers.step_guard import NonFiniteStepGuardMixin
from halo.trainers.token_metrics import TokenMetricsMixin

# name -> the sub-mixin that must own it, so an owner change stays MRO-visible from the composed class.
_OWNERSHIP = {
    AcceleratorMixin: (
        "_disable_native_amp_logits_upcast",
        "_inner_training_loop",
        "_needs_custom_accelerator",
        "_warn_knobs_unapplied_under_accelerate",
        "create_accelerator_and_postprocess",
    ),
    EvaluationMixin: (
        "_dp_gather_for_metrics",
        "_eval_ranks_have_unequal_batches",
        "_install_dp_metric_gather",
        "eval_split_rows",
        "evaluate",
        "evaluation_loop",
        "suspended_dp_metric_gather",
    ),
    ModeSetupMixin: (
        "_adopt_cp_wrapper",
        "_apply_cp_fsdp2",
        "_apply_dp_fsdp2",
        "_apply_ep_aware_dp_fsdp2",
        "_apply_ep_fsdp2",
        "_apply_ep_tp_fsdp2",
        "_apply_tp_fsdp2",
        "_cast_peft_params_to_compute_dtype",
        "_dtype_excluded_params",
        "_find_cp_wrapper",
        "_fsdp_excluded_params",
        "_get_tp_rank",
        "_install_qlora_grad_sync",
        "_upcast_non_ep_params_to_fp32",
        "_wrap_for_parallel_mode",
    ),
    TPLeaderBroadcastMixin: (
        "_broadcast_object_from_tp_leader",
        "_broadcast_tensors_from_tp_leader",
        "_get_tp_group_src_rank",
        "_get_tp_or_etp_process_group",
    ),
    CheckpointingMixin: (
        "_checkpoint_context",
        "_checkpoint_load_context",
        "_load_best_model",
        "_load_from_checkpoint",
        "_load_optimizer_and_scheduler",
        "_load_weights_through_base",
        "_optimizer_store",
        "_persist_trainer_sidecars",
        "_publish_checkpoint",
        "_restore_state_beside_weights",
        "_restore_trainer_sidecars",
        "_rotate_checkpoints_after_sidecars",
        "_save_checkpoint",
        "_weights_load_context",
        "_write_model_payload",
        "create_model_card",
        "save_model",
        "train",
    ),
    GradientClippingMixin: (
        "_compute_global_grad_norm",
        "_has_mixed_mesh_grads",
        "_install_ep_clip",
        "_install_qlora_clip",
        "_install_tp_clip",
    ),
    GradientSyncMixin: (
        "_finish_fsdp2_backward",
        "_install_grad_accum_window",
        "_register_grad_sync_step_hooks",
        "_run_deferred_grad_sweep",
        "_sync_qlora_grads",
        "_sync_tp_replicated_grads",
        "_tp_per_head_norm_param_ids",
        "_tp_sharded_plain_param_ids",
    ),
    NonFiniteStepGuardMixin: (
        "_finalize_training",
        "_get_grad_norm",
        "_hand_over_verdict",
        "_init_training_state",
        "_install_nonfinite_step_skip",
        "_maybe_log_save_evaluate",
        "_raise_on_nonfinite_weights",
        "_settle_device_verdict",
        "_record_verdict",
        "get_batch_samples",
    ),
}

# Owner -> the methods it extends with a zero-arg ``super()``: the HF Trainer implementations they reach.
_SUPER_DELEGATED = {
    AcceleratorMixin: ("_inner_training_loop", "create_accelerator_and_postprocess"),
    CheckpointingMixin: (
        "_load_from_checkpoint",
        "_load_optimizer_and_scheduler",
        "_save_checkpoint",
        "create_model_card",
        "save_model",
        "train",
    ),
    EvaluationMixin: ("evaluate", "evaluation_loop"),
    NonFiniteStepGuardMixin: (
        "_finalize_training",
        "_get_grad_norm",
        "_init_training_state",
        "_maybe_log_save_evaluate",
        "get_batch_samples",
        "log",
    ),
}


class _BaseTrainer:
    """Stands in for ``transformers.Trainer``, the target of every ``super()`` above."""

    def _load_from_checkpoint(self, resume_from_checkpoint, model=None):
        return "base"

    def _load_optimizer_and_scheduler(self, checkpoint):
        return "base"

    def _save_checkpoint(self, model, trial):
        return "base"

    def save_model(self, output_dir=None, _internal_call=False):
        return "base"

    def create_model_card(self, tags=None, **kwargs):
        return "base"

    def train(self, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None):
        return "base"

    def create_accelerator_and_postprocess(self):
        return "base"

    def _inner_training_loop(self, *args, **kwargs):
        return "base"

    def evaluate(self, *args, **kwargs):
        return "base"

    def evaluation_loop(self, *args, **kwargs):
        return "base"

    def _init_training_state(self, *args, **kwargs):
        return "base"

    def _get_grad_norm(self, model, grad_norm=None):
        return "base"

    def _maybe_log_save_evaluate(self, *args, **kwargs):
        return "base"

    def get_batch_samples(self, *args, **kwargs):
        return "base"

    def log(self, logs, start_time=None):
        return "base"

    def _finalize_training(self, *args, **kwargs):
        return "base"


class _ComposedTrainer(DistributedTrainerMixin, _BaseTrainer):
    """The shape every concrete trainer has: the mixin first, a Trainer base behind it."""


def _owner(cls: type, name: str) -> type:
    return next(base for base in cls.__mro__ if name in base.__dict__)


def test_every_method_is_owned_by_its_sub_mixin_and_reachable_from_the_composed_class():
    for mixin, names in _OWNERSHIP.items():
        assert mixin in DistributedTrainerMixin.__mro__, f"{mixin.__name__} is not a base of the trainer mixin"
        for name in names:
            assert _owner(mixin, name) is mixin, f"{name} is not defined by {mixin.__name__}"
            assert _owner(DistributedTrainerMixin, name) is mixin, (
                f"DistributedTrainerMixin.{name} resolves to {_owner(DistributedTrainerMixin, name).__name__}, "
                f"not {mixin.__name__} — the bases tuple or an override shadows the sub-mixin."
            )


def test_super_calls_reach_the_base_trainer():
    """No class between a delegating sub-mixin and the Trainer base may define the delegated names."""
    mro = _ComposedTrainer.__mro__
    for mixin, names in _SUPER_DELEGATED.items():
        behind = mro[mro.index(mixin) + 1 :]
        for name in names:
            assert _owner(mixin, name) is mixin, f"{name} is not defined by {mixin.__name__}"
            owner = next(base for base in behind if name in base.__dict__)
            assert owner is _BaseTrainer, (
                f"super().{name}() from {mixin.__name__} resolves to {owner.__name__}, not the base Trainer — "
                f"a sibling mixin now intercepts it."
            )


def test_step_guard_timing_seam_is_carried_by_the_trainer_mixin():
    """The guard's ``get_batch_samples`` calls ``time_batch_fetches``, which only the token-metrics mixin owns."""
    assert _owner(DistributedTrainerMixin, "time_batch_fetches") is TokenMetricsMixin
    assert _owner(DistributedTrainerMixin, "get_batch_samples") is NonFiniteStepGuardMixin


def test_sibling_mixins_declare_no_overlapping_methods():
    """Two bases defining one method would make resolution depend on base order."""
    bases = DistributedTrainerMixin.__bases__
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
