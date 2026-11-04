"""The trainer spine every distributed trainer composes before its TRL/HF base; each concern lives in a sibling
module."""

import logging
from typing import Any, ClassVar, Literal

import torch
import torch.nn as nn

from halo.checkpoint.parallel.ep import reject_unmergeable_ep_sharded_save
from halo.configs.script_arguments import DEFAULT_MAX_CONSECUTIVE_NONFINITE_STEPS
from halo.distributed.expert_parallel.deepep.backends import free_retired_ep_buffers, free_unclaimed_ep_buffers
from halo.distributed.expert_parallel.deepep.capacity import register_forward_generation_hook
from halo.distributed.expert_parallel.moe_layer.expert_lora import (
    install_ep_aware_disable_adapter,
    model_has_expert_lora,
)
from halo.distributed.fsdp import (
    install_fsdp2_safe_disable_adapter,
    reshard_fsdp2_modules,
)
from halo.distributed.parallelism_rules import (
    reject_cross_domain_determinism,
    reject_custom_parallelism_under_accelerate,
)
from halo.distributed.rank_uniform_env import agree_rank_uniform_env
from halo.model_loading.liger.dispatch import (
    sanitize_liger_config,
    trl_reapplication_config,
    warn_if_flce_unreachable,
)
from halo.model_loading.parallel.frozen_models import (
    PREFERENCE_REFERENCE_ALTERNATIVES,
    ReferenceAlternatives,
    warn_unparallelized_reference,
)
from halo.models.attention.sdpa_flex_sliding import reject_full_determinism_after_warmup
from halo.models.config_access import (
    ROUTER_AUX_FIELDS,
    SPECIAL_TOKEN_ID_FIELDS,
    restore_config_fields,
    set_config_field_run_scoped,
    snapshot_config_fields,
)
from halo.models.moe.config_fields import resolve_active_router_aux_loss_coef
from halo.models.moe.router_logits import mark_router_logits_forced_off
from halo.models.unwrapping import find_peft_model, unwrap_framework_wrappers
from halo.optimizers.registry import (
    NAMED_OPTIMIZER_BUILDERS,
    optim_name,
)
from halo.runtime.collectives import barrier
from halo.runtime.filesystem import verify_output_filesystem_sharing
from halo.runtime.ranks import (
    get_global_rank,
    is_global_main_process,
)
from halo.trainers.accelerator import AcceleratorMixin, accelerate_manages_ddp, accelerate_manages_fsdp
from halo.trainers.checkpointing import CheckpointingMixin
from halo.trainers.ctor_slots import get_ctor_model_and_config
from halo.trainers.dataloaders import DataParallelLoaderMixin
from halo.trainers.evaluation import EvaluationMixin
from halo.trainers.optimizer_selection import build_optimizer, resolve_bf16_optimizer
from halo.trainers.parallelism.adapter_rules import (
    reject_expert_lora_peft_conflicts,
    reject_lora_under_tp,
    reject_peft_lora_in_ep_layers,
    reject_unmergeable_expert_lora_save,
    reject_unrealized_expert_lora,
    reject_unsaved_grown_vocabulary,
)
from halo.trainers.parallelism.ep_layers import EPLayersMixin, resolve_use_reentrant
from halo.trainers.parallelism.grad_clipping import GradientClippingMixin
from halo.trainers.parallelism.grad_sync import GradientSyncMixin
from halo.trainers.parallelism.mode_setup import ModeSetupMixin, reject_unsynced_trainable_params
from halo.trainers.parallelism.pipeline_rules import reject_pp_named_optimizer
from halo.trainers.parallelism.pipeline_trainer import PipelineTrainerMixin
from halo.trainers.parallelism.tp_leader import TPLeaderBroadcastMixin
from halo.trainers.step_guard import NonFiniteStepGuardMixin
from halo.trainers.token_metrics import TokenMetricsMixin
from halo.trainers.trainer_rules import (
    INERT_AUX_LOSS_DETAIL,
    reject_inert_aux_loss_balancing,
    reject_reference_sinks_mismatch,
    reject_stock_adamw_on_mixed_params,
    reject_sync_ref_model_on_sharded_policy,
    reject_trainer_router_aux_loss_coef,
    reject_unreachable_mtp_term,
    reject_unreloadable_best_model,
    reject_unsaveable_external_tables,
)
from halo.trainers.training_args_overrides import align_save_on_each_node, disable_trl_liger

logger = logging.getLogger(__name__)

# Peak-allocated device fraction: a rank this close to full after the first optimizer step OOMs on a later backward.
MEMORY_MARGIN_WARN_RATIO = 0.92


def thin_memory_margin_message(peak_bytes: int, total_bytes: int, rank: int) -> str | None:
    if total_bytes <= 0 or peak_bytes < MEMORY_MARGIN_WARN_RATIO * total_bytes:
        return None
    gib = 1024**3
    return (
        f"RANK {rank}: peak allocated {peak_bytes / gib:.1f} GiB is {peak_bytes / total_bytes:.0%} of "
        f"the device's {total_bytes / gib:.1f} GiB after the first optimizer step — "
        f"{(total_bytes - peak_bytes) / gib:.1f} GiB of margin for the rest of the run. Later batches "
        f"routinely peak higher (routing skew on MoE, longer packed rows), so expect an OOM mid-run: "
        f"lower per_device_train_batch_size (scale gradient_accumulation_steps to keep the effective "
        f"batch), or shrink max_length."
    )


def verify_no_meta_tensors(model: nn.Module) -> None:
    """Raise on parameters or buffers still on meta at placement: allocating one would train or score uninitialized
    memory that differs per rank (a zero ``inv_freq`` degenerates RoPE to NoPE at a plausible loss)."""
    meta_buffer_names = []
    meta_param_names = []
    for name, module in model.named_modules():
        for attr_name, buf in module._buffers.items():
            if buf is not None and buf.device.type == "meta":
                meta_buffer_names.append(f"{name}.{attr_name}" if name else attr_name)
        for attr_name, param in module._parameters.items():
            if param is not None and param.device.type == "meta":
                meta_param_names.append(f"{name}.{attr_name}" if name else attr_name)

    if meta_param_names:
        raise RuntimeError(
            f"{len(meta_param_names)} model PARAMETER(s) reached device placement still on the "
            f"meta device, so they hold no weights at all: {meta_param_names[:20]}. Training them "
            f"would optimize uninitialized memory that also differs across ranks. The checkpoint "
            f"load lost these tensors — check the loader's key mapping, or load with "
            f"ep_lazy_loading=False to route through from_pretrained."
        )

    if meta_buffer_names:
        raise RuntimeError(
            f"{len(meta_buffer_names)} model BUFFER(s) reached device placement still on the "
            f"meta device, so they hold no values at all: {meta_buffer_names[:20]}. Allocating "
            f"them would run on uninitialized memory that also differs across ranks. The load "
            f"path skipped finalize_loaded_model, or this buffer belongs to a family its fixes "
            f"do not recompute — extend the _ROTARY_FIXERS / _NON_PERSISTENT_FIXERS chains that "
            f"finalize_loaded_model runs."
        )


def _parallelism_log_fields(config, ep_config, has_ep_layers: bool) -> dict[str, float]:
    fields = {"parallelism/dp_size": config.data_parallel_size}
    if config.has_ep_group and config.is_tp_mode:
        fields["parallelism/ep_tp_mode"] = 1.0
    elif config.has_ep_group and config.is_cp_mode:
        fields["parallelism/ep_cp_mode"] = 1.0
    if config.is_tp_mode:
        fields["parallelism/tp_size"] = config.tp_size
    if has_ep_layers:
        fields["parallelism/ep_size"] = config.ep_size
        fields["parallelism/num_ep_groups"] = getattr(ep_config, "num_ep_groups", 1)
        if config.use_grouped_gemm and not config.has_ep_group:
            fields["parallelism/grouped_gemm_only"] = 1.0
    if config.is_expert_tp_mode:
        fields["parallelism/expert_tp_size"] = config.expert_tp_size
        fields["parallelism/ep_group_size"] = config.ep_group_size
    if config.is_cp_mode:
        fields["parallelism/cp_size"] = config.cp_size
    return fields


class DistributedTrainerMixin(
    AcceleratorMixin,
    CheckpointingMixin,
    DataParallelLoaderMixin,
    EPLayersMixin,
    EvaluationMixin,
    GradientClippingMixin,
    GradientSyncMixin,
    ModeSetupMixin,
    NonFiniteStepGuardMixin,
    PipelineTrainerMixin,
    TokenMetricsMixin,
    TPLeaderBroadcastMixin,
):
    """EP/CP/TP/PP machinery for a TRL/HF trainer: call ``_init_distributed_config()`` before ``super().__init__()``
    and ``_configure_distributed_modes()`` after it. No trainer refuses EP, ETP or TP; CP and PP run only where
    declared below. ``cp_config`` is set only under CP, ``_ep_config`` whenever the model carries EP layers."""

    # CP and PP are declare-to-enable: both fail silently on a loss nobody verified under them.
    SUPPORTS_CP: ClassVar[bool] = False
    # Why this trainer refuses PP; None admits it.
    PP_REFUSAL: ClassVar[str | None] = (
        "a pipeline stage holds only a contiguous subset of the layers, so this trainer's "
        "loss cannot be expressed as a last-stage function of logits and labels"
    )
    SUPPORTS_LOWP_PRECISION: ClassVar[bool] = False
    # Sequence rows one model call carries per dataset example; the EP dispatch gate multiplies by it.
    FORWARD_ROWS_PER_EXAMPLE: ClassVar[int] = 1
    # The loss runs outside the forward with no labels (the GRPO family): no router aux term, no Liger CE.
    POLICY_GRADIENT_LOSS: ClassVar[bool] = False
    SYNCS_TO_EXTERNAL_GENERATOR: ClassVar[bool] = False
    # What adds the router aux loss: the model's ``labels`` forward, TRL's own loss, or nothing.
    ROUTER_AUX_LOSS_ADDED_BY: ClassVar[Literal["model_forward", "trl_loss"] | None] = None
    # compute_loss returns the batch's own mean: HF must neither rescale it nor drop its /grad_accum division.
    LOSS_IS_OWN_MEAN: ClassVar[bool] = False
    # A trainer's way around holding a frozen reference, named by every reference warning and refusal.
    REFERENCE_ALTERNATIVES: ClassVar[ReferenceAlternatives] = PREFERENCE_REFERENCE_ALTERNATIVES

    @classmethod
    def reject_undeclared_axes(cls, *, cp_size: int, pp_size: int) -> None:
        if cp_size > 1 and not cls.SUPPORTS_CP:
            raise ValueError(
                f"{cls.__name__} does not support Context Parallelism (CP). Use Expert Parallelism (EP) and/or "
                "Tensor Parallelism (TP) instead, or remove context_parallel_size."
            )
        if pp_size > 1 and cls.PP_REFUSAL is not None:
            raise ValueError(
                f"{cls.__name__} does not support Pipeline Parallelism (PP): {cls.PP_REFUSAL}. Use Expert "
                "Parallelism (EP) and/or Tensor Parallelism (TP) instead, or remove pipeline_parallel_size."
            )

    def _init_distributed_config(
        self,
        kwargs: dict,
        training_args=None,
        ctor_args: tuple = (),
        ctor_positions: dict[str, int] | None = None,
        **explicit,
    ) -> dict:
        """COLLECTIVE. Pop the distributed kwargs, take over the TrainingArguments fields the toolkit owns and run the
        construction refusals; returns the remaining kwargs. A ``(*args, **kwargs)`` trainer passes its positionals as
        ``ctor_args`` with its base's ``ctor_positions``."""
        if explicit:
            kwargs = {**explicit, **kwargs}
        parallelism_config = kwargs.pop("parallelism_config", None)
        if parallelism_config is None:
            raise ValueError(
                "parallelism_config is required. Pass a ParallelismConfig instance.\n"
                "For standard DDP (no parallelism): ParallelismConfig()\n"
                "For EP: ParallelismConfig(ep_size=8)\n"
                "For TP: ParallelismConfig(tp_size=8)"
            )
        self.parallelism_config = parallelism_config
        self.save_sharded_ep = kwargs.pop("save_sharded_ep", False)
        self.save_max_shard_size = kwargs.pop("save_max_shard_size", None)
        self.allow_optimizer_warm_restart = kwargs.pop("allow_optimizer_warm_restart", False)
        self._max_consecutive_nonfinite_steps = kwargs.pop(
            "max_consecutive_nonfinite_steps", DEFAULT_MAX_CONSECUTIVE_NONFINITE_STEPS
        )
        # The PP split gate and the inert-aux-loss refusal key raise-vs-warn off this value.
        self._moe_balancing = kwargs.pop("moe_balancing", "auto")
        # Datasets pre-sharded per DP rank: re-sharding in the DataLoader drops ~(N-1)/N of each slice.
        self._dataset_presharded = kwargs.pop("dataset_presharded", False)
        self._eval_dataloaders: dict[str, Any] = {}

        model, ctor_training_args = get_ctor_model_and_config(ctor_args, kwargs, ctor_positions)
        if training_args is None:
            training_args = ctor_training_args
        model_config = getattr(model, "config", None)
        self._accelerate_manages_fsdp = accelerate_manages_fsdp(parallelism_config)
        self._accelerate_manages_ddp = accelerate_manages_ddp(parallelism_config)
        self._bf16_optimizer = resolve_bf16_optimizer(
            kwargs.pop("bf16_optimizer", None),
            parallelism_config,
            training_args,
            replicated_ddp=self._accelerate_manages_ddp,
        )
        self._snapshot_model_config_state(model_config)
        self._apply_training_args_overrides(training_args, model)

        self.cp_config = None
        self._ep_config = None
        self._device_mesh = None
        self._fsdp_wrapped = False
        self._param_id_caches = {}
        self._window_modules = []
        self._window_end_armed = True
        self._warned_empty_labels = False
        self._memory_margin_checked = False

        type(self).reject_undeclared_axes(cp_size=parallelism_config.cp_size, pp_size=parallelism_config.pp_size)
        reject_pp_named_optimizer(training_args, parallelism_config)
        reject_custom_parallelism_under_accelerate(parallelism_config)
        full_determinism = getattr(training_args, "full_determinism", False)
        # Before HF's Trainer.__init__ turns on the deterministic mode the warmed graphs were compiled without.
        reject_full_determinism_after_warmup(full_determinism)
        reject_cross_domain_determinism(parallelism_config, full_determinism)
        return self._split_pipeline_stage(kwargs, training_args)

    def _snapshot_model_config_state(self, model_config) -> None:
        """Snapshot what TRL's constructor writes: the special-token ids every save restores (HF's
        ``align_special_tokens`` also collapses a list-valued ``eos_token_id`` at train()) and the router aux-loss
        fields."""
        self._pristine_special_token_ids = (
            snapshot_config_fields(model_config, SPECIAL_TOKEN_ID_FIELDS) if model_config else []
        )
        self._router_aux_fields = snapshot_config_fields(model_config, ROUTER_AUX_FIELDS) if model_config else []

    def _apply_training_args_overrides(self, training_args, model) -> None:
        config = self.parallelism_config
        model_config = getattr(model, "config", None)
        if training_args is None:
            self._deferred_liger_kernel = False
            return
        if hasattr(training_args, "router_aux_loss_coef"):
            # TRL reads an unset coefficient off the model config and would train a hub coefficient under any
            # balancing mode; it gets the mode's coefficient only where its own loss adds the term.
            reject_trainer_router_aux_loss_coef(training_args)
            trl_adds_term = self.ROUTER_AUX_LOSS_ADDED_BY == "trl_loss"
            training_args.router_aux_loss_coef = resolve_active_router_aux_loss_coef(model) if trl_adds_term else 0.0
        # Every entry point, the Python API included; free after an entry script's own probe.
        verify_output_filesystem_sharing(training_args.output_dir)
        align_save_on_each_node(training_args)
        # The mixin owns the sharding.
        training_args.fsdp = ""
        # HF re-applies Liger at train() on the wrapped model, bypassing the load-time filtering.
        if getattr(training_args, "use_liger_kernel", False):
            safe_config = dict(getattr(training_args, "liger_kernel_config", None) or {})
            safe_config.setdefault("fused_linear_cross_entropy", False)
            training_args.liger_kernel_config = trl_reapplication_config(
                model_config,
                sanitize_liger_config(
                    safe_config,
                    needs_ep_layers=config.needs_ep_layers,
                    model_config=model_config,
                    tp_size=config.tp_size,
                    cp_size=config.cp_size,
                    pp_size=config.pp_size,
                ),
            )
        # Deferred past TRL.__init__ (which would re-apply Liger on EP-wrapped experts); restored before train().
        self._deferred_liger_kernel = config.needs_ep_layers and disable_trl_liger(training_args)
        # Force use_reentrant before super().__init__ enables GC. Not under PP, which requires non-reentrant.
        if getattr(training_args, "gradient_checkpointing", False):
            gc_kwargs = getattr(training_args, "gradient_checkpointing_kwargs", None) or {}
            configured = gc_kwargs.get("use_reentrant")
            use_reentrant = resolve_use_reentrant(config, model_config, configured)
            if use_reentrant != configured:
                training_args.gradient_checkpointing_kwargs = {**gc_kwargs, "use_reentrant": use_reentrant}

    def _configure_distributed_modes(self) -> None:
        """COLLECTIVE. Wrap the model for its parallel mode and run the post-wrap gates; call after
        ``super().__init__()``."""
        config = self.parallelism_config
        self._restore_trl_overrides()
        model_config = getattr(self.model, "config", None)
        if self.POLICY_GRADIENT_LOSS and model_config is not None:
            warn_if_flce_unreachable(model_config, type(self).__name__)
        if is_global_main_process():
            optim = optim_name(self.args.optim)
            if self._bf16_optimizer:
                logger.info("  Optimizer: AdamWBF16 (stochastic rounding, 6B/param)")
            elif optim in NAMED_OPTIMIZER_BUILDERS:
                logger.info(f"  Optimizer: {NAMED_OPTIMIZER_BUILDERS[optim][1]}")
        # Latched, so an entry script's own pre-load call makes this free; a trainer built without the script
        # scaffold needs it for the bucketed grad sweeps (TP replicated, QLoRA) over groups spanning OS nodes.
        agree_rank_uniform_env()

        # Before any wrapping: adapters sit outside the TP graph and would train rank-inconsistently.
        if config.is_tp_mode:
            reject_lora_under_tp(self._top_level_model())
        self._disable_inert_router_logits()
        reject_unreachable_mtp_term(
            self._top_level_model(),
            trainer=type(self).__name__,
            labelled_forward=self.ROUTER_AUX_LOSS_ADDED_BY == "model_forward",
            is_cp_mode=config.is_cp_mode,
            pops_labels=self.label_smoother is not None or self.compute_loss_func is not None,
            rows_per_forward=self.args.per_device_train_batch_size,
        )
        reject_unreloadable_best_model(self.args, config, self.model)
        reject_unsaveable_external_tables(
            self._top_level_model(), accelerate_manages_fsdp=self._accelerate_manages_fsdp
        )
        reject_unsaved_grown_vocabulary(self._top_level_model(), accelerate_manages_fsdp=self._accelerate_manages_fsdp)
        # FSDP2 needs one dtype per group; before the fp32_non_ep upcast so that mode wins.
        self._cast_peft_params_to_compute_dtype()
        # FP32 master weights before FSDP2 wrapping so it sees fp32 params and produces fp32 grads.
        if config.fp32_non_ep_params:
            self._upcast_non_ep_params_to_fp32()
        # Captured once, ahead of the parallel-mode setup: the FSDP2 wrap, the deferred grad sweep, the
        # global grad norm and the ETP generation broadcast all read the process groups it holds.
        ep_layers = self._ep_layers()
        self._ep_config = ep_layers[0].ep_config if ep_layers else None

        self._wrap_for_parallel_mode()

        if config.needs_ep_layers:
            reject_peft_lora_in_ep_layers(self._top_level_model())
            self._install_ep_layer_hooks()
        # Every run and never gated on the model: under PP a stage may hold no EP layer and the check is
        # collective, and a model can carry EP layers its config does not declare.
        reject_unsynced_trainable_params(self.model, self._fsdp_excluded_params(), self._ep_layers(), self._ep_config)
        reject_unrealized_expert_lora(self._top_level_model(), config.expert_lora)
        # After the wrap and after TRL's dropout disabling, so they read what the run will actually use.
        reject_expert_lora_peft_conflicts(self._top_level_model(), config.expert_lora)
        self._warn_on_zeroed_lora_dropout()
        self._configure_wrapped_model()

        # Setup varies per rank (TP materialization, EP patching), so ranks re-align before the first step.
        barrier()

    def _restore_trl_overrides(self) -> None:
        """Take back what TRL's constructor changed: the router aux-loss fields, the deferred Liger flag, HF's inferred
        loss-kwargs scaling and the TP width TRL's token counter divides by."""
        if self.LOSS_IS_OWN_MEAN:
            self.model_accepts_loss_kwargs = False
            # A compute_loss_func, which these losses never call, would otherwise also drop the /grad_accum division.
            self.loss_is_scaled_for_ga = False
        # TRL reads the ranks sharing a batch off accelerate's parallelism config, which the toolkit's accelerator
        # never carries, so its world-gathered attended-token counter would count a TP/ETP/CP batch once per rank.
        if hasattr(self, "_tp_size"):
            self._tp_size = self.parallelism_config.non_dp_replication_factor
        # TRL's entropy guard needs the flag to skip logits access when FusedLinearCE yields logits=None.
        if self._deferred_liger_kernel:
            self.args.use_liger_kernel = True
        restore_config_fields(getattr(self, "_router_aux_fields", []))
        if not getattr(self, "_pristine_special_token_ids", None):
            self._snapshot_model_config_state(getattr(getattr(self, "model", None), "config", None))

    def _disable_inert_router_logits(self) -> None:
        """Refuse, or turn off, MoE balancing that rides an aux loss this trainer never adds. Under ``auto`` the router
        logits go back off; set outside the balancing strategy, they stay on with a warning."""
        if self.ROUTER_AUX_LOSS_ADDED_BY is not None:
            return
        model = self._top_level_model()
        coef = resolve_active_router_aux_loss_coef(model)
        if coef <= 0:
            return
        trainer = type(self).__name__
        reject_inert_aux_loss_balancing(trainer, self._moe_balancing)
        if self._moe_balancing == "auto":
            # ``auto`` set the flag before this trainer was built; clear it rather than pay for the tensor.
            # The stamp is what makes it stick: MoEMetricsCallback re-enables the flag at train begin.
            set_config_field_run_scoped(model.config, "output_router_logits", False)
            mark_router_logits_forced_off(model.config)
            action = (
                "moe_balancing='auto' resolved to aux_loss before this trainer was built, so "
                "output_router_logits is being turned back OFF for this run (moe/* metrics go with it)."
            )
        else:
            action = (
                f"output_router_logits was set outside the balancing strategy "
                f"(moe_balancing='{self._moe_balancing}'), so it is left ON and keeps costing that "
                f"tensor every forward."
            )
        if is_global_main_process():
            logger.warning(
                f"output_router_logits is ON with router_aux_loss_coef={coef}, but that aux loss is "
                f"INERT here. {INERT_AUX_LOSS_DETAIL.format(trainer=trainer)} {action} Set "
                f"moe_balancing='bias_update' for real balancing on families whose EP layers support it."
            )

    def _warn_on_zeroed_lora_dropout(self) -> None:
        """Warn when ``lora_dropout`` was zeroed: TRL's ``disable_dropout`` runs after the adapter wrap and zeroes
        PEFT's dropout too."""
        peft_model = find_peft_model(self.model)
        spec = self.parallelism_config.expert_lora
        configured = max(
            [float(getattr(cfg, "lora_dropout", 0) or 0) for cfg in getattr(peft_model, "peft_config", {}).values()]
            + [spec.dropout if spec is not None else 0.0]
        )
        if configured <= 0:
            return
        # One substring covers PEFT's ``lora_dropout.<adapter>`` and the EP layers' ``expert_lora_dropout``.
        live = [
            module.p
            for name, module in self._top_level_model().named_modules()
            if isinstance(module, nn.Dropout) and "lora_dropout" in name
        ]
        if live and not any(p > 0 for p in live) and is_global_main_process():
            logger.warning(
                f"lora_dropout={configured} is configured but every LoRA dropout in the live model is 0: "
                f"this trainer disables dropout after the adapter wrap (disable_dropout=True), which "
                f"zeroes PEFT's lora_dropout along with the model's own. Set disable_dropout=False to keep "
                f"it, or set lora_dropout: 0.0 to stop expecting regularization that is not applied."
            )

    def _install_ep_layer_hooks(self) -> None:
        if self._ep_layers():
            # EP patching registers the generation hook on the model it patched, which a PEFT wrapper reaches
            # through .forward() without a pre-hook, freezing the generation at the first forward's capacity.
            register_forward_generation_hook(self.model)
        # modules_to_save swaps in a router copy the ctor-time DP-sync hook never saw.
        if find_peft_model(self.model) is not None:
            attached = sum(module.reattach_dp_average_grad_hooks() for module in self._ep_layers())
            if attached and is_global_main_process():
                logger.info(f"  Re-attached EP router DP-sync hook to {attached} trainable modules_to_save params")

    def _configure_wrapped_model(self) -> None:
        """Everything that must follow the wrap, in order."""
        config = self.parallelism_config
        # FSDP2 replaces managed Parameter objects, so an id memoized pre-wrap names a dead object.
        self._param_id_caches.clear()
        self._install_grad_accum_window()
        self._install_ep_gradient_checkpointing()
        # The reference passes run under peft's disable_adapter(): it must drop the native expert adapters too,
        # so they see a frozen base, and restore trainability on the sharded params.
        peft_model = find_peft_model(self.model)
        if peft_model is not None:
            if model_has_expert_lora(self.model):
                install_ep_aware_disable_adapter(peft_model)
            install_fsdp2_safe_disable_adapter(peft_model, self.model)
        reject_unmergeable_expert_lora_save(
            self.model,
            merge_expert_lora_on_save=config.merge_expert_lora_on_save,
            accelerate_manages_fsdp=self._accelerate_manages_fsdp,
        )
        # Before any evaluate(): a DP-sharded loader needs a DP-scoped metric gather.
        self._install_dp_metric_gather()
        if self.save_sharded_ep:
            reject_unmergeable_ep_sharded_save(self.model, merge_expert_lora_on_save=config.merge_expert_lora_on_save)
        reject_sync_ref_model_on_sharded_policy(
            self.args, sharded=self._fsdp_wrapped or config.has_ep_group or config.is_tp_mode
        )
        # FSDP2's dynamic subclass is absent from _CAN_RECORD_REGISTRY, so router logits go uncaptured.
        if self._fsdp_wrapped:
            self._register_output_capturing_for_fsdp()
        self._apply_torch_compile()

    def _move_model_to_device(self, model, device):
        verify_no_meta_tensors(model)
        model.to(device)

    def _backbone_model(self) -> nn.Module:
        """``model.model``, without ``lm_head``; work covering every trainable parameter takes
        :meth:`_top_level_model`."""
        model = self._top_level_model()
        if hasattr(model, "model"):
            return model.model
        return model

    def _top_level_model(self) -> nn.Module:
        """The full model under accelerate/DDP and ``torch.compile``; its parameters include an untied ``lm_head``."""
        return unwrap_framework_wrappers(self.model)

    def get_total_train_batch_size(self, args) -> int:
        """Over the toolkit's DP width: the base Trainer's divisors read state the toolkit never sets and have no PP or
        ETP term."""
        return self._train_batch_size * args.gradient_accumulation_steps * self.parallelism_config.data_parallel_size

    def _validate_reference_model(self, ref_model) -> None:
        if ref_model is not None:
            warn_unparallelized_reference(self.parallelism_config, self.REFERENCE_ALTERNATIVES)

    def _validate_held_reference_model(self) -> None:
        """A sinks mismatch biases the KL and raises; an unsharded replica wastes memory and warns. TRL nulls
        ``ref_model`` where it holds none."""
        ref_model = getattr(self, "ref_model", None)
        if ref_model is None:
            return
        reject_reference_sinks_mismatch(
            getattr(self, "model", None), ref_model, self.REFERENCE_ALTERNATIVES.under(self.parallelism_config)
        )
        warn_unparallelized_reference(self.parallelism_config, self.REFERENCE_ALTERNATIVES)

    def log(self, logs: dict[str, float], start_time: float | None = None):
        """COLLECTIVE. After the first step, add the parallelism fields and the world-summed token and FLOPs counts
        (:meth:`_drain_token_counts`). TRL's attended-token counter logs as ``num_attended_tokens_seen``, beside
        ``num_input_tokens_seen`` (padding included) and ``num_unmasked_output_tokens_seen`` (loss tokens)."""
        if hasattr(self, "_tp_size"):
            for mode_metrics in self._metrics.values():
                if "num_tokens" in mode_metrics:
                    mode_metrics["num_attended_tokens_seen"] = mode_metrics.pop("num_tokens")
        if hasattr(self, "state") and self.state.global_step != 0:
            logs.update(_parallelism_log_fields(self.parallelism_config, self._ep_config, bool(self._ep_layers())))
            self._drain_token_counts()
            total_output_tokens = getattr(self, "_cumulative_unmasked_output_tokens", None)
            if total_output_tokens is not None:
                logs["num_unmasked_output_tokens_seen"] = total_output_tokens
        super().log(logs, start_time)

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Arm FSDP2's per-window toggles and complete its post-backward after the inherited step. COLLECTIVE once a
        step's last backward is done, which frees the DeepEP buffers its grows retired."""
        self._accumulate_unmasked_output_tokens(self._extract_output_token_count(inputs))
        self._accumulate_attention_flops(inputs)
        if self._pp_runner is not None:
            # The schedule drives forward and backward, so the inherited step must not also run them.
            loss = self._pp_training_step(inputs)
            free_retired_ep_buffers()
            return loss
        self._warn_once_on_thin_memory_margin()
        # HF sets sync_gradients before every training_step, so this is the window's last microbatch.
        self._set_window_end(self.accelerator.sync_gradients)
        try:
            loss = super().training_step(model, inputs, num_items_in_batch)
        except torch.OutOfMemoryError as e:
            # The peers die on launch failures in whatever kernel is on-stream (DeepEP combine, cuBLAS; Xid 43),
            # which read like a compute bug, so the rank that ran out of memory says so, whichever rank it is.
            logger.error(
                f"RANK {get_global_rank()}: torch.OutOfMemoryError in training_step — THIS RANK IS THE PRIMARY "
                f"FAILURE. Peer ranks will report DeepEP/NCCL/cuBLAS launch failures (Xid 43) as collateral of this "
                f"context tearing down; diagnose this OOM, not those. {e}"
            )
            raise
        self._finish_fsdp2_backward(model)
        if self.accelerator.sync_gradients:
            free_retired_ep_buffers()
        return loss

    def _warn_once_on_thin_memory_margin(self):
        """After the first optimizer step, where weights, grads, optimizer state and a backward's activations first
        coexist, warn per rank if the peak is near device memory; MoE routing skew loads the hot ranks."""
        if self._memory_margin_checked or self.state.global_step < 1:
            return
        self._memory_margin_checked = True
        if not torch.cuda.is_available():
            return
        message = thin_memory_margin_message(
            torch.cuda.max_memory_allocated(),
            torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory,
            get_global_rank(),
        )
        if message is not None:
            logger.warning(message)

    def prediction_step(self, model, inputs, prediction_loss_only=True, ignore_keys=None):
        """Under PP the eval step must run through the schedule (no rank holds the whole model)."""
        if self._pp_runner is not None:
            return self._pp_prediction_step(inputs, prediction_loss_only)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

    def create_optimizer(self, model=None):  # base Trainer passes it; the mixin builds from self
        if self.optimizer is None:
            # An evaluate() before train() leaves FSDP2 modules gathered (reshard_after_forward=False), and param
            # groups built over their transient unsharded params update orphans the next unshard discards.
            reshard_fsdp2_modules(self.model)
            self.optimizer = build_optimizer(
                self.model,
                self.args,
                self.get_decay_parameter_names(self.model),
                bf16_optimizer=self._bf16_optimizer,
                config=self.parallelism_config,
                has_ep_layers=bool(self._ep_layers()),
                get_optimizer_cls_and_kwargs=self.get_optimizer_cls_and_kwargs,
            )
            if self.optimizer is None:
                reject_stock_adamw_on_mixed_params(
                    optim_name(self.args.optim), self.parallelism_config, getattr(self.model, "config", None)
                )
                super().create_optimizer()

        self._register_grad_sync_step_hooks()
        self._install_nonfinite_step_skip()
        return self.optimizer

    def cleanup_ep(self):
        """COLLECTIVE. Destroy the DeepEP buffers; call when done with the trainer, before GC tears down the CUDA
        context."""
        # Align first: destroy is a device-side barrier (~100 s spin), so the wait belongs under the PG timeout.
        barrier()
        count = 0
        # Through the EP-layer registry rather than a hasattr check on ``dispatcher``, which matches too widely.
        for module in self._ep_layers():
            try:
                module.dispatcher.destroy()
                count += 1
            except Exception as e:
                if is_global_main_process():
                    logger.warning(f"Failed to destroy DeepEP buffer for {type(module).__name__}: {e}")
        # Layers already collected freed nothing (a finalizer must not run a collective); swept here instead.
        freed = free_unclaimed_ep_buffers()
        if (count > 0 or freed > 0) and is_global_main_process():
            logger.info(f"✓ Released {count} DeepEP layers, freed {freed} buffers")
        barrier()
