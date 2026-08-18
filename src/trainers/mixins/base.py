"""Distributed Trainer Mixin — the EP/CP/TP lifecycle spine of every distributed trainer.

Subclass alongside a Trainer; the mixin owns accelerator creation and FSDP setup, so subclasses
must NOT override create_accelerator_and_postprocess(). The sibling sub-mixins (gradient sync and
clipping, checkpointing, dataloaders, EP introspection, validation, pipeline) compose into the
class below.
"""

import logging
import os
import time
from abc import ABC
from collections.abc import Iterable
from dataclasses import fields
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DataLoaderConfiguration
from transformers.utils.output_capturing import _CAN_RECORD_REGISTRY
from trl.trainer.utils import disable_dropout_in_model

from src.data.collators.packing import DataCollatorWithFlattening, DataCollatorWithPacking
from src.data.pipeline.processing import resolve_map_num_proc
from src.data.spans import LABEL_IGNORE_INDEX
from src.distributed.checkpoint.peft import find_peft_model
from src.distributed.context_parallel.validation import validate_trainer_args_for_cp
from src.distributed.context_parallel.wrapper import UlyssesCPModelWrapper
from src.distributed.expert_parallel.base_layer import make_disable_adapter_ep_aware
from src.distributed.expert_parallel.dispatcher import (
    free_unclaimed_ep_buffers,
    register_forward_generation_hook,
    verify_rank_uniform_env,
)
from src.distributed.expert_parallel.expert_weights import has_ep_lora
from src.distributed.expert_parallel.saving import validate_ep_sharded_save
from src.distributed.fsdp import (
    reshard_fsdp2_modules,
    reshard_label,
    setup_fsdp2_for_dp,
    setup_fsdp2_for_tp,
)
from src.distributed.mesh import ParallelDims
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.pipeline_parallel.losses import PPLossAdapter, causal_lm_token_loss
from src.distributed.runtime import (
    barrier,
    get_global_rank,
    get_global_world_size,
    is_global_main_process,
    is_output_shared_filesystem,
    nccl_safe_broadcast,
    rank_consensus,
)
from src.env import is_accelerate_fsdp_launch, is_accelerate_launch
from src.kernels.liger.orchestrator import (
    apply_liger_parallelism_overrides,
    liger_ep_disables_fused_glu,
    liger_parallelism_overrides,
    warn_if_flce_unreachable,
)
from src.models.loading.config_levels import config_sources, snapshot_special_token_ids
from src.models.moe_balancing import ep_wraps_experts
from src.models.structure import model_has_quantized_params, unwrap_framework_wrappers
from src.optimizers.adamw_bf16 import build_bf16_optimizer
from src.optimizers.param_groups import build_tensor_type_grouped_optimizer
from src.optimizers.registry import (
    NAMED_OPTIMIZER_BUILDERS,
    STOCK_ADAMW_OPTIMIZERS,
    optim_name,
    register_custom_optimizers,
)
from src.trainers.mixins.checkpointing import CheckpointingMixin
from src.trainers.mixins.dataloader import DataParallelDataLoaderMixin
from src.trainers.mixins.ep_introspection import EpIntrospectionMixin
from src.trainers.mixins.grad_sync import GradientSyncMixin
from src.trainers.mixins.pipeline import PipelineTrainerMixin
from src.trainers.mixins.token_metrics import TokenMetricsMixin
from src.trainers.mixins.validation import ParallelismValidationMixin, disable_trl_liger

register_custom_optimizers()

logger = get_logger(__name__, log_level="info")

# FSDP2 shaping knobs the mixin's own wrap implements; NAMES only — each knob's "not requested"
# value is its own ParallelismConfig dataclass default, read at call time so a flipped default
# cannot leave a stale copy here gating on the wrong value.
_FSDP_SHAPING_KNOBS = ("use_hsdp", "fsdp_reshard_after_forward", "fsdp_reshard_after_backward")

# ParallelismConfig knobs only the mixin-managed (torchrun) FSDP2 wrap implements.
_ACCELERATE_UNSUPPORTED_KNOBS = (*_FSDP_SHAPING_KNOBS, "fp32_grad_reduce")

# Peak-allocated fraction of device memory above which the post-first-step margin warning fires.
# A rank this close to full after the first optimizer step OOMs on a later backward.
_MEMORY_MARGIN_WARN_RATIO = 0.92


def thin_memory_margin_message(peak_bytes: int, total_bytes: int, rank: int) -> str | None:
    """Warning text when ``peak_bytes`` leaves less than ``1 - _MEMORY_MARGIN_WARN_RATIO`` of the
    device free, else ``None``. Pure so the threshold contract is CPU-testable."""
    if total_bytes <= 0 or peak_bytes < _MEMORY_MARGIN_WARN_RATIO * total_bytes:
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


def oom_banner(rank: int, exc: BaseException) -> str:
    """One unmissable line for a rank-local OOM inside a distributed step.

    In EP/FSDP runs the OOMing rank abandons collectives its peers are waiting in, and the peers
    then die on launch failures inside whatever kernel is on-stream (DeepEP combine, cuBLAS) with
    Xid 43 — noise that reads like a compute bug and buries the real cause. This banner marks the
    primary failure so post-mortems grep it first.
    """
    return (
        f"RANK {rank}: torch.OutOfMemoryError in training_step — THIS RANK IS THE PRIMARY FAILURE. "
        f"Peer ranks will report DeepEP/NCCL/cuBLAS launch failures (Xid 43) as collateral of this "
        f"context tearing down; diagnose this OOM, not those. {exc}"
    )


def emit_primary_failure(rank: int, exc: BaseException) -> None:
    """Log the OOM banner from the rank that OOMed, whichever rank that is.

    ``main_process_only=False`` is the whole point: this module logs through accelerate's
    ``MultiProcessAdapter``, which DROPS a record on every non-main rank by default. The banner
    names the OOMing rank, so the default would print it only when rank 0 is the one that died —
    never in the case it was written for, leaving a post-mortem with peer watchdog timeouts and no
    primary cause.
    """
    logger.error(oom_banner(rank, exc), main_process_only=False)


def _knobs_set_by_user(config, names: Iterable[str]) -> list[str]:
    """``names`` whose value on ``config`` differs from :class:`ParallelismConfig`'s own default.

    The defaults are read off the dataclass that declares them rather than restated here, so a
    flipped default moves both gates with it instead of leaving them comparing against a stale value.
    """
    defaults = {f.name: f.default for f in fields(ParallelismConfig)}
    return [name for name in names if getattr(config, name) != defaults[name]]


class DistributedTrainerMixin(
    CheckpointingMixin,
    DataParallelDataLoaderMixin,
    EpIntrospectionMixin,
    GradientSyncMixin,
    ParallelismValidationMixin,
    PipelineTrainerMixin,
    TokenMetricsMixin,
    ABC,
):  # base mixin: not directly instantiable, but has no abstract methods
    """Distributed training infrastructure (EP/CP/TP) for a Trainer subclass.

    Lifecycle: subclasses call ``_init_distributed_config()`` to extract parallelism
    kwargs BEFORE ``super().__init__()``, then ``_setup_distributed_modes()`` after it,
    and override the ``_supports_*`` flags to declare supported modes.

    ``parallelism_config`` is the single source of truth; ``cp_config`` / ``_ep_config``
    are set only when CP / EP are enabled.
    """

    _supports_ep: bool = True
    # CP/PP default OFF: both fail SILENTLY — CP mis-pools log-probs, a PP stage holds only some layers.
    _supports_cp: bool = False
    _supports_tp: bool = True
    # A subclass restates a flag only where it CHANGES the default — except ``_supports_pp = False``,
    # which every PP-refusing trainer keeps as the anchor of its paired ``_pp_unsupported_reason``
    # (the reason is what the gate quotes, and the two must be read together).
    _supports_pp: bool = False
    _pp_unsupported_reason: str = ""
    # True only where ``labels`` go through ``*ForCausalLM.forward`` and the aux term lands in ``outputs.loss``.
    _consumes_router_aux_loss: bool = False
    # True where the loss runs outside the model's forward with no labels passed (the GRPO family):
    # Liger's CE/FLCE can never fire there, so an applied FLCE draws a warning at construction.
    _loss_outside_model_forward: bool = False
    # True where compute_loss already returns the batch's OWN mean. HF then must not rescale it by
    # num_items_in_batch, and must keep its /gradient_accumulation_steps — see _setup_distributed_modes.
    _loss_is_own_mean: bool = False

    def _pp_loss_adapter(self) -> PPLossAdapter:
        """This trainer's pipeline-loss contract; overridden by trainers that enable PP.

        The default is the causal-LM contract (summed shifted CE, token-count normalizer) — what
        SFT needs. A trainer flipping ``_supports_pp`` with a different objective overrides this
        to declare its loss, pair layout, batch transform, normalizer and per-example side tensors
        (see :class:`~src.distributed.pipeline_parallel.losses.PPLossAdapter`).
        """
        return PPLossAdapter(token_loss_fn=causal_lm_token_loss)

    def _init_distributed_config(self, kwargs: dict, training_args=None, ctor_args: tuple = (), **explicit) -> dict:
        """Extract ParallelismConfig from kwargs and set up distributed state.

        Call BEFORE super().__init__() to extract/validate parallelism args (modifies kwargs
        in-place, removing parallelism args). training_args defaults to kwargs["args"]; trainers
        with an explicit `args` param (Classification, SMPO) should pass it. Trainers whose
        signatures name the distributed params explicitly pass them via ``**explicit``
        (``parallelism_config=..., save_sharded_ep=...``) instead of packing a dict themselves;
        kwargs-style values win over explicit ones. ``ctor_args`` are the trainer's own ctor
        positionals, forwarded to the ``_validate_pp_mode`` hook.
        """
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

        # Never defaulted silently: the PP split gate and _validate_router_aux_loss_consumable key raise-vs-warn off it.
        self._moe_balancing = kwargs.pop("moe_balancing", "auto")

        # Datasets pre-sharded per DP rank: re-sharding in the DataLoader drops ~(N-1)/N of each slice.
        self._dataset_presharded = kwargs.pop("dataset_presharded", False)

        # Persistent-worker eval-loader cache, holding the PREPARED loader as the base Trainer does.
        # Declared here so every trainer's eval path caches the same kind under the same key.
        self._eval_dataloaders: dict[str, Any] = {}

        if training_args is None:
            training_args = kwargs.get("args")
        self._configure_mixed_precision(kwargs, training_args)

        # Non-shared FS: without a per-node write, nodes 1..N resume at global_step=0 → step desync → hang.
        if (
            training_args is not None
            and not is_output_shared_filesystem()
            and not getattr(training_args, "save_on_each_node", False)
        ):
            training_args.save_on_each_node = True
            if is_global_main_process():
                logger.info(
                    "Non-shared output filesystem → forcing save_on_each_node=True so "
                    "trainer_state.json / scheduler.pt / rng_state are written on every "
                    "node (otherwise non-zero nodes resume at global_step=0 → step desync)."
                )

        # HF re-applies Liger at train() on the wrapped model, bypassing the load-time filtering.
        if training_args is not None and getattr(training_args, "use_liger_kernel", False):
            safe_config = dict(getattr(training_args, "liger_kernel_config", None) or {})
            if "fused_linear_cross_entropy" not in safe_config:
                safe_config["fused_linear_cross_entropy"] = False
            forced_off = liger_parallelism_overrides(
                has_ep_wrapped_experts=liger_ep_disables_fused_glu(
                    parallelism_config.needs_ep_wrappers, getattr(kwargs.get("model"), "config", None)
                ),
                tp_size=parallelism_config.tp_size,
                cp_size=parallelism_config.cp_size,
                pp_size=parallelism_config.pp_size,
            )
            training_args.liger_kernel_config = apply_liger_parallelism_overrides(safe_config, forced_off)

        # Deferred past TRL.__init__ (which would re-apply Liger on EP-wrapped experts); restored before train().
        self._deferred_liger_kernel = parallelism_config.needs_ep_wrappers and disable_trl_liger(training_args)

        self.cp_config = None
        self._ep_config = None
        self._device_mesh = None
        self._fsdp_wrapped = False
        self._backward_reshard_modules = []
        self._backward_reshard_armed = True
        self._warned_empty_labels = False
        self._memory_margin_checked = False

        self._accelerate_manages_fsdp = self._should_accelerate_manage_fsdp()
        self._accelerate_manages_ddp = self._should_accelerate_manage_ddp()

        self._validate_parallelism_modes()
        kwargs = self._maybe_prepare_pipeline_model(kwargs, training_args, ctor_args)

        # Force use_reentrant before super().__init__ enables GC. Not under PP, which requires non-reentrant.
        if (
            (parallelism_config.is_ep_mode or parallelism_config.is_cp_mode)
            and not parallelism_config.is_pp_mode
            and training_args is not None
            and getattr(training_args, "gradient_checkpointing", False)
        ):
            gc_kwargs = dict(getattr(training_args, "gradient_checkpointing_kwargs", None) or {})
            if gc_kwargs.get("use_reentrant") is False:
                logger.warning(
                    "Expert/Context Parallelism uses use_reentrant=True for gradient checkpointing; "
                    "overriding the configured use_reentrant=False. CP's sequence all-to-alls do "
                    "not survive non-reentrant recompute; non-reentrant is validated only under "
                    "pipeline parallelism, which requires it."
                )
            gc_kwargs["use_reentrant"] = True
            training_args.gradient_checkpointing_kwargs = gc_kwargs

        return kwargs

    def _configure_mixed_precision(self, kwargs: dict, training_args: Any):
        """Surface fp32_grad_reduce onto training_args and resolve whether to use AdamWBF16
        (``self._bf16_optimizer``). Whether the selected stock optimizer can be BUILT over a
        mixed plain/DTensor parameter set is create_optimizer's decision, not this one's."""
        # Ctor kwarg wins (tests construct trainers directly); else the YAML/CLI value on the config.
        bf16_optimizer_explicit = kwargs.pop("bf16_optimizer", None)
        if bf16_optimizer_explicit is None:
            bf16_optimizer_explicit = self.parallelism_config.bf16_optimizer

        # create_mixed_precision_policy_v2 sees only args, so surface it there for every FSDP path.
        if training_args is not None:
            training_args.fp32_grad_reduce = self.parallelism_config.fp32_grad_reduce

        # Auto-enable only on the default optimizer — an explicit choice (adamw_bnb_8bit) must stand.
        # load_best_model_at_end (needs post-PEFT state) and stock-optimizer buildability are decided later.
        if bf16_optimizer_explicit is not None:
            requested_optim = optim_name(training_args.optim if training_args is not None else None)
            if bf16_optimizer_explicit and requested_optim in NAMED_OPTIMIZER_BUILDERS:
                raise ValueError(
                    f"bf16_optimizer=True and optim={requested_optim!r} both select an optimizer, and "
                    f"the bf16 path wins — {requested_optim} would be silently discarded with nothing "
                    f"in the log to say so. Pick one: drop bf16_optimizer to run {requested_optim}, or "
                    f"drop optim to run AdamWBF16."
                )
            self._bf16_optimizer = bf16_optimizer_explicit
        elif training_args is not None and getattr(training_args, "bf16", False):
            optim = optim_name(training_args.optim)
            use_bf16_opt = optim in STOCK_ADAMW_OPTIMIZERS
            # Replicated DDP is outside the validated SR matrix (FSDP/EP/TP/HSDP) — conservative only.
            if use_bf16_opt and self._should_accelerate_manage_ddp():
                logger.warning(
                    "bf16=True with replicated DDP (accelerate MULTI_GPU): "
                    "skipping AdamWBF16 stochastic-rounding auto-enable to avoid "
                    "cross-replica weight drift. Use FSDP/EP/TP for SR, or set "
                    "bf16_optimizer=True explicitly to override."
                )
                use_bf16_opt = False
            self._bf16_optimizer = use_bf16_opt
        else:
            self._bf16_optimizer = False

    def _no_custom_parallelism(self) -> bool:
        """Whether no custom parallelism (EP/CP/TP/PP) is configured.

        use_grouped_gemm is excluded — it only activates for MoE models during loading.
        """
        return (
            self.parallelism_config.ep_group_size <= 1
            and self.parallelism_config.cp_size == 1
            and self.parallelism_config.tp_size == 1
            and self.parallelism_config.pp_size == 1
        )

    def _should_accelerate_manage_fsdp(self) -> bool:
        """Check if accelerate should manage FSDP (launched with accelerate launch + FSDP config)."""
        has_accelerate_fsdp = is_accelerate_fsdp_launch()
        return has_accelerate_fsdp and self._no_custom_parallelism()

    def _should_accelerate_manage_ddp(self) -> bool:
        """Whether accelerate manages DDP (MULTI_GPU, no custom parallelism): accelerate launcher
        detected without FSDP enabled."""
        return is_accelerate_launch() and not is_accelerate_fsdp_launch() and self._no_custom_parallelism()

    def _enable_input_require_grads(self, model: nn.Module) -> None:
        """Make embedding outputs require grad so gradient checkpointing keeps a grad path when the
        embeddings are frozen (PEFT)."""
        model.enable_input_require_grads()

    def _disable_dropout_for_onpolicy(self):
        """Force dropout OFF for on-policy RL. MUST be called AFTER ``_setup_distributed_modes`` so
        it also reaches the EP grouped expert-LoRA dropout realized there.

        The recomputed log-probs must match vLLM's dropout-free sampling: active dropout lowers every
        recomputed log-prob (Jensen), so the importance ratio drifts below 1 as the adapter grows,
        corrupting the IS correction, k3 KL estimator, and (num_iterations>1) PPO ratio. Idempotent.
        """
        if not self.args.disable_dropout:
            logger.info("On-policy RL: disabling dropout for importance-sampling / vLLM consistency.")
            self.args.disable_dropout = True
        disable_dropout_in_model(self.model)
        if getattr(self, "ref_model", None) is not None:
            disable_dropout_in_model(self.ref_model)

        # A config-float dropout inside attention/MLP survives disable_dropout_in_model and still perturbs log-probs.
        config = getattr(self.model, "config", None)
        if config is None:
            return
        # Through the shared seam, not a raw ``.text_config`` read: a composite config may nest the
        # text model under another name, and only ``get_text_config()`` resolves all of them.
        for cfg in config_sources(config):
            for field in ("attention_dropout", "attn_pdrop", "hidden_dropout", "resid_pdrop", "dropout"):
                val = getattr(cfg, field, 0) or 0
                if isinstance(val, (int, float)) and val > 0:
                    logger.warning(
                        "On-policy RL: model config %s=%s is a non-module dropout that cannot be zeroed by "
                        "disable_dropout_in_model — it perturbs recomputed log-probs vs vLLM's dropout-free "
                        "sampling (biases the importance ratio). Set it to 0 in the model config.",
                        field,
                        val,
                    )

    def _setup_distributed_modes(self):
        """Setup the enabled parallelism modes and data parallel gradient sync.

        Call AFTER super().__init__() to setup EP, CP, TP, or standard FSDP DP. All torchrun modes
        use FSDP v2 (fully_shard); EP/CP add ignored_params for EP modules, TP uses a 2D mesh,
        EP+TP adds a DP process group.
        """
        # Declared-only, so a trainer that has not adopted the attribute keeps whatever HF inferred.
        if self._loss_is_own_mean:
            self.model_accepts_loss_kwargs = False

        # TRL's entropy guard needs the flag to skip logits access when FusedLinearCE yields logits=None.
        if self._deferred_liger_kernel:
            self.args.use_liger_kernel = True

        # HF's align_special_tokens collapses a list-valued config.eos_token_id; every save restores this snapshot.
        model_config = getattr(getattr(self, "model", None), "config", None)
        self._pristine_special_token_ids = snapshot_special_token_ids(model_config) if model_config else []

        if self._loss_outside_model_forward and model_config is not None:
            warn_if_flce_unreachable(model_config, type(self).__name__)

        self._log_parallelism_config()

        # Latched, so the entry scripts' own pre-load call makes this free. It is here because the
        # bucketed grad sweeps this mixin owns (the TP replicated sweep, the QLoRA sweep) read
        # HALO_GRAD_BUCKET_MB over groups that span OS nodes, and neither goes anywhere near DeepEP:
        # a trainer built without the script scaffold would otherwise have no uniformity guard.
        verify_rank_uniform_env()

        config = self.parallelism_config

        # Before any wrapping: adapters sit outside the TP DTensor graph and would train rank-inconsistently.
        if config.is_tp_mode:
            self._validate_lora_tp_compatibility()

        self._validate_router_aux_loss_consumable()
        self._validate_load_best_model_reloadable()

        # FSDP2 needs one dtype per group; must run BEFORE the fp32_non_ep upcast so that mode wins.
        self._cast_peft_params_to_compute_dtype()

        # FP32 master weights BEFORE FSDP2 wrapping so it sees fp32 params and produces fp32 grads.
        if config.fp32_non_ep_params:
            self._upcast_non_ep_params_to_fp32()

        # TP/CP before needs_ep_wrappers (defaults True), else a TP-only dense model shards FSDP2 over world.
        if config.is_pp_mode:
            self._setup_pipeline_parallel()
        elif config.is_ep_tp_mode:
            self._setup_ep_tp()
        elif config.is_ep_cp_mode:
            self._setup_ep_cp()
        elif config.is_cp_mode:
            self._setup_cp_only()
        elif config.is_tp_mode:
            self._setup_tp_only()
        elif config.is_ep_mode:
            self._setup_ep_only()
        elif self._accelerate_manages_fsdp or self._accelerate_manages_ddp:
            # Before needs_ep_wrappers, whose branch would hijack an accelerate launch into mixin FSDP2.
            if self._find_ep_modules():
                raise ValueError(
                    "The model carries grouped-GEMM/EP expert wrappers but this is an "
                    "'accelerate launch' run: the wrappers require the mixin-managed FSDP2 path "
                    "(torchrun). Launch with torchrun, or load the model with "
                    "use_grouped_gemm=false to train it under accelerate."
                )
            if self._accelerate_manages_fsdp:
                fsdp_strategy = os.environ.get("FSDP_SHARDING_STRATEGY", "unknown")
                fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
                is_fsdp2 = fsdp_plugin is not None and getattr(fsdp_plugin, "fsdp_version", 1) == 2
                fsdp_label = "FSDP2" if is_fsdp2 else "FSDP"
                if is_global_main_process():
                    logger.info("=" * 60)
                    logger.info(f"{fsdp_label} managed by accelerate (standard data parallelism)")
                    logger.info(f"  FSDP strategy: {fsdp_strategy}")
                    logger.info("=" * 60)
                    if not is_fsdp2 and fsdp_strategy in ("SHARD_GRAD_OP", "FULL_SHARD", "2", "1"):
                        logger.warning(
                            "FSDP v1 SHARD_GRAD_OP/FULL_SHARD has a known PyTorch bug that can corrupt "
                            "model state after checkpoint saves during training. Consider using:\n"
                            "  - accelerate with distributed_type: MULTI_GPU (DDP, recommended)\n"
                            "  - accelerate with fsdp_version: 2 (FSDP2)\n"
                            "  - torchrun (uses FSDP2, safe for all save operations)"
                        )
            elif is_global_main_process():
                logger.info("=" * 60)
                logger.info("DDP managed by accelerate (standard data parallelism)")
                logger.info("=" * 60)
        elif config.needs_ep_wrappers:
            # Grouped-GEMM-only: a reorder letting TP/CP/ETP/EP fall through here mis-shards FSDP2.
            if config.is_tp_mode or config.is_cp_mode or config.is_expert_tp_mode or config.is_ep_mode:
                raise RuntimeError(
                    "Dispatch ordering broken: needs_ep_wrappers branch reached while a dedicated "
                    f"mode is active (tp={config.is_tp_mode}, cp={config.is_cp_mode}, "
                    f"etp={config.is_expert_tp_mode}, ep={config.is_ep_mode}). This would mis-shard FSDP2."
                )
            if self._find_ep_modules():
                self._setup_ep_only()
            else:
                self._setup_standard_data_parallel()
        else:
            self._setup_standard_data_parallel()

        if config.needs_ep_wrappers:
            self._validate_lora_ep_compatibility()
            if self._has_ep_layers:
                # The EP capacity generation rides the module the loop calls. EP patching registers it
                # on the model it patched, which a PEFT wrapper then reaches through .forward() —
                # running no pre-hook, so the generation freezes at the first forward's capacity.
                register_forward_generation_hook(self.model)
            # modules_to_save swaps in a router copy the ctor-time DP-sync hook never saw.
            self._reattach_ep_router_grad_sync_for_peft()
            self._validate_ep_peft_trainable_params_synced()
        self._validate_expert_lora_realized()
        # After the wrap and after TRL's dropout disabling, so they read what the run will actually use.
        self._validate_expert_lora_peft_config()
        self._validate_lora_dropout_live()

        # FSDP2 REPLACES managed Parameter objects, so an id memoized pre-wrap names an object nobody sees again.
        self._invalidate_param_id_caches()

        # After every wrap, for the same reason: only fully_shard makes a module an FSDPModule.
        self._setup_backward_reshard_window()

        self._setup_ep_gradient_checkpointing()

        # Make peft's disable_adapter() drop native expert adapters too, so KL references see a frozen base.
        if has_ep_lora(self.model):
            peft_model = find_peft_model(self.model)
            if peft_model is not None:
                make_disable_adapter_ep_aware(peft_model)

        self._validate_merge_expert_lora_save()

        # After the wrap, before any evaluate(): a DP-sharded loader needs a DP-scoped metric gather.
        self._install_dp_metric_gather()

        # Fail fast at construction on an unsupported sharded-save shape rather than hours in.
        if self.save_sharded_ep:
            validate_ep_sharded_save(
                self.model, merge_expert_lora_on_save=self.parallelism_config.merge_expert_lora_on_save
            )

        # Liger's fused loss matmuls the FSDP2-sharded lm_head.weight; the INSTANCE flag TRL caches must go too.
        has_liger_loss = hasattr(self, "liger_loss_fn") or getattr(self, "liger_grpo_loss", None) is not None
        if self._fsdp_wrapped and getattr(self, "use_liger_kernel", False) and has_liger_loss:
            self.use_liger_kernel = False
            self.args.use_liger_kernel = False
            if is_global_main_process():
                logger.info("  Disabled Liger fused loss (incompatible with FSDP2 DTensors)")

        # TRL's SyncRefModelCallback zips policy/ref params with plain .data ops — crashes on a DTensor/EP policy.
        wrapped = self._fsdp_wrapped or self.parallelism_config.is_ep_mode or self.parallelism_config.is_tp_mode
        if wrapped and getattr(self.args, "sync_ref_model", False):
            raise ValueError(
                "sync_ref_model is not supported under FSDP2/EP/TP: TRL's SyncRefModelCallback "
                "assumes an unwrapped policy. Disable sync_ref_model (use PEFT with ref_model=None, "
                "or precompute_ref_log_probs where available)."
            )

        # FSDP2's dynamic subclass is absent from _CAN_RECORD_REGISTRY → router logits go uncaptured.
        if self._fsdp_wrapped:
            self._register_output_capturing_for_fsdp()

        self._apply_torch_compile()

        # Setup is per-rank-varying (TP materialization, EP patching); without this, fast ranks race ahead → hang.
        barrier()

    def _cast_peft_params_to_compute_dtype(self):
        """Align trainable PEFT params (LoRA adapters + ``modules_to_save`` copies) to the surrounding
        storage dtype before FSDP2 wrapping, so each ``fully_shard`` group has one dtype.

        Target = fp32 under ``fp32_non_ep_params`` (the base is upcast to fp32, so adapters/copies must
        join it), else the bf16/fp16 compute dtype. Only trainable floating params are touched.
        """
        if self.parallelism_config.fp32_non_ep_params:
            target_dtype = torch.float32
        elif getattr(self.args, "bf16", False):
            target_dtype = torch.bfloat16
        elif getattr(self.args, "fp16", False):
            target_dtype = torch.float16
        else:
            return
        peft_model = find_peft_model(self.model)
        if peft_model is None:
            return
        # EP params: FSDP-ignored and deliberately fp32 — downcasting breaks the router forward.
        ep_param_ids = {id(p) for module in self._find_ep_modules() for p in module.parameters()}
        cast = 0
        for param in peft_model.parameters():
            if id(param) in ep_param_ids:
                continue
            if param.requires_grad and param.is_floating_point() and param.dtype != target_dtype:
                param.data = param.data.to(target_dtype)
                cast += 1
        if cast and is_global_main_process():
            logger.info(
                f"  Aligned {cast} trainable PEFT param tensors (adapters + modules_to_save) to {target_dtype}"
            )

    def _reattach_ep_router_grad_sync_for_peft(self):
        """Re-attach EP router DP-sync hooks to PEFT ``modules_to_save`` router/replicated copies.

        No-op without a PEFT model. Must run AFTER PEFT wrapping and EP setup.
        """
        if find_peft_model(self.model) is None:
            return
        attached = sum(module.reattach_router_grad_sync() for module in self._find_ep_modules())
        if attached and is_global_main_process():
            logger.info(f"  Re-attached EP router DP-sync hook to {attached} trainable modules_to_save params")

    def _validate_ep_peft_trainable_params_synced(self):
        """Fail loud if a trainable param inside an EP module has no gradient sync (any EP run).

        EP modules are FSDP-ignored; their grads stay DP-consistent only via the EP layer's own hooks
        / deferred sweep, which cover exactly the params each family declares (experts + LoRA, router,
        replicated submodules). A trainable param outside that set — PEFT ``modules_to_save`` on an
        unexpected EP submodule, or a family wrapper failing to declare a new weight in
        ``expert_named_params`` / ``replicated_named_params`` — would silently drift across DP ranks.
        """
        offenders = []
        for module in self._find_ep_modules():
            synced = module.synced_trainable_param_ids()
            offenders += [name for name, p in module.named_parameters() if p.requires_grad and id(p) not in synced]
        if offenders:
            shown = "\n".join(f"  - {n}" for n in offenders[:10])
            more = f"\n  ... and {len(offenders) - 10} more" if len(offenders) > 10 else ""
            if find_peft_model(self.model) is not None:
                remedy = (
                    "EP experts are trained via native grouped-LoRA (list expert projections in "
                    "lora_target_modules); the router/gate is trainable via lora_modules_to_save (its "
                    "DP-sync hook is re-attached automatically). modules_to_save on any other EP-internal "
                    "submodule is unsupported — remove it or target it through the native EP-LoRA path."
                )
            else:
                remedy = (
                    "The EP family wrapper must declare every trainable weight it owns: expert shards in "
                    "expert_named_params() (via _EXPERT_WEIGHT_ATTR_ROOTS) and replicated submodules in "
                    "replicated_named_params(), so the layer's grad-sync hooks cover it."
                )
            raise RuntimeError(
                f"{len(offenders)} trainable param(s) inside EP-wrapped modules have no gradient "
                f"sync — they would drift across DP ranks and silently corrupt the run:\n"
                f"{shown}{more}\n\n{remedy}"
            )

    def _validate_merge_expert_lora_save(self):
        """Fail fast where ``merge_expert_lora_on_save`` cannot produce the merged checkpoint.

        The flag folds both adapter halves into a gathered base checkpoint written by the EP
        strategy. Two shapes cannot deliver that, and both are read off the LIVE model rather than
        off a mode list, so every parallelism/model combination is covered by construction:

        - **No native expert adapters built.** Dense models, MoE runs with no expert projection in
          ``lora_target_modules``, and ``expert_tp_size > 1`` (where ``EPConfig`` rejects expert
          LoRA) all land here. There is nothing for the flag to fold, and an attention-only run
          saves adapters via :class:`PeftAdapterSaver`, which never merges.
        - **Accelerate-managed FSDP v1.** Params are flat-param shards the base Trainer's save
          serializes, so the gather would write shard views and the fold would land nowhere.
        """
        if not self.parallelism_config.merge_expert_lora_on_save:
            return
        if not has_ep_lora(self.model):
            raise ValueError(
                "merge_expert_lora_on_save=True but this model built no native grouped expert "
                "adapters, so there is nothing to fold and the flag would be silently ignored. It "
                "requires an EP/grouped-GEMM MoE run whose lora_target_modules names expert "
                "projections (gate_proj/up_proj/down_proj/gate_up_proj/experts). Attention-only "
                "LoRA saves adapters instead — merge those with "
                "scripts/after_training/merge_peft_adapters.py."
            )
        if self._accelerate_manages_fsdp:
            raise ValueError(
                "merge_expert_lora_on_save=True is not supported under accelerate-managed FSDP: "
                "adapters are flat-param shards there and the base Trainer's save owns the layout, "
                "so the merged gather would write shard views. Launch with torchrun (mixin-managed "
                "FSDP2), or drop merge_expert_lora_on_save and keep adapter checkpoints."
            )

    def _setup_ep_only(self):
        """Setup Expert Parallelism mode."""
        self._patch_gradient_clipping_for_ep()
        self._setup_ep_gradient_sync()
        logger.info("✓ EP mode configured")

    def _setup_cp_only(self):
        """Setup Context Parallelism mode."""
        self._configure_cp_wrapper()
        self._setup_cp_gradient_sync()
        logger.info("✓ CP mode configured")

    def _setup_tp_only(self):
        """Setup Tensor Parallelism mode.

        Pass the TOP-LEVEL model (not the ``.model`` backbone) to FSDP: the loader attaches the 2D
        ``(dp, tp)`` mesh there (handing FSDP the backbone would mint a conflicting fresh mesh), and
        wrapping it puts ``lm_head`` under FSDP so its TP-sharded grad is DP-synced (else the head
        drifts across DP replicas on TP+DP with untied embeddings). Pure TP (DP=1) skips FSDP.
        """
        model = self._top_level_model()
        self._fsdp_wrapped, self._device_mesh = setup_fsdp2_for_tp(
            model,
            self.parallelism_config.tp_size,
            self.args,
            dp_size=self.parallelism_config.data_parallel_size,
            fp32_master_weights=self.parallelism_config.fp32_non_ep_params,
            reshard_after_forward=self.parallelism_config.fsdp_reshard_after_forward,
        )
        if self._device_mesh is None:
            # Pure TP skips FSDP, so recover the loader's mesh or clipping/sync miss the TP group.
            self._device_mesh = getattr(model, "_device_mesh", None)
        self._require_tp_mesh()
        self._patch_gradient_clipping_for_tp()
        logger.info("✓ TP mode configured")

    def _setup_ep_cp(self):
        """Setup EP+CP combined mode."""
        self._configure_cp_wrapper()
        self._patch_gradient_clipping_for_ep()
        self._setup_ep_gradient_sync()
        logger.info("✓ EP+CP mode configured")

    def _setup_ep_tp(self):
        """Setup EP+TP combined mode."""
        self._patch_gradient_clipping_for_ep()
        self._device_mesh = getattr(self.model, "_device_mesh", None)
        self._require_tp_mesh()
        self._setup_ep_tp_gradient_sync()
        logger.info("✓ EP+TP mode configured")

    def _require_tp_mesh(self) -> None:
        """The TP mesh is what names the TP group; without it every TP reduce silently no-ops.

        ``tp_group()`` would return None, so the replicated-grad sync returns early (replicas drift
        apart across the TP axis, forever) and the TP shard-norm bucket is never reduced. Nothing
        downstream can tell that apart from "no TP", so it has to raise here.
        """
        if self._device_mesh is not None:
            return
        raise RuntimeError(
            f"Tensor parallelism is active (tp_size={self.parallelism_config.tp_size}) but no device "
            "mesh reached the trainer: FSDP did not build one (pure TP) and the model carries no "
            "`_device_mesh` from the loader. Without it the TP group is unknown, so replicated "
            "gradients are never averaged and the TP gradient-norm bucket is never reduced. Load the "
            "model through load_distributed_model, which attaches the mesh."
        )

    @staticmethod
    def _ignored_params(modules: Iterable[nn.Module]) -> list[nn.Parameter] | None:
        """Flatten modules into the ``ignored_params`` list FSDP2 takes, or None when there are none."""
        params = [p for module in modules for p in module.parameters()]
        return params or None

    def _apply_dp_fsdp2(
        self,
        model: nn.Module,
        dp_size: int,
        *,
        ignored_modules: Iterable[nn.Module] = (),
        dp_group: dist.ProcessGroup | None = None,
        dp_replicate_size: int = 1,
        topo: str = "",
        detail: str,
    ) -> bool:
        """Apply FSDP2 for DP gradient sync, record it, and log it. Returns whether it ran.

        Single seam for every parallelism mode's DP wrap: ``reshard_after_forward`` and
        ``fp32_master_weights`` always come from ``parallelism_config``, so a mode cannot silently
        diverge from the others on either. ``topo``/``detail`` carry only the mode-specific log text.
        """
        config = self.parallelism_config
        applied = setup_fsdp2_for_dp(
            model,
            dp_size,
            self.args,
            ignored_params=self._ignored_params(ignored_modules),
            dp_group=dp_group,
            reshard_after_forward=config.fsdp_reshard_after_forward,
            fp32_master_weights=config.fp32_non_ep_params,
            dp_replicate_size=dp_replicate_size,
        )
        if applied:
            self._fsdp_wrapped = True
            logger.info(
                f"✓ FSDP2 ({reshard_label(self.parallelism_config.fsdp_reshard_after_forward)}{topo}) {detail}"
            )
        return applied

    def _setup_standard_data_parallel(self):
        """Setup standard data parallel gradient sync (no EP/CP/TP) via FSDP v2.

        PEFT/QLoRA models with mixed/quantized dtypes pass incompatible params as ignored_params.
        """
        # Rank-block width, NOT the global world: a PP stage's FSDP must never span other stages.
        world_size = self.parallelism_config.stage_world_size
        if world_size <= 1:
            return

        # QLoRA: fully_shard cannot wrap non-float Params4bit, and the base is replicated anyway.
        if model_has_quantized_params(self.model):
            self._reject_fsdp_knobs_under_qlora()
            self._setup_qlora_gradient_sync()
            self._patch_gradient_clipping_for_qlora()
            logger.info("✓ QLoRA gradient sync (FSDP2 skipped: quantized 4-bit base weights)")
            return

        incompatible_modules = self._find_fsdp_incompatible_modules()
        dp_replicate_size = self.parallelism_config.dp_replicate_size
        extra = f", {len(incompatible_modules)} incompatible modules ignored" if incompatible_modules else ""
        topo = (
            f", HSDP {dp_replicate_size}×{world_size // dp_replicate_size}"
            if dp_replicate_size > 1
            else ", 1D full-shard"
        )
        self._apply_dp_fsdp2(
            self.model,
            world_size,
            ignored_modules=incompatible_modules,
            dp_replicate_size=dp_replicate_size,
            topo=topo,
            detail=f"applied for data parallel gradient sync ({world_size} ranks{extra})",
        )

    def _find_fsdp_incompatible_modules(self) -> list[nn.Module]:
        """Modules whose params differ from the trainable dtype (frozen bf16 base under fp32 adapters,
        or uint8 quantized layers). Passed as ignored_params so only trainable-dtype modules sync.
        """
        trainable_dtypes = {p.dtype for p in self.model.parameters() if p.requires_grad}
        all_dtypes = {p.dtype for p in self.model.parameters()}

        if len(all_dtypes) <= 1:
            return []

        if not trainable_dtypes:
            return []

        incompatible = []
        for module in self.model.modules():
            module_params = list(module.parameters(recurse=False))
            if not module_params:
                continue

            module_dtypes = {p.dtype for p in module_params}
            if not module_dtypes.issubset(trainable_dtypes):
                incompatible.append(module)

        return incompatible

    def _reject_fsdp_knobs_under_qlora(self) -> None:
        """QLoRA skips FSDP2 entirely (``fully_shard`` cannot wrap bnb's non-float Params4bit), so
        the FSDP-shaping knobs have nothing to act on — a multi-node run asking for ``use_hsdp`` would
        otherwise silently do a flat replicated all-reduce instead of the topology the config names.
        """
        set_knobs = _knobs_set_by_user(self.parallelism_config, _FSDP_SHAPING_KNOBS)
        if set_knobs:
            raise ValueError(
                f"{', '.join(set_knobs)} cannot take effect under QLoRA: quantized 4-bit base "
                f"weights skip FSDP2 entirely (fully_shard cannot wrap bnb Params4bit), so the run "
                f"would silently use plain replicated gradient sync instead of the sharding the "
                f"config names. Remove the flag(s), or use plain LoRA / full fine-tuning."
            )

    def _configure_cp_wrapper(self):
        """Set the CPConfig on the already-loaded CP wrapper and validate it is wrapped
        (CP model wrapping happens during model loading), plus the trainer settings whose loss or
        metric path would pair this rank's sequence chunk with full-length labels."""
        validate_trainer_args_for_cp(
            self.args,
            compute_metrics=getattr(self, "compute_metrics", None),
            preprocess_logits_for_metrics=getattr(self, "preprocess_logits_for_metrics", None),
        )
        self.cp_config = self.parallelism_config.create_cp_config()

        cp_wrapper = self._find_cp_wrapper()
        if cp_wrapper is None:
            raise ValueError(
                f"CP mode requires model to be wrapped with UlyssesCPModelWrapper "
                f"during loading (via load_model_for_cp / load_model_for_ep_cp / "
                f"load_distributed_model). Got: {type(self.model).__name__}"
            )

        cp_wrapper.cp_config = self.cp_config

        logger.info("✓ CP configured (NVLink-domain-local)")

    def _find_cp_wrapper(self) -> UlyssesCPModelWrapper | None:
        """Find the CP wrapper in the model hierarchy."""
        if isinstance(self.model, UlyssesCPModelWrapper):
            return self.model

        inner = getattr(self.model, "base_model", None)
        if inner is not None:
            inner_model = getattr(inner, "model", inner)
            if isinstance(inner_model, UlyssesCPModelWrapper):
                return inner_model

        return None

    def _ep_fsdp_ignored_modules(self) -> tuple[list[nn.Module], list[nn.Module], list[nn.Module]]:
        """``(ep_modules, dtype_incompatible, merged)`` — the FSDP ignored-module inputs every
        EP-aware wrap (EP/EP+TP modes and the PP stage wrap) shares.

        fsdp_shard_ep1_experts: only when experts are truly replicated (ep_group_size==1). Drop
        them from ignored_params so FSDP shards them; the EP layer skips its own grad hooks so
        FSDP reduce-scatter is the sole expert sync. No effect when ep_group_size>1 (incl pure ETP).
        """
        config = self.parallelism_config
        ep_modules = self._find_ep_modules()
        if config.experts_fsdp_managed:
            # The fp32_non_ep_params × managed-experts clash is refused at config time, rank-symmetrically.
            ep_modules = []
        dtype_incompatible = self._find_fsdp_incompatible_modules()
        merged = list({id(m): m for m in ep_modules + dtype_incompatible}.values())
        return ep_modules, dtype_incompatible, merged

    def _apply_ep_aware_dp_fsdp2(
        self,
        model: nn.Module,
        *,
        ignored: tuple[list[nn.Module], list[nn.Module], list[nn.Module]],
        fallback_dp_size: int,
        fallback_dp_group: dist.ProcessGroup | None = None,
        dp_replicate_size: int = 1,
        topo: str = "",
    ) -> None:
        """FSDP2 wrap for a model carrying EP wrappers, shared by the EP mode and the PP stage wrap.

        ``ignored`` is :meth:`_ep_fsdp_ignored_modules`' triple, passed in rather than re-derived:
        both callers already need part of it, and that derivation walks every parameter's dtype.

        Deferred-DP topologies (``is_deferred_dp``) shard the non-expert params over the EP group —
        the reduce-scatter must share the DeepEP combine's membership, with the cross-replica DP
        average deferred to :meth:`_sync_deferred_expert_grads` — everything else shards over
        the caller's DP scope (the world without PP, the stage group under PP).
        """
        ep_modules, dtype_incompatible, all_ignored_modules = ignored
        ep_cfg = self._ep_config
        if ep_cfg is not None and ep_cfg.is_deferred_dp:
            # Sharded over the EP GROUP, not the stage: non-expert memory per rank is `replicas`x the stage-FSDP figure.
            replicas = max(1, ep_cfg.world_size // ep_cfg.ep_group_size)
            if replicas > 1 and is_global_main_process():
                logger.info(
                    f"FSDP2 shards non-expert params {ep_cfg.ep_group_size}-way (the EP group) inside a "
                    f"stage of {ep_cfg.world_size}: non-expert memory per rank is {replicas}x the "
                    f"stage-wide-FSDP figure, and the cross-replica average is deferred over "
                    f"{replicas} replicas."
                )
            self._apply_dp_fsdp2(
                model,
                ep_cfg.ep_group_size,
                ignored_modules=all_ignored_modules,
                dp_group=ep_cfg.process_group,
                topo=f", EP-group {ep_cfg.ep_group_size}-way{topo}",
                detail=(
                    f"for EP gradient sync; cross-replica DP average deferred post-backward "
                    f"({ep_cfg.num_ep_groups} EP groups, {len(ep_modules)} EP modules ignored)"
                ),
            )
        else:
            extra = f", {len(dtype_incompatible)} dtype-incompatible" if dtype_incompatible else ""
            self._apply_dp_fsdp2(
                model,
                fallback_dp_size,
                ignored_modules=all_ignored_modules,
                dp_group=fallback_dp_group,
                dp_replicate_size=dp_replicate_size,
                topo=topo,
                detail=(
                    f"for EP gradient sync ({fallback_dp_size} ranks, {len(ep_modules)} EP modules{extra} ignored)"
                ),
            )

    @property
    def parallel_dims(self) -> ParallelDims:
        """Typed view over the active device mesh, rebuilt each access from current state (cheap;
        holds a reference) so there is no staleness across the setup lifecycle. EP/ETP groups are
        read off ``_ep_config`` directly — the object that owns them."""
        return ParallelDims(device_mesh=self._device_mesh)

    def _get_tp_process_group(self) -> dist.ProcessGroup | None:
        """Get the TP process group from the device mesh."""
        return self.parallel_dims.tp_group()

    def _get_tp_rank(self) -> int:
        """Get TP rank from DeviceMesh if available, fallback to modular arithmetic."""
        tp_local_rank = self.parallel_dims.tp_local_rank()
        if tp_local_rank is not None:
            return tp_local_rank
        return self.accelerator.process_index % self.parallelism_config.tp_size

    def _get_tp_or_etp_process_group(self) -> dist.ProcessGroup | None:
        """The process group within which ranks must share identical generation data: TP group (TP
        mode), expert-TP group (ETP mode), or None."""
        if self.parallelism_config.is_tp_mode:
            return self._get_tp_process_group()
        if self.parallelism_config.is_expert_tp_mode and self._ep_config is not None:
            return self._ep_config.expert_tp_group
        return None

    def _get_tp_group_src_rank(self) -> int:
        """Global rank of tp_rank=0 in this TP/ETP group — the source for broadcasting generation
        results so all group ranks process identical inputs."""
        if self.parallelism_config.is_tp_mode:
            return get_global_rank() - self._get_tp_rank()
        if self.parallelism_config.is_expert_tp_mode and self._ep_config is not None:
            group = self._ep_config.expert_tp_group
            if group is not None:
                return dist.get_process_group_ranks(group)[0]
        return get_global_rank()

    def _broadcast_object_from_tp_leader(self, obj):
        """Replace ``obj`` with the TP/ETP-group leader's copy. No-op outside TP/ETP.

        Siblings share one sharded replica and run collectives inside ``model.forward``; independently
        sampled rollouts would deadlock them.
        """
        group = self._get_tp_or_etp_process_group()
        if group is None:
            return obj
        container = [obj]
        dist.broadcast_object_list(container, src=self._get_tp_group_src_rank(), group=group)
        return container[0]

    def _broadcast_tensors_from_tp_leader(self, result: dict) -> dict:
        """Broadcast every tensor value of ``result`` in place from the TP/ETP leader. No-op outside TP/ETP.

        Dtype-safe: NCCL-unsupported dtypes (the int16 routing-replay masks) ride a uint8 bit view.
        """
        group = self._get_tp_or_etp_process_group()
        if group is None:
            return result
        src = self._get_tp_group_src_rank()
        for value in result.values():
            if isinstance(value, torch.Tensor):
                nccl_safe_broadcast(value, src=src, group=group)
        return result

    def _register_output_capturing_for_fsdp(self):
        """Re-register _can_record_outputs for FSDP2-wrapped model classes.

        FSDP2 creates dynamic subclasses not in transformers' _CAN_RECORD_REGISTRY; capture_outputs
        keys on str(self.__class__), so without this the wrapped class captures no outputs.
        """
        model = self._get_unwrapped_model()
        for sub in model.modules():
            cls_str = str(sub.__class__)
            if cls_str not in _CAN_RECORD_REGISTRY:
                record_outputs = getattr(sub, "_can_record_outputs", None)
                if record_outputs:
                    _CAN_RECORD_REGISTRY[cls_str] = record_outputs
                    if is_global_main_process():
                        logger.info(
                            f"  Registered output capturing for {sub.__class__.__name__} "
                            f"(keys: {list(record_outputs.keys())})"
                        )

    def _apply_torch_compile(self):
        """Apply torch.compile after FSDP wrapping and distributed setup."""
        training_args = getattr(self, "args", None)
        if training_args is None or not getattr(training_args, "torch_compile", False):
            return

        compile_mode = getattr(training_args, "torch_compile_mode", None) or "reduce-overhead"
        compile_backend = getattr(training_args, "torch_compile_backend", None) or "inductor"

        # Packed batches at bs>1 flatten to a VARIABLE-width row, so reduce-overhead's CUDA graphs stop applying.
        if (
            compile_mode == "reduce-overhead"
            and getattr(training_args, "packing", False)
            and getattr(training_args, "per_device_train_batch_size", 1) > 1
            and is_global_main_process()
        ):
            logger.warning(
                "torch_compile mode 'reduce-overhead' with packing at per_device_train_batch_size>1: "
                "the flattened row's width varies per step, which forces dynamic-shape recompiles and "
                "disables CUDA graphs. Set torch_compile_mode: 'default' (or keep batch 1) to avoid "
                "paying compile time for no speedup."
            )

        compile_start = time.time()
        self.model = torch.compile(self.model, mode=compile_mode, backend=compile_backend)
        # The training loop wraps self.model_wrapped; leaving it at the eager module would train eager.
        self.model_wrapped = self.model

        if is_global_main_process():
            logger.info(
                f"✓ torch.compile applied ({compile_mode}/{compile_backend}) in {time.time() - compile_start:.1f}s"
            )

    def _reject_cp_incompatible_collator(self, collator: Any):
        """Reject packed/varlen batches under CP, for trainers that build their own collator.

        ``collators/factory.py`` owns this rule for configs that go through it; this covers a collator
        handed straight to the trainer. Ulysses drops ``attention_mask`` and calls flash-attn with no
        ``cu_seqlens``, so packed documents attend across each other and the loss silently differs from
        the non-CP run. ``isinstance`` against the two roots — the completion-only variants subclass
        them, and a class-name substring missed both.
        """
        if not self.parallelism_config.is_cp_mode or collator is None:
            return
        if isinstance(collator, DataCollatorWithPacking | DataCollatorWithFlattening):
            raise ValueError(
                f"{type(collator).__name__} is incompatible with context parallelism: the Ulysses "
                f"path has no per-document boundaries, so packed sequences attend across each other. "
                f"Use a padding collator (padding_free/packing off) under CP."
            )

    def _move_model_to_device(self, model, device):
        """Reject leftover meta tensors — parameters AND buffers — before ``model.to(device)``.

        No loader path leaves either behind: the lazy loaders random-init whatever the checkpoint
        does not carry, ``from_pretrained`` initializes its MISSING set, ``init_from_scratch``
        builds on a real device, and every load path runs
        :func:`~src.models.patches.buffer_fixes.finalize_loaded_model`, which recomputes the
        known non-persistent buffer families. A tensor arriving here on meta therefore holds no
        values at all; allocating it would train/score uninitialized memory that also DIFFERS per
        rank (a zero inv_freq degenerates RoPE to NoPE with a plausible loss), so this is the last
        place it can be caught.
        """
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
                f"do not recompute — extend fix_rotary_inv_freq / fix_non_persistent_buffers "
                f"(src/models/patches/buffer_fixes.py)."
            )

        model.to(device)

    def _get_unwrapped_model(self) -> nn.Module:
        """The transformer BACKBONE (``model.model``, without ``lm_head``), through HF wrappers.

        Use only for backbone-specific work (EP-layer / router introspection, output-capture
        registration). For anything covering every trainable param (grad-norm, fp32 upcast, FSDP
        wrapping) use :meth:`_top_level_model` — the backbone drops ``lm_head`` on untied models.
        """
        model = self._top_level_model()
        if hasattr(model, "model"):
            return model.model
        return model

    def _top_level_model(self) -> nn.Module:
        """The full top-level model (``*ForCausalLM`` / CP wrapper), unwrapped from accelerate/DDP
        and from ``torch.compile``. Its ``named_parameters()`` covers the whole trainable surface
        including ``lm_head``.

        ``keep_torch_compile=False`` is not cosmetic: accelerate keeps the ``OptimizedModule`` by
        default, whose parameters are named ``_orig_mod.*`` — a save taken through it writes a
        checkpoint no loader can read, and an isinstance check for the CP wrapper fails through it.
        """
        return unwrap_framework_wrappers(self.model)

    def _dataset_map_num_proc(self, configured: int | None) -> int:
        """Worker count for this trainer's own ``dataset.map``/``filter`` passes.

        Same count in every parallelism mode: the map callables are module-level functions taking
        their state through ``fn_kwargs``, so a worker never pickles ``self``.
        """
        return resolve_map_num_proc(configured)

    @property
    def cp_size(self) -> int:
        """Context parallel size."""
        return self.parallelism_config.cp_size

    @property
    def is_tp_mode(self) -> bool:
        """Whether TP is enabled."""
        return self.parallelism_config.is_tp_mode

    @property
    def is_ep_mode(self) -> bool:
        """Whether EP is enabled."""
        return self.parallelism_config.is_ep_mode

    @property
    def is_cp_mode(self) -> bool:
        """Whether CP is enabled."""
        return self.parallelism_config.is_cp_mode

    def get_data_parallel_size(self) -> int:
        """Number of distinct batches being processed."""
        return self.parallelism_config.data_parallel_size

    def get_data_parallel_rank(self) -> int:
        """Data parallel rank (which batch this rank processes)."""
        return self.parallelism_config.get_data_parallel_rank()

    def get_total_train_batch_size(self, args) -> int:
        """Samples per optimizer step across the run — over the toolkit's DP width.

        The base Trainer divides ``args.world_size`` by ``get_tp_size()`` / ``get_cp_size()`` /
        ``get_sp_size()``, which read ``model._tp_size`` (transformers never sets it for its own
        ``tp_plan`` TP) and accelerate's ``parallelism_config`` (the toolkit builds a plain
        ``Accelerator``), and it carries no PP or ETP term — so every non-DP axis would count as data
        parallelism. Step and epoch counts come from the DP-sharded dataloader and are unaffected;
        the "Total train batch size" banner, ``num_train_samples`` and
        ``train_samples_per_second`` under ``max_steps`` read this.
        """
        return self._train_batch_size * args.gradient_accumulation_steps * self.get_data_parallel_size()

    def _log_parallelism_config(self):
        """Log parallelism configuration using ParallelismConfig."""
        if is_global_main_process():
            logger.info(self.parallelism_config.summary())
            optim = optim_name(self.args.optim)
            if self._bf16_optimizer:
                logger.info("  Optimizer: AdamWBF16 (stochastic rounding, 6B/param)")
            elif optim in NAMED_OPTIMIZER_BUILDERS:
                logger.info(f"  Optimizer: {NAMED_OPTIMIZER_BUILDERS[optim][1]}")

    def log(self, logs: dict[str, float], start_time: float | None = None):
        """Add parallelism info to logs before delegating to base Trainer."""
        self._add_parallelism_to_logs(logs)
        super().log(logs, start_time)

    def _add_parallelism_to_logs(self, logs: dict[str, float]):
        """Add parallelism info to logs dict."""
        if not hasattr(self, "state") or self.state.global_step == 0:
            return

        config = self.parallelism_config
        logs["parallelism/dp_size"] = config.data_parallel_size

        if config.is_ep_tp_mode:
            logs["parallelism/ep_tp_mode"] = 1.0
        elif config.is_ep_cp_mode:
            logs["parallelism/ep_cp_mode"] = 1.0

        if config.is_tp_mode:
            logs["parallelism/tp_size"] = config.tp_size
        if self._has_ep_layers:
            logs["parallelism/ep_size"] = config.ep_size
            logs["parallelism/num_ep_groups"] = getattr(self._ep_config, "num_ep_groups", 1)
            if config.use_grouped_gemm and not config.is_ep_mode:
                logs["parallelism/grouped_gemm_only"] = 1.0
        if config.is_expert_tp_mode:
            logs["parallelism/expert_tp_size"] = config.expert_tp_size
            logs["parallelism/ep_group_size"] = config.ep_group_size
        if config.is_cp_mode:
            logs["parallelism/cp_size"] = config.cp_size

        self._drain_unmasked_output_tokens()
        total_output_tokens = getattr(self, "_cumulative_unmasked_output_tokens", None)
        if total_output_tokens is not None:
            logs["num_unmasked_output_tokens_seen"] = total_output_tokens

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Count loss-contributing tokens (accumulated as ``num_unmasked_output_tokens_seen``) and arm
        FSDP2's per-window backward reshard, before delegating the actual step."""
        self._accumulate_unmasked_output_tokens(self._extract_output_token_count(inputs))
        if self._pp_runtime is not None:
            # The schedule owns forward AND backward, so the inherited step must not also run them.
            return self._pp_training_step(inputs, num_items_in_batch)
        self._warn_once_on_thin_memory_margin()
        # HF sets sync_gradients before every training_step, so this is the window's last microbatch.
        self._set_backward_reshard(self.accelerator.sync_gradients)
        try:
            return super().training_step(model, inputs, num_items_in_batch)
        except torch.OutOfMemoryError as e:
            emit_primary_failure(get_global_rank(), e)
            raise

    def _warn_once_on_thin_memory_margin(self):
        """After the first optimizer step — the earliest point where weights, gradients, optimizer
        state AND a full backward's activations have all coexisted — warn if this rank's peak sits
        within a few percent of its device. Per-rank by design: MoE routing skew concentrates
        dispatch buffers and expert activations on hot ranks, so only they see the thin margin."""
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
            # Per-rank verdict, so it must escape the adapter's main-process-only default.
            logger.warning(message, main_process_only=False)

    def prediction_step(self, model, inputs, prediction_loss_only=True, ignore_keys=None):
        """Under PP the eval step must run through the schedule (no rank holds the whole model)."""
        if self._pp_runtime is not None:
            return self._pp_prediction_step(inputs, prediction_loss_only, ignore_keys)
        return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys=ignore_keys)

    def create_accelerator_and_postprocess(self):
        """Create the accelerator for distributed training.

        Under custom parallelism, a plain Accelerator that does NOT apply DDP (gradient sync is via
        FSDP); otherwise delegate to the base Trainer via MRO.
        """
        if not self._needs_custom_accelerator():
            super().create_accelerator_and_postprocess()
            self._warn_knobs_unapplied_under_accelerate()
        else:
            # torch.compile is applied manually after FSDP wrapping (_apply_torch_compile).
            compile_keys = ["ACCELERATE_DYNAMO_BACKEND", "ACCELERATE_DYNAMO_MODE"]
            env_backup = {k: os.environ.pop(k) for k in compile_keys if k in os.environ}
            try:
                self._create_plain_accelerator()
            finally:
                os.environ.update(env_backup)

        # Both branches: the fp32 logits upcast costs the same under an accelerate-managed launch.
        self._configure_fp32_output_conversion()

    def _configure_fp32_output_conversion(self) -> None:
        """Turn off accelerate's autocast output conversion when the config asks for it.

        ``native_amp`` makes ``prepare`` wrap the forward in ``convert_outputs_to_fp32``, which
        upcasts the full ``[B, S, V]`` logits — the dominant allocation at long sequence × large
        vocab. bf16 compute comes from the FSDP mixed-precision policy (or from bf16 weights), not
        from autocast, so dropping it changes only the output dtype.

        Under fp16 it must stay on: ``native_amp`` also gates ``unscale_gradients``, so clearing it
        would leave clipping and the optimizer step operating on GradScaler-scaled gradients.
        """
        if self.parallelism_config.fp32_output_conversion:
            return
        if getattr(self.args, "fp16", False):
            logger.warning(
                "fp32_output_conversion=false is IGNORED under fp16: accelerate's native_amp also "
                "owns GradScaler unscaling, so disabling it would clip and step on scaled gradients. "
                "Use bf16 (the toolkit default) to drop the fp32 logits upcast."
            )
            return
        self.accelerator.native_amp = False
        logger.info("Disabled native_amp to prevent fp32 output conversion (avoids OOM for long sequences)")

    def _warn_knobs_unapplied_under_accelerate(self) -> None:
        """Warn for parallelism knobs an accelerate-managed FSDP/DDP launch silently drops.

        The mixin implements them inside its own FSDP2 wrap, which this launch never reaches: the
        sharding strategy, the HSDP mesh and the reduce dtype all come from accelerate's plugin.
        """
        config = self.parallelism_config
        requested = _knobs_set_by_user(config, _ACCELERATE_UNSUPPORTED_KNOBS)
        if not requested or not is_global_main_process():
            return
        logger.warning(
            f"These parallelism knobs are IGNORED under 'accelerate launch' (accelerate owns the "
            f"FSDP/DDP wrap, so the mixin never applies them): {sorted(requested)}. Launch with "
            f"torchrun to get them, or drop them from the config."
        )

    def _mixin_manages_gmm(self) -> bool:
        """Whether the grouped-GEMM wrapper path is mixin-managed (torchrun) rather than an
        accelerate-managed launch.

        ``needs_ep_wrappers`` is True at the default ``use_grouped_gemm`` even for dense models, so
        on an accelerate MULTI_GPU/FSDP launch it must not pull the run onto the custom
        accelerator / skip-DDP path — that would silently convert the user's DDP choice into mixin
        FSDP2 while ``_configure_mixed_precision`` skips AdamWBF16 on the "replicated DDP" premise.
        """
        return self.parallelism_config.needs_ep_wrappers and not (
            self._accelerate_manages_fsdp or self._accelerate_manages_ddp
        )

    def _needs_custom_accelerator(self) -> bool:
        """Check if we need custom accelerator (any parallelism or multi-GPU without accelerate management)."""
        config = self.parallelism_config
        has_parallelism = config.is_ep_mode or config.is_cp_mode or config.is_tp_mode or self._mixin_manages_gmm()
        accelerate_manages = self._accelerate_manages_fsdp or self._accelerate_manages_ddp
        multi_gpu_no_accelerate = get_global_world_size() > 1 and not accelerate_manages
        return has_parallelism or multi_gpu_no_accelerate

    def _create_plain_accelerator(self):
        """Create plain accelerator without DDP wrapping."""
        accelerator_config = self.args.accelerator_config.to_dict()

        dataloader_params = ["split_batches", "dispatch_batches", "even_batches", "use_seedable_sampler"]
        dataloader_config = DataLoaderConfiguration(
            **{param: accelerator_config.pop(param) for param in dataloader_params},
            data_seed=self.args.data_seed,
            non_blocking=accelerator_config.pop("non_blocking", False),
        )

        kwargs = {
            "gradient_accumulation_steps": 1,  # Trainer drives accumulation
            "dataloader_config": dataloader_config,
        }
        if self.args.bf16:
            kwargs["mixed_precision"] = "bf16"
        elif self.args.fp16:
            kwargs["mixed_precision"] = "fp16"

        self.accelerator = Accelerator(**kwargs)
        self.gather_function = self.accelerator.gather_for_metrics

        # Suppress accelerate's "dataset had no length" INFO — harmless here.
        logging.getLogger("accelerate.accelerator").setLevel(logging.WARNING)

        self.is_deepspeed_enabled = False
        # Runs before any FSDP2 wrapping, so _fsdp_wrapped is False — the checkpoint path relies on it.
        self.is_fsdp_enabled = getattr(self, "is_fsdp_enabled", False)

        logger.info(f"Created Accelerator with distributed_type: {self.accelerator.state.distributed_type}")

    def create_optimizer(self, model=None):  # base Trainer passes it; the mixin builds from self
        """Route to the AdamWBF16 / Muon / FlashAdamW / tensor-type-grouped builders (each in its
        own :mod:`src.optimizers` module) per config; owns the ``self.optimizer`` assignment and the
        TP replicated-grad-sync hook registration."""
        if self.optimizer is None:
            # A gathered FSDP2 module registers its transient unsharded params, and an evaluate() before
            # train() leaves it gathered (reshard_after_forward=False; the root never reshards post-forward):
            # param groups built over those objects update orphans the next unshard discards — never the shards.
            reshard_fsdp2_modules(self.model)
            decay_parameters = self.get_decay_parameter_names(self.model)
            optim = optim_name(self.args.optim)
            if self._bf16_optimizer:
                self.optimizer = build_bf16_optimizer(self.model, self.args, decay_parameters)
            elif optim in NAMED_OPTIMIZER_BUILDERS:
                builder = NAMED_OPTIMIZER_BUILDERS[optim][0]
                self.optimizer = builder(self.model, self.args, decay_parameters)
            # Stock SGD's foreach cannot mix FSDP2 DTensors with plain EP tensors; kept narrow to spare other optims.
            elif self.parallelism_config.fp32_non_ep_params or (optim == "sgd" and self._has_ep_layers):
                self.optimizer = build_tensor_type_grouped_optimizer(
                    self.model, self.args, decay_parameters, self.get_optimizer_cls_and_kwargs
                )
            else:
                self._refuse_stock_optimizer_on_mixed_params(optim)
                super().create_optimizer()

        self._register_tp_replicated_grad_sync_hook()
        self._register_deferred_ep_grad_sync_hook()
        self._register_qlora_grad_sync_hook()
        return self.optimizer

    def _refuse_stock_optimizer_on_mixed_params(self, optim: str) -> None:
        """Refuse the stock fused/foreach AdamW where plain expert tensors coexist with DTensor peers.

        ``aten._fused_adamw_`` raises 'mixed torch.Tensor and DTensor' at the first step over that
        parameter set. Refused at the BUILD rather than at trainer construction: the mix defeats the
        optimizer only — forward, backward and the EP-aware clip are correct over the same
        parameters, which is exactly what the fp32 PP+ETP equivalence gate exercises with no
        optimizer in play. ``train()`` builds the optimizer before the first step, so a real training
        run still dies within seconds of starting rather than mid-epoch.

        Narrower than "the model is MoE", and read from the config so every PP stage agrees (a
        module walk would refuse on MoE stages while dense stages walked into the collective):
        :func:`ep_wraps_experts` is the same predicate the Liger gate uses — at ``ep_group_size ==
        1`` with ``fsdp_shard_ep1_experts`` (the default) FSDP2 shards the experts too, and a MoE
        family with no registered EP wrapper class (qwen3_next) never gets wrapped — both are
        uniformly DTensor and step fine. ``fp32_non_ep_params`` never reaches here; it routes to
        the tensor-type-grouped builder above, which is the remedy this message names.
        """
        config = self.parallelism_config
        if optim not in STOCK_ADAMW_OPTIMIZERS:
            return  # named/per-parameter optimizers carry their own gates
        if not (
            config.ep_group_size > 1
            or (
                ep_wraps_experts(config.needs_ep_wrappers, getattr(self.model, "config", None))
                and not config.experts_fsdp_managed
            )
        ):
            return
        raise ValueError(
            f"optim={optim!r} cannot mix the plain-tensor expert FFN params (EP rank-local experts, "
            f"or the grouped-GEMM gate_proj_gmm/up_proj_gmm split at ep_size=1) with the FSDP2 "
            f"DTensor non-expert params: aten._fused_adamw_ raises 'mixed torch.Tensor and DTensor' "
            f"at the first step. Use AdamWBF16 (bf16=true, the production default, or "
            f"bf16_optimizer=true — its per-parameter path is DTensor-aware), or "
            f"fp32_non_ep_params=true for fp32 masters on the non-expert params only (experts stay "
            f"bf16). Full fp32 master weights are supported on dense (non-MoE) models."
        )

    def evaluation_loop(self, *args, **kwargs):
        """The eval loop's cross-cutting seams, owned here because two sub-mixins each hold one.

        Before: the DP-scoped metric gather is re-armed for THIS loop (HF hands ``gather_function``
        back to accelerate at the end of every one). After: a pipeline stage is re-sharded, the
        forward-only drives having left it gathered.
        """
        self._rearm_dp_metric_gather()
        output = super().evaluation_loop(*args, **kwargs)
        self._reshard_after_evaluation_loop()
        return output

    def evaluate(self, *args, **kwargs):
        """Prevent AllGather deadlock during evaluation.

        When custom data distribution gives ranks unequal eval batch counts, the default eval loop's
        per-step gather_for_metrics deadlocks — swap gather_function to identity (each rank scores its
        own shard). Gated on actually-unequal counts: for equal-batch DP the identity swap would
        report rank-0's shard as the global metric, so there we keep the cross-rank gather.
        """
        if (
            self._needs_custom_accelerator()
            and dist.is_initialized()
            and self._eval_ranks_have_unequal_batches(args, kwargs)
        ):
            if getattr(self.args, "top_entropy_quantile", 1.0) < 1.0:
                # TRL's entropy threshold pads-and-gathers through this same accelerator; identity padding → SIGABRT.
                raise ValueError(
                    "Evaluation with unequal (or unmeasurable) per-rank batch counts cannot run "
                    "with entropy masking (top_entropy_quantile < 1.0): the eval escape hatch "
                    "disables cross-rank padding, which that gather needs. Provide a sized, evenly "
                    "divisible eval dataset or set top_entropy_quantile: 1.0."
                )
            original_gather = self.gather_function
            self.gather_function = lambda x: x
            # evaluation_loop also pads logits/labels outside gather_function; identity is safe (own shard only).
            self.accelerator.pad_across_processes = lambda tensor, *pad_args, **pad_kwargs: tensor
            try:
                # Holds the identity across evaluation_loop, which otherwise re-arms the DP-scoped gather.
                with self.suspended_dp_metric_gather():
                    result = super().evaluate(*args, **kwargs)
            finally:
                self.gather_function = original_gather
                # del, not re-assignment: assigning the bound method back pins a permanent instance attribute.
                self.accelerator.__dict__.pop("pad_across_processes", None)
            barrier()
            return result
        return super().evaluate(*args, **kwargs)

    def _eval_ranks_have_unequal_batches(self, eval_args: tuple, eval_kwargs: dict) -> bool:
        """True if the eval dataloader yields a different number of batches across ranks.

        Min/max all-reduce of the local batch count. When unmeasurable (iterable dataset, no len),
        returns True so the gather is disabled rather than risking a deadlock.
        """
        eval_dataset = eval_kwargs.get("eval_dataset", eval_args[0] if eval_args else None)
        try:
            local_batches = len(self.get_eval_dataloader(eval_dataset))
            measurable = True
        except (TypeError, AttributeError):
            local_batches, measurable = 0, False

        # Agree measurability first: a rank that early-returns strands peers at the reduces below.
        if not rank_consensus(measurable)[0]:
            return True

        device = getattr(self.args, "device", None) or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        count = torch.tensor([local_batches], device=device, dtype=torch.long)
        cmin = count.clone()
        cmax = count.clone()
        dist.all_reduce(cmin, op=dist.ReduceOp.MIN)
        dist.all_reduce(cmax, op=dist.ReduceOp.MAX)
        return int(cmin.item()) != int(cmax.item())

    def _inner_training_loop(
        self, batch_size=None, args=None, resume_from_checkpoint=None, trial=None, ignore_keys_for_eval=None
    ):
        """Override to skip DDP wrapping when FSDP handles gradient sync."""
        if not self._should_skip_ddp_wrapping():
            return super()._inner_training_loop(batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)

        original_prepare = self.accelerator.prepare
        self.accelerator.prepare = self._create_no_wrap_prepare()

        try:
            return super()._inner_training_loop(batch_size, args, resume_from_checkpoint, trial, ignore_keys_for_eval)
        finally:
            self.accelerator.prepare = original_prepare

    def _should_skip_ddp_wrapping(self) -> bool:
        """Check if we should skip DDP wrapping (FSDP, EP, or TP handles sync)."""
        return (
            self._fsdp_wrapped
            or self._mixin_manages_gmm()
            or self.parallelism_config.is_tp_mode
            or self.parallelism_config.is_expert_tp_mode
            or self.parallelism_config.is_pp_mode
        )

    def _create_no_wrap_prepare(self):
        """Create a prepare function that marks objects as prepared without wrapping."""
        original_prepare = self.accelerator.prepare

        def no_wrap_prepare(*objects):
            prepared = []
            for obj in objects:
                if isinstance(obj, (torch.nn.Module, torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler)):
                    obj._is_accelerate_prepared = True
                    prepared.append(obj)
                elif obj is None:
                    prepared.append(None)
                else:
                    prepared.append(original_prepare(obj))
            return prepared[0] if len(prepared) == 1 else tuple(prepared)

        return no_wrap_prepare

    def _validate_inputs(self, inputs: dict[str, Any]):
        """Warn once if a batch has no training signal (all labels masked).

        Only on the first optimizer step: the collator config is static (a systemic bug shows
        immediately) and the ``.all()`` reduction is a device→host sync not worth paying every step.
        """
        if self._warned_empty_labels:
            return
        if getattr(self, "state", None) is not None and self.state.global_step > 0:
            return
        labels = inputs.get("labels")
        if labels is None:
            return
        non_training = labels == LABEL_IGNORE_INDEX
        tokenizer = getattr(self, "processing_class", None)
        for attr in ("pad_token_id", "eos_token_id"):
            token_id = getattr(tokenizer, attr, None)
            if token_id is not None:
                non_training = non_training | (labels == token_id)
        if non_training.all():
            self._warned_empty_labels = True
            logger.warning(
                "Batch contains no training signal — all labels are -100, pad, or eos. "
                "This typically means the collator could not find the assistant_message_template "
                "in the tokenized input, or the eval set is too small after packing. "
                f"Labels shape: {list(labels.shape)}"
            )

    def cleanup_ep(self):
        """Destroy DeepEP buffers. Call when done with the trainer to avoid CUDA context issues during GC."""
        # Align first: destroy is a device-side barrier (~100 s spin), so the wait belongs under the PG timeout.
        barrier()
        count = 0
        # Through the EP-layer registry, not a hasattr sniff on ``dispatcher``, which only widened the match.
        for module in self._find_ep_modules():
            try:
                module.dispatcher.destroy()
                count += 1
            except Exception as e:
                logger.warning(f"Failed to destroy DeepEP buffer for {type(module).__name__}: {e}")
        # Layers the GC already took freed nothing (a finalizer must not run a collective); swept here instead.
        freed = free_unclaimed_ep_buffers()
        if count > 0 or freed > 0:
            logger.info(f"✓ Released {count} DeepEP layers, freed {freed} buffers")
        barrier()
