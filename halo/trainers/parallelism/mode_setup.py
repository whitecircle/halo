"""Wrapping the trainer's model for its parallel mode: the mode ladder, the FSDP2 wraps and what they leave out."""

import logging
import os
import time
from collections.abc import Iterable

import torch
import torch.distributed as dist
import torch.nn as nn
from transformers.utils.output_capturing import _CAN_RECORD_REGISTRY

from halo.distributed.context_parallel.ulysses_model import UlyssesModel, find_ulysses_model
from halo.distributed.fsdp import IdentityParamSet, apply_fsdp2_for_dp, apply_fsdp2_for_tp
from halo.distributed.mesh import tp_local_rank
from halo.log import KEY_PREVIEW_COUNT
from halo.model_loading.dtype import has_packed_quantized_params, resolve_training_dtype
from halo.models.mtp.head import frozen_trunk_params
from halo.models.unwrapping import find_peft_model
from halo.runtime.collectives import nccl_safe_broadcast
from halo.runtime.consensus import reject_across_ranks
from halo.runtime.ranks import is_global_main_process
from halo.trainers.parallelism.ep_layers import require_ep_config
from halo.trainers.trainer_rules import (
    reject_cp_trainer_args,
    reject_ep_layers_under_accelerate,
    reject_fsdp_knobs_under_qlora,
)

logger = logging.getLogger(__name__)

# Accelerate's TorchDynamoPlugin default too; reduce-overhead's CUDA graphs measure slower on EP MoE.
DEFAULT_TORCH_COMPILE_MODE = "default"


def require_tp_mesh(mesh, tp_size: int):
    """``mesh``, or a raise: without the TP mesh the TP group is unknown and every TP reduce is a silent no-op."""
    if mesh is not None:
        return mesh
    raise RuntimeError(
        f"Tensor parallelism is active (tensor_parallel_size={tp_size}) "
        "but no device mesh reached the trainer: the model carries no `_device_mesh` from the "
        "loader. Without it the TP group is unknown, so "
        "replicated gradients are never averaged and the TP gradient-norm bucket is never "
        "reduced. Load the model through load_distributed_model, which attaches the mesh."
    )


def require_ulysses_model(model: nn.Module) -> UlyssesModel:
    wrapper = find_ulysses_model(model)
    if wrapper is None:
        raise ValueError(
            f"CP mode requires model to be wrapped with UlyssesModel "
            f"during loading (via load_cp_model / load_ep_cp_model / "
            f"load_distributed_model). Got: {type(model).__name__}"
        )
    return wrapper


def reject_unsynced_trainable_params(
    model: nn.Module, candidates: Iterable[nn.Parameter], ep_layers, ep_config
) -> None:
    """COLLECTIVE. Raise on every rank if a trainable parameter the FSDP2 wraps leave out has no gradient sync; it
    would drift across DP ranks while every loss stays finite."""
    candidate_set = IdentityParamSet(candidates)
    # The deferred sweep averages every trainable non-DTensor parameter, and a parameter outside the
    # shard groups is never a DTensor.
    deferred = ep_config is not None and ep_config.sweep_reduces_across_replicas
    covered: set[int] = set()
    for layer in ep_layers:
        if not layer.ep_config.experts_fsdp_managed:
            covered |= layer.synced_trainable_param_ids()
    unsynced = [
        name
        for name, param in model.named_parameters()
        if param.requires_grad and param in candidate_set and not deferred and id(param) not in covered
    ]
    reason = None
    if unsynced:
        shown = "\n".join(f"  - {name}" for name in unsynced[:KEY_PREVIEW_COUNT])
        hidden = len(unsynced) - KEY_PREVIEW_COUNT
        more = f"\n  ... and {hidden} more" if hidden > 0 else ""
        if find_peft_model(model) is not None:
            remedy = (
                "EP experts are trained via native grouped-LoRA (list expert projections in "
                "lora_target_modules); the router/gate is trainable via lora_modules_to_save (its "
                "DP-sync hook is re-attached automatically). modules_to_save on any other EP-internal "
                "submodule is unsupported — remove it or target it through the native EP-LoRA path."
            )
        else:
            remedy = (
                "The family's EP layer must declare every trainable weight it owns: expert shards in "
                "expert_named_params() (via FAMILY.expert_weight_roots) and replicated submodules in "
                "replicated_named_params(), so the layer's grad-sync hooks cover it."
            )
        reason = (
            f"{len(unsynced)} trainable parameter(s) outside FSDP2's shard groups have no other gradient "
            f"sync — they would drift across DP ranks and silently corrupt the run:\n{shown}{more}\n\n{remedy}"
        )
    reject_across_ranks(reason, "unsynced trainable parameters")


class ModeSetupMixin:
    def _wrap_for_parallel_mode(self) -> None:
        """PP, EP+TP, EP+CP, CP, TP, EP, an accelerate-managed launch, or plain DP. TP and CP precede the EP branch,
        which ``needs_ep_layers`` (on by default) would otherwise take for a dense TP-only model."""
        config = self.parallelism_config
        mode = None
        if config.is_pp_mode:
            self._configure_pipeline_parallel()
        elif config.has_ep_group and config.is_tp_mode:
            self._install_ep_clip()
            self._device_mesh = require_tp_mesh(getattr(self.model, "_device_mesh", None), config.tp_size)
            self._apply_ep_tp_fsdp2()
            mode = "EP+TP"
        elif config.has_ep_group and config.is_cp_mode:
            self._adopt_cp_wrapper()
            self._install_ep_clip()
            self._apply_ep_fsdp2()
            mode = "EP+CP"
        elif config.is_cp_mode:
            self._adopt_cp_wrapper()
            self._apply_cp_fsdp2()
            mode = "CP"
        elif config.is_tp_mode:
            self._device_mesh = require_tp_mesh(getattr(self._top_level_model(), "_device_mesh", None), config.tp_size)
            self._apply_tp_fsdp2()
            self._install_tp_clip()
            mode = "TP"
        elif config.has_ep_group or (
            not self._accelerate_manages_wrap and config.needs_ep_layers and self._ep_layers()
        ):
            self._install_ep_clip()
            self._apply_ep_fsdp2()
            mode = "EP"
        elif self._accelerate_manages_wrap:
            reject_ep_layers_under_accelerate(self._ep_layers())
            self._log_accelerate_managed_wrap()
        elif config.stage_world_size > 1:
            if has_packed_quantized_params(self.model):
                self._install_qlora_grad_sync("data parallel")
            else:
                self._apply_dp_fsdp2(
                    self.model,
                    config.stage_world_size,
                    excluded_params=self._fsdp_excluded_params(),
                    dp_replicate_size=config.dp_replicate_size,
                )
        if mode and is_global_main_process():
            logger.info(f"✓ {mode} mode configured")

    def _log_accelerate_managed_wrap(self) -> None:
        """Name the wrap accelerate applies; FSDP v1's sharded strategies draw a warning."""
        if not is_global_main_process():
            return
        if not self._accelerate_manages_fsdp:
            logger.info("=" * 60)
            logger.info("DDP managed by accelerate (standard data parallelism)")
            logger.info("=" * 60)
            return
        fsdp_strategy = os.environ.get("FSDP_SHARDING_STRATEGY", "unknown")
        fsdp_plugin = getattr(self.accelerator.state, "fsdp_plugin", None)
        is_fsdp2 = fsdp_plugin is not None and getattr(fsdp_plugin, "fsdp_version", 1) == 2
        logger.info("=" * 60)
        logger.info(f"{'FSDP2' if is_fsdp2 else 'FSDP'} managed by accelerate (standard data parallelism)")
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

    def _adopt_cp_wrapper(self) -> None:
        """Adopt the loader's CP wrapper, after refusing settings that would pair this rank's chunk with full-length
        labels."""
        reject_cp_trainer_args(
            self.args,
            compute_metrics=getattr(self, "compute_metrics", None),
            preprocess_logits_for_metrics=getattr(self, "preprocess_logits_for_metrics", None),
        )
        self.cp_config = require_ulysses_model(self._top_level_model()).cp_config
        if is_global_main_process():
            logger.info("✓ CP configured (NVLink-domain-local)")

    def _find_cp_wrapper(self) -> UlyssesModel | None:
        return find_ulysses_model(self._top_level_model())

    def _apply_tp_fsdp2(self) -> None:
        """Pure TP (DP=1) skips FSDP. Over the top level, since ``lm_head`` must sit under FSDP for its DP sync."""
        config = self.parallelism_config
        self._fsdp_wrapped = apply_fsdp2_for_tp(
            self._top_level_model(),
            self._device_mesh,
            compute_dtype=resolve_training_dtype(self.args),
            fp32_reduce=config.fsdp_reduces_in_fp32,
            dp_size=config.data_parallel_size,
            reshard_after_forward=config.fsdp_reshard_after_forward,
        )

    def _apply_ep_fsdp2(self) -> None:
        """FSDP2 for the EP (and EP+CP) gradient sync: experts FSDP-ignored, everything else sharded."""
        config = self.parallelism_config
        # Rank-block width, not the global world (identical without PP).
        if config.stage_world_size <= 1:
            return
        self._apply_ep_aware_dp_fsdp2(
            self.model,
            excluded_params=self._fsdp_excluded_params(),
            fallback_dp_size=config.stage_world_size,
            dp_replicate_size=config.dp_replicate_size,
        )

    def _apply_ep_tp_fsdp2(self) -> None:
        """The EP+TP DP leg: the 2D (dp, tp) mesh with EP modules left out. Through the TP entry point, so it runs the
        shared wrap body and takes the mesh the DTensors are sharded on, never a fresh one."""
        config = self.parallelism_config
        self._fsdp_wrapped = apply_fsdp2_for_tp(
            self.model,
            self._device_mesh,
            compute_dtype=resolve_training_dtype(self.args),
            fp32_reduce=config.fsdp_reduces_in_fp32,
            dp_size=config.data_parallel_size,
            ignored_params=self._fsdp_excluded_params(),
            reshard_after_forward=config.fsdp_reshard_after_forward,
        )

    def _apply_cp_fsdp2(self) -> None:
        """FSDP2 (or the QLoRA all-reduce) for the CP-only gradient sync."""
        config = self.parallelism_config
        if config.stage_world_size <= 1:
            return
        if has_packed_quantized_params(self.model):
            self._install_qlora_grad_sync("context parallel")
            return
        self._apply_dp_fsdp2(
            self._top_level_model(),
            config.stage_world_size,
            excluded_params=self._fsdp_excluded_params(),
            dp_replicate_size=config.dp_replicate_size,
        )

    def _install_qlora_grad_sync(self, mode: str) -> None:
        """A flag rather than a hook per parameter: ``register_post_accumulate_grad_hook`` fires only for params with a
        grad on this rank, so its all-reduce would hang whenever a microbatch skips a branch on one rank."""
        reject_fsdp_knobs_under_qlora(self.parallelism_config)
        self._qlora_grad_sync = True
        self._install_qlora_clip()
        broadcast = self._start_qlora_replicas_identical()
        if is_global_main_process():
            logger.info(
                f"✓ QLoRA gradient sync for {mode} (FSDP2 skipped: quantized 4-bit base weights), "
                f"{broadcast} floating-point parameters starting from rank 0's copy"
            )

    def _start_qlora_replicas_identical(self) -> int:
        """COLLECTIVE over the world. Every floating-point parameter takes rank 0's; returns how many. The sync
        averages gradients only, so an adapter drawn from each rank's RNG would keep that draw for the whole run."""
        params = [param for param in self._top_level_model().parameters() if param.is_floating_point()]
        with torch.no_grad():
            for param in params:
                nccl_safe_broadcast(param.data, src=0)
        return len(params)

    def _apply_dp_fsdp2(
        self,
        model: nn.Module,
        dp_size: int,
        *,
        excluded_params: Iterable[nn.Parameter] = (),
        dp_group: dist.ProcessGroup | None = None,
        dp_replicate_size: int = 1,
    ) -> None:
        """FSDP2 for DP gradient sync, shared by every mode's DP wrap so none diverges on dtype, reduce or reshard."""
        config = self.parallelism_config
        excluded_params = list(excluded_params)
        if apply_fsdp2_for_dp(
            model,
            dp_size,
            compute_dtype=resolve_training_dtype(self.args),
            fp32_reduce=config.fsdp_reduces_in_fp32,
            ignored_params=excluded_params or None,
            dp_group=dp_group,
            reshard_after_forward=config.fsdp_reshard_after_forward,
            dp_replicate_size=dp_replicate_size,
            # No expert is distributed, so whatever the wrap leaves out is a whole copy on every DP rank.
            ignored_params_replicated=config.ep_group_size == 1,
        ):
            self._fsdp_wrapped = True

    def _apply_ep_aware_dp_fsdp2(
        self,
        model: nn.Module,
        *,
        excluded_params: Iterable[nn.Parameter],
        fallback_dp_size: int,
        fallback_dp_group: dist.ProcessGroup | None = None,
        dp_replicate_size: int = 1,
    ) -> None:
        """Deferred-DP topologies shard the non-expert params over the EP group, whose membership the reduce-scatter
        must share with the DeepEP combine, and defer the cross-replica average to the post-backward sweep."""
        ep_cfg = require_ep_config(self._ep_config)
        dp_size, dp_group = fallback_dp_size, fallback_dp_group
        if ep_cfg.is_deferred_dp:
            dp_size, dp_group, dp_replicate_size = ep_cfg.ep_group_size, ep_cfg.process_group, 1
            replicas = max(1, ep_cfg.stage_world_size // ep_cfg.ep_group_size)
            if replicas > 1 and is_global_main_process():
                logger.info(
                    f"FSDP2 shards non-expert params {ep_cfg.ep_group_size}-way (the EP group) inside a "
                    f"stage of {ep_cfg.stage_world_size}: non-expert memory per rank is {replicas}x the "
                    f"stage-wide-FSDP figure, and the cross-replica average is deferred over "
                    f"{replicas} replicas."
                )
        self._apply_dp_fsdp2(
            model, dp_size, excluded_params=excluded_params, dp_group=dp_group, dp_replicate_size=dp_replicate_size
        )

    def _fsdp_excluded_params(self) -> list[nn.Parameter]:
        """EP modules sync their own gradients and are left out whole, except under ``experts_fsdp_managed``. Frozen
        parameters whose dtype no trainable parameter shares are taken one by one, so a trainable child of their module
        stays in its shard group. A ``head_only`` run's frozen trunk joins them: it takes no gradient to reduce and would
        otherwise re-gather every step."""
        ep_modules = self._ep_layers()
        if self.parallelism_config.experts_fsdp_managed:
            ep_modules = []
        ep_params = [p for module in ep_modules for p in module.parameters()]
        params = ep_params + self._dtype_excluded_params() + frozen_trunk_params(self.model)
        return list({id(p): p for p in params}.values())

    def _dtype_excluded_params(self) -> list[nn.Parameter]:
        """Frozen parameters whose dtype no trainable parameter shares (a frozen bf16 base under fp32 adapters, uint8
        quantized weights). Per parameter, so the owning module's trainable children stay in the shard groups."""
        params = list(self.model.parameters())
        trainable_dtypes = {p.dtype for p in params if p.requires_grad}
        if not trainable_dtypes:
            return []
        return [p for p in params if not p.requires_grad and p.dtype not in trainable_dtypes]

    def _cast_peft_params_to_compute_dtype(self) -> None:
        """Align trainable PEFT params to the surrounding storage dtype, so each ``fully_shard`` group has one dtype:
        fp32 under ``fp32_non_ep_params``, else the bf16/fp16 compute dtype."""
        if self.parallelism_config.fp32_non_ep_params:
            target_dtype = torch.float32
        else:
            target_dtype = resolve_training_dtype(self.args)
            if target_dtype == torch.float32:
                return
        peft_model = find_peft_model(self.model)
        if peft_model is None:
            return
        # EP params: FSDP-ignored and deliberately fp32; downcasting breaks the router forward.
        ep_param_ids = self._get_ep_param_ids()
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

    def _upcast_non_ep_params_to_fp32(self):
        """Upcast the non-EP parameters to FP32 masters. The loaders already hold checkpoint-backed masters at FP32, so
        this changes only PEFT and wrapper-added parameters and a model handed in past those loaders."""
        ep_param_ids = self._get_ep_param_ids()
        upcast_count = 0
        skipped_ep_count = 0
        for _name, param in self._top_level_model().named_parameters():
            if id(param) in ep_param_ids:
                skipped_ep_count += 1
                continue
            # A QLoRA base is bnb Params4bit over uint8-packed storage; .float() would reinterpret nibbles as values.
            if param.dtype != torch.float32 and param.is_floating_point():
                param.data = param.data.float()
                upcast_count += 1
        if is_global_main_process():
            logger.info(
                f"✓ FP32 master weights for {upcast_count} non-EP params "
                f"(skipped {skipped_ep_count} EP params, compute stays BF16)"
            )

    def _get_tp_rank(self) -> int:
        rank = tp_local_rank(self._device_mesh)
        if rank is not None:
            return rank
        return self.accelerator.process_index % self.parallelism_config.tp_size

    def _register_output_capturing_for_fsdp(self) -> None:
        """transformers keys output capture on ``str(self.__class__)``, and FSDP2's dynamic subclasses are absent from
        its registry."""
        model = self._backbone_model()
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

    def _apply_torch_compile(self) -> None:
        """Runs after FSDP wrapping and the rest of the distributed setup."""
        training_args = self.args
        if not training_args.torch_compile:
            return

        compile_mode = training_args.torch_compile_mode or DEFAULT_TORCH_COMPILE_MODE
        compile_backend = training_args.torch_compile_backend or "inductor"

        # Packed batches at bs>1 flatten to a variable-width row, so reduce-overhead's CUDA graphs stop applying.
        if (
            compile_mode == "reduce-overhead"
            and getattr(training_args, "packing", False)
            and training_args.per_device_train_batch_size > 1
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
