"""Gradient synchronization: the FSDP2 grad-accum window and post-backward completion, the post-backward sweeps, and
the step-pre-hooks that run them on a step that missed the clip."""

import logging
from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor import DTensor

from halo.distributed.fsdp import (
    complete_fsdp2_post_backward,
    fsdp2_modules,
    fsdp2_param_groups,
    reject_unfinished_fsdp2_backward,
)
from halo.distributed.grad_reduce import agree_grad_presence, reduce_grads_bucketed
from halo.distributed.mesh import MeshDim, mesh_dim_names, tp_group
from halo.distributed.tensor_parallel.plain_params import hand_sliced_suffixes, per_head_norm_param_names
from halo.runtime.ranks import is_global_main_process
from halo.trainers.parallelism.ep_layers import require_ep_config
from halo.trainers.parallelism.grad_clipping import is_clipping_enabled

logger = logging.getLogger(__name__)


def register_grad_sync_step_hook(
    trainer, flag_attr: str, enabled: Callable[[], bool], action: Callable[[], None]
) -> None:
    """Sweeps live in the patched ``clip_grad_norm_``, which HF reaches at ``max_grad_norm <= 0`` only through its
    unclipped norm pass, so a backstop acts only on a step that missed it. Each sits out torch's zero-LR
    ``_init_optim_state`` step."""
    if trainer.optimizer is None or getattr(trainer, flag_attr, False) or not enabled():
        return

    def _pre_step_hook(optimizer, args, kwargs):
        if getattr(trainer, "_materializing_optimizer_state", False):
            return
        action()

    trainer.optimizer.register_step_pre_hook(_pre_step_hook)
    setattr(trainer, flag_attr, True)


def same_optimizer_step(marker: tuple | None, state) -> bool:
    """``marker`` is a sweep's ``(TrainerState, global_step)``. State compares by identity: ``global_step`` restarts
    with every ``train()``'s fresh state, and a step taken before one must not mark the run's first step done."""
    return marker is not None and marker[0] is state and marker[1] == state.global_step


def claim_optimizer_step(trainer, marker_attr: str) -> bool:
    """False when the step already carries this mark, which a sweep that is not idempotent must not run under twice."""
    state = trainer.state
    if same_optimizer_step(getattr(trainer, marker_attr, None), state):
        return False
    setattr(trainer, marker_attr, (state, state.global_step))
    return True


class GradientSyncMixin:
    def _install_grad_accum_window(self) -> None:
        """Empty unless ``fsdp_reshard_after_backward=False`` or ``fsdp_defer_grad_sync`` is set on this trainer's
        wrap. Called after every wrap, which is what makes an ``FSDPModule``."""
        config = self.parallelism_config
        toggled = self._fsdp_wrapped and (not config.fsdp_reshard_after_backward or config.fsdp_defer_grad_sync)
        # The outermost model, not ``_top_level_model()``: every mode wraps at or below it.
        self._window_modules = fsdp2_modules(self.model) if toggled else []
        # fully_shard's own default, and what the modules carry until the first microstep toggles them.
        self._window_end_armed = True
        if self._window_modules and is_global_main_process():
            logger.info(
                f"  ✓ {len(self._window_modules)} FSDP2 modules toggle per grad-accum window "
                f"(fsdp_reshard_after_backward={config.fsdp_reshard_after_backward}, fsdp_defer_grad_sync="
                f"{config.fsdp_defer_grad_sync}): only the window's LAST backward runs the skipped work"
            )

    def _set_window_end(self, is_last: bool) -> None:
        """Off for microbatches 1..n-1, on for the last. A reshard left off on the last leaves grad-less unsharded
        params the optimizer never captured; a reduce left off never reaches the sharded grads."""
        if not self._window_modules or is_last == self._window_end_armed:
            return
        config = self.parallelism_config
        for module in self._window_modules:
            if not config.fsdp_reshard_after_backward:
                module.set_reshard_after_backward(is_last, recurse=False)
            if config.fsdp_defer_grad_sync:
                module.set_requires_gradient_sync(is_last, recurse=False)
        self._window_end_armed = is_last

    def _finish_fsdp2_backward(self, model: nn.Module) -> None:
        """COLLECTIVE where a group reduces. Complete the post-backward a reentrant recompute skipped, then refuse one
        still undone, before the clip."""
        cached_model, groups = getattr(self, "_fsdp2_groups_of", (None, None))
        if cached_model is not model:
            groups = fsdp2_param_groups(model)
            self._fsdp2_groups_of = (model, groups)
        if not groups:
            return
        reduced = complete_fsdp2_post_backward(groups)
        reject_unfinished_fsdp2_backward(groups)
        if reduced and not getattr(self, "_fsdp2_completion_logged", False) and is_global_main_process():
            self._fsdp2_completion_logged = True
            logger.info(
                f"  FSDP2: {reduced} shard group(s) recomputed more than once by reentrant gradient checkpointing "
                f"had their post-backward completed after the backward (one extra reduce-scatter each per backward)"
            )

    def _register_grad_sync_step_hooks(self) -> None:
        """Back each sweep (TP-replicated, deferred EP, QLoRA, in that order) with a step-pre-hook for a step that
        never reaches the clip. Idempotent."""

        def sync_tp_replicated() -> None:
            if is_clipping_enabled(self.args.max_grad_norm):
                return  # the installed TP / EP clip already synced
            # Over model.parameters(), since Muon nests its optimizer.
            self._sync_tp_replicated_grads(self._tp_replicated_params(self.model.parameters()))

        def sweep_deferred_experts() -> None:
            # Gated on whether the sweep ran this step, never on max_grad_norm: at max_grad_norm == 0
            # transformers still calls _get_grad_norm -> the clip, which already swept.
            if not same_optimizer_step(getattr(self, "_deferred_sweep_last_step", None), self.state):
                self._sync_deferred_expert_grads()

        register_grad_sync_step_hook(
            self, "_tp_grad_sync_hook_registered", lambda: self.parallelism_config.is_tp_mode, sync_tp_replicated
        )
        register_grad_sync_step_hook(
            self,
            "_deferred_ep_grad_sync_hook_registered",
            lambda: self._ep_config is not None and self._ep_config.runs_post_backward_sweep,
            sweep_deferred_experts,
        )
        # The QLoRA sweep's per-step marker makes the clip's second call a no-op.
        register_grad_sync_step_hook(
            self,
            "_qlora_grad_sync_hook_registered",
            lambda: getattr(self, "_qlora_grad_sync", False),
            self._sync_qlora_grads,
        )

    def _sync_qlora_grads(self) -> None:
        """DP-average every trainable grad once per optimizer step, over the membership :func:`agree_grad_presence`
        agrees."""
        if not getattr(self, "_qlora_grad_sync", False):
            return
        if not claim_optimizer_step(self, "_qlora_sweep_last_step"):
            return
        params = [p for _name, p in self._top_level_model().named_parameters() if p.requires_grad]
        grads = [p.grad for p in agree_grad_presence(params)]
        reduce_grads_bucketed(grads, op=dist.ReduceOp.AVG, fp32=self.parallelism_config.fp32_grad_reduce)

    def _sync_deferred_expert_grads(self) -> None:
        """COLLECTIVE, once per optimizer step. Deferred because in-backward hooks would race the EP group's DeepEP
        combine, re-fire per microbatch (PP), or re-fire per nested backward of a reentrant recompute, which the expert
        divide cannot absorb."""
        ep_cfg = require_ep_config(self._ep_config)
        if not ep_cfg.runs_post_backward_sweep:
            return
        # The expert leg (SUM/world_size) is not idempotent: a second pass halves every expert grad.
        if not claim_optimizer_step(self, "_deferred_sweep_last_step"):
            raise RuntimeError(
                f"_sync_deferred_expert_grads ran twice at global_step={self.state.global_step}: the expert "
                f"SUM/world_size leg is not idempotent — a second pass halves every expert gradient."
            )
        expert_params, world_params, replica_params = self._classify_deferred_params(ep_cfg)
        replica_group = ep_cfg.expert_replica_group  # None ⇔ one EP group per rank block (R==1)
        # ETP partners slice one expert and share a batch, so the DP divisor drops the ETP factor.
        expert_divisor = ep_cfg.stage_world_size // ep_cfg.expert_tp_size
        fp32 = ep_cfg.fp32_grad_reduce

        if not ep_cfg.sweep_reduces_across_replicas:
            # One EP group, its router averaged by the in-backward hook: the combine already summed every rank's
            # tokens, so the experts take only the divide, with no collective. Per-tensor div_, whose CUDA kernel
            # multiplies by the reciprocal unlike _foreach_div_'s true divide, is the rounding single-group expert
            # grads train with.
            for grad in (param.grad for param in expert_params if param.grad is not None):
                grad.div_(expert_divisor)
            return

        if replica_group is not None:
            expert_grads = [p.grad for p in agree_grad_presence(expert_params, replica_group)]
            reduce_grads_bucketed(
                expert_grads, op=dist.ReduceOp.SUM, divisor=expert_divisor, group=replica_group, fp32=fp32
            )
        else:
            # R==1: the combine already summed every replica, so only the storage-dtype divide remains,
            # and with no collective an expert this rank never touched keeps grad None.
            expert_grads = [p.grad for p in expert_params if p.grad is not None]
            if expert_grads:
                torch._foreach_div_(expert_grads, expert_divisor)
        # DP-scope AVG, not data_parallel_size: the per-rank loss is mean-normalized.
        world_grads = [p.grad for p in agree_grad_presence(world_params, ep_cfg.dp_scope_group)]
        reduce_grads_bucketed(world_grads, op=dist.ReduceOp.AVG, group=ep_cfg.dp_scope_group, fp32=fp32)
        if replica_params and replica_group is None:
            # group=None is the world group, which under PP spans stages holding different layers.
            raise RuntimeError(
                f"Deferred cross-replica DP sync collected {len(replica_params)} sharded expert "
                f"gradient(s) but has no expert replica group to average them over. Reducing over "
                f"the default world group would average across pipeline stages / dispatch groups "
                f"that hold different parameters."
            )
        replica_grads = [p.grad._local_tensor for p in agree_grad_presence(replica_params, replica_group)]
        reduce_grads_bucketed(replica_grads, op=dist.ReduceOp.AVG, group=replica_group, fp32=fp32)

    def _classify_deferred_params(self, ep_cfg) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
        """``(expert, world, replica)``: expert FFN shards (SUM over replicas, then the DP divide), router / replicated
        EP / plain non-expert params (AVG over the DP scope), and non-expert FSDP shards within the EP group
        (``is_deferred_dp``: replica-group AVG)."""
        expert_ids = self._get_sharded_expert_param_ids()
        ep_ids = self._get_ep_param_ids()
        expert_params: list[nn.Parameter] = []
        world_params: list[nn.Parameter] = []
        replica_params: list[nn.Parameter] = []
        for name, param in self._top_level_model().named_parameters():
            if not param.requires_grad:
                continue
            if id(param) in expert_ids:
                expert_params.append(param)
            elif id(param) in ep_ids or not isinstance(param.data, DTensor):
                world_params.append(param)
            elif ep_cfg.is_deferred_dp:
                # Valid only when sharded over the EP group (1D mesh); any other mesh averages different slices.
                mesh = param.data.device_mesh
                if mesh.ndim != 1 or mesh.size() != ep_cfg.ep_group_size:
                    raise RuntimeError(
                        f"Deferred cross-replica DP sync expects FSDP sharding over the EP group "
                        f"(1D mesh of {ep_cfg.ep_group_size}), got a {tuple(mesh.shape)} mesh for "
                        f"'{name}'. This topology must not set is_deferred_dp."
                    )
                replica_params.append(param)
        return expert_params, world_params, replica_params

    def _tp_replicated_params(self, params) -> list[nn.Parameter]:
        """``params`` minus the rank-owned EP/ETP expert shards; selected structurally, never by rank-local grad
        presence."""
        expert_ids = self._get_sharded_expert_param_ids()
        return [p for p in params if id(p) not in expert_ids]

    def _sync_tp_replicated_grads(self, params: list) -> None:
        """Plain replicated params are AVG-reduced; per-head attention norms are SUM-reduced, since each rank's
        gradient covers its own heads; hand-sliced plain shards sit out. Membership is agreed over the TP group: a
        router tie-break can leave grad=None on one rank only."""
        group = tp_group(self._device_mesh)
        if group is None or dist.get_world_size(group=group) <= 1:
            return
        # Once per step: max_grad_norm == 0 still reaches the clip via _get_grad_norm, and the SUM is not idempotent.
        if not claim_optimizer_step(self, "_tp_sync_last_step"):
            return
        sharded_plain_ids = self._tp_sharded_plain_param_ids()
        per_head_norm_ids = self._tp_per_head_norm_param_ids()
        candidates = []
        for p in params:
            if not p.requires_grad or id(p) in sharded_plain_ids:
                continue
            if (
                id(p) not in per_head_norm_ids
                and isinstance(p.data, DTensor)
                and MeshDim.TP in mesh_dim_names(p.data.device_mesh)
            ):
                # Plan-sharded: the TP forward's collectives leave each rank's gradient complete for its shard. A
                # DTensor without a tp dim is FSDP's dp shard of a TP-replicated param, whose reduce-scatter runs per
                # TP column, so without the AVG below the siblings' replicas drift apart.
                continue
            candidates.append(p)
        replicated_grads = []
        per_head_norm_grads = []
        for p in agree_grad_presence(candidates, group):
            # TP siblings hold the same dp shard of the same replica, so reducing local shards is the exact sync.
            local = p.grad.to_local() if isinstance(p.grad, DTensor) else p.grad
            (per_head_norm_grads if id(p) in per_head_norm_ids else replicated_grads).append(local)
        fp32 = bool(self.parallelism_config.fp32_grad_reduce)
        reduce_grads_bucketed(replicated_grads, op=dist.ReduceOp.AVG, group=group, fp32=fp32)
        reduce_grads_bucketed(per_head_norm_grads, op=dist.ReduceOp.SUM, group=group, fp32=fp32)

    def _tp_sharded_plain_param_ids(self) -> set[int]:
        """Ids of the hand-sliced params (GptOss sinks) TP sharded as plain tensors: disjoint slices that look
        like replicas, whose grads must never be TP-averaged and whose norms sum over the TP group."""

        def build() -> set[int]:
            model = self._top_level_model()
            suffixes = hand_sliced_suffixes(model)
            return {id(param) for name, param in model.named_parameters() if suffixes and name.endswith(suffixes)}

        return self._memoized_param_ids("tp_sharded_plain", build)

    def _tp_per_head_norm_param_ids(self) -> set[int]:
        """Ids of the attention norms whose gradient covers only this rank's heads; their true gradient is the sum over
        the TP group. Registered as names, since FSDP2 replaces Parameter objects after TP is applied."""

        def build() -> set[int]:
            model = self._top_level_model()
            names = set(per_head_norm_param_names(model))
            ids = {id(param) for name, param in model.named_parameters() if name in names}
            if names and not ids:
                raise RuntimeError(
                    f"TP registered {len(names)} per-head attention norms for the step-time gradient SUM, but "
                    f"none of those names resolve against the trainer's model (e.g. {sorted(names)[:3]}). "
                    "Their gradients would stay divided across the TP group and the norms would train on a "
                    "1/tensor_parallel_size gradient. The registry is keyed on the model "
                    "apply_tp_to_attention_only saw."
                )
            return ids

        return self._memoized_param_ids("tp_per_head_norms", build)
