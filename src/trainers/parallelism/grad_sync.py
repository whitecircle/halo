"""Gradient synchronization for the distributed trainers: the FSDP2 grad-accum window, the QLoRA /
deferred-EP / TP-replicated post-backward sweeps, and the step-pre-hooks that run the sweeps on steps
where clipping is off.
"""

from collections.abc import Callable

import torch
import torch.distributed as dist
import torch.nn as nn
from accelerate.logging import get_logger
from torch.distributed.tensor import DTensor

from src.distributed.fsdp import fsdp2_modules
from src.distributed.grad_reduce import agree_grad_presence, reduce_grads_bucketed
from src.distributed.mesh import MeshDim, mesh_dim_names, tp_group
from src.distributed.tensor_parallel.plain_params import hand_sliced_suffixes, per_head_norm_param_names
from src.trainers.parallelism.ep_layers import require_ep_config
from src.trainers.parallelism.grad_clipping import is_clipping_enabled

logger = get_logger(__name__, log_level="info")


def register_grad_sync_step_hook(
    trainer, flag_attr: str, enabled: Callable[[], bool], action: Callable[[], None]
) -> None:
    """Register ``action`` as a once-per-trainer optimizer step-pre-hook, marked by ``flag_attr``.

    Each grad-sync backstop exists because HF skips ``clip_grad_norm_`` at ``max_grad_norm == 0``, and
    each sits out torch's zero-LR ``_init_optim_state`` step during optimizer-state restore, which carries
    no gradients. ``enabled`` is a callable so its state reads stay behind the ``optimizer is None`` check.
    """
    if trainer.optimizer is None or getattr(trainer, flag_attr, False) or not enabled():
        return

    def _pre_step_hook(optimizer, args, kwargs):
        if getattr(trainer, "_restoring_optimizer_state", False):
            return
        action()

    trainer.optimizer.register_step_pre_hook(_pre_step_hook)
    setattr(trainer, flag_attr, True)


class GradientSyncMixin:
    """The post-backward gradient sync of :class:`DistributedTrainerMixin`."""

    def _install_grad_accum_window(self) -> None:
        """Cache the FSDP2 modules whose post-backward work toggles per grad-accum window: empty, leaving
        :meth:`_set_window_end` inert, unless ``fsdp_reshard_after_backward=False`` or ``fsdp_defer_grad_sync``
        is set on a wrap this trainer applied. Called after every wrap, which is what makes an ``FSDPModule``."""
        config = self.parallelism_config
        toggled = self._fsdp_wrapped and (not config.fsdp_reshard_after_backward or config.fsdp_defer_grad_sync)
        # The outermost model, not ``_top_level_model()``: every mode wraps at or below it.
        self._window_modules = fsdp2_modules(self.model) if toggled else []
        # fully_shard's own default, and what the modules carry until the first microstep toggles them.
        self._window_end_armed = True
        if self._window_modules:
            logger.info(
                f"  ✓ {len(self._window_modules)} FSDP2 modules toggle per grad-accum window "
                f"(fsdp_reshard_after_backward={config.fsdp_reshard_after_backward}, fsdp_defer_grad_sync="
                f"{config.fsdp_defer_grad_sync}): only the window's LAST backward runs the skipped work"
            )

    def _set_window_end(self, is_last: bool) -> None:
        """Arm FSDP2's post-backward reshard and/or gradient reduce for the microstep about to run: off for
        microbatches 1..n-1 of the grad-accum window, on for the last. A reshard left off on the last one
        leaves grad-less unsharded params the optimizer never captured; a reduce left off never reaches
        the sharded grads."""
        if not self._window_modules or is_last == self._window_end_armed:
            return
        config = self.parallelism_config
        for module in self._window_modules:
            if not config.fsdp_reshard_after_backward:
                module.set_reshard_after_backward(is_last, recurse=False)
            if config.fsdp_defer_grad_sync:
                module.set_requires_gradient_sync(is_last, recurse=False)
        self._window_end_armed = is_last

    def _register_grad_sync_step_hooks(self) -> None:
        """Back each post-backward sweep (TP-replicated, deferred EP, QLoRA, in that order) with an
        optimizer step-pre-hook for the steps that never reach the clip. Idempotent."""

        def sync_tp_replicated() -> None:
            if is_clipping_enabled(self.args.max_grad_norm):
                return  # the installed TP / EP clip already synced
            # Over model.parameters(), robust to Muon's nested optimizer.
            self._sync_tp_replicated_grads(self._tp_replicated_params(self.model.parameters()))

        def sweep_deferred_experts() -> None:
            # Gated on whether the sweep ran this step, never on max_grad_norm: at max_grad_norm == 0
            # transformers still calls _get_grad_norm -> the clip, which already swept.
            if getattr(self, "_deferred_sweep_last_step", None) != self.state.global_step:
                self._sync_deferred_expert_grads()

        register_grad_sync_step_hook(
            self, "_tp_grad_sync_hook_registered", lambda: self.parallelism_config.is_tp_mode, sync_tp_replicated
        )
        register_grad_sync_step_hook(
            self,
            "_deferred_ep_grad_sync_hook_registered",
            lambda: self._ep_config is not None and self._ep_config.defer_grad_sync,
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
        """DP-average every trainable grad once per optimizer step, over the membership
        :func:`agree_grad_presence` agrees across the world."""
        if not getattr(self, "_qlora_grad_sync", False):
            return
        current_step = self.state.global_step
        if getattr(self, "_qlora_sweep_last_step", None) == current_step:
            return
        self._qlora_sweep_last_step = current_step
        params = [p for _name, p in self._top_level_model().named_parameters() if p.requires_grad]
        grads = [p.grad for p in agree_grad_presence(params)]
        reduce_grads_bucketed(grads, op=dist.ReduceOp.AVG, fp32=self.parallelism_config.fp32_grad_reduce)

    def _sync_deferred_expert_grads(self) -> None:
        """COLLECTIVE: the post-backward EP grad sync of every deferred topology (``EPConfig.defer_grad_sync``).

        Deferred because in-backward hooks would race the EP group's DeepEP combine (multi-group EP
        across nodes) or re-fire per microbatch (PP). Each grad ends as the DP average over its rank
        block, by the legs of :meth:`_classify_deferred_params`. Collectives issue in a fixed
        ``named_parameters`` order, each bucket's membership agreed over its own reduce group. Both
        callers run only with EP layers present, so a missing EP config raises.
        """
        ep_cfg = require_ep_config(self._ep_config)
        if not ep_cfg.defer_grad_sync:
            return
        # The expert leg (SUM/world_size) is not idempotent: a second pass halves every expert grad.
        current_step = self.state.global_step
        if getattr(self, "_deferred_sweep_last_step", None) == current_step:
            raise RuntimeError(
                f"_sync_deferred_expert_grads ran twice at global_step={current_step}: the expert "
                f"SUM/world_size leg is not idempotent — a second pass halves every expert gradient."
            )
        self._deferred_sweep_last_step = current_step
        expert_params, world_params, replica_params = self._classify_deferred_params(ep_cfg)
        replica_group = ep_cfg.expert_replica_group  # None ⇔ one EP group per rank block (R==1)
        # ETP partners slice one expert and share a batch, so the DP divisor drops the ETP factor.
        expert_divisor = ep_cfg.stage_world_size // ep_cfg.expert_tp_size
        fp32 = ep_cfg.fp32_grad_reduce

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
            # A contract check, not a fallback: group=None is the world group, which under PP spans
            # stages holding different layers.
            raise RuntimeError(
                f"Deferred cross-replica DP sync collected {len(replica_params)} sharded expert "
                f"gradient(s) but has no expert replica group to average them over. Reducing over "
                f"the default world group would average across pipeline stages / dispatch groups "
                f"that hold different parameters."
            )
        replica_grads = [p.grad._local_tensor for p in agree_grad_presence(replica_params, replica_group)]
        reduce_grads_bucketed(replica_grads, op=dist.ReduceOp.AVG, group=replica_group, fp32=fp32)

    def _classify_deferred_params(self, ep_cfg) -> tuple[list[nn.Parameter], list[nn.Parameter], list[nn.Parameter]]:
        """The trainable parameters by deferred-sweep leg, as ``(expert, world, replica)``: expert FFN shards
        (combine already summed the dispatch group: ``SUM`` over the replicas, then the DP divide), router /
        replicated EP / plain non-expert params (``AVG`` over the DP scope), and non-expert FSDP shards
        sharded within the EP group (``is_deferred_dp``: a replica-group ``AVG``; otherwise reduce-scatter
        already averaged them and they sit out)."""
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
        """``params`` minus the EP/ETP expert shards, which are rank-owned rather than TP replicas;
        selected structurally, never by rank-local grad presence."""
        expert_ids = self._get_sharded_expert_param_ids()
        return [p for p in params if id(p) not in expert_ids]

    def _sync_tp_replicated_grads(self, params: list) -> None:
        """Reduce the grads the TP graph leaves un-reduced, once per optimizer step; no-op without a TP group.

        Plain replicated params (norms, embeddings, ``lm_head``, pure-TP replicated experts) are
        AVG-reduced, since nothing else reduces their per-rank grads; the per-head attention norms
        (:meth:`_tp_per_head_norm_param_ids`) are SUM-reduced, since each rank's gradient covers its own
        heads. Hand-sliced plain shards (:meth:`_tp_sharded_plain_param_ids`) sit out both. Membership is
        agreed over the TP group: a router tie-break can leave grad=None on one rank only.
        """
        group = tp_group(self._device_mesh)
        if group is None or dist.get_world_size(group=group) <= 1:
            return
        # Once per step: max_grad_norm == 0 still reaches the clip via _get_grad_norm, and the SUM is not idempotent.
        step = getattr(getattr(self, "state", None), "global_step", None)
        if step is not None and getattr(self, "_tp_sync_last_step", None) == step:
            return
        self._tp_sync_last_step = step
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
                # Plan-sharded: the DTensor graph reduces these over the TP mesh itself. A DTensor
                # without a tp dim is FSDP's dp shard of a TP-replicated param; its reduce-scatter
                # runs per TP column, so without the AVG below the TP siblings' replicas drift apart.
                continue
            candidates.append(p)
        replicated_grads = []
        per_head_norm_grads = []
        for p in agree_grad_presence(candidates, group):
            # TP siblings hold the same dp shard of the same replica, so reducing local shards is the
            # exact cross-replica sync.
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
        """Ids of the attention norms whose gradient covers only this rank's heads.

        Registered by :func:`apply_tp_to_attention_only` as names, since FSDP2 replaces Parameter
        objects after TP is applied. Their true gradient is the sum over the TP group: a colwise
        projection's ``use_local_output=True`` ends the DTensor graph, so nothing else reduces them.
        """

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
