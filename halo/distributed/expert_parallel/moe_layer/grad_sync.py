"""Gradient sync for EP MoE layers: post-accumulate hooks on the FSDP-ignored router and replicated params. Expert
grads take no hook; their divide and cross-replica sum run once per optimizer step in the trainer's post-backward
sweep, which every rank enters."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist
from accelerate.state import GradientState

from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.grad_reduce import reduce_grad


def should_sync_gradients() -> bool:
    """True on the final accumulation step; also True when GradientState is uninitialized (no Accelerator)."""
    return GradientState().sync_gradients


def build_dp_average_grad_hook(ep_config: EPConfig) -> Callable[[torch.Tensor], None]:
    """Router-param hook: ``all_reduce(SUM)`` over the DP-scope group (under PP this stage's rank block),
    divided by its size ``stage_world_size``."""
    divisor, group, fp32 = ep_config.stage_world_size, ep_config.dp_scope_group, ep_config.fp32_grad_reduce

    def router_grad_hook(param):
        if param.grad is not None and should_sync_gradients():
            reduce_grad(param.grad, op=dist.ReduceOp.SUM, divisor=divisor, group=group, fp32=fp32)

    return router_grad_hook


class EPGradSyncMixin:
    """The layer's grad-sync hooks, mixed into :class:`EPMoELayerBase`."""

    ep_config: EPConfig
    _grad_sync_hooks: list

    def reattach_dp_average_grad_hooks(self) -> int:
        """Hook the trainable router/replicated params again for PEFT ``modules_to_save``, which trains a copy the
        construction hooks never saw; returns how many."""
        return self._hook_dp_averaged_params()

    def synced_trainable_param_ids(self) -> set:
        """Ids of the trainable params this layer keeps DP/EP-consistent: all under the FSDP-sharded or deferred sweep,
        else the experts and adapters (the sweep's divide), router and replicated submodules (the router hook)."""
        trainable = {id(p) for p in self.parameters() if p.requires_grad}
        if not self.ep_config.uses_backward_grad_hooks:
            return trainable
        ids = {id(p) for _n, p in self.expert_named_params() if p.requires_grad}
        ids |= {id(p) for _n, p in self._dp_averaged_named_params() if p.requires_grad}
        return ids

    def _install_gradient_sync_hooks(self) -> None:
        self._hook_dp_averaged_params()

    def _hook_dp_averaged_params(self) -> int:
        # FSDP reduce-scatters fully replicated experts (ep_group_size==1, not pure ETP), and the deferred
        # sweep owns every cross-replica topology and PP rank block: a hook would double-sync or race it.
        if not (self.ep_config.uses_backward_grad_hooks and self.ep_config.has_grad_sync_peers):
            return 0
        dp_hook = build_dp_average_grad_hook(self.ep_config)
        trainable = [param for _, param in self._dp_averaged_named_params() if param.requires_grad]
        for param in trainable:
            self._grad_sync_hooks.append(param.register_post_accumulate_grad_hook(dp_hook))
        return len(trainable)

    def _dp_averaged_named_params(self) -> list:
        """What the router hook DP-averages: the live router, ``modules_to_save`` copy included, and replicas."""
        router = self._live_router_module()
        router_params = list(router.named_parameters()) if router is not None else []
        return router_params + self.replicated_named_params()
