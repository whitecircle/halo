"""Gradient sync for EP MoE layers: post-accumulate hooks on the FSDP-ignored router and replicated params, inert
during gradient accumulation and in a single process. The expert grads take no hook: their divide, and the
cross-replica sum multiple EP groups add, run once per optimizer step in the trainer's post-backward sweep
(``EPConfig.runs_post_backward_sweep``), which every rank enters."""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.distributed as dist
from accelerate.state import GradientState

from halo.distributed.expert_parallel.config import EPConfig
from halo.distributed.grad_reduce import reduce_grad
from halo.runtime.ranks import has_process_group


def should_sync_gradients() -> bool:
    """True on the final accumulation step; also True when GradientState is uninitialized (no Accelerator)."""
    return GradientState().sync_gradients


def has_grad_sync_peers(ep_config: EPConfig) -> bool:
    """Whether this process has peer ranks to sync gradients with, read live: the default process group is
    initialized independently of the config."""
    return has_process_group() and ep_config.stage_world_size > 1


def build_router_grad_hook(ep_config: EPConfig) -> Callable[[torch.Tensor], None]:
    """Router-param hook: ``all_reduce(SUM)`` over the DP-scope group (under PP this stage's rank block),
    divided by its size ``stage_world_size``."""
    if not has_grad_sync_peers(ep_config):
        return lambda param: None
    divisor, group, fp32 = ep_config.stage_world_size, ep_config.dp_scope_group, ep_config.fp32_grad_reduce

    def router_grad_hook(param):
        if param.grad is not None and should_sync_gradients():
            reduce_grad(param.grad, op=dist.ReduceOp.SUM, divisor=divisor, group=group, fp32=fp32)

    return router_grad_hook


class EPGradSyncMixin:
    """The layer's grad-sync hooks, mixed into :class:`EPMoELayerBase`, whose construction builds the state
    annotated here. Trainers call its two public methods on layers."""

    ep_config: EPConfig
    _grad_sync_hooks: list

    def reattach_router_grad_sync(self) -> int:
        """Hook the trainable router/replicated params again for PEFT ``modules_to_save``, which trains a copy
        the construction hooks never saw; returns how many. Only on the in-backward hook path.
        """
        if not self.ep_config.uses_backward_grad_hooks:
            return 0
        if not has_grad_sync_peers(self.ep_config):
            return 0
        return self._hook_dp_averaged_params()

    def synced_trainable_param_ids(self) -> set:
        """Ids of the trainable params this layer keeps DP/EP-consistent: all of them under the FSDP-sharded
        or deferred sweep, else the experts and their adapters (the sweep's expert divide), the router and
        replicated submodules (the router hook).
        """
        trainable = {id(p) for p in self.parameters() if p.requires_grad}
        if not self.ep_config.uses_backward_grad_hooks:
            return trainable
        ids = {id(p) for _n, p in self.expert_named_params() if p.requires_grad}
        ids |= {id(p) for _n, p in self._dp_averaged_named_params() if p.requires_grad}
        return ids

    def _install_gradient_sync_hooks(self) -> None:
        """Register the router hook on the router and replicated submodules, the only in-backward sync these
        FSDP-ignored params get."""
        # FSDP reduce-scatters fully replicated experts (ep_group_size==1, not pure ETP), and the deferred
        # sweep owns every cross-replica topology and PP rank block: a hook would double-sync or race it.
        if not self.ep_config.uses_backward_grad_hooks:
            return
        if has_grad_sync_peers(self.ep_config):
            self._hook_dp_averaged_params()

    def _hook_dp_averaged_params(self) -> int:
        """Register the router hook on every trainable :meth:`_dp_averaged_named_params` param; returns how
        many."""
        dp_hook = build_router_grad_hook(self.ep_config)
        trainable = [param for _, param in self._dp_averaged_named_params() if param.requires_grad]
        for param in trainable:
            self._grad_sync_hooks.append(param.register_post_accumulate_grad_hook(dp_hook))
        return len(trainable)

    def _dp_averaged_named_params(self) -> list:
        """``(name, param)`` the router hook DP-averages: the live router (a PEFT ``modules_to_save`` copy
        included) and the replicated submodules."""
        router = self._live_router_module()
        router_params = list(router.named_parameters()) if router is not None else []
        return router_params + self.replicated_named_params()
