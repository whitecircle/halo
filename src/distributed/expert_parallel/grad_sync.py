"""Gradient synchronization hooks used by EP MoE layers.

Two post-accumulation grad hooks, built straight off :class:`EPConfig`: a router hook (DP-average the
router grad, since the router runs on the local sequence chunk) and an expert hook (scale expert grads
to match DDP/FSDP averaging). Both no-op during gradient accumulation.

The expert hook carries no collective: the cross-replica sum multiple EP groups need is deferred to the
trainer's post-backward sweep (``EPConfig.defer_grad_sync``), which every rank enters structurally.
"""

from __future__ import annotations

import torch.distributed as dist
from accelerate.state import GradientState

from src.distributed.expert_parallel.config import EPConfig
from src.distributed.grad_reduce import reduce_grad


def should_sync_gradients() -> bool:
    """True on the final accumulation step; also True when GradientState is uninitialized (no Accelerator)."""
    return GradientState().sync_gradients


def has_grad_sync_peers(ep_config: EPConfig) -> bool:
    """Whether this process has peer ranks to sync gradients with.

    A live-process predicate, not config state: the default group is initialized independently of the
    config's construction, so it is read where the hook is built (and where the layer decides whether
    to register one) rather than latched. Single-process runs have no peers and no default group, so
    every hook must be inert there.
    """
    return dist.is_available() and dist.is_initialized() and ep_config.world_size > 1


def _noop_sync_hook(param):
    """Single-process grad hook: no peer ranks, so there is nothing to reduce or rescale."""


def _make_hook(*, divisor: int, reduce: bool, group=None, fp32: bool = False):
    """Post-accumulation grad hook: optionally ``all_reduce(SUM)`` over ``group`` then scale by
    ``1/divisor``. ``reduce=False`` = the expert case (the shard already holds its full grad, so only
    the DP-averaging divide is needed, and ``group``/``fp32`` do not apply)."""

    def sync_hook(param):
        if param.grad is None or not should_sync_gradients():
            return
        if reduce:
            reduce_grad(param.grad, op=dist.ReduceOp.SUM, divisor=divisor, group=group, fp32=fp32)
        else:
            if not param.grad.is_contiguous():
                param.grad = param.grad.contiguous()
            param.grad.div_(divisor)

    return sync_hook


def create_router_grad_hook(ep_config: EPConfig):
    """Router-param grad sync: ``all_reduce(SUM)`` over the DP-scope group (the world group without
    PP; under PP this stage's rank block only) then divide by ``world_size`` — the rank-block width,
    which is that group's size.

    No-op in single-process mode (no peers to average with)."""
    if not has_grad_sync_peers(ep_config):
        return _noop_sync_hook
    return _make_hook(
        divisor=ep_config.world_size,
        group=ep_config.dp_scope_group,
        reduce=True,
        fp32=ep_config.fp32_grad_reduce,
    )


def create_expert_grad_hook(ep_config: EPConfig):
    """Expert-param grad sync scaled to match FSDP/DDP averaging: divide by
    ``world_size // expert_tp_size`` (ETP partners share a batch, so they are not DP replicas).

    Collective-free by construction. The cross-replica sum that multiple EP groups need cannot ride
    this hook: it fires only where a grad was accumulated, and a rank whose dispatch delivered no
    tokens for a layer never touches its expert weights — its replicas would then hang inside a
    collective it never enters. Those topologies defer to the trainer's post-backward sweep, which
    contributes every param structurally. No-op in single-process mode (divisor 1, nothing to scale).
    """
    if not has_grad_sync_peers(ep_config):
        return _noop_sync_hook
    if ep_config.needs_expert_grad_sync:
        raise RuntimeError(
            "EP expert gradients need a cross-replica all-reduce (multiple EP groups) but the "
            "in-backward hook path was selected. That reduce fires only on ranks whose experts "
            "received tokens, so a zero-token rank would leave its replicas hanging in it. "
            "EPConfig.defer_grad_sync must cover every multi-EP-group topology."
        )
    return _make_hook(divisor=ep_config.world_size // ep_config.expert_tp_size, reduce=False)
