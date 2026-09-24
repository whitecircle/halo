"""The router aux-loss gradient of reentrant-checkpointed MoE layers.

transformers computes a MoE family's router aux loss from ``outputs.router_logits``, which forward
hooks on the declared router collect while the backbone runs. A reentrant checkpoint runs a layer's
original forward under ``no_grad`` and re-runs it in backward where nothing collects, so every
checkpointed layer hands the aux loss a tensor with no graph behind it: the term reaches the logged
loss and adds no router gradient.

:func:`install_router_aux_gradient` restores that gradient in two hops without re-deriving the loss.
At the backbone output, each collected tensor that a checkpoint's ``no_grad`` pass produced is
swapped for a carrier holding the same values, whose backward keeps the gradient the loss hands it:
the exact per-layer gradient of the family's own aux loss, with the pooled load, the attention mask,
the coefficient and the trainer's loss scaling all applied, whichever consumer added the term. In the
recompute, an identity node on the output of the block owning the router hands the kept gradient to
the router's live logits. A tensor collected with a graph (no checkpoint, a non-reentrant one, a
layer ``every_n_layers`` leaves unchecked) keeps its native path, so no layer's term counts twice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn as nn
from peft.utils.other import AuxiliaryTrainingWrapper

from src.models.moe_balancing import ROUTER_LOGITS_KEY, declared_routers, declares_router_logits

# The attribute marking a backbone as already carrying the hooks, so a second install is a no-op.
_INSTALLED_ATTR = "_router_aux_gradient"

# Module levels between a router and the block that calls it which never run the block's forward:
# module containers, and PEFT's ``modules_to_save`` wrapper around a trained router copy.
_PASS_THROUGH_LEVELS = (nn.ModuleDict, nn.ModuleList, AuxiliaryTrainingWrapper)


def _graph_task() -> int:
    """The autograd graph task executing on this thread, ``-1`` outside backward.

    A module forward running while it is set is a checkpoint recompute; the id also names the one
    outer backward that a carrier's gradient and the recompute consuming it both belong to.
    """
    return torch._C._current_graph_task_id()


@dataclass(eq=False)
class _BackboneForward:
    """One grad-enabled backbone forward: the collected tensors a checkpoint's ``no_grad`` pass
    produced, by ``id`` to their router, then the gradient the loss hands back for each router."""

    producers: dict[int, nn.Module] = field(default_factory=dict)
    gradients: dict[nn.Module, torch.Tensor] = field(default_factory=dict)


class _KeepCollectedGradient(torch.autograd.Function):
    """Identity on a collected ``no_grad`` router-logits tensor whose backward keeps its gradient.

    ``anchor`` is the backbone output. Taking it as an input gives the carrier a graph, and it makes
    the backbone's own backward wait for every carrier, so each kept gradient exists before the first
    checkpointed layer recomputes rather than by the engine's scheduling order.
    """

    @staticmethod
    def forward(ctx, logits, anchor, keeper, record, router):
        ctx.keeper, ctx.record, ctx.router = keeper, record, router
        return logits.detach()

    @staticmethod
    def backward(ctx, grad):
        ctx.keeper.keep(ctx.record, ctx.router, grad)
        return None, None, None, None, None


class _InjectRouterGradient(torch.autograd.Function):
    """Identity on a block output that, in backward, hands ``gradient`` to the router's ``logits``."""

    @staticmethod
    def forward(ctx, activation, logits, gradient):
        ctx.save_for_backward(gradient)
        return activation.view_as(activation)

    @staticmethod
    def backward(ctx, grad_output):
        (gradient,) = ctx.saved_tensors
        return grad_output, gradient, None


def _inject(output, logits: torch.Tensor, gradient: torch.Tensor):
    """``output`` with the injection node on its hidden-state tensor (a block returning a tuple leads
    with it, as GPT-OSS's ``(hidden_states, router_scores)`` does)."""
    if isinstance(output, torch.Tensor):
        return _InjectRouterGradient.apply(output, logits, gradient)
    if type(output) is tuple and output and isinstance(output[0], torch.Tensor):
        return (_InjectRouterGradient.apply(output[0], logits, gradient), *output[1:])
    raise TypeError(
        f"The block owning a checkpointed router returned {type(output).__name__}, not a tensor or a "
        f"tuple led by one, so the router aux-loss gradient has no hidden state to ride back on."
    )


def _owning_block(root: nn.Module, router_name: str) -> nn.Module:
    """The nearest ancestor of the router at ``router_name`` whose forward calls it."""
    parts = router_name.split(".")
    for depth in range(len(parts) - 1, -1, -1):
        owner = root.get_submodule(".".join(parts[:depth]))
        if not isinstance(owner, _PASS_THROUGH_LEVELS):
            return owner
    raise ValueError(f"router {router_name!r} has no owning block under {type(root).__name__}")


class _RouterAuxGradient:
    """The hooks and per-backward state of one collecting backbone.

    Routers and their owning blocks are resolved at the first forward rather than at install:
    installation precedes the trainer, and a PEFT ``modules_to_save`` wrap applied there replaces
    the router the forward calls with a copy.
    """

    def __init__(self, backbone: nn.Module):
        self._backbone = backbone
        self._owned: dict[nn.Module, list[nn.Module]] | None = None
        self._logits_index: dict[nn.Module, int] = {}
        self._names: dict[nn.Module, str] = {}
        self._open: _BackboneForward | None = None
        self._live: dict[nn.Module, torch.Tensor] = {}
        self._task: int | None = None
        self._task_forwards: list[_BackboneForward] = []
        backbone.register_forward_pre_hook(self._on_backbone_input)
        backbone.register_forward_hook(self._on_backbone_output, always_call=True)

    def _resolve_routers(self) -> None:
        self._owned = {}
        for router in declared_routers(self._backbone):
            owner = _owning_block(self._backbone, router.name)
            self._owned.setdefault(owner, []).append(router.module)
            self._logits_index[router.module] = router.logits_index
            self._names[router.module] = router.name
            router.module.register_forward_hook(self._on_router)
        for owner in self._owned:
            owner.register_forward_hook(self._on_owner)

    def _on_backbone_input(self, backbone, args):
        if self._owned is None:
            self._resolve_routers()
        self._live.clear()
        self._open = _BackboneForward() if torch.is_grad_enabled() else None

    def _on_router(self, router, args, output):
        logits = output[self._logits_index[router]] if isinstance(output, (tuple, list)) else output
        if not isinstance(logits, torch.Tensor):
            return
        if _graph_task() != -1:
            self._live[router] = logits
        elif self._open is not None and not torch.is_grad_enabled():
            # A no_grad router inside a grad-enabled backbone forward: a reentrant checkpoint's first pass.
            self._open.producers[id(logits)] = router

    def _on_backbone_output(self, backbone, args, output):
        record, self._open = self._open, None
        if record is None or not record.producers or output is None:
            return
        collected = getattr(output, ROUTER_LOGITS_KEY, None)
        anchor = getattr(output, "last_hidden_state", None)
        if not collected or anchor is None or not anchor.requires_grad:
            return
        carried, seen = [], set()
        for logits in collected:
            router = record.producers.get(id(logits))
            if router is None:
                if not logits.requires_grad:
                    raise RuntimeError(
                        f"A router_logits tensor was collected under a checkpoint's no_grad pass from a "
                        f"router these hooks never saw (resolved at the first forward: "
                        f"{sorted(self._names.values())}), so its aux-loss gradient has no way back. "
                        f"Was a router module replaced after the model's first forward?"
                    )
                carried.append(logits)
                continue
            if router in seen:
                raise RuntimeError(
                    f"Router {self._names[router]} ran twice in one checkpointed forward; its "
                    f"recompute could not tell which collected aux-loss gradient belongs to which call."
                )
            seen.add(router)
            carried.append(_KeepCollectedGradient.apply(logits, anchor, self, record, router))
        output[ROUTER_LOGITS_KEY] = tuple(carried)

    def keep(self, record: _BackboneForward, router: nn.Module, gradient: torch.Tensor) -> None:
        """Hold ``gradient`` for ``router``'s recompute in the backward now running."""
        task = _graph_task()
        if task != self._task:
            self._task, self._task_forwards = task, []
            torch.autograd.Variable._execution_engine.queue_callback(partial(self._finish, task))
        if record not in self._task_forwards:
            self._task_forwards.append(record)
        record.gradients[router] = gradient

    def _take(self, router: nn.Module) -> torch.Tensor | None:
        if self._task is None or self._task != _graph_task():
            return None
        holders = [record for record in self._task_forwards if router in record.gradients]
        if len(holders) > 1:
            raise RuntimeError(
                f"One backward carries the router aux loss of {len(holders)} forwards through the "
                f"checkpointed router {self._names[router]}; its recompute cannot tell whose "
                f"gradient it owes. Backpropagate each forward's loss separately."
            )
        return holders[0].gradients.pop(router) if holders else None

    def _on_owner(self, owner, args, output):
        if _graph_task() == -1:
            return None
        for router in self._owned[owner]:
            logits = self._live.pop(router, None)
            gradient = self._take(router) if logits is not None else None
            if gradient is None or not logits.requires_grad:
                continue
            if gradient.shape != logits.shape:
                raise RuntimeError(
                    f"Router {self._names[router]} recomputed logits of shape {tuple(logits.shape)}, but "
                    f"its collected aux-loss gradient has shape {tuple(gradient.shape)}."
                )
            output = _inject(output, logits, gradient)
        return output

    def _finish(self, task: int) -> None:
        """End of the backward: every kept gradient must have reached its router's recompute."""
        if task != self._task:
            return
        unreached = sorted(self._names[router] for record in self._task_forwards for router in record.gradients)
        self._task, self._task_forwards = None, []
        self._live.clear()
        if unreached:
            raise RuntimeError(
                f"The router aux loss handed a gradient to {unreached}, collected under a checkpoint's "
                f"no_grad pass, but no recompute of the block owning those routers came to take it: "
                f"their aux-loss gradient was dropped. Expected every such router to re-run inside "
                f"its checkpoint's recompute."
            )


def _collecting_backbones(model: nn.Module) -> list[nn.Module]:
    """The module whose forward collects each declared router's logits: its innermost ancestor
    declaring ``router_logits`` (a causal-LM head inherits the declaration without collecting)."""
    declaring = {name: module for name, module in model.named_modules() if declares_router_logits(module)}
    backbones: dict[str, nn.Module] = {}
    for router in declared_routers(model):
        holders = [name for name in declaring if not name or router.name.startswith(f"{name}.")]
        if holders:
            name = max(holders, key=len)
            backbones[name] = declaring[name]
    return list(backbones.values())


def install_router_aux_gradient(model: nn.Module) -> int:
    """Route the router aux-loss gradient through reentrant-checkpointed MoE layers of ``model``.

    Idempotent. Returns the number of collecting backbones now carrying the hooks; ``0`` means no
    module declares a ``router_logits`` capture, so there is nothing to route.
    """
    backbones = _collecting_backbones(model)
    for backbone in backbones:
        if getattr(backbone, _INSTALLED_ATTR, None) is None:
            setattr(backbone, _INSTALLED_ATTR, _RouterAuxGradient(backbone))
    return len(backbones)
