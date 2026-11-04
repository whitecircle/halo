"""The router aux-loss gradient of reentrant-checkpointed MoE layers. A reentrant checkpoint runs a layer under
``no_grad`` and recomputes it uncollected, so a collected tensor's gradient is kept and injected in the recompute."""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import torch
import torch.nn as nn
from peft.utils.other import AuxiliaryTrainingWrapper

from halo.models.gradient_checkpointing import current_graph_task
from halo.models.moe.router_logits import ROUTER_LOGITS_KEY, declared_routers, declares_router_logits

_INSTALLED_ATTR = "_router_aux_gradient"

# Module levels between a router and the block that calls it which never run the block's forward:
# module containers, and PEFT's ``modules_to_save`` wrapper around a trained router copy.
_PASS_THROUGH_LEVELS = (nn.ModuleDict, nn.ModuleList, AuxiliaryTrainingWrapper)


@dataclass(eq=False)
class _BackboneForward:
    """``producers`` maps a checkpoint's ``no_grad`` tensors, by ``id``, to their router."""

    producers: dict[int, nn.Module] = field(default_factory=dict)
    gradients: dict[nn.Module, torch.Tensor] = field(default_factory=dict)


class _KeepCollectedGradient(torch.autograd.Function):
    """Taking the backbone output ``anchor`` as an input gives the carrier a graph and makes the backbone's backward
    wait for every carrier, so each kept gradient exists before the first checkpointed layer recomputes."""

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
    """A block returning a tuple leads with its hidden state, as GPT-OSS's ``(hidden_states, router_scores)`` does."""
    if isinstance(output, torch.Tensor):
        return _InjectRouterGradient.apply(output, logits, gradient)
    if type(output) is tuple and output and isinstance(output[0], torch.Tensor):
        return (_InjectRouterGradient.apply(output[0], logits, gradient), *output[1:])
    raise TypeError(
        f"The block owning a checkpointed router returned {type(output).__name__}, not a tensor or a "
        f"tuple led by one, so the router aux-loss gradient has no hidden state to ride back on."
    )


def _owning_block(root: nn.Module, router_name: str) -> nn.Module:
    parts = router_name.split(".")
    for depth in range(len(parts) - 1, -1, -1):
        owner = root.get_submodule(".".join(parts[:depth]))
        if not isinstance(owner, _PASS_THROUGH_LEVELS):
            return owner
    raise ValueError(f"router {router_name!r} has no owning block under {type(root).__name__}")


class _RouterAuxGradient:
    """Routers and owning blocks resolve at the first forward, as installation precedes the trainer and a PEFT
    ``modules_to_save`` wrap replaces the router the forward calls with a copy."""

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
        if current_graph_task() != -1:
            self._live[router] = logits
        elif self._open is not None and not torch.is_grad_enabled():
            # A no_grad router inside a grad-enabled backbone forward is a reentrant checkpoint's first pass.
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
        task = current_graph_task()
        if task != self._task:
            self._task, self._task_forwards = task, []
            torch.autograd.Variable._execution_engine.queue_callback(partial(self._finish, task))
        if record not in self._task_forwards:
            self._task_forwards.append(record)
        record.gradients[router] = gradient

    def _take(self, router: nn.Module) -> torch.Tensor | None:
        if self._task is None or self._task != current_graph_task():
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
        if current_graph_task() == -1:
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
    """Each router's innermost ancestor declaring the capture; a causal-LM head inherits it without collecting."""
    declaring = {name: module for name, module in model.named_modules() if declares_router_logits(module)}
    backbones: dict[str, nn.Module] = {}
    for router in declared_routers(model):
        holders = [name for name in declaring if not name or router.name.startswith(f"{name}.")]
        if holders:
            name = max(holders, key=len)
            backbones[name] = declaring[name]
    return list(backbones.values())


def install_router_aux_gradient(model: nn.Module) -> int:
    """Idempotent. Returns the collecting backbones now carrying the hooks; ``0`` means nothing declares a capture."""
    backbones = _collecting_backbones(model)
    for backbone in backbones:
        if getattr(backbone, _INSTALLED_ATTR, None) is None:
            setattr(backbone, _INSTALLED_ATTR, _RouterAuxGradient(backbone))
    return len(backbones)
