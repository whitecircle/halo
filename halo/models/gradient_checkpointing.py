"""Recompute scopes shared by an activation checkpoint's two passes, and the gradient-checkpointing enable that
installs them. A scope makes the pass count an exact recompute signal and lets a layer replay first-pass results. Also
the input a reentrant checkpoint needs for its layer's trainable parameters to take a gradient."""

from __future__ import annotations

import functools
import inspect
import logging
import threading
from collections.abc import Callable

import torch
import torch.nn as nn
from torch.utils._pytree import tree_leaves
from transformers.modeling_layers import GradientCheckpointingLayer

logger = logging.getLogger(__name__)

_local = threading.local()


def _stack() -> list[RecomputeScope]:
    """Per thread, as the recompute pass runs on an autograd engine thread."""
    stack = getattr(_local, "stack", None)
    if stack is None:
        stack = _local.stack = []
    return stack


class RecomputeScope:
    """``slot(owner, name)`` hands out that owner's cache dict in call order within the pass: the first pass creates
    slots, the recompute re-reads them positionally. Keyed by checkpoint frame, as a 1F1B pipeline schedule keeps
    several microbatches in flight and the non-reentrant checkpoint runs both passes with grad enabled."""

    __slots__ = ("_slots", "_cursors", "_pass", "grad_enabled")

    def __init__(self):
        self._slots: dict[tuple[int, str], list[dict]] = {}
        self._cursors: dict[tuple[int, str], int] = {}
        self._pass = 0
        # Read before the checkpoint's internal no_grad, so this tells a training microbatch from a no_grad reference
        # or eval forward through the same checkpointed block.
        self.grad_enabled = torch.is_grad_enabled()

    @property
    def is_recompute(self) -> bool:
        return self._pass > 1

    def slot(self, owner: object, name: str) -> dict:
        key = (id(owner), name)
        index = self._cursors.get(key, 0)
        self._cursors[key] = index + 1
        slots = self._slots.setdefault(key, [])
        if index == len(slots):
            if self.is_recompute:
                raise RuntimeError(
                    f"{type(owner).__name__}: checkpoint recompute reached call {index + 1} of "
                    f"'{name}' but the original forward made only {len(slots)} — the recompute took "
                    f"a different path than the forward it must reproduce."
                )
            slots.append({})
        return slots[index]

    def __enter__(self) -> RecomputeScope:
        self._pass += 1
        self._cursors.clear()
        _stack().append(self)
        return self

    def __exit__(self, *exc_info) -> bool:
        _stack().pop()
        # The last pass that reads the slots; the reentrant checkpoint keeps this scope until the loss is dropped, and
        # with it every layer's replay cache, DeepEP receive buffers included.
        if self.is_recompute:
            self._slots.clear()
        return False


def current_graph_task() -> int:
    """``-1`` outside backward."""
    return torch._C._current_graph_task_id()


def in_backward_pass() -> bool:
    """A training forward reaching a layer here is a checkpoint recompute (the signal torch's FSDP2 uses)."""
    return current_graph_task() != -1


def active_recompute_scope() -> RecomputeScope | None:
    stack = _stack()
    return stack[-1] if stack else None


def counts_toward_expert_load() -> bool:
    """Inside a scope, the original pass of a grad-driven invocation; outside one, a grad-enabled pass, as a frozen
    reference forward would skew the balance. The scope's outer grad mode is read, as the reentrant checkpoint runs its
    original forward under ``no_grad``."""
    scope = active_recompute_scope()
    if scope is not None:
        return scope.grad_enabled and not scope.is_recompute
    return torch.is_grad_enabled()


def scoped_checkpoint_func(inner: Callable) -> Callable:
    """Wrapping the installed function keeps the checkpoint kwargs the model resolved, ``use_reentrant`` above all."""
    if getattr(inner, "_ep_scoped", False):
        return inner

    @functools.wraps(inner)
    def scoped(function: Callable, *args, **kwargs):
        scope = RecomputeScope()

        def in_scope(*call_args, **call_kwargs):
            with scope:
                return function(*call_args, **call_kwargs)

        return inner(in_scope, *args, **kwargs)

    scoped._ep_scoped = True
    return scoped


def install_recompute_scopes(model) -> int:
    """Run after the model's own ``gradient_checkpointing_enable``. Returns the checkpoint functions now scoped,
    already-scoped ones included, so a caller re-enabling GC can tell scopes in place from no checkpoint function."""
    installed = 0
    for module in model.modules():
        inner = getattr(module, "_gradient_checkpointing_func", None)
        if inner is None:
            continue
        module._gradient_checkpointing_func = scoped_checkpoint_func(inner)
        installed += 1
    return installed


def _split_enable_kwargs(
    gradient_checkpointing_kwargs: dict | None, every_n_layers: int = 1, offload: bool = False
) -> tuple[dict | None, int, bool]:
    """Left in the checkpoint kwargs they would reach ``torch.utils.checkpoint`` and fail the first recompute (TRL's
    re-enable passes the args dict unchanged)."""
    kwargs = dict(gradient_checkpointing_kwargs or {})
    every_n_layers = kwargs.pop("every_n_layers", every_n_layers)
    offload = kwargs.pop("offload", offload)
    return kwargs or None, every_n_layers, offload


def _rescope_on_reenable(model: nn.Module) -> None:
    """A re-enable installs HF's bare checkpoint function, and online GRPO re-enables every step."""
    enable = model.gradient_checkpointing_enable
    if getattr(enable, "_ep_scoped", False):
        return

    @functools.wraps(enable)
    def enable_with_scopes(gradient_checkpointing_kwargs=None, every_n_layers: int = 1, offload: bool = False):
        kwargs, every_n_layers, offload = _split_enable_kwargs(gradient_checkpointing_kwargs, every_n_layers, offload)
        result = enable(gradient_checkpointing_kwargs=kwargs, every_n_layers=every_n_layers, offload=offload)
        install_recompute_scopes(model)
        return result

    enable_with_scopes._ep_scoped = True
    model.gradient_checkpointing_enable = enable_with_scopes


def enable_scoped_gradient_checkpointing(model: nn.Module, *, gradient_checkpointing_kwargs: dict) -> None:
    """An EP layer replays its DeepEP dispatch/combine from the scope, as a fresh all-to-all in the recompute would
    invalidate the handle the first pass's backward still holds."""
    if not hasattr(model, "gradient_checkpointing_enable"):
        raise RuntimeError(
            f"gradient_checkpointing=True but {type(model).__name__} has no "
            f"gradient_checkpointing_enable(): the EP path has already disabled HF Trainer's own GC "
            f"setup, so returning here would train with NO gradient checkpointing at the memory "
            f"profile of a run that asked for it. Disable gradient_checkpointing for this model."
        )

    kwargs, every_n_layers, offload = _split_enable_kwargs(gradient_checkpointing_kwargs)
    model.gradient_checkpointing_enable(
        gradient_checkpointing_kwargs=kwargs, every_n_layers=every_n_layers, offload=offload
    )
    scoped = install_recompute_scopes(model)
    if scoped == 0:
        raise RuntimeError(
            f"gradient_checkpointing=True but {type(model).__name__}.gradient_checkpointing_enable() "
            f"installed no checkpoint function: EP's dispatch replay hangs off that function, so the "
            f"run would issue a second DeepEP dispatch inside backward and corrupt every gradient."
        )
    _rescope_on_reenable(model)
    logger.info(
        f"Gradient checkpointing enabled on {scoped} modules (kwargs={kwargs}, every_n_layers={every_n_layers})"
    )


def _declares_non_reentrant(checkpoint_func: Callable) -> bool:
    """Read through :func:`scoped_checkpoint_func` to the ``partial`` transformers installs. A function that declares
    nothing counts as reentrant, as the input it then gets changes no gradient under a non-reentrant one."""
    while (wrapped := getattr(checkpoint_func, "__wrapped__", None)) is not None:
        checkpoint_func = wrapped
    return getattr(checkpoint_func, "keywords", {}).get("use_reentrant") is False


def _with_gradient_carrying_input(layer: GradientCheckpointingLayer, args: tuple, kwargs: dict) -> tuple[tuple, dict]:
    """A reentrant checkpoint's outputs carry a gradient only if a positional input does. transformers' input-grad hook
    gives that to the text embedding and to a tower declaring ``get_input_embeddings``; any other layer behind frozen
    modules, a tower block behind its patch embedding above all, would keep its trainable parameters at no gradient, and
    a frozen one would drop a gradient arriving in another argument. A hidden state passed by keyword moves to the
    positional slot it names, and a detached alias leaves the caller's tensor out of the graph. A layer with nothing
    trainable and no gradient arriving takes no new input."""
    if any(isinstance(arg, torch.Tensor) and arg.requires_grad for arg in args):
        return args, kwargs
    if _declares_non_reentrant(layer._gradient_checkpointing_func):
        return args, kwargs
    if not args:
        first_parameter = next(iter(inspect.signature(layer.forward).parameters), None)
        if isinstance(kwargs.get(first_parameter), torch.Tensor):
            args = (kwargs[first_parameter],)
            kwargs = {name: value for name, value in kwargs.items() if name != first_parameter}
            if args[0].requires_grad:
                return args, kwargs
    arriving = any(isinstance(leaf, torch.Tensor) and leaf.requires_grad for leaf in tree_leaves((args, kwargs)))
    if not arriving and not any(param.requires_grad for param in layer.parameters()):
        return args, kwargs
    for index, arg in enumerate(args):
        if isinstance(arg, torch.Tensor) and arg.is_floating_point():
            return (*args[:index], arg.detach().requires_grad_(), *args[index + 1 :]), kwargs
    raise ValueError(
        f"{type(layer).__name__} trains or takes a gradient under reentrant gradient checkpointing, and takes no "
        f"floating tensor as its first argument or positionally, so its checkpoint's outputs could carry no gradient. "
        f"Pass its hidden state positionally, or train it without gradient checkpointing."
    )


_stock_layer_call = GradientCheckpointingLayer.__call__


@functools.wraps(_stock_layer_call)
def _call_with_gradient_carrying_input(self, *args, **kwargs):
    if self.gradient_checkpointing and self.training and torch.is_grad_enabled():
        args, kwargs = _with_gradient_carrying_input(self, args, kwargs)
    return _stock_layer_call(self, *args, **kwargs)


# At import, as every trainer imports this module and every transformers layer that checkpoints itself runs this call.
GradientCheckpointingLayer.__call__ = _call_with_gradient_carrying_input
