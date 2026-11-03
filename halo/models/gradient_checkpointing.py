"""Recompute scopes shared by an activation checkpoint's two passes, and the gradient-checkpointing enable that
installs them. A scope makes the pass count an exact recompute signal and lets a layer replay first-pass results."""

from __future__ import annotations

import functools
import logging
import threading
from collections.abc import Callable

import torch
import torch.nn as nn

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
        if not getattr(inner, "_ep_scoped", False):
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
