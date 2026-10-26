"""Recompute scopes shared by an activation checkpoint's two passes, and the gradient-checkpointing
enable that installs them. One :class:`RecomputeScope` per checkpoint invocation, entered by the
original forward and by the backward recompute, makes the pass count an exact recompute signal and
lets a layer cache first-pass results that the recompute must replay rather than recompute.
"""

from __future__ import annotations

import functools
import logging
import threading
from collections.abc import Callable

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_local = threading.local()


class RecomputeScope:
    """One checkpoint invocation's replay cache, entered once per pass.

    ``slot(owner, name)`` hands out that owner's cache dict for the current call, in call order
    within the pass: the first pass creates slots, later passes re-read them positionally. Keyed by
    checkpoint frame rather than by layer or grad mode: a 1F1B pipeline schedule keeps several
    microbatches in flight, and the non-reentrant checkpoint runs both passes with grad enabled.
    """

    __slots__ = ("_slots", "_cursors", "_pass", "grad_enabled")

    def __init__(self):
        self._slots: dict[tuple[int, str], list[dict]] = {}
        self._cursors: dict[tuple[int, str], int] = {}
        self._pass = 0
        # The outer grad mode: scopes are constructed at checkpoint invocation, before the checkpoint
        # machinery's internal no_grad, so this distinguishes a training microbatch from a no_grad
        # reference/eval forward through the same checkpointed block.
        self.grad_enabled = torch.is_grad_enabled()

    @property
    def is_recompute(self) -> bool:
        """True once the body is running for the second time — the backward recompute."""
        return self._pass > 1

    def slot(self, owner: object, name: str) -> dict:
        """This call's cache dict for ``owner``'s ``name`` cache."""
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
        return False


def _stack() -> list[RecomputeScope]:
    """The active scopes on this thread (the recompute pass runs on an autograd engine thread)."""
    stack = getattr(_local, "stack", None)
    if stack is None:
        stack = _local.stack = []
    return stack


def current_graph_task() -> int:
    """The autograd graph task executing on this thread, ``-1`` outside backward."""
    return torch._C._current_graph_task_id()


def in_backward_pass() -> bool:
    """True while an autograd graph task is executing on this thread: a training forward reaching a
    layer here is a checkpoint recompute (the signal torch's FSDP2 uses, ``_fsdp_common.is_bw``)."""
    return current_graph_task() != -1


def active_recompute_scope() -> RecomputeScope | None:
    """The innermost recompute scope currently executing, or None outside one."""
    stack = _stack()
    return stack[-1] if stack else None


def counts_toward_expert_load() -> bool:
    """Whether the current forward is the single per-microbatch pass that records expert load: inside
    a scope, the original pass of a grad-driven invocation; outside one, a grad-enabled pass (a frozen
    reference forward would skew the balance). The scope's outer grad mode is read rather than the live
    one, since the reentrant checkpoint runs its original forward under ``no_grad`` and HF checkpoints a
    ``torch.no_grad()`` forward through a train-mode model too.
    """
    scope = active_recompute_scope()
    if scope is not None:
        return scope.grad_enabled and not scope.is_recompute
    return torch.is_grad_enabled()


def scoped_checkpoint_func(inner: Callable) -> Callable:
    """Wrap an installed ``_gradient_checkpointing_func`` so both of its passes share one scope.

    Wrapping the installed function rather than building a new ``checkpoint`` partial keeps the
    checkpoint kwargs the model resolved, ``use_reentrant`` above all. Idempotent via ``_ep_scoped``.
    """
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
    """Re-point every checkpoint function in ``model`` at its scoped wrapper.

    Run after the model's own ``gradient_checkpointing_enable``, which sets the attribute. Returns the
    number of checkpoint functions now scoped, counting already-scoped ones, so a caller re-enabling GC
    can distinguish "scopes in place" from "no checkpoint function anywhere".
    """
    installed = 0
    for module in model.modules():
        inner = getattr(module, "_gradient_checkpointing_func", None)
        if inner is None:
            continue
        if not getattr(inner, "_ep_scoped", False):
            module._gradient_checkpointing_func = scoped_checkpoint_func(inner)
        installed += 1
    return installed


def _split_every_n_layers(
    gradient_checkpointing_kwargs: dict | None, every_n_layers: int = 1
) -> tuple[dict | None, int]:
    """Lift ``every_n_layers``, ``gradient_checkpointing_enable``'s own keyword, out of the checkpoint
    kwargs, where it would reach ``torch.utils.checkpoint`` and fail the first recompute (TRL's
    re-enable passes the args dict unchanged)."""
    kwargs = dict(gradient_checkpointing_kwargs or {})
    every_n_layers = kwargs.pop("every_n_layers", every_n_layers)
    return kwargs or None, every_n_layers


def _rescope_on_reenable(model: nn.Module) -> None:
    """Bind the recompute scopes to the model's ``gradient_checkpointing_enable`` for every later caller:
    a re-enable installs HF's bare checkpoint function, and online GRPO re-enables every step (TRL's
    ``disable_gradient_checkpointing`` exit)."""
    enable = model.gradient_checkpointing_enable
    if getattr(enable, "_ep_scoped", False):
        return

    @functools.wraps(enable)
    def enable_with_scopes(gradient_checkpointing_kwargs=None, every_n_layers: int = 1):
        kwargs, every_n_layers = _split_every_n_layers(gradient_checkpointing_kwargs, every_n_layers)
        result = enable(gradient_checkpointing_kwargs=kwargs, every_n_layers=every_n_layers)
        install_recompute_scopes(model)  # idempotent: already-scoped functions are skipped
        return result

    enable_with_scopes._ep_scoped = True
    model.gradient_checkpointing_enable = enable_with_scopes


def enable_scoped_gradient_checkpointing(model: nn.Module, *, gradient_checkpointing_kwargs: dict) -> None:
    """Enable gradient checkpointing whose every invocation runs inside one :class:`RecomputeScope`.

    The model's own ``gradient_checkpointing_enable`` resolves HF's policy (input-grad requirement,
    per-module flags, ``every_n_layers``), then every installed checkpoint function is re-pointed at
    its scoped wrapper. An EP layer replays its DeepEP dispatch/combine from the scope, because a
    fresh all-to-all in the recompute would invalidate the handle the first pass's backward still holds.
    """
    if not hasattr(model, "gradient_checkpointing_enable"):
        raise RuntimeError(
            f"gradient_checkpointing=True but {type(model).__name__} has no "
            f"gradient_checkpointing_enable(): the EP path has already disabled HF Trainer's own GC "
            f"setup, so returning here would train with NO gradient checkpointing at the memory "
            f"profile of a run that asked for it. Disable gradient_checkpointing for this model."
        )

    kwargs, every_n_layers = _split_every_n_layers(gradient_checkpointing_kwargs)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs=kwargs, every_n_layers=every_n_layers)
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
