"""Per-invocation checkpoint scope shared by an activation checkpoint's two passes.

A checkpointed block runs its body twice (forward, then backward recompute), and EP's DeepEP
dispatch/combine must not run the second time: a fresh all-to-all reuses the same ``ElasticBuffer``
and invalidates the handle the first forward's backward node still holds. The layer therefore caches
its dispatch/combine results on the first pass and replays them on the second
(:meth:`EPMoELayerBase._gc_dispatch`).

The cache is keyed by checkpoint frame rather than by layer or by grad mode. A 1F1B pipeline schedule
keeps up to ``pp_size`` microbatches in flight, so one microbatch's forward would overwrite the entry
another's recompute is about to replay; and only the reentrant checkpoint runs its original forward
under ``no_grad``, while the non-reentrant one runs both passes with grad enabled.
:func:`scoped_checkpoint_func` wraps the checkpoint function a model has installed so one scope object
is created per invocation and entered by both passes. The pass counter is then an exact recompute
signal, and the cache dies with the frame.
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable

import torch

_local = threading.local()


def in_backward_pass() -> bool:
    """True while an autograd graph task is executing on this thread.

    A training forward reaching an EP layer here is a checkpoint recompute; the same signal torch's
    FSDP2 uses (``_fsdp_common.is_bw``).
    """
    return torch._C._current_graph_task_id() != -1


def _stack() -> list[EPCheckpointScope]:
    """The active scopes on this thread (the recompute pass runs on an autograd engine thread)."""
    stack = getattr(_local, "stack", None)
    if stack is None:
        stack = _local.stack = []
    return stack


def active_checkpoint_scope() -> EPCheckpointScope | None:
    """The innermost checkpoint scope currently executing, or None outside one."""
    stack = _stack()
    return stack[-1] if stack else None


def counts_toward_expert_load() -> bool:
    """Whether the current forward is the single per-microbatch pass that records expert load.

    Inside a checkpoint scope the original pass of a grad-driven invocation counts and the recompute
    does not, its tokens being already counted. Outside one, only a grad-enabled pass counts: a frozen
    reference/teacher forward drives no optimizer step, so its load would skew the per-step balance the
    bias update corrects.

    Grad mode alone is not enough on either side. The reentrant checkpoint, forced for every non-PP
    EP/CP run, executes its original forward under ``no_grad``, so the scope's
    :attr:`~EPCheckpointScope.grad_enabled` — the outer mode, captured at invocation — is read instead.
    A ``torch.no_grad()`` forward through a train-mode GC model (offline GRPO's KL reference pass) still
    enters checkpoint scopes, since HF gates checkpointing on ``self.training`` rather than grad mode,
    and must not record: the bias would balance a policy+reference mixture.
    """
    scope = active_checkpoint_scope()
    if scope is not None:
        return scope.grad_enabled and not scope.is_recompute
    return torch.is_grad_enabled()


class EPCheckpointScope:
    """One checkpoint invocation's replay cache, entered once per pass.

    ``slot(owner, name)`` hands out that owner's cache dict for the current call, in call order
    within the pass: the first pass creates slots, later passes re-read them positionally.
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

    def __enter__(self) -> EPCheckpointScope:
        self._pass += 1
        self._cursors.clear()
        _stack().append(self)
        return self

    def __exit__(self, *exc_info) -> bool:
        _stack().pop()
        return False


def _in_scope(function: Callable, scope: EPCheckpointScope, *args, **kwargs):
    with scope:
        return function(*args, **kwargs)


def scoped_checkpoint_func(inner: Callable) -> Callable:
    """Wrap an installed ``_gradient_checkpointing_func`` so both of its passes share one scope.

    Wrapping the installed function rather than building a new ``checkpoint`` partial keeps the
    checkpoint kwargs the model resolved, ``use_reentrant`` above all. Idempotent via ``_ep_scoped``.
    """
    if getattr(inner, "_ep_scoped", False):
        return inner

    @functools.wraps(inner)
    def scoped(function: Callable, *args, **kwargs):
        return inner(functools.partial(_in_scope, function, EPCheckpointScope()), *args, **kwargs)

    scoped._ep_scoped = True
    return scoped


def install_ep_checkpoint_scopes(model) -> int:
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
