"""Per-invocation checkpoint scope shared by an activation checkpoint's two passes.

A checkpointed block runs its body twice — once to produce activations, once during backward to
recompute them — and EP's DeepEP dispatch/combine must NOT run the second time: a fresh all-to-all
reuses the same ``ElasticBuffer``, invalidating the handle the ORIGINAL forward's backward node
still holds. So the layer caches its dispatch/combine results on the first pass and replays them on
the second (:meth:`EPMoELayerBase._gc_dispatch`).

That cache cannot live on the layer. A pipeline stage keeps up to ``pp_size`` microbatches in flight
under a 1F1B schedule, so microbatch *j*'s forward would overwrite the entry microbatch *i*'s
recompute is about to replay. It also cannot key on grad mode: only the REENTRANT checkpoint runs
its original forward under ``no_grad``; the non-reentrant one (the only mode pipeline parallelism
accepts) runs both passes with grad enabled.

The scope solves both. :func:`scoped_checkpoint_func` wraps the checkpoint function a model has
installed so that one scope object is created per invocation and entered by BOTH passes — reentrant
and non-reentrant alike, since each re-invokes the same wrapped callable. Slots are therefore keyed
by checkpoint frame, concurrent frames cannot see each other, the pass counter is an exact recompute
signal, and the cache dies with the frame (no manual clearing, no cross-step leak).
"""

from __future__ import annotations

import functools
import threading
from collections.abc import Callable

import torch

_local = threading.local()


def in_backward_pass() -> bool:
    """True while an autograd graph task is executing on this thread.

    A training forward that reaches an EP layer here can only be a checkpoint recompute — the same
    signal torch's own FSDP2 uses to spot one (``_fsdp_common.is_bw``).
    """
    return torch._C._current_graph_task_id() != -1


def _stack() -> list[EPCheckpointScope]:
    """The active scopes on THIS thread (the recompute pass runs on an autograd engine thread)."""
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

    Inside a checkpoint scope the ORIGINAL pass of a grad-driven invocation counts; the recompute
    does not — its tokens were already counted. Outside one, only a grad-enabled pass counts: a
    frozen reference/teacher forward drives no optimizer step, so its load would skew the per-step
    balance the bias update corrects.

    Two grad-mode subtleties, both load-bearing:

    * The REENTRANT checkpoint — forced for every non-PP EP/CP run — executes its original forward
      under ``no_grad``, so a bare ``torch.is_grad_enabled()`` gate here records nothing on the
      configurations that actually ship. The scope's :attr:`~EPCheckpointScope.grad_enabled` is the
      OUTER mode, captured at invocation before the checkpoint's internal ``no_grad``.
    * A ``torch.no_grad()`` forward through a train-mode GC model (offline GRPO's KL reference
      pass) still enters checkpoint scopes — HF gates checkpointing on ``self.training``, not grad
      mode — and must not record, else the bias balances a policy+reference mixture.
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
        # The OUTER grad mode: scopes are constructed at checkpoint invocation, before the
        # checkpoint machinery's internal no_grad, so this is what distinguishes a training
        # microbatch from a no_grad reference/eval forward through the same checkpointed block.
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

    Wrapping the installed function (rather than building a new ``checkpoint`` partial) keeps
    whatever checkpoint kwargs the model resolved — ``use_reentrant`` above all — and works for any
    module tree HF has already set the attribute on. Idempotent via ``_ep_scoped``.
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

    Runs AFTER the model's own ``gradient_checkpointing_enable``, which is what sets the attribute.
    Returns the number of checkpoint functions now scoped — an already-scoped one counts, so a
    caller re-enabling GC (the rescope hook has already wrapped the fresh install) can still tell
    "scopes in place" apart from "no checkpoint function anywhere".
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
