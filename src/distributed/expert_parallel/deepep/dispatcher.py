"""The DeepEP token dispatcher an EP MoE layer routes through, its dispatch/combine autograd functions and their
checkpoint-recompute replay."""

from __future__ import annotations

import contextlib
import logging
import sys
import weakref

import torch
import torch.nn as nn

from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.deepep.backends import (
    DeepEPBackend,
    build_ep_backend,
    deep_ep,
    deep_ep_available,
    free_all_ep_buffers,
)
from src.distributed.expert_parallel.deepep.capacity import capacity_scope
from src.models.gradient_checkpointing import active_recompute_scope, in_backward_pass
from src.runtime.collectives import barrier
from src.runtime.ranks import is_global_main_process

logger = logging.getLogger(__name__)

# The dispatchers holding a claim, which teardown drops. Weak, so its contents follow each rank's garbage
# collector: it never decides which buffers are freed or in what order.
_live_dispatchers: weakref.WeakValueDictionary = weakref.WeakValueDictionary()


def _deterministic_dispatch() -> bool:
    """Whether received tokens must land in a fixed order: torch's deterministic-algorithms mode is on,
    which HF's ``Trainer.__init__`` turns on for ``full_determinism``."""
    return torch.are_deterministic_algorithms_enabled()


def _release_uninitialized_memory_fill() -> None:
    """Switch off ``torch.utils.deterministic.fill_uninitialized_memory`` under deterministic mode, for the
    process: DeepEP asserts the two are not on together (the fill races its communication-stream allocations).
    """
    if torch.are_deterministic_algorithms_enabled() and torch.utils.deterministic.fill_uninitialized_memory:
        torch.utils.deterministic.fill_uninitialized_memory = False
        if is_global_main_process():
            logger.warning(
                "Deterministic algorithms are on with an EP model: setting "
                "torch.utils.deterministic.fill_uninitialized_memory=False for the rest of the process, "
                "which DeepEP's dispatch and combine require under deterministic mode."
            )


def _dispatch_backward(ctx, grad_recv_x: torch.Tensor, grad_recv_topk_weights) -> tuple[torch.Tensor, torch.Tensor]:
    """The gradient of a dispatch: a combine over the layout cached on ``ctx.handle``."""
    grad_x, grad_topk_weights = ctx.dispatcher.backend.dispatch_backward(
        grad_recv_x.contiguous(), grad_recv_topk_weights, ctx.handle
    )
    if grad_x.dtype != ctx.input_dtype:
        grad_x = grad_x.to(ctx.input_dtype)
    return grad_x, grad_topk_weights


def _combine_backward(ctx, grad_combined_x: torch.Tensor) -> torch.Tensor:
    """The gradient of a combine: a dispatch over the layout cached on ``ctx.handle``."""
    grad_x = ctx.dispatcher.backend.combine_backward(grad_combined_x.contiguous(), ctx.handle)
    if grad_x.dtype != ctx.input_dtype:
        grad_x = grad_x.to(ctx.input_dtype)
    return grad_x


class DeepEPDispatchFunction(torch.autograd.Function):
    """DeepEP dispatch; its backward is a combine."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor, dispatcher):
        x = x.contiguous()
        topk_idx = topk_idx.contiguous().to(deep_ep().topk_idx_t)
        # DeepEP asserts contiguous float32 top-k weights.
        topk_weights = topk_weights.to(dtype=torch.float32).contiguous()
        recv_x, recv_topk_idx, recv_topk_weights, handle = dispatcher.backend.dispatch_forward(
            x, topk_idx, topk_weights
        )
        ctx.dispatcher = dispatcher
        ctx.handle = handle
        ctx.input_dtype = x.dtype
        return recv_x, recv_topk_idx, recv_topk_weights, handle

    @staticmethod
    def backward(ctx, grad_recv_x, grad_recv_topk_idx, grad_recv_topk_weights, grad_handle):
        grad_x, grad_topk_weights = _dispatch_backward(ctx, grad_recv_x, grad_recv_topk_weights)
        return grad_x, None, grad_topk_weights, None


class DeepEPCombineFunction(torch.autograd.Function):
    """DeepEP combine; its backward is a dispatch reusing the layout ``handle`` carries."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, topk_weights: torch.Tensor, handle: object, dispatcher):
        x = x.contiguous()
        topk_weights = topk_weights.to(dtype=torch.float32).contiguous()
        combined_x = dispatcher.backend.combine_forward(x, topk_weights, handle)
        ctx.dispatcher = dispatcher
        ctx.handle = handle
        ctx.input_dtype = x.dtype
        return combined_x

    @staticmethod
    def backward(ctx, grad_combined_x):
        return _combine_backward(ctx, grad_combined_x), None, None, None


class ReplayDispatchFunction(torch.autograd.Function):
    """A dispatch replayed from its cached results during the GC recompute; its backward is the combine."""

    @staticmethod
    def forward(
        ctx,
        x,
        topk_idx,
        topk_weights,
        dispatcher,
        cached_recv_x,
        cached_recv_topk_idx,
        cached_recv_topk_weights,
        cached_handle,
    ):
        ctx.dispatcher = dispatcher
        ctx.handle = cached_handle
        ctx.input_dtype = x.dtype
        return cached_recv_x, cached_recv_topk_idx, cached_recv_topk_weights, cached_handle

    @staticmethod
    def backward(ctx, grad_recv_x, grad_recv_topk_idx, grad_recv_topk_weights, grad_handle):
        grad_x, grad_topk_weights = _dispatch_backward(ctx, grad_recv_x, grad_recv_topk_weights)
        return grad_x, None, grad_topk_weights, None, None, None, None, None


class ReplayCombineFunction(torch.autograd.Function):
    """A combine replayed from its cached result during the GC recompute; its backward is the dispatch."""

    @staticmethod
    def forward(ctx, x, topk_weights, handle, dispatcher, cached_combined_x):
        ctx.dispatcher = dispatcher
        ctx.handle = handle
        ctx.input_dtype = x.dtype
        return cached_combined_x

    @staticmethod
    def backward(ctx, grad_combined_x):
        return _combine_backward(ctx, grad_combined_x), None, None, None, None


class DeepEPDispatcher:
    """One MoE layer's token dispatcher: its EP topology, and the transport backend it claims a buffer through.

    At ``ep_size <= 1`` every expert is local (and DeepEP cannot run): dispatch and combine are the
    identity and no backend is built.
    """

    def __init__(self, ep_config: EPConfig, num_experts: int, hidden_dim: int):
        self.ep_config = ep_config
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        # Under expert TP each ETP group dispatches within its own sub-EP group; without it this is the EP group.
        self.ep_group = ep_config.dispatch_ep_group
        self.ep_size = ep_config.ep_size
        self.spans_domains = ep_config.spans_domains
        self._noop = self.ep_size <= 1
        self._live = False
        # Public: the autograd functions drive the backend's ops directly.
        self.backend: DeepEPBackend | None = None

        if self._noop:
            # Main process only: every MoE layer builds its own dispatcher.
            if is_global_main_process():
                logger.info(f"DeepEPDispatcher (noop): all experts local, ep_size={self.ep_size}")
            return
        if not deep_ep_available():
            raise ImportError(
                "DeepEP is required for Expert Parallelism but is not installed. "
                "See agent-docs/infrastructure/deepep.md for installation instructions."
            )
        self.backend = build_ep_backend(self, ep_config.ep_buffer_backend)

    def dispatch(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor, *, owner: nn.Module):
        """Dispatch tokens to their experts' ranks; the identity in no-op mode, where the indices are local.

        Inside a recompute scope the original pass caches its results under ``owner`` and the recompute replays
        them: a second DeepEP dispatch invalidates the handle the original backward holds and silently corrupts
        every gradient.
        """
        if self._noop:
            return x, topk_idx, topk_weights, None
        scope = active_recompute_scope()
        if scope is None and owner.training and in_backward_pass():
            raise RuntimeError(
                f"{type(owner).__name__}: training forward inside backward with no EP checkpoint "
                f"scope — gradient checkpointing was re-enabled without install_recompute_scopes "
                f"(TRL's disable_gradient_checkpointing context manager re-installs a bare "
                f"checkpoint on exit). A second DeepEP dispatch here corrupts every gradient."
            )
        slot = scope.slot(owner, "dispatch") if scope is not None else None
        if scope is not None and scope.is_recompute:
            return ReplayDispatchFunction.apply(
                x,
                topk_idx,
                topk_weights,
                self,
                slot["recv_x"],
                slot["recv_topk_idx"],
                slot["recv_topk_weights"],
                slot["handle"],
            )
        self._ensure_buffer(x.shape[0], topk_idx.shape[1])
        recv_x, recv_topk_idx, recv_topk_weights, handle = DeepEPDispatchFunction.apply(
            x, topk_idx, topk_weights, self
        )
        if slot is not None:
            # Detached aliases, not clones: DeepEP's copy epilogue lands every result in an allocator-owned
            # tensor (never a view of the shared arena) and nothing downstream writes into them in place.
            slot.update(
                recv_x=recv_x.detach(),
                recv_topk_idx=recv_topk_idx.detach() if recv_topk_idx is not None else None,
                recv_topk_weights=recv_topk_weights.detach() if recv_topk_weights is not None else None,
                handle=handle,
            )
        return recv_x, recv_topk_idx, recv_topk_weights, handle

    def _ensure_buffer(self, num_tokens: int, num_topk: int) -> None:
        """COLLECTIVE over the EP group. Build or grow the buffer to hold ``num_tokens`` per rank."""
        _release_uninitialized_memory_fill()
        self.backend.ensure(num_tokens, num_topk, deterministic=_deterministic_dispatch())
        if not self._live:
            _live_dispatchers[id(self)] = self
            self._live = True

    def combine(
        self, x: torch.Tensor, topk_weights: torch.Tensor, handle: object, *, owner: nn.Module
    ) -> torch.Tensor:
        """Combine expert outputs back to their tokens' positions; the identity in no-op mode. Replayed under
        ``owner`` in a recompute like :meth:`dispatch`, since the combine writes the same shared buffer."""
        if self._noop:
            return x
        scope = active_recompute_scope()
        if scope is None:
            return DeepEPCombineFunction.apply(x, topk_weights, handle, self)
        slot = scope.slot(owner, "combine")
        if scope.is_recompute:
            return ReplayCombineFunction.apply(x, topk_weights, handle, self, slot["combined"])
        combined = DeepEPCombineFunction.apply(x, topk_weights, handle, self)
        slot["combined"] = combined.detach()
        return combined

    def destroy(self, *, free_buffer: bool = True) -> None:
        """Drop this layer's claim on its buffer (idempotent); a later dispatch claims one again.

        With ``free_buffer``, COLLECTIVE over the EP group: the buffer is freed once no layer on any rank
        of the group claims it, so call it only from a rank-synchronized point.
        """
        if not self._live:
            return
        self._live = False  # first, against re-entry
        _live_dispatchers.pop(id(self), None)
        self.backend.destroy(free_buffer=free_buffer)

    def __del__(self, _is_finalizing=sys.is_finalizing):
        # Interpreter teardown can clear this module's globals before the finalizer runs, so the
        # shutdown probe is bound as a default; past it the CUDA context may already be gone.
        if _is_finalizing():
            return
        # Drop the claim only: freeing is a collective, and the garbage collector fires at a moment no
        # other rank agrees on. destroy_all_dispatchers() frees what this leaves, at a barriered teardown.
        with contextlib.suppress(Exception):
            self.destroy(free_buffer=False)


def destroy_all_dispatchers() -> None:
    """COLLECTIVE over the world. Free every DeepEP buffer; call before ``dist.destroy_process_group()``, as
    Gin frees through the group communicator (after it, ``cudaErrorIllegalAddress``).

    The unconditional leading barrier aligns ranks after rank-asymmetric work (a gathered save, an eval)
    under the process-group timeout rather than the device-side ``HALO_DEEPEP_GPU_TIMEOUT_SECONDS`` spin
    of ``buffer.destroy()``; a pipeline stage without MoE layers enters it too. The live dispatchers only
    drop their claims and every arena is then freed in build order: which dispatchers are live depends on
    each rank's garbage collector, so a free on the last claim would spin ranks in different barriers.
    """
    barrier()
    for dispatcher in list(_live_dispatchers.values()):
        dispatcher.destroy(free_buffer=False)
    free_all_ep_buffers()
    # The capacity entries hold their EP groups; kept past teardown they would pin every communicator
    # the job ever built.
    capacity_scope.clear()
