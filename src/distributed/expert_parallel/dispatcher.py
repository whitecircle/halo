"""DeepEP token dispatcher used by EP MoE layers.

:class:`DeepEPDispatcher` manages the EP communication buffer's lifetime, exposes autograd-aware
``dispatch`` / ``combine``, and short-circuits to no-op mode when ``ep_size <= 1`` (all experts local).

Buffer lifecycle and the four dispatch/combine ops sit behind :class:`_DeepEPBackend`, selected once per
run by ``ep_buffer_backend`` (``"auto"`` resolves to elastic; no fallback):

* :class:`_ElasticBackend` (``"elastic"``/``"auto"``, default) — DeepEP V2 :class:`deep_ep.ElasticBuffer`
  over NCCL Gin (RDMA) cross-node, non-Gin over NVLink intra-node. Arbitrary sequence length; the only
  guarded limit is DeepEP's 32-bit wire index (~175k tokens/rank). Its TMA combine kernel requires
  ``hidden % 256 == 0``, so non-conforming hidden is zero-padded on the wire and sliced back (symmetric
  across fwd/bwd, invisible to callers).
* :class:`_LegacyBackend` (``"legacy"``) — DeepEP V1 :class:`deep_ep.Buffer` (CUDA IPC over NVLink),
  intranode only (cross-node rejected, ``num_rdma_bytes=0``), no hidden padding, numerically identical to
  elastic. Use it for long-context ep8, where the elastic combine barrier races FSDP2's reduce-scatter
  and deadlocks.

See ``agent-docs/infrastructure/deepep.md`` for backend trade-offs and the wire-index limit.
"""

from __future__ import annotations

import contextlib
import itertools
import logging
import os
import sys
import weakref
from abc import ABC, abstractmethod
from typing import Self

import torch
import torch.distributed as dist
import torch.nn.functional as F

from src.distributed.expert_parallel.autograd import (
    DeepEPCombineFunction,
    DeepEPDispatchFunction,
)
from src.distributed.expert_parallel.config import (
    GIN_MAX_TOKENS_PER_RANK,
    EPConfig,
    ep_dispatch_capacity,
    padded_wire_hidden,
    reject_legacy_backend_topology,
    reject_oversized_dispatch,
)
from src.distributed.expert_parallel.extension import deep_ep, deep_ep_available
from src.distributed.grad_reduce import GRAD_BUCKET_MB
from src.distributed.runtime import (
    barrier,
    get_local_world_size,
    is_global_main_process,
    reject_divergent_settings,
)
from src.env import env_flag, env_int, env_positive_int, resolve_nccl_timeout_minutes, resolve_store_timeout_hours
from src.log import warn_once

logger = logging.getLogger(__name__)

# ``warn_once`` scope for the process-global DeepEP environment warnings: one line per process rather
# than one per MoE layer (each layer builds its own backend instance, so an instance flag cannot
# bound it).
_WARNED_ENV: set = set()

# Insertion-ordered (build order == forward order, rank-identical) so the collective buffer.destroy() matches.
_LIVE_DISPATCHERS: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

# Fallback SM count when get_theoretical_num_sms divides by zero on an inter-node Blackwell topology.
_DEFAULT_INTERNODE_NUM_SMS = 24

# ``HALO_EP_CAPACITY_DEDUP=0`` restores the per-layer all-reduce, for a model whose later MoE layers grow.
_CAPACITY_DEDUP_ENABLED = env_flag("HALO_EP_CAPACITY_DEDUP", True)
# Device-side spin budget for the dispatch/combine NVLink barrier, in seconds (DeepEP's own default is
# 100). The clock starts when a rank enters the barrier, so it bounds rank skew rather than idle time
# between steps. Save paths end on a collective (``DeferredRankFailure.reject``) and teardown opens
# with a barrier, both host-blocking, so that skew is absorbed under the larger process-group timeout;
# raise this only for a legitimately skewed phase that can outlast it.
_DEEPEP_DEFAULT_GPU_TIMEOUT_SECONDS = 100
_GPU_TIMEOUT_SECONDS = env_positive_int(
    "HALO_DEEPEP_GPU_TIMEOUT_SECONDS",
    _DEEPEP_DEFAULT_GPU_TIMEOUT_SECONDS,
)
# Opt-in override of the dispatch/combine SM count (0 => auto). Read at module level because both
# transports honour it.
_NUM_SMS = env_int("HALO_DEEPEP_NUM_SMS", None) or 0
# RDMA queue-pair count (0 => auto); more QPs help the latency-bound all-to-all on EFA Gin. Elastic
# backend only: V1 takes no per-call QP count and its RDMA path is not wired up here, so the legacy
# backend warns rather than letting the knob look applied.
_NUM_QPS = env_int("HALO_DEEPEP_NUM_QPS", None) or 0
# Bumped once per top-level model forward by ``register_forward_generation_hook``: every MoE layer in a forward
# sees the same tokens/rank, so the capacity all-reduce runs once (first layer) and the rest reuse the size.
_FORWARD_GENERATION = 0
# id(ep_group) -> (group, generation, num_topk, capacity), decided at the first MoE layer of each
# forward. The group is held as a strong ref because the key is its id, which the allocator could
# otherwise recycle onto a rebuilt group and serve it another group's capacity.
# ``destroy_all_dispatchers`` clears the cache, so the ref never outlives EP teardown.
_CAPACITY_CACHE: dict[int, tuple[object, int, int, int]] = {}
# Distinguishes the per-layer arenas built when sharing is off, which have no shape to key them apart.
_ARENA_SERIAL = itertools.count()
# Latched by verify_rank_uniform_env once the join has run; the settings cannot change after import,
# so one collective per process is enough.
_ENV_UNIFORMITY_VERIFIED = False


def _is_group_leader(group) -> bool:
    """Whether this rank is rank 0 of ``group``, which logs the per-EP-group events."""
    if not (dist.is_available() and dist.is_initialized()):
        return True
    return dist.get_rank(group=group) == 0


def rank_uniform_ep_settings() -> dict[str, object]:
    """The toolkit settings every rank of a collective path must agree on, as resolved values.

    Resolved rather than raw, because each knob is consumed through :mod:`src.env`, where an absent
    variable and the variable set to its default mean the same thing; comparing raw strings would
    reject a launcher that exports a default on the head node only. ``EP_DISABLE_GIN`` is DeepEP-owned
    and read there as a raw string, so it is compared as one; left unset it is derived from topology
    by :meth:`_ElasticBackend._configure_env`, identically on every rank of a group.

    A divergence in any of these desynchronizes ranks: ``HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK`` gates a
    raise against an all-reduced capacity, ``HALO_GRAD_BUCKET_MB`` sets the chunk boundaries
    :func:`~src.distributed.grad_reduce.reduce_grads_bucketed` needs every rank of the reduction group
    to build identically, and the two coordination timeouts decide which rank gives up on a collective
    or a store key first.
    """
    return {
        "HALO_EP_CAPACITY_DEDUP": _CAPACITY_DEDUP_ENABLED,
        "HALO_DEEPEP_GPU_TIMEOUT_SECONDS": _GPU_TIMEOUT_SECONDS,
        "HALO_DEEPEP_NUM_SMS": _NUM_SMS,
        "HALO_DEEPEP_NUM_QPS": _NUM_QPS,
        "HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK": GIN_MAX_TOKENS_PER_RANK,
        # Reported back in the unit the operator sets (MB), so a divergence message names the value
        # they can grep their --env-file for.
        "HALO_GRAD_BUCKET_MB": GRAD_BUCKET_MB,
        "DIST_NCCL_TIMEOUT_MINUTES": resolve_nccl_timeout_minutes(),
        "DIST_STORE_TIMEOUT_HOURS": resolve_store_timeout_hours(),
        "EP_DISABLE_GIN": os.environ.get("EP_DISABLE_GIN"),
    }


def verify_rank_uniform_env(force: bool = False) -> None:
    """Collective. Raise when ranks disagree on a setting in :func:`rank_uniform_ep_settings`.

    A divergence is a hang, not a slow run: ranks with ``HALO_EP_CAPACITY_DEDUP`` off run the
    capacity all-reduce on every MoE layer while ranks with it on run it once, so from layer 2 the
    collective counts diverge inside one EP group.

    Runs at most once per process; the entry scripts call it in distributed setup, before the weight
    load. ``force`` re-runs it for a caller that reaches DeepEP without the script scaffold (a
    standalone test harness). The latch closes only once the join could actually run: before
    ``init_process_group`` :func:`reject_divergent_settings` no-ops, and latching on such a call
    would compare nothing and then permanently disarm
    :func:`~src.distributed.expert_parallel.patching.create_ep_buffers`'s backstop.
    """
    global _ENV_UNIFORMITY_VERIFIED
    if _ENV_UNIFORMITY_VERIFIED and not force:
        return
    reject_divergent_settings(
        rank_uniform_ep_settings(),
        "Rank-uniform toolkit environment",
        "Every rank of the job must agree on these — the check is world-wide, because "
        "HALO_EP_CAPACITY_DEDUP changes how many collectives a rank runs, HALO_GRAD_BUCKET_MB sets "
        "the chunk boundaries of the bucketed gradient reductions (EP, TP and QLoRA sweeps alike), "
        "DIST_NCCL_TIMEOUT_MINUTES / DIST_STORE_TIMEOUT_HOURS decide which rank gives up on a join "
        "first, and the rest are wire parameters shared by both ends of the all-to-all.",
    )
    _ENV_UNIFORMITY_VERIFIED = dist.is_available() and dist.is_initialized()


def destroy_all_dispatchers() -> None:
    """Destroy every live DeepEP buffer; call before ``dist.destroy_process_group()``.

    Gin frees each buffer's symmetric heap through the group communicator, so a free after the group is
    gone faults with ``cudaErrorIllegalAddress``.

    The leading barrier is load-bearing: ``buffer.destroy()`` spins on the device-side NVLink barrier
    for ``HALO_DEEPEP_GPU_TIMEOUT_SECONDS``, and teardown is reached after rank-asymmetric work (a
    gathered save streaming tens of GB on the writer rank, an adapter unmerge, an eval). NCCL's
    barrier blocks the host, so aligning here puts that wait under the process-group timeout rather
    than the device one; without it the first rank in spends the whole budget and the job fails with
    ``DeepEP NVLink barrier timeout`` followed by ``cudaErrorLaunchFailure``. It only aligns ranks
    that still agree on the collective order; an already desynchronized job arrives here skewed
    regardless.

    Unconditional, never gated on ``_LIVE_DISPATCHERS``: a pipeline stage holding no MoE layers has
    no dispatchers, and a gated barrier would be rank-asymmetric.
    """
    barrier()
    for dispatcher in list(_LIVE_DISPATCHERS.values()):
        dispatcher.destroy()
    free_unclaimed_ep_buffers()
    # The capacity entries hold their EP group (see ``_CAPACITY_CACHE``); keeping them past teardown
    # would pin every communicator the job ever built for the life of the process.
    _CAPACITY_CACHE.clear()


def free_unclaimed_ep_buffers() -> int:
    """Free every DeepEP buffer no live MoE layer still claims. Returns how many were freed.

    Collective — call only where every rank arrives, and only after a barrier. Sweeps up claims
    dropped by the garbage collector, which cannot free a buffer itself
    (:meth:`_SharedArena.release`); without it a model that went out of scope leaks its arena for the
    rest of the process.
    """
    return _SharedArena.free_unclaimed()


def bump_forward_generation() -> None:
    """Open a new per-forward capacity scope. Call once per backbone forward, on every rank.

    :func:`register_forward_generation_hook` covers the forwards that enter through a module's
    ``__call__``; a caller that enters the backbone directly (TRL's chunked log-prob path, which peels
    ``base_model`` off the wrapper) has to open the scope itself.
    """
    global _FORWARD_GENERATION
    _FORWARD_GENERATION += 1


def _bump_forward_generation(*_args) -> None:
    """Pre-hook signature over :func:`bump_forward_generation`."""
    bump_forward_generation()


def register_forward_generation_hook(model: torch.nn.Module) -> None:
    """Let :class:`_ElasticBackend` dedup its capacity all-reduce to once per forward.

    Every MoE layer in a forward sees the same per-rank token count, so only the first all-reduces the
    global max and the rest reuse it, saving N-1 collectives and device syncs per forward. The
    generation is rank-uniform, so the first-layer cache miss lands on all ranks together and the
    collective stays matched.

    Register on the outermost module the loop calls, and re-register whenever a wrapper lands above
    that module: a task-typed ``PeftModel`` reaches the model it wraps through ``.forward()``, which
    runs no pre-hook. Idempotence is read off this module's own hooks rather than a marker attribute,
    which PEFT would forward down to the wrapped model. A forward that enters the backbone directly
    opens its own scope through :func:`bump_forward_generation`.
    """
    if not _CAPACITY_DEDUP_ENABLED or _bump_forward_generation in model._forward_pre_hooks.values():
        return
    model.register_forward_pre_hook(_bump_forward_generation)


class _SharedArena:
    """One DeepEP buffer, shared by every MoE layer that dispatches the same shape on one EP group.

    A buffer per layer costs the arena size times the layer count (~4 GiB each for the V2
    receive-side arena at ep64 with 8k tokens/rank, a flat ~100 MiB for the V1 chunked pipeline).
    Sharing is safe on both because ``dispatch`` copies its results into allocator-owned tensors and
    the returned handle carries the routing layout rather than buffer state, so a later layer's
    dispatch leaves an earlier layer's pending backward intact.

    Subclasses build the buffer and define the shape key that decides who may share one; this class
    holds the refcount and frees the buffer once the last layer releases it. Every arena is
    registered, shared or not, so :meth:`free_unclaimed` sees all of them.
    """

    _REGISTRY: dict[tuple, _SharedArena] = {}

    def __init__(self, group, key: tuple):
        # Strong ref: the key holds id(group), which the allocator could otherwise recycle.
        self.group = group
        self.key = key
        self.buffer = None
        self._users = 0

    @classmethod
    def is_shared(cls) -> bool:
        """Whether the layers on one EP group share a buffer, or each layer keeps its own."""
        return True

    @classmethod
    def acquire(cls, group, *shape) -> Self:
        """The arena of this class serving ``group`` at ``shape``, created on first use."""
        key = (cls.__name__, id(group), *shape)
        arena = cls._REGISTRY.get(key) if cls.is_shared() else None
        # isinstance narrows the base-typed registry value; the key carries the class name, so
        # anything filed under it is of this class.
        if not isinstance(arena, cls):
            # A private arena is registered too, under a key nothing else can collide with.
            arena = cls(group, key if cls.is_shared() else (*key, next(_ARENA_SERIAL)), *shape)
            cls._REGISTRY[arena.key] = arena
        arena._users += 1
        return arena

    def _owned_buffers(self) -> list:
        """Every buffer this arena must free; subclasses may retain ones they superseded."""
        return [self.buffer] if self.buffer is not None else []

    def release(self, *, free_buffer: bool) -> None:
        """Drop one layer's claim; the last claim released frees the buffer when ``free_buffer``.

        Freeing is collective: ``buffer.destroy()`` spins on the device-side NVLink barrier for
        ``HALO_DEEPEP_GPU_TIMEOUT_SECONDS``. A finalizer must therefore pass ``free_buffer=False``,
        since it runs whenever this rank's garbage collector fires, at a moment no other rank agrees
        on, and one rank entering that barrier alone spends the whole budget and aborts the job with
        ``cudaErrorLaunchFailure``. :func:`free_unclaimed_ep_buffers` sweeps up what a finalizer
        leaves behind, at a host-aligned teardown.
        """
        self._users -= 1
        if self._users > 0 or not free_buffer:
            return
        self._free()

    def _free(self) -> None:
        """Destroy every buffer this arena holds (collective; host-aligned callers only)."""
        _SharedArena._REGISTRY.pop(self.key, None)
        for buffer in self._owned_buffers():
            try:
                buffer.destroy()
            except Exception as e:
                logger.warning(f"Failed to destroy DeepEP buffer: {e}")
        self.buffer = None

    @classmethod
    def free_unclaimed(cls) -> int:
        """Free every arena no live layer still claims. Returns how many were freed.

        A dispatcher dropped by the garbage collector decrements its arena without freeing it (see
        :meth:`release`), leaving it at zero claims with nobody left to free it. Claimed arenas are
        left alone: a second model in the same process (a distillation teacher, a reference policy)
        may still be dispatching through one.
        """
        unclaimed = [arena for arena in cls._REGISTRY.values() if arena._users <= 0]
        for arena in unclaimed:
            arena._free()
        return len(unclaimed)


class _ElasticArena(_SharedArena):
    """A DeepEP V2 :class:`deep_ep.ElasticBuffer` arena: receive-side, tokens/rank-sized, grow-only.

    Sized for every peer sending its whole batch, so it is linear in the dispatch-group width and
    independent of the layer count: ``ep_size × tokens_per_rank × (2·hidden + 4·top_k)``, ~4 GiB at
    ep64 with 8k tokens/rank.

    Sharing needs one capacity for the whole forward, which the capacity dedup provides. With dedup
    off a later layer may present more tokens than the first and grow the arena mid-forward, so each
    layer then keeps a private one instead.

    Retained superseded arenas bound the memory of a variable-length run: a grow retires the previous
    buffer rather than freeing it (an in-flight microbatch's backward may still name it) and only
    teardown frees it, so the HBM footprint is the sum of every capacity the arena ever held. Freeing
    a retirement earlier needs a point where no live handle can name it, and no rank-uniform such
    point exists here. Pre-size instead: pack the corpus, or keep the first forward's tokens/rank at
    the run's maximum (``max_length × per_device_train_batch_size``).
    """

    def __init__(self, group, key: tuple, padded_hidden: int, num_topk: int, num_qps: int):
        super().__init__(group, key)
        self.padded_hidden = padded_hidden
        self.num_topk = num_topk
        self.num_qps = num_qps
        self.capacity = 0
        # A grow cannot free the old arena in place: under a pipeline schedule an earlier microbatch's
        # backward may still hold handles against it. Retired arenas are freed at teardown instead.
        self._retired: list = []

    @classmethod
    def is_shared(cls) -> bool:
        return _CAPACITY_DEDUP_ENABLED

    def _owned_buffers(self) -> list:
        return [*self._retired, *super()._owned_buffers()]

    def ensure(self, needed: int) -> bool:
        """Build or grow the arena to ``needed`` tokens per rank (collective). True if it (re)built."""
        if self.buffer is not None and needed <= self.capacity:
            return False
        if self.buffer is not None:
            self._retired.append(self.buffer)
            # One line per EP group, not per job: capacity is all-reduced within the dispatch group,
            # so a multi-group topology grows independently and a global-rank-0 gate would hide all
            # groups but its own.
            if _is_group_leader(self.group):
                logger.info(
                    f"DeepEP arena grew {self.capacity} -> {needed} tokens/rank; the previous arena is "
                    f"retired rather than freed (handles from in-flight microbatches may still name it) "
                    f"and is released at teardown — {len(self._retired)} retained so far."
                )
        self.buffer = deep_ep().ElasticBuffer(
            self.group,
            num_max_tokens_per_rank=needed,
            hidden=self.padded_hidden,
            num_topk=self.num_topk,
            explicitly_destroy=True,
            # A pinned QP count must also be allocated: dispatch asserts `num_qps <= num_allocated_qps`
            # and DeepEP's automatic allocation is 17 (65/129 in hybrid mode). 0 keeps that sizing.
            num_allocated_qps=self.num_qps,
            num_gpu_timeout_secs=_GPU_TIMEOUT_SECONDS,
        )
        self.capacity = needed
        return True

    def _free(self) -> None:
        super()._free()
        self._retired.clear()
        self.capacity = 0


class _LegacyArena(_SharedArena):
    """A DeepEP V1 :class:`deep_ep.Buffer` arena: a fixed chunked pipeline sized by hidden and ep_size.

    Token-count-independent, so it never grows and every layer on the group can share one whatever
    the capacity dedup is set to.
    """

    def __init__(self, group, key: tuple, num_nvl_bytes: int):
        super().__init__(group, key)
        self.num_nvl_bytes = num_nvl_bytes

    def ensure(self) -> bool:
        """Build the buffer on first use (collective). True if it built."""
        if self.buffer is not None:
            return False
        self.buffer = deep_ep().Buffer(
            self.group,
            num_nvl_bytes=self.num_nvl_bytes,
            num_rdma_bytes=0,
            low_latency_mode=False,
            explicitly_destroy=True,
        )
        return True


class _DeepEPBackend(ABC):
    """A DeepEP transport backend held by a :class:`DeepEPDispatcher`.

    Each subclass implements its buffer's lifetime and the four dispatch/combine ops, where
    ``dispatch.backward`` is a combine and ``combine.backward`` is a handle-reuse dispatch (the
    ``handle`` carries the routing layout for both). EP topology is read from the dispatcher.
    """

    def __init__(self, dispatcher: DeepEPDispatcher):
        self._d = dispatcher
        self._arena: _SharedArena | None = None

    @property
    def buffer(self):
        """The shared arena's live DeepEP buffer (``None`` before the first dispatch)."""
        return self._arena.buffer if self._arena is not None else None

    def destroy(self, *, free_buffer: bool) -> None:
        """Drop this layer's claim on its buffer, freeing it when unclaimed and ``free_buffer``."""
        arena, self._arena = self._arena, None
        if arena is not None:
            arena.release(free_buffer=free_buffer)

    @abstractmethod
    def prepare(self) -> None:
        """Non-collective setup, safe to call during sequential model loading (env selection, etc.)."""

    @abstractmethod
    def ensure(self, num_tokens: int, num_topk: int) -> None:
        """Create (or grow) the buffer to serve ``num_tokens`` per rank. Collective over the EP group."""

    @abstractmethod
    def dispatch_fwd(self, x, topk_idx, topk_weights):
        """Forward dispatch → ``(recv_x, recv_topk_idx, recv_topk_weights, handle)``."""

    @abstractmethod
    def combine_grad(self, grad_recv_x, grad_recv_topk_weights, handle):
        """``dispatch.backward`` == combine → ``(grad_x, grad_topk_weights)``."""

    @abstractmethod
    def combine_fwd(self, x, topk_weights, handle):
        """Forward combine → ``combined_x``."""

    @abstractmethod
    def dispatch_grad(self, grad_combined_x, handle):
        """``combine.backward`` == dispatch reusing the cached layout via ``handle`` → ``grad_x``."""


class _ElasticBackend(_DeepEPBackend):
    """DeepEP V2 :class:`deep_ep.ElasticBuffer` (NCCL Gin), the default backend.

    Grow-only sizing from ``num_max_tokens_per_rank``, pads hidden to a multiple of 256, guards DeepEP's
    32-bit wire-index limit. Cross-node capable over RDMA.
    """

    _arena: _ElasticArena | None

    def __init__(self, dispatcher: DeepEPDispatcher):
        super().__init__(dispatcher)
        self._num_topk = 0
        self._resolved_num_sms = 0
        self._env_configured = False
        self._padded_hidden = padded_wire_hidden(dispatcher.hidden_dim)
        self._needs_pad = self._padded_hidden != dispatcher.hidden_dim

    @property
    def _capacity(self) -> int:
        return self._arena.capacity if self._arena is not None else 0

    def prepare(self) -> None:
        self._configure_env()

    def _configure_env(self) -> None:
        """Set the ElasticBuffer env from the EP topology (idempotent, process-global).

        ``EP_DISABLE_GIN``: Gin (RDMA) is unavailable on a single NVLink node, so it is disabled
        intra-node and enabled inter-node; an explicit user value is honored.
        ``EP_SUPPRESS_NCCL_CHECK`` cannot be set from here — DeepEP reads it inside
        ``check_nccl_so()`` at ``import deep_ep``, which dispatcher construction has already
        triggered — so this warns instead. Both images set it in the environment.

        The environment is process-global but ``_env_configured`` is per backend, and every MoE layer
        builds its own, so the lines below are bounded by :data:`_WARNED_ENV` (the warning, which
        every rank must be able to see) and by the main-process gate (the two info lines, identical
        on every rank of a group); a 92-layer model at 512 ranks would otherwise open with ~47k
        duplicate lines.
        """
        if self._env_configured:
            return
        self._env_configured = True
        if os.environ.get("EP_SUPPRESS_NCCL_CHECK") != "1":
            warn_once(
                logger,
                _WARNED_ENV,
                "suppress_nccl_check",
                "EP_SUPPRESS_NCCL_CHECK != 1. The NGC image ships HPC-X libnccl-net alongside the pip "
                "libnccl.so, and DeepEP's duplicate-NCCL guard flags the plugin as a second runtime. "
                "Set it in the process environment (both prebuilt images do); a Python-level write "
                "here is too late — DeepEP latches it at import.",
            )
        if "EP_DISABLE_GIN" not in os.environ:
            os.environ["EP_DISABLE_GIN"] = "0" if self._d.is_inter_node else "1"
        elif is_global_main_process():
            logger.info(f"DeepEP Gin backend: honoring explicit EP_DISABLE_GIN={os.environ['EP_DISABLE_GIN']}")
        if is_global_main_process():
            backend = "NCCL Gin (RDMA)" if self._d.is_inter_node else "NVLink intranode (Gin disabled)"
            logger.info(f"DeepEP ElasticBuffer backend: {backend} (EP_DISABLE_GIN={os.environ['EP_DISABLE_GIN']})")

    def ensure(self, num_tokens: int, num_topk: int) -> None:
        d = self._d
        # Reuse the first layer's capacity; the generation is rank-uniform, so a miss hits every rank together.
        gid = id(d.ep_group)
        cached = _CAPACITY_CACHE.get(gid) if _CAPACITY_DEDUP_ENABLED else None
        if cached is not None and cached[1] == _FORWARD_GENERATION and cached[2] == num_topk:
            # Judged against the aligned world capacity, not this rank's own first-layer count:
            # per-layer token counts are data-dependent per rank, so "does it fit" is the only
            # invariant the shared arena needs. A rank that outgrows the arena raises here while its
            # peers reach the dispatch collective; HALO_DEEPEP_GPU_TIMEOUT_SECONDS bounds that wait.
            needed = cached[3]
            if num_tokens > needed:
                raise RuntimeError(
                    f"EP capacity dedup: a MoE layer dispatched {num_tokens} tokens/rank, over the "
                    f"capacity {needed} cached at forward generation {_FORWARD_GENERATION}. Either this "
                    f"model's later MoE layers dispatch more tokens than its first, or this forward "
                    f"never opened a capacity scope of its own (see bump_forward_generation). Set "
                    f"HALO_EP_CAPACITY_DEDUP=0 to size every layer with its own all-reduce."
                )
        else:
            cap = torch.tensor([num_tokens], device="cuda", dtype=torch.int64)
            dist.all_reduce(cap, op=dist.ReduceOp.MAX, group=d.ep_group)
            needed = ep_dispatch_capacity(int(cap.item()))
            if _CAPACITY_DEDUP_ENABLED:
                _CAPACITY_CACHE[gid] = (d.ep_group, _FORWARD_GENERATION, num_topk, needed)

        reject_oversized_dispatch(
            needed, num_topk=num_topk, padded_hidden=self._padded_hidden, is_inter_node=d.is_inter_node
        )

        if self.buffer is not None and needed <= self._capacity and num_topk == self._num_topk:
            return

        self._configure_env()
        # A changed top-k reshapes the wire, so this layer moves to the arena for the new shape.
        if self._arena is not None and num_topk != self._num_topk:
            self._arena.release(free_buffer=True)
            self._arena = None
        if self._arena is None:
            self._arena = _ElasticArena.acquire(d.ep_group, self._padded_hidden, num_topk, _NUM_QPS)
        built = self._arena.ensure(needed)
        self._num_topk = num_topk
        # get_theoretical_num_sms zero-divides on some cross-NVLink-domain Blackwell topologies; fall back.
        if _NUM_SMS:
            self._resolved_num_sms = _NUM_SMS
        else:
            try:
                self._resolved_num_sms = self.buffer.get_theoretical_num_sms(d.num_experts, num_topk)
            except ZeroDivisionError:
                self._resolved_num_sms = _DEFAULT_INTERNODE_NUM_SMS
                logger.warning(
                    "DeepEP get_theoretical_num_sms hit a zero-division on this inter-node topology; "
                    f"falling back to num_sms={_DEFAULT_INTERNODE_NUM_SMS} (override with HALO_DEEPEP_NUM_SMS)."
                )

        if built:
            mode = "inter-node (Gin)" if d.is_inter_node else "intra-node (NVLink)"
            ep_tp = d.ep_config.expert_tp_size
            tp_info = f", expert_tp={ep_tp}" if ep_tp > 1 else ""
            pad_info = f", hidden_padded={self._padded_hidden}" if self._needs_pad else ""
            scope = "shared by every MoE layer" if self._arena.is_shared() else "private to this layer"
            logger.info(
                f"DeepEP ElasticBuffer ({mode}, {scope}): ep_size={d.ep_size}, hidden={d.hidden_dim}"
                f"{pad_info}, num_max_tokens_per_rank={needed}, num_sms={self._resolved_num_sms}{tp_info}"
            )

    # Hidden padding lives on the wire only: stripped before results re-enter autograd, so callers
    # see hidden_dim.

    def dispatch_fwd(self, x, topk_idx, topk_weights):
        recv_x, recv_topk_idx, recv_topk_weights, handle, _ = self.buffer.dispatch(
            self._pad(x),
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=self._d.num_experts,
            num_max_tokens_per_rank=self._capacity,
            num_sms=self._resolved_num_sms,
            num_qps=_NUM_QPS,
        )
        return self._slice(recv_x), recv_topk_idx, recv_topk_weights, handle

    # ``num_qps`` is passed on the combines too: combine does not inherit it from the handle (only
    # ``num_sms`` is) but recomputes ``get_theoretical_num_qps``, so without it the knob would cover
    # only the dispatch half of the all-to-all.

    def combine_grad(self, grad_recv_x, grad_recv_topk_weights, handle):
        grad_x, grad_w, _ = self.buffer.combine(
            self._pad(grad_recv_x), handle=handle, topk_weights=grad_recv_topk_weights, num_qps=_NUM_QPS
        )
        return self._slice(grad_x), grad_w

    def combine_fwd(self, x, topk_weights, handle):
        combined_x, _, _ = self.buffer.combine(
            self._pad(x), handle=handle, topk_weights=topk_weights, num_qps=_NUM_QPS
        )
        return self._slice(combined_x)

    def dispatch_grad(self, grad_combined_x, handle):
        # num_sms must be explicit here: a handle-reuse dispatch resolves it too late, so 0 raises NoneType % int.
        grad_x, _, _, _, _ = self.buffer.dispatch(
            self._pad(grad_combined_x), handle=handle, num_sms=self._resolved_num_sms, num_qps=_NUM_QPS
        )
        return self._slice(grad_x)

    def _pad(self, t: torch.Tensor) -> torch.Tensor:
        """Zero-pad the feature dim up to ``_padded_hidden`` for the all-to-all transport."""
        if not self._needs_pad:
            return t
        return F.pad(t, (0, self._padded_hidden - self._d.hidden_dim))

    def _slice(self, t: torch.Tensor) -> torch.Tensor:
        """Drop the transport padding, returning the real ``hidden_dim`` width."""
        if not self._needs_pad:
            return t
        return t[..., : self._d.hidden_dim].contiguous()


class _LegacyBackend(_DeepEPBackend):
    """DeepEP V1 :class:`deep_ep.Buffer` (CUDA IPC P2P over NVLink): intranode / node-local EP only.

    Opt-in via ``ep_buffer_backend="legacy"``. Fixed-size chunked pipeline sized by hidden (not tokens),
    built once, no ``hidden % 256`` padding, numerically identical to elastic. Use it for long-context
    ep8 training where elastic deadlocks at extreme tok/rank (combine barrier races FSDP2's reduce-scatter);
    this CUDA-IPC buffer has no such barrier. Cross-node is rejected (``num_rdma_bytes=0``).
    """

    _arena: _LegacyArena | None

    def prepare(self) -> None:
        """Intranode legacy uses CUDA IPC over NVLink — no Gin/IBGDA env to select."""
        if _NUM_QPS:
            logger.warning(
                "HALO_DEEPEP_NUM_QPS=%d is ignored under ep_buffer_backend='legacy': the V1 buffer is "
                "intranode CUDA-IPC and takes no per-call queue-pair count. It applies to the default "
                "'elastic' backend only.",
                _NUM_QPS,
            )
        # Compares the parsed value, like the QP gate above: a docker-compose pass-through exports
        # the name with an empty value when it is absent, which resolves to the default.
        if _GPU_TIMEOUT_SECONDS != _DEEPEP_DEFAULT_GPU_TIMEOUT_SECONDS:
            logger.warning(
                "HALO_DEEPEP_GPU_TIMEOUT_SECONDS is ignored under ep_buffer_backend='legacy': the V1 "
                "buffer's ctor takes no timeout (its CUDA-IPC path has no NVLink spin barrier). It "
                "applies to the default 'elastic' backend only."
            )

    def ensure(self, num_tokens: int, num_topk: int) -> None:
        if self.buffer is not None:
            return
        d = self._d
        # Size hint takes a per-token byte size (hidden*2 for bf16), not a token count; the buffer is
        # token-independent.
        hidden_bytes = d.hidden_dim * 2
        num_nvl_bytes = 0
        buffer_cls = deep_ep().Buffer
        if _NUM_SMS:
            # V1 carries the SM count inside its per-rank-count Config tables rather than as a
            # dispatch argument, so the pin has to be applied before they are read. DeepEP asserts an
            # even count; surface that as a config error rather than an assert from inside the buffer.
            if _NUM_SMS % 2:
                raise ValueError(
                    f"HALO_DEEPEP_NUM_SMS={_NUM_SMS} must be even on ep_buffer_backend='legacy' "
                    f"(DeepEP V1 splits SMs into send/recv channel pairs)."
                )
            buffer_cls.set_num_sms(_NUM_SMS)
        for cfg in (buffer_cls.get_dispatch_config(d.ep_size), buffer_cls.get_combine_config(d.ep_size)):
            num_nvl_bytes = max(num_nvl_bytes, cfg.get_nvl_buffer_size_hint(hidden_bytes, d.ep_size))
        # The byte size is the whole shape here: it already folds in hidden, ep_size and the SM pin.
        self._arena = _LegacyArena.acquire(d.ep_group, num_nvl_bytes)
        if self._arena.ensure():
            logger.info(
                f"DeepEP legacy Buffer (intranode CUDA-IPC, shared by every MoE layer): "
                f"ep_size={d.ep_size}, hidden={d.hidden_dim}, num_nvl_bytes={num_nvl_bytes:,} "
                f"(token-count-independent buffer)"
            )

    def dispatch_fwd(self, x, topk_idx, topk_weights):
        n_per_rank, n_per_rdma, n_per_expert, is_in_rank, _ = self.buffer.get_dispatch_layout(
            topk_idx, self._d.num_experts
        )
        recv_x, recv_topk_idx, recv_topk_weights, _, handle, _ = self.buffer.dispatch(
            x,
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_tokens_per_rank=n_per_rank,
            num_tokens_per_rdma_rank=n_per_rdma,
            is_token_in_rank=is_in_rank,
            num_tokens_per_expert=n_per_expert,
        )
        return recv_x, recv_topk_idx, recv_topk_weights, handle

    def combine_grad(self, grad_recv_x, grad_recv_topk_weights, handle):
        grad_x, grad_w, _ = self.buffer.combine(grad_recv_x, handle=handle, topk_weights=grad_recv_topk_weights)
        return grad_x, grad_w

    def combine_fwd(self, x, topk_weights, handle):
        combined_x, _, _ = self.buffer.combine(x, handle=handle, topk_weights=topk_weights)
        return combined_x

    def dispatch_grad(self, grad_combined_x, handle):
        grad_x, _, _, _, _, _ = self.buffer.dispatch(grad_combined_x, handle=handle)
        return grad_x


_BACKENDS: dict[str, type[_DeepEPBackend]] = {"elastic": _ElasticBackend, "legacy": _LegacyBackend}


class DeepEPDispatcher:
    """Token dispatcher for EP communication: holds the EP topology and the selected transport backend.

    When ep_size <= 1 (all experts local) no inter-rank communication is needed; the dispatcher operates
    in no-op mode (dispatch/combine are identity) and builds no backend.
    """

    def __init__(self, ep_config: EPConfig, num_experts: int, hidden_dim: int):
        self.ep_config = ep_config
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim

        # Under expert-TP, dispatch/combine runs within the sub-EP group so each ETP group serves its subset.
        if ep_config.expert_tp_size > 1:
            self.ep_group = ep_config.dispatch_ep_group
        else:
            self.ep_group = ep_config.process_group
        self.ep_size = ep_config.ep_size

        # No-op mode: ep_size <= 1 → all experts local (DeepEP doesn't support ep_size=1; no comm needed).
        self._noop = self.ep_size <= 1
        self._destroyed = True
        # Public: the autograd Functions drive the backend's four dispatch/combine ops directly
        # (:class:`_DeepEPBackend`); the dispatcher only manages its lifetime.
        self.backend: _DeepEPBackend | None = None

        # No compile-sync warmup barrier: ElasticBuffer's long (100 s) dispatch/combine timeout absorbs FA4's
        # first-use CuTe JIT mid-forward, and the first capacity all-reduce already syncs the EP group.

        if self._noop:
            # Main-process only: one line per MoE layer per rank floods the log.
            if is_global_main_process():
                logger.info(f"DeepEPDispatcher (noop): all experts local, ep_size={self.ep_size}")
            return

        if not deep_ep_available():
            raise ImportError(
                "DeepEP is required for Expert Parallelism but is not installed. "
                "See agent-docs/infrastructure/deepep.md for installation instructions."
            )

        self.is_inter_node = ep_config.num_nodes > 1 and not ep_config.node_local
        self.backend = self._select_backend()

    def _select_backend(self) -> _DeepEPBackend:
        """Resolve ``ep_buffer_backend`` to a backend instance, rejecting unsupported combinations.

        ``auto``/``elastic`` pick ElasticBuffer (cross-node default); ``legacy`` is the V1 CUDA-IPC buffer
        (intranode / node-local). Fixed for the run — no fallback.
        """
        choice = (self.ep_config.ep_buffer_backend or "auto").strip().lower()
        if choice == "auto":
            choice = "elastic"
        if choice not in _BACKENDS:
            expected = ", ".join(repr(name) for name in sorted({"auto", *_BACKENDS}))
            raise ValueError(f"Unknown ep_buffer_backend={choice!r}; expected one of {expected}.")
        if choice == "legacy":
            # ParallelismConfig gates the same rules before any weight is read; this is the backstop
            # for a hand-built EPConfig that bypassed it. The dispatch group is what rides the
            # buffer: under expert-TP it is narrower than the EP group.
            reject_legacy_backend_topology(
                self.ep_size,
                is_cross_node=self.is_inter_node,
                gpus_per_node=get_local_world_size(),
                scope=f"ep_size={self.ep_size}, cross-node={self.is_inter_node}",
            )
        return _BACKENDS[choice](self)

    def create_buffer(self) -> None:
        """Prepare the dispatcher for buffer creation.

        The buffer is sized per-step and built lazily on the first dispatch; this only does the
        non-collective backend setup, so it is safe during sequential model loading.
        """
        if self._noop:
            return
        self.backend.prepare()

    def _ensure_buffer(self, num_tokens: int, num_topk: int) -> None:
        """Build/grow the transport buffer to hold ``num_tokens`` per rank (collective)."""
        self.backend.ensure(num_tokens, num_topk)
        _LIVE_DISPATCHERS[id(self)] = self
        self._destroyed = False

    def destroy(self, *, free_buffer: bool = True) -> None:
        """Release this layer's DeepEP buffer, freeing it when no layer is left to use it (idempotent).

        Collective by default: call it only from a rank-synchronized point.
        """
        if self._destroyed:
            return
        self._destroyed = True  # mark destroyed first to prevent re-entry
        _LIVE_DISPATCHERS.pop(id(self), None)
        if self.backend is not None:
            self.backend.destroy(free_buffer=free_buffer)

    def __del__(self, _is_finalizing=sys.is_finalizing):
        # Skip destroy() during interpreter shutdown (the CUDA context may already be torn down).
        # Bound as a default arg because interpreter teardown can clear this module's globals before
        # the finalizer runs, making a bare ``sys.is_finalizing()`` raise.
        if _is_finalizing():
            return
        # Never free the buffer from a finalizer: freeing runs a collective on DeepEP's device-side
        # NVLink barrier, and the garbage collector fires at a moment no other rank agrees on, so the
        # first rank in spends its whole timeout budget and aborts the job. Drop the claim only;
        # free_unclaimed_ep_buffers() collects what this leaves behind, at a barriered teardown.
        with contextlib.suppress(Exception):
            self.destroy(free_buffer=False)

    def dispatch(self, x: torch.Tensor, topk_idx: torch.Tensor, topk_weights: torch.Tensor):
        """Dispatch tokens to experts.

        No-op mode (ep_size <= 1): returns inputs unchanged (all experts local; indices already valid
        local indices).
        """
        if self._noop:
            return x, topk_idx, topk_weights, None
        self._ensure_buffer(x.shape[0], topk_idx.shape[1])
        return DeepEPDispatchFunction.apply(x, topk_idx, topk_weights, self, self.num_experts)

    def combine(self, x: torch.Tensor, topk_weights: torch.Tensor, handle: object) -> torch.Tensor:
        """Combine expert outputs back to original positions.

        No-op mode: returns output unchanged (contributions already accumulated locally via index_add_).
        """
        if self._noop:
            return x
        return DeepEPCombineFunction.apply(x, topk_weights, handle, self)
