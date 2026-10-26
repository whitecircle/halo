"""DeepEP buffer backends: the arenas that own DeepEP buffers, and the transport ops through them.

``elastic`` (the default; ``auto``) is V2 ``ElasticBuffer``, Gin across NVLink domains and NVLink within
one; ``legacy`` is the V1 CUDA-IPC ``Buffer``, intranode only, for long-context ep8 where the elastic
combine barrier races FSDP2's reduce-scatter. The one ``deep_ep`` importer, lazily: it runs ``cuInit``.
"""

from __future__ import annotations

import itertools
import logging
import os
from abc import ABC, abstractmethod
from functools import lru_cache
from types import ModuleType
from typing import TYPE_CHECKING, Self, get_args

import torch
import torch.distributed as dist

from src.distributed.expert_parallel.config import EPBufferBackend
from src.distributed.expert_parallel.deepep.capacity import (
    capacity_scope,
    ep_dispatch_capacity,
    padded_wire_hidden,
    reject_legacy_backend_topology,
    reject_oversized_dispatch,
)
from src.distributed.expert_parallel.deepep.settings import (
    DEEPEP_DEFAULT_GPU_TIMEOUT_S,
    DEEPEP_DEFAULT_INTERNODE_NUM_SMS,
    DEEPEP_GPU_TIMEOUT_S,
    DEEPEP_NUM_QPS,
    DEEPEP_NUM_SMS,
    EP_CAPACITY_DEDUP,
)
from src.log import warn_once
from src.runtime.consensus import rank_consensus
from src.runtime.ranks import get_local_world_size, is_global_main_process

if TYPE_CHECKING:
    from src.distributed.expert_parallel.deepep.dispatcher import DeepEPDispatcher

logger = logging.getLogger(__name__)

# ``warn_once`` scope of the knob warnings: every MoE layer builds its own backend.
_warned_env: set = set()
# Keys apart the private per-layer arenas built when sharing is off.
_arena_serial = itertools.count()
_deepep_env_configured = False


@lru_cache(maxsize=1)
def deep_ep() -> ModuleType:
    """The native ``deep_ep`` module, imported on first use. Raises ``ImportError`` when absent."""
    import deep_ep  # noqa: PLC0415 — optional native extension; importing it runs cuInit

    return deep_ep


def deep_ep_available() -> bool:
    """Whether the native extension imports on this host."""
    try:
        deep_ep()
    except (ImportError, OSError):
        return False
    return True


class _Arena:
    """One DeepEP buffer shared by every MoE layer dispatching one shape on one EP group: dispatch copies its
    results out and the handle carries the routing layout, so a later layer leaves a pending backward intact."""

    # Every arena, shared or private, in build order: the one registry every free walks.
    _registry: dict[tuple, _Arena] = {}

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
        arena = _Arena._registry.get(key) if cls.is_shared() else None
        # The key carries the class name, so anything filed under it is of this class.
        if not isinstance(arena, cls):
            # A private arena is registered too, under a key nothing else can collide with.
            arena = cls(group, key if cls.is_shared() else (*key, next(_arena_serial)), *shape)
            _Arena._registry[arena.key] = arena
        arena._users += 1
        return arena

    def release(self, *, free_buffer: bool) -> None:
        """Drop one layer's claim; with ``free_buffer``, COLLECTIVE over the group, free the buffer once no rank
        claims it. Freeing spins on DeepEP's device-side barrier, so a finalizer passes ``free_buffer=False``."""
        self._users -= 1
        if free_buffer and self._unclaimed_on_every_rank():
            self._free()

    def _unclaimed_on_every_rank(self) -> bool:
        """COLLECTIVE over the group. Whether no rank of the group still claims this arena."""
        return rank_consensus(self._users <= 0, group=self.group).all_ok

    def _free(self) -> None:
        """Destroy every buffer this arena holds (collective; host-aligned callers only)."""
        _Arena._registry.pop(self.key, None)
        for buffer in self._owned_buffers():
            try:
                buffer.destroy()
            except Exception as e:
                logger.warning(f"Failed to destroy DeepEP buffer: {e}")
        self.buffer = None

    def _owned_buffers(self) -> list:
        return [self.buffer] if self.buffer is not None else []


class _ElasticArena(_Arena):
    """A DeepEP V2 ``ElasticBuffer``: receive-side and grow-only, ``ep_size × tokens/rank × (2·hidden + 4·top_k)``;
    shared only under the capacity dedup, which fixes one capacity for the whole forward."""

    def __init__(self, group, key: tuple, padded_hidden: int, num_topk: int, num_qps: int, deterministic: bool):
        super().__init__(group, key)
        self.padded_hidden = padded_hidden
        self.num_topk = num_topk
        self.num_qps = num_qps
        # DeepEP's deterministic mode places each received token by prefix sums, not an atomic claim.
        self.deterministic = deterministic
        self.capacity = 0
        # A grow retires the old buffer rather than freeing it: an in-flight microbatch's backward may
        # still name it, and no rank-uniform point exists where none can. Teardown frees them.
        self._retired: list = []

    @classmethod
    def is_shared(cls) -> bool:
        return EP_CAPACITY_DEDUP

    def ensure(self, needed: int) -> bool:
        """Build or grow the buffer to ``needed`` tokens per rank (collective). True if it (re)built."""
        if self.buffer is not None and needed <= self.capacity:
            return False
        if self.buffer is not None:
            self._retired.append(self.buffer)
            # Logged per EP group: each group grows its own capacity.
            if not (dist.is_available() and dist.is_initialized()) or dist.get_rank(group=self.group) == 0:
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
            deterministic=self.deterministic,
            explicitly_destroy=True,
            # Dispatch asserts `num_qps <= num_allocated_qps`; 0 keeps DeepEP's automatic sizing.
            num_allocated_qps=self.num_qps,
            num_gpu_timeout_secs=DEEPEP_GPU_TIMEOUT_S,
        )
        self.capacity = needed
        return True

    def _free(self) -> None:
        super()._free()
        self._retired.clear()
        self.capacity = 0

    def _owned_buffers(self) -> list:
        return [*self._retired, *super()._owned_buffers()]


class _LegacyArena(_Arena):
    """A DeepEP V1 ``Buffer``: a fixed chunked pipeline sized by hidden and ep_size, never grown, so
    every layer on the group shares one whatever the capacity dedup is set to."""

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


def _configure_deepep_env(spans_domains: bool) -> None:
    """Set DeepEP's process-global env once per process: Gin only across NVLink domains unless ``EP_DISABLE_GIN``
    is set; ``EP_SUPPRESS_NCCL_CHECK`` is latched at DeepEP import, so a missing one only warns."""
    global _deepep_env_configured
    if _deepep_env_configured:
        return
    _deepep_env_configured = True
    if os.environ.get("EP_SUPPRESS_NCCL_CHECK") != "1":
        logger.warning(
            "EP_SUPPRESS_NCCL_CHECK != 1. The NGC image ships HPC-X libnccl-net alongside the pip "
            "libnccl.so, and DeepEP's duplicate-NCCL guard flags the plugin as a second runtime. "
            "Set it in the process environment (both prebuilt images do); a Python-level write "
            "here is too late — DeepEP latches it at import."
        )
    if "EP_DISABLE_GIN" not in os.environ:
        os.environ["EP_DISABLE_GIN"] = "0" if spans_domains else "1"
    elif is_global_main_process():
        logger.info(f"DeepEP Gin backend: honoring explicit EP_DISABLE_GIN={os.environ['EP_DISABLE_GIN']}")
    if is_global_main_process():
        backend = "NCCL Gin (RDMA)" if spans_domains else "NVLink intranode (Gin disabled)"
        logger.info(f"DeepEP ElasticBuffer backend: {backend} (EP_DISABLE_GIN={os.environ['EP_DISABLE_GIN']})")


class DeepEPBackend(ABC):
    """A dispatcher's transport: its claim on an arena and the four dispatch/combine ops, reading the EP
    topology off the dispatcher."""

    def __init__(self, dispatcher: DeepEPDispatcher):
        self._d = dispatcher
        self._arena: _Arena | None = None

    @property
    def buffer(self):
        """The arena's live DeepEP buffer (None before the first dispatch)."""
        return self._arena.buffer if self._arena is not None else None

    @abstractmethod
    def ensure(self, num_tokens: int, num_topk: int, *, deterministic: bool) -> None:
        """COLLECTIVE over the EP group. Build or grow the buffer to serve ``num_tokens`` per rank, with
        received tokens placed in a fixed order when ``deterministic``."""

    @abstractmethod
    def dispatch_forward(self, x, topk_idx, topk_weights):
        """Dispatch → ``(recv_x, recv_topk_idx, recv_topk_weights, handle)``."""

    @abstractmethod
    def dispatch_backward(self, grad_recv_x, grad_recv_topk_weights, handle):
        """A dispatch's backward, a combine → ``(grad_x, grad_topk_weights)``."""

    @abstractmethod
    def combine_forward(self, x, topk_weights, handle):
        """Combine → ``combined_x``."""

    @abstractmethod
    def combine_backward(self, grad_combined_x, handle):
        """A combine's backward, a dispatch reusing the routing layout ``handle`` carries → ``grad_x``."""

    def destroy(self, *, free_buffer: bool) -> None:
        """Drop this layer's claim on its buffer, freeing it when unclaimed and ``free_buffer``."""
        arena, self._arena = self._arena, None
        if arena is not None:
            arena.release(free_buffer=free_buffer)


class _ElasticBackend(DeepEPBackend):
    """DeepEP V2 ``ElasticBuffer``, the default: grow-only sizing, hidden zero-padded to a multiple of
    256 on the wire and sliced back before results re-enter autograd, the 32-bit wire index guarded."""

    _arena: _ElasticArena | None

    def __init__(self, dispatcher: DeepEPDispatcher):
        super().__init__(dispatcher)
        self._num_topk = 0
        self._resolved_num_sms = 0
        self._padded_hidden = padded_wire_hidden(dispatcher.hidden_dim)
        self._needs_pad = self._padded_hidden != dispatcher.hidden_dim

    def ensure(self, num_tokens: int, num_topk: int, *, deterministic: bool) -> None:
        d = self._d
        needed = self._resolve_capacity(num_tokens, num_topk)
        reject_oversized_dispatch(
            needed, num_topk=num_topk, padded_hidden=self._padded_hidden, spans_domains=d.spans_domains
        )

        # A changed top-k reshapes the wire and a changed mode (a forward ahead of Trainer.__init__)
        # rebuilds the buffer, so this layer moves to the arena for the new kind.
        mode_changed = self._arena is not None and self._arena.deterministic != deterministic
        new_kind = num_topk != self._num_topk or mode_changed
        if self.buffer is not None and needed <= self._arena.capacity and not new_kind:
            return

        _configure_deepep_env(d.spans_domains)
        if self._arena is not None and new_kind:
            self._arena.release(free_buffer=True)
            self._arena = None
        if self._arena is None:
            self._arena = _ElasticArena.acquire(
                d.ep_group, self._padded_hidden, num_topk, DEEPEP_NUM_QPS, deterministic
            )
        built = self._arena.ensure(needed)
        self._num_topk = num_topk
        self._resolved_num_sms = self._resolve_num_sms(num_topk)
        if built:
            self._log_built(needed, deterministic)

    def _resolve_capacity(self, num_tokens: int, num_topk: int) -> int:
        """The arena capacity this forward needs: the first MoE layer all-reduces it, the rest reuse it."""
        d = self._d
        gid = id(d.ep_group)
        cached = capacity_scope.lookup(gid, num_topk) if EP_CAPACITY_DEDUP else None
        if cached is not None:
            # Judged against the group capacity, as per-layer counts are data-dependent per rank: the rank
            # that outgrows it raises while its peers wait in the dispatch for HALO_DEEPEP_GPU_TIMEOUT_SECONDS.
            if num_tokens > cached:
                raise RuntimeError(
                    f"EP capacity dedup: a MoE layer dispatched {num_tokens} tokens/rank, over the "
                    f"capacity {cached} cached at forward generation {capacity_scope.generation}. Either this "
                    f"model's later MoE layers dispatch more tokens than its first, or this forward "
                    f"never opened a capacity scope of its own (see bump_forward_generation). Set "
                    f"HALO_EP_CAPACITY_DEDUP=0 to size every layer with its own all-reduce."
                )
            return cached
        cap = torch.tensor([num_tokens], device="cuda", dtype=torch.int64)
        dist.all_reduce(cap, op=dist.ReduceOp.MAX, group=d.ep_group)
        needed = ep_dispatch_capacity(int(cap.item()))
        if EP_CAPACITY_DEDUP:
            capacity_scope.store(d.ep_group, gid, num_topk, needed)
        return needed

    def _resolve_num_sms(self, num_topk: int) -> int:
        if DEEPEP_NUM_SMS:
            return DEEPEP_NUM_SMS
        try:
            return self.buffer.get_theoretical_num_sms(self._d.num_experts, num_topk)
        except ZeroDivisionError:
            # Seen on some cross-NVLink-domain Blackwell topologies.
            logger.warning(
                "DeepEP get_theoretical_num_sms hit a zero-division on this inter-node topology; "
                f"falling back to num_sms={DEEPEP_DEFAULT_INTERNODE_NUM_SMS} (override with HALO_DEEPEP_NUM_SMS)."
            )
            return DEEPEP_DEFAULT_INTERNODE_NUM_SMS

    def _log_built(self, needed: int, deterministic: bool) -> None:
        d = self._d
        mode = "inter-node (Gin)" if d.spans_domains else "intra-node (NVLink)"
        ep_tp = d.ep_config.expert_tp_size
        tp_info = f", expert_tp={ep_tp}" if ep_tp > 1 else ""
        pad_info = f", hidden_padded={self._padded_hidden}" if self._needs_pad else ""
        scope = "shared by every MoE layer" if self._arena.is_shared() else "private to this layer"
        det_info = ", deterministic dispatch" if deterministic else ""
        logger.info(
            f"DeepEP ElasticBuffer ({mode}, {scope}): ep_size={d.ep_size}, hidden={d.hidden_dim}"
            f"{pad_info}, num_max_tokens_per_rank={needed}, num_sms={self._resolved_num_sms}{tp_info}"
            f"{det_info}"
        )

    def dispatch_forward(self, x, topk_idx, topk_weights):
        recv_x, recv_topk_idx, recv_topk_weights, handle, _ = self.buffer.dispatch(
            self._pad(x),
            topk_idx=topk_idx,
            topk_weights=topk_weights,
            num_experts=self._d.num_experts,
            num_max_tokens_per_rank=self._arena.capacity,
            num_sms=self._resolved_num_sms,
            num_qps=DEEPEP_NUM_QPS,
        )
        return self._slice(recv_x), recv_topk_idx, recv_topk_weights, handle

    # ``num_qps`` is passed on the combines too: a combine inherits only ``num_sms`` from the handle and
    # recomputes the QP count, so without it the knob would cover only the dispatch half.

    def dispatch_backward(self, grad_recv_x, grad_recv_topk_weights, handle):
        grad_x, grad_w, _ = self.buffer.combine(
            self._pad(grad_recv_x), handle=handle, topk_weights=grad_recv_topk_weights, num_qps=DEEPEP_NUM_QPS
        )
        return self._slice(grad_x), grad_w

    def combine_forward(self, x, topk_weights, handle):
        combined_x, _, _ = self.buffer.combine(
            self._pad(x), handle=handle, topk_weights=topk_weights, num_qps=DEEPEP_NUM_QPS
        )
        return self._slice(combined_x)

    def combine_backward(self, grad_combined_x, handle):
        # num_sms must be explicit here: a handle-reuse dispatch resolves it too late, so 0 raises NoneType % int.
        grad_x, _, _, _, _ = self.buffer.dispatch(
            self._pad(grad_combined_x), handle=handle, num_sms=self._resolved_num_sms, num_qps=DEEPEP_NUM_QPS
        )
        return self._slice(grad_x)

    def _pad(self, t: torch.Tensor) -> torch.Tensor:
        """Zero-pad the feature dim to the wire width in one pass (``F.pad`` makes two)."""
        if not self._needs_pad:
            return t
        hidden = self._d.hidden_dim
        out = t.new_empty((*t.shape[:-1], self._padded_hidden))
        out[..., :hidden].copy_(t)
        out[..., hidden:].zero_()
        return out

    def _slice(self, t: torch.Tensor) -> torch.Tensor:
        """Drop the wire padding as a view: every consumer reads the padded row stride like a contiguous row."""
        if not self._needs_pad:
            return t
        return t[..., : self._d.hidden_dim]


class _LegacyBackend(DeepEPBackend):
    """DeepEP V1 ``Buffer`` (CUDA IPC over NVLink), intranode only: a token-count-independent chunked
    pipeline with no NVLink spin barrier, built once, no hidden padding."""

    _arena: _LegacyArena | None

    def __init__(self, dispatcher: DeepEPDispatcher):
        super().__init__(dispatcher)
        if DEEPEP_NUM_QPS:
            warn_once(
                logger,
                _warned_env,
                "legacy_num_qps",
                "HALO_DEEPEP_NUM_QPS=%d is ignored under ep_buffer_backend='legacy': the V1 buffer is "
                "intranode CUDA-IPC and takes no per-call queue-pair count. It applies to the default "
                "'elastic' backend only.",
                DEEPEP_NUM_QPS,
            )
        # The parsed value, not the variable's presence: a docker-compose pass-through exports the
        # name with an empty value when it is absent, which resolves to the default.
        if DEEPEP_GPU_TIMEOUT_S != DEEPEP_DEFAULT_GPU_TIMEOUT_S:
            warn_once(
                logger,
                _warned_env,
                "legacy_gpu_timeout",
                "HALO_DEEPEP_GPU_TIMEOUT_SECONDS is ignored under ep_buffer_backend='legacy': the V1 "
                "buffer's ctor takes no timeout (its CUDA-IPC path has no NVLink spin barrier). It "
                "applies to the default 'elastic' backend only.",
            )

    def ensure(self, num_tokens: int, num_topk: int, *, deterministic: bool) -> None:
        # V1 places received tokens at offsets prefix-summed over source rank and channel, so the
        # receive order is fixed whatever ``deterministic`` asks.
        if self.buffer is not None:
            return
        d = self._d
        # The size hint takes a per-token byte size (bf16), not a token count.
        hidden_bytes = d.hidden_dim * 2
        num_nvl_bytes = 0
        buffer_cls = deep_ep().Buffer
        if DEEPEP_NUM_SMS:
            # V1 carries the SM count in its per-rank-count Config tables, so the pin lands before they
            # are read; DeepEP asserts an even count, surfaced here as a config error.
            if DEEPEP_NUM_SMS % 2:
                raise ValueError(
                    f"HALO_DEEPEP_NUM_SMS={DEEPEP_NUM_SMS} must be even on ep_buffer_backend='legacy' "
                    f"(DeepEP V1 splits SMs into send/recv channel pairs)."
                )
            buffer_cls.set_num_sms(DEEPEP_NUM_SMS)
        for cfg in (buffer_cls.get_dispatch_config(d.ep_size), buffer_cls.get_combine_config(d.ep_size)):
            num_nvl_bytes = max(num_nvl_bytes, cfg.get_nvl_buffer_size_hint(hidden_bytes, d.ep_size))
        # The byte size is the whole shape: it already folds in hidden, ep_size and the SM pin.
        self._arena = _LegacyArena.acquire(d.ep_group, num_nvl_bytes)
        if self._arena.ensure():
            logger.info(
                f"DeepEP legacy Buffer (intranode CUDA-IPC, shared by every MoE layer): "
                f"ep_size={d.ep_size}, hidden={d.hidden_dim}, num_nvl_bytes={num_nvl_bytes:,} "
                f"(token-count-independent buffer)"
            )

    def dispatch_forward(self, x, topk_idx, topk_weights):
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

    def dispatch_backward(self, grad_recv_x, grad_recv_topk_weights, handle):
        grad_x, grad_w, _ = self.buffer.combine(grad_recv_x, handle=handle, topk_weights=grad_recv_topk_weights)
        return grad_x, grad_w

    def combine_forward(self, x, topk_weights, handle):
        combined_x, _, _ = self.buffer.combine(x, handle=handle, topk_weights=topk_weights)
        return combined_x

    def combine_backward(self, grad_combined_x, handle):
        grad_x, _, _, _, _, _ = self.buffer.dispatch(grad_combined_x, handle=handle)
        return grad_x


def resolve_ep_buffer_backend(choice: str | None) -> str:
    """The backend ``choice`` selects, fixed for the run: ``auto`` is ``elastic``; an unknown name raises."""
    choice = (choice or "auto").strip().lower()
    if choice == "auto":
        choice = "elastic"
    if choice not in get_args(EPBufferBackend):
        expected = ", ".join(repr(name) for name in sorted(get_args(EPBufferBackend)))
        raise ValueError(f"Unknown ep_buffer_backend={choice!r}; expected one of {expected}.")
    return choice


def build_ep_backend(dispatcher: DeepEPDispatcher, backend: str | None) -> DeepEPBackend:
    """The transport ``backend`` selects for ``dispatcher``. A legacy one is checked against the dispatch
    group first, the backstop for a hand-built ``EPConfig`` that bypassed ``ParallelismConfig``."""
    if resolve_ep_buffer_backend(backend) == "elastic":
        return _ElasticBackend(dispatcher)
    reject_legacy_backend_topology(
        dispatcher.ep_size,
        is_cross_node=dispatcher.spans_domains,
        gpus_per_node=get_local_world_size(),
        scope=f"expert_parallel_size={dispatcher.ep_size}, cross-node={dispatcher.spans_domains}",
    )
    return _LegacyBackend(dispatcher)


def free_unclaimed_ep_buffers() -> int:
    """COLLECTIVE over each arena's EP group; call after a barrier. Free every DeepEP buffer no MoE layer on
    any rank of its group claims, agreed arena by arena in build order; returns how many it freed.

    It sweeps up the claims the garbage collector dropped, which a finalizer cannot free; an arena still
    claimed on some rank (a teacher or reference model dispatching) stays until teardown.
    """
    unclaimed = [arena for arena in _Arena._registry.values() if arena._unclaimed_on_every_rank()]
    for arena in unclaimed:
        arena._free()
    return len(unclaimed)


def free_all_ep_buffers() -> None:
    """COLLECTIVE over each arena's EP group; teardown only. Free every arena, claimed or not, in build
    order, which every rank of a group shares whatever its collector reclaimed."""
    for arena in list(_Arena._registry.values()):
        arena._free()
