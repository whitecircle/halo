"""DeepEP buffer sizing: wire tables and alignments, transport ceilings, the per-forward capacity."""

from __future__ import annotations

import torch

from halo.distributed.expert_parallel.deepep.settings import EP_CAPACITY_DEDUP
from halo.env import env_int

# DeepEP V1 (legacy) drives its CUDA-IPC buffer over at most this many NVLink peers; its NVSHMEM
# internode path is not wired up here, hence ``num_rdma_bytes=0``.
DEEPEP_V1_MAX_NVL_PEERS = 8
# Dispatch-group widths V1 ships a tuned ``Config`` for (its ``config_map`` keys);
# ``Buffer.get_dispatch_config`` asserts on any other width at the first dispatch.
DEEPEP_V1_CONFIG_RANKS = frozenset({2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 144, 160})
# DeepEP V2 sizes its per-rank arena in whole blocks of this many tokens.
EP_CAPACITY_ALIGN = 256
# The V2 combine kernel's TMA path requires a wire hidden width that is a multiple of this.
DEEPEP_HIDDEN_ALIGN = 256
# DeepEP offsets the per-rank wire buffer with 32-bit indices; extent >= 2**31 wraps into an illegal access.
DEEPEP_INDEX_LIMIT = 2**31
# Cross-node (Gin) dispatch above this many tokens/rank wedges instead of erroring; 0 disables.
# Boundary and fault signatures: agent-docs/infrastructure/deepep.md#expert-parallelism-over-aws-efa.
GIN_MAX_TOKENS_PER_RANK = env_int("HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK", 8192)


class CapacityScope:
    """The capacity the first MoE layer of a forward all-reduced per EP group, reused by later layers. Every MoE layer
    sees the same tokens/rank and the generation advances once per forward on every rank, so a miss and its all-reduce
    land on all ranks of a group together."""

    def __init__(self):
        self.generation = 0
        # id(group) -> (group, generation, top_k, capacity). The group is held because its id is the
        # key, which the allocator could otherwise recycle onto a rebuilt group.
        self._entries: dict[int, tuple[object, int, int, int]] = {}

    def bump(self) -> None:
        self.generation += 1

    def lookup(self, group_id: int, top_k: int) -> int | None:
        entry = self._entries.get(group_id)
        if entry is None or entry[1] != self.generation or entry[2] != top_k:
            return None
        return entry[3]

    def store(self, group, group_id: int, top_k: int, capacity: int) -> None:
        self._entries[group_id] = (group, self.generation, top_k, capacity)

    def clear(self) -> None:
        """Drop every entry and the groups they hold (EP teardown)."""
        self._entries.clear()


capacity_scope = CapacityScope()


def ep_dispatch_capacity(tokens_per_rank: int) -> int:
    """Round up to :data:`EP_CAPACITY_ALIGN`, so similar-sized batches reuse one buffer instead of rebuilding it every
    few tokens of growth."""
    return ((tokens_per_rank + EP_CAPACITY_ALIGN - 1) // EP_CAPACITY_ALIGN) * EP_CAPACITY_ALIGN


def padded_wire_hidden(hidden_dim: int) -> int:
    return ((hidden_dim + DEEPEP_HIDDEN_ALIGN - 1) // DEEPEP_HIDDEN_ALIGN) * DEEPEP_HIDDEN_ALIGN


def bump_forward_generation(*_hook_args) -> None:
    """Open a new per-forward capacity scope; call once per backbone forward, on every rank. A caller that enters the
    backbone directly (TRL's chunked log-prob path peels ``base_model`` off the wrapper) calls it itself."""
    capacity_scope.bump()


def register_forward_generation_hook(model: torch.nn.Module) -> None:
    """Bump the capacity generation once per ``model`` forward. Register on the outermost module the loop calls, and
    again whenever a wrapper lands above it: a task-typed ``PeftModel`` reaches the model it wraps through
    ``.forward()``, which runs no pre-hook. Idempotence is read off these hooks, since PEFT forwards a marker attribute
    to the wrapped model."""
    if not EP_CAPACITY_DEDUP or bump_forward_generation in model._forward_pre_hooks.values():
        return
    model.register_forward_pre_hook(bump_forward_generation)


def reject_legacy_backend_topology(dispatch_size: int, *, is_cross_node: bool, gpus_per_node: int, scope: str) -> None:
    """Raise for a dispatch topology a V1 (``legacy``) buffer cannot drive. The gate is the dispatch group, which
    expert TP makes narrower than the EP group."""
    if is_cross_node:
        raise ValueError(
            f"ep_buffer_backend='legacy' cannot drive a CROSS-NODE EP group ({scope}): the V1 buffer "
            f"is built with num_rdma_bytes=0 and its NVSHMEM internode path is not wired up. Use the "
            f"default 'elastic' backend, or keep the EP group node-local."
        )
    if dispatch_size > DEEPEP_V1_MAX_NVL_PEERS or dispatch_size > gpus_per_node:
        raise ValueError(
            f"ep_buffer_backend='legacy' cannot drive a {dispatch_size}-rank dispatch group ({scope}): "
            f"the V1 buffer is intranode CUDA-IPC over at most {DEEPEP_V1_MAX_NVL_PEERS} NVLink peers "
            f"within one OS node ({gpus_per_node} GPUs here). Use the default 'elastic' backend."
        )
    if dispatch_size not in DEEPEP_V1_CONFIG_RANKS:
        usable = sorted(w for w in DEEPEP_V1_CONFIG_RANKS if w <= min(DEEPEP_V1_MAX_NVL_PEERS, gpus_per_node))
        raise ValueError(
            f"ep_buffer_backend='legacy' has no tuned Config for a {dispatch_size}-rank dispatch "
            f"group; DeepEP V1 asserts on anything outside its shipped tables at the first dispatch. "
            f"Usable widths here: {usable}. Or use the default 'elastic' backend."
        )


def reject_oversized_dispatch(capacity: int, *, num_topk: int, padded_hidden: int, spans_domains: bool) -> None:
    """Raise past the 32-bit wire index or, on a group spanning NVLink domains, past the proxy-Gin ceiling."""
    index_extent = capacity * num_topk * padded_hidden
    if index_extent >= DEEPEP_INDEX_LIMIT:
        raise ValueError(
            f"DeepEP dispatch exceeds the 32-bit wire-index limit: num_max_tokens_per_rank={capacity} × "
            f"num_topk={num_topk} × padded_hidden={padded_hidden} = {index_extent:,} ≥ "
            f"{DEEPEP_INDEX_LIMIT:,}. The dispatch kernel would illegal-access. The buffer is per-rank, "
            f"so expert_parallel_size does not lower it — reduce the tokens in one MoE forward: "
            f"per_device_train_batch_size=1, a shorter sequence (max_length), or in env-GRPO fewer "
            f"generations / shorter trajectories (reasoning effort / max_turns / rollout length). At "
            f"~175k tokens/rank, far beyond training."
        )
    if spans_domains and GIN_MAX_TOKENS_PER_RANK and capacity > GIN_MAX_TOKENS_PER_RANK:
        raise ValueError(
            f"Cross-node DeepEP GIN dispatch: num_max_tokens_per_rank={capacity} exceeds the validated "
            f"EFA proxy-GIN ceiling ({GIN_MAX_TOKENS_PER_RANK}). A larger dispatch wedges in transit "
            f"instead of erroring — the receive counts never arrive ('Dispatch CPU wait ... received "
            f"count 0', or Xid 109 CTX SWITCH TIMEOUT cascading into Xid 43). Reduce the tokens in one "
            f"MoE forward (per_device_train_batch_size, max_length), or prefer node-local EP "
            f"(ep_scope=node) with DP across nodes — cross-node dispatch is also latency-bound and "
            f"several times slower. HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK overrides (0 disables) after "
            f"validating a larger dispatch end-to-end on your fabric."
        )
