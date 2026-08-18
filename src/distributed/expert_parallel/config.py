"""Expert Parallelism configuration and process-group management.

:class:`EPConfig` owns the EP topology (group sizes, scope, expert TP) and creates the process groups
for token routing, gradient sync, and expert TP.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import torch.distributed as dist

from src.distributed.group_layout import (
    cross_node_group_ranks,
    cross_node_layout,
    cross_node_rank_and_group,
    cross_node_replica_ranks,
    etp_dispatch_coords,
    node_local_group_ranks,
    node_local_groups_per_domain,
    node_local_rank_and_group,
    node_local_replica_ranks,
    reject_cross_node_ep_group,
    reject_cross_node_etp_shape,
    reject_node_local_ep_group,
)
from src.distributed.runtime import (
    get_global_rank,
    get_global_world_size,
    get_local_world_size,
    get_nccl_timeout,
    is_global_main_process,
)
from src.env import env_int
from src.models.moe_balancing import ROUTER_EXPERT_COUNT_FIELDS, get_first_router_field

logger = logging.getLogger(__name__)

# DeepEP V1 (``ep_buffer_backend: legacy``) capability limits.
# V1 drives its CUDA-IPC buffer over at most this many NVLink peers (its NVSHMEM internode path is not
# wired up here, hence ``num_rdma_bytes=0``). The one home for the number, since both the config-time
# gate (``ParallelismConfig._validate_ep_buffer_backend``) and the runtime backend selection
# (``DeepEPDispatcher``) read it, and a drift between the two would move a rejection back to
# post-model-load.
DEEPEP_V1_MAX_NVL_PEERS = 8
# Dispatch-group widths V1 ships a tuned ``Config`` for; ``Buffer.get_dispatch_config`` asserts on any
# other value, at the first dispatch. Mirrors the shipped ``config_map`` keys so the config-time gate
# rejects a width that would otherwise fail inside DeepEP after the whole multi-GPU load.
DEEPEP_V1_CONFIG_RANKS = frozenset({2, 4, 8, 16, 24, 32, 48, 64, 96, 128, 144, 160})

# DeepEP V2 (``ep_buffer_backend: elastic``) transport limits. Here rather than in the dispatcher for
# the same reason as the V1 numbers above: ``ParallelismConfig`` judges a run's declared token budget
# against them before any weight is read, and the dispatcher judges the batch in hand against them at
# buffer sizing — one home, so an early refusal and the runtime backstop cannot disagree. (The
# dispatcher module itself is deliberately NOT importable from ``ParallelismConfig`` — see
# ``src/distributed/__init__.py``.)
#
# DeepEP's per-rank arena is sized in whole blocks of this many tokens.
EP_CAPACITY_ALIGN = 256
# The V2 combine kernel's TMA path requires a wire hidden width that is a multiple of this.
DEEPEP_HIDDEN_ALIGN = 256
# DeepEP offsets the per-rank wire buffer with 32-bit indices; extent >= 2**31 wraps into an illegal access.
DEEPEP_INDEX_LIMIT = 2**31
# Cross-node (Gin) dispatch wedges above this many tokens/rank instead of erroring — the receive
# counts never arrive — so an oversized shape is rejected at buffer sizing (and, when the run's token
# budget is known, at config time). Intra-node NVLink dispatch is unaffected; 0 disables. Boundary and
# fault signatures: agent-docs/infrastructure/deepep.md#expert-parallelism-over-aws-efa.
GIN_MAX_TOKENS_PER_RANK = env_int("HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK", 8192)


def ep_dispatch_capacity(tokens_per_rank: int) -> int:
    """Per-rank arena capacity DeepEP is built for when a rank presents ``tokens_per_rank`` tokens.

    The one home for the sizing arithmetic, so the two places that judge a capacity cannot disagree
    about the number: the dispatcher applies it to the all-reduced group MAX at buffer sizing, and
    ``ParallelismConfig`` applies it to the run's declared per-rank token budget at config time.
    The rounding is slack, not waste — it lets similar-sized batches reuse one buffer instead of
    retiring and rebuilding the arena on every few tokens of growth.
    """
    return ((tokens_per_rank + EP_CAPACITY_ALIGN - 1) // EP_CAPACITY_ALIGN) * EP_CAPACITY_ALIGN


def padded_wire_hidden(hidden_dim: int) -> int:
    """Hidden width on the DeepEP wire — ``hidden_dim`` rounded up to :data:`DEEPEP_HIDDEN_ALIGN`.
    Non-conforming hidden is zero-padded onto the wire and sliced back, symmetrically across
    forward and backward."""
    return ((hidden_dim + DEEPEP_HIDDEN_ALIGN - 1) // DEEPEP_HIDDEN_ALIGN) * DEEPEP_HIDDEN_ALIGN


def reject_legacy_backend_topology(dispatch_size: int, *, is_cross_node: bool, gpus_per_node: int, scope: str) -> None:
    """Refuse a DeepEP V1 (``ep_buffer_backend='legacy'``) buffer this dispatch topology cannot drive.

    V1 is wired as intranode CUDA-IPC (``num_rdma_bytes=0``; its NVSHMEM internode path is not wired
    up here) over at most :data:`DEEPEP_V1_MAX_NVL_PEERS` NVLink peers within one OS node, and
    ``Buffer.get_dispatch_config`` asserts on any width outside :data:`DEEPEP_V1_CONFIG_RANKS` at the
    first dispatch. The one owner of all three rules: ``ParallelismConfig`` applies them to the
    declared topology before a weight is read, and ``DeepEPDispatcher`` re-applies them to a
    hand-built ``EPConfig`` that never passed through it. ``scope`` names the topology in the caller's
    own terms; the gate is the DISPATCH group, which expert TP makes narrower than the EP group.
    """
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
        # Only the widths that also clear the peer limit above — advertising the full table would
        # send the reader to a dispatch size the previous check rejects.
        usable = sorted(w for w in DEEPEP_V1_CONFIG_RANKS if w <= min(DEEPEP_V1_MAX_NVL_PEERS, gpus_per_node))
        raise ValueError(
            f"ep_buffer_backend='legacy' has no tuned Config for a {dispatch_size}-rank dispatch "
            f"group; DeepEP V1 asserts on anything outside its shipped tables at the first dispatch. "
            f"Usable widths here: {usable}. Or use the default 'elastic' backend."
        )


def reject_oversized_gin_dispatch(needed: int) -> None:
    """Raise for a cross-node dispatch above the validated proxy-GIN ceiling — it wedges, not errors.

    Called only for an inter-node (Gin) EP group; the intra-node NVLink path has no such limit
    (validated to 65k tokens/rank). See :data:`GIN_MAX_TOKENS_PER_RANK` for the measurements.
    """
    if GIN_MAX_TOKENS_PER_RANK and needed > GIN_MAX_TOKENS_PER_RANK:
        raise ValueError(
            f"Cross-node DeepEP GIN dispatch: num_max_tokens_per_rank={needed} exceeds the validated "
            f"EFA proxy-GIN ceiling ({GIN_MAX_TOKENS_PER_RANK}). A larger dispatch wedges in transit "
            f"instead of erroring — the receive counts never arrive ('Dispatch CPU wait ... received "
            f"count 0', or Xid 109 CTX SWITCH TIMEOUT cascading into Xid 43). Reduce the tokens in one "
            f"MoE forward (per_device_train_batch_size, max_length), or prefer node-local EP "
            f"(ep_scope=node) with DP across nodes — cross-node dispatch is also latency-bound and "
            f"several times slower. HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK overrides (0 disables) after "
            f"validating a larger dispatch end-to-end on your fabric."
        )


def reject_oversized_dispatch(capacity: int, *, num_topk: int, padded_hidden: int, is_inter_node: bool) -> None:
    """Refuse a per-rank arena capacity DeepEP cannot carry — both transport ceilings, in one place.

    Called at buffer sizing with the capacity the batch in hand needs, and at config time with the
    capacity the run's declared token budget implies
    (``ParallelismConfig.validate_against_model_config``), so the early refusal and the runtime
    backstop can never disagree about where the boundary is.
    """
    index_extent = capacity * num_topk * padded_hidden
    if index_extent >= DEEPEP_INDEX_LIMIT:
        # Crossing INT32_MAX wraps the wire index into an illegal memory access (CUDA fault).
        raise ValueError(
            f"DeepEP dispatch exceeds the 32-bit wire-index limit: num_max_tokens_per_rank={capacity} × "
            f"num_topk={num_topk} × padded_hidden={padded_hidden} = {index_extent:,} ≥ "
            f"{DEEPEP_INDEX_LIMIT:,}. The dispatch kernel would illegal-access. The buffer is per-rank, "
            f"so EP size does not lower it — reduce the tokens in one MoE forward: per_device_train_batch_size=1, "
            f"a shorter sequence (max_length), or in env-GRPO fewer generations / shorter trajectories "
            f"(reasoning effort / max_turns / rollout length). At ~175k tokens/rank, far beyond training."
        )
    if is_inter_node:
        reject_oversized_gin_dispatch(capacity)


# Logical projections a stored expert weight carries, keyed on its leading attribute segment. The one
# home for that fact: the user-facing ``lora_target_modules`` vocabulary is derived from it
# (:func:`expert_target_projections`), and a family adding a weight root is CI-guarded into it.
LORA_PROJECTION_COVERAGE: dict[str, frozenset[str]] = {
    "gate_up_proj": frozenset({"gate", "up"}),
    "gate_proj": frozenset({"gate"}),
    "up_proj": frozenset({"up"}),
    "down_proj": frozenset({"down"}),
    "gate_proj_gmm": frozenset({"gate"}),
    "up_proj_gmm": frozenset({"up"}),
}

# Every logical projection any family stores — the "all of them" a container-level target requests.
EXPERT_PROJECTIONS: frozenset[str] = frozenset().union(*LORA_PROJECTION_COVERAGE.values())

# Container-level ``lora_target_modules`` aliases: the module holding the experts, not a weight, so
# they name no projection of their own and stand for the whole FFN.
EXPERT_CONTAINER_TARGETS: frozenset[str] = frozenset({"experts", "mlp.experts"})


def expert_target_projections(target: str) -> frozenset[str] | None:
    """Logical projections a ``lora_target_modules`` entry requests, or ``None`` if it names no expert
    weight. Derived from :data:`LORA_PROJECTION_COVERAGE`, so a family adding a stored weight root
    gets its user-facing target name without a second list to keep in sync."""
    if target in EXPERT_CONTAINER_TARGETS:
        return EXPERT_PROJECTIONS
    return LORA_PROJECTION_COVERAGE.get(target)


def reject_expert_lora_with_expert_tp() -> None:
    """Raise the one refusal for native expert LoRA under expert TP.

    Checked twice on purpose — by ``ParallelismConfig`` at config time (before a multi-hundred-GB
    checkpoint downloads) and by :class:`EPConfig` at group construction (for hand-built configs
    that never pass through it). The message is built here so the two timings cannot drift into
    contradicting each other about what the user should change.
    """
    raise ValueError(
        "Expert LoRA is not supported with expert_tp_size > 1: the replicated adapter half "
        "receives partial gradients under expert TP and drifts across ranks. Use EP without "
        "expert TP for expert adapters, or remove expert projections from lora_target_modules."
    )


@dataclass(frozen=True)
class ExpertLoraSpec:
    """Native grouped-LoRA recipe for EP-distributed experts.

    ``projections`` is the logical set (``gate``/``up``/``down``), resolved to stored attrs via
    :data:`LORA_PROJECTION_COVERAGE`. Mirrors the ``LoraConfig`` knobs the grouped adapters can
    honour; the ones they cannot are rejected where the spec is built
    (:func:`src.distributed.loading.peft_setup.split_expert_lora_targets`), never dropped silently.
    """

    r: int
    alpha: float
    dropout: float = 0.0
    projections: frozenset[str] = field(default_factory=lambda: EXPERT_PROJECTIONS)
    use_rslora: bool = False

    @property
    def scaling(self) -> float:
        """PEFT's adapter scaling: ``alpha / sqrt(r)`` under rsLoRA, else ``alpha / r``."""
        return self.alpha / math.sqrt(self.r) if self.use_rslora else self.alpha / self.r

    def adapts(self, weight_attr: str) -> bool:
        """Whether the stored expert weight ``weight_attr`` should receive an adapter."""
        coverage = LORA_PROJECTION_COVERAGE.get(weight_attr)
        return coverage is not None and bool(coverage & self.projections)

    def peft_config_conflicts(self, peft_config) -> list[str]:
        """``LoraConfig`` settings this spec cannot reproduce on the grouped expert adapters.

        The grouped adapters express exactly the fields above, so anything else on a co-resident
        attention ``LoraConfig`` would apply to that half alone — two different adapters under one
        config. Returned rather than raised: the caller owns the message and the timing.
        """
        conflicts = []
        if getattr(peft_config, "use_rslora", False) != self.use_rslora:
            conflicts.append(
                f"use_rslora={getattr(peft_config, 'use_rslora', False)} on the attention adapters vs "
                f"{self.use_rslora} on the expert adapters — the two halves would scale differently"
            )
        for field_name, honoured in (("use_dora", False), ("lora_bias", False), ("init_lora_weights", True)):
            value = getattr(peft_config, field_name, honoured)
            if value != honoured:
                conflicts.append(f"{field_name}={value!r} has no grouped-expert implementation")
        return conflicts


class EPConfig:
    """Independent Expert Parallelism configuration (Megatron-LM style).

    ``node_local=True`` keeps each EP group within one NVLink domain; ``False`` spans domains over RDMA.
    EP is orthogonal to DP (experts receive tokens from any batch).
    """

    def __init__(
        self,
        ep_size: int,
        ep_group_size: int = 0,
        world_size: int | None = None,
        rank_offset: int = 0,
        node_local: bool = True,
        gpus_per_node: int | None = None,
        fp32_router: bool = False,
        fp32_experts: bool = False,
        expert_tp_size: int = 1,  # Expert FFN TP (independent of attention TP)
        use_grouped_gemm: bool = True,
        fp32_grad_reduce: bool = False,
        expert_lora: ExpertLoraSpec | None = None,
        fsdp_shard_ep1_experts: bool = True,
        ep_buffer_backend: str = "auto",
    ):
        """Initialize Expert Parallelism configuration.

        ``ep_size`` = expert distribution size; ``ep_group_size`` (= ``ep_size * expert_tp_size``,
        auto-computed when 0) = full EP process-group size. ``expert_tp_size > 1`` shards expert FFN
        weights across node-local ranks. ``fp32_grad_reduce`` reduces router/expert grads in fp32 with
        bf16 storage.

        ``world_size`` is the width of ONE rank block (a pipeline stage; the whole job without PP) and
        ``rank_offset`` is that block's first global rank, so all layout math runs in block-local rank
        space and every emitted membership list is lifted back into global rank space.
        """
        self.ep_size = ep_size
        self.expert_tp_size = expert_tp_size
        self.ep_group_size = ep_group_size if ep_group_size > 0 else ep_size * expert_tp_size
        self.node_local = node_local
        self.fp32_router = fp32_router
        self.fp32_experts = fp32_experts
        self.fp32_grad_reduce = fp32_grad_reduce
        self.use_grouped_gemm = use_grouped_gemm
        # The LoRA gather/load paths assume this: adapters are never ETP-sharded, so lifting the
        # restriction means re-adding the per-projection A/B shard split those paths dropped.
        if expert_lora is not None and expert_tp_size > 1:
            reject_expert_lora_with_expert_tp()
        self.expert_lora = expert_lora
        # ep_size==1: FSDP shards the replicated experts and its reduce-scatter is the sole sync.
        self.fsdp_shard_ep1_experts = fsdp_shard_ep1_experts
        # "auto"/"elastic" = ElasticBuffer (NCCL Gin, cross-node); "legacy" = CUDA-IPC (intranode).
        self.ep_buffer_backend = ep_buffer_backend
        # The four expert-assignment fields are declared together and filled together by
        # ``finalize_expert_assignment`` once the model's expert count is known.
        self.num_experts = None
        self.experts_per_rank = None
        self.expert_start_idx = None
        self.expert_end_idx = None

        if world_size is None:
            world_size = get_global_world_size()
        self.world_size = world_size
        self.rank_offset = rank_offset

        # ``gpus_per_node`` is the NVLINK-DOMAIN size, not the OS node's GPU count: production always
        # passes ``ParallelismConfig.nvlink_domain_size`` (which on NVL72 is a whole 72-GPU rack
        # spanning ~18 nodes). The fallback below is the wrong unit for such a box, so prefer
        # passing the size in.
        if gpus_per_node is None:
            gpus_per_node = get_local_world_size()
            logger.warning(
                "EPConfig built without an explicit domain size — assuming LOCAL_WORLD_SIZE "
                f"({gpus_per_node}). On a multi-node NVLink domain (NVL72) that is the wrong unit; "
                "pass ParallelismConfig.nvlink_domain_size."
            )
        self.gpus_per_node = gpus_per_node
        # NVLink DOMAINS and this rank's domain index, despite the node-shaped names: both divide by
        # ``gpus_per_node``, which is the domain size above. On NVL72 one domain is ~18 OS nodes, so
        # a 144-GPU job reports 2 here, not 36.
        self.num_nodes = world_size // gpus_per_node if gpus_per_node > 0 else 1

        # Same predicates ParallelismConfig gates the run with, so a directly built EPConfig cannot
        # accept a shape the run-level config refuses; only the remedy is spelled in this API's terms.
        if node_local:
            reject_node_local_ep_group(
                self.ep_group_size,
                gpus_per_node,
                ep_size=ep_size,
                expert_tp_size=expert_tp_size,
                remedy="Use node_local=False for cross-node EP.",
            )
        else:
            reject_cross_node_ep_group(
                self.ep_group_size,
                world_size,
                scope_desc="world size",
                ep_size=ep_size,
                expert_tp_size=expert_tp_size,
            )

        self._create_process_groups()

        # EVERY cross-replica topology defers, not just the cross-node one: a post-accumulate hook fires
        # only where the expert weight accumulated a grad, and a rank whose dispatch delivered no tokens
        # for a layer never touches those weights — its replicas would hang inside a collective it never
        # enters. ``is_deferred_dp`` is a subset of ``needs_expert_grad_sync``, so it needs no term here.
        # PP defers too: it pins gradient_accumulation_steps to 1, so per-backward hooks would fire on
        # EVERY microbatch and re-scale already-synced grads (the hook is not idempotent).
        self.defer_grad_sync = (
            self.needs_expert_grad_sync or self.num_rank_blocks > 1
        ) and not self.experts_fsdp_managed

        scope_str = "node-local" if node_local else "cross-node"
        if is_global_main_process():
            logger.info(
                f"EP Config ({scope_str}): ep_size={ep_size}, ep_group_size={self.ep_group_size}, world_size={world_size}"
            )
            logger.info(f"  NVLink domains={self.num_nodes}, GPUs/domain={gpus_per_node}")
            if expert_tp_size > 1:
                logger.info(f"  Expert TP: size={expert_tp_size}, ep_size (distribution)={ep_size}")
            logger.info(f"  Gradient sync: {world_size} ranks")  # non-EP FSDP2 grad sync spans all ranks
            logger.info("EP and DP work orthogonally - experts can receive tokens from any batch")

    @property
    def experts_fsdp_managed(self) -> bool:
        """Whether FSDP2 owns the experts (and the router) instead of the EP layer's own grad hooks.

        True only when the experts are TRULY replicated (``ep_group_size == 1``) and
        ``fsdp_shard_ep1_experts`` lets ``fully_shard`` shard them, making its reduce-scatter the
        sole expert sync. This one predicate decides FSDP-ignored membership, whether the expert and
        router hooks register at all, the grad-norm bucket, the dtype of a stored expert weight, and
        whether ``fp32_experts`` has any effect — so every consumer must read it from here. Splitting
        it means double-syncing the experts or not syncing them at all, neither of which raises.
        """
        return self.fsdp_shard_ep1_experts and self.ep_group_size == 1

    def _create_process_groups(self):
        """Create EP token-routing groups and expert-replica grad-sync groups.

        All ranks must call ``dist.new_group`` for ALL groups — of every pipeline stage, not just their
        own (:meth:`_iter_all_ep_group_ranks`) — in the same order.

        Without a process group the config still resolves its rank math as rank 0 of every group —
        the CPU tests plan expert assignments this way. A real run cannot reach that branch: every
        EP script launches under torchrun, which initializes the default group before any config is
        built.
        """
        if not dist.is_initialized():
            self.process_group = None
            self.dispatch_ep_group = None
            self.expert_tp_group = None
            self.expert_replica_group = None
            self.needs_expert_grad_sync = False
            self.ep_rank = 0
            self.ep_group_idx = 0
            self.num_ep_groups = 1
            self.expert_tp_rank = 0
            self.dispatch_ep_rank = 0
            self.expert_replica_ranks = [0]
            self.is_deferred_dp = False
            self.num_rank_blocks = 1
            self.dp_scope_group = None
            logger.warning("Distributed not initialized - using single process mode")
            return

        global_rank = get_global_rank()
        # world_size is ONE rank block's width (a pipeline stage), so >1 blocks == running under PP.
        self.num_rank_blocks = get_global_world_size() // self.world_size

        if self.node_local:
            self._create_node_local_groups(global_rank)
        else:
            self._create_cross_node_groups(global_rank)

        # Narrower than ``defer_grad_sync`` (which every cross-replica topology sets): this flag marks
        # the multi-domain multi-group case where non-expert DP additionally shards over the EP group.
        # Attention TP cannot reach it — ParallelismConfig._validate_tp rejects that exact shape.
        self.is_deferred_dp = (
            self.num_ep_groups > 1 and self.num_nodes > 1 and self.expert_tp_size == 1 and self.ep_group_size > 1
        )

        if self.expert_tp_size > 1:
            self._create_expert_tp_groups(global_rank)
        else:
            self.dispatch_ep_group = self.process_group
            self.expert_tp_group = None
            self.expert_tp_rank = 0
            self.dispatch_ep_rank = self.ep_rank

        # None (= world group) without PP; under PP the world group would blend stages owning
        # DIFFERENT layers, so each rank block gets its own DP-average domain.
        self.dp_scope_group = None
        if self.num_rank_blocks > 1:
            for base in self._rank_block_bases():
                block_ranks = list(range(base, base + self.world_size))
                group = dist.new_group(block_ranks, timeout=get_nccl_timeout())
                if base == self.rank_offset:
                    self.dp_scope_group = group
            if self.dp_scope_group is None:
                raise RuntimeError(
                    f"DP-scope group selection inconsistency: rank_offset={self.rank_offset} matched "
                    f"no rank-block base in {list(self._rank_block_bases())} "
                    f"(world_size={self.world_size}). This indicates a rank-block/offset mismatch."
                )

        self._log_process_groups(global_rank)

    def _create_node_local_groups(self, global_rank: int):
        """Create EP groups within each NVLink domain (node-local EP) — contiguous rank blocks.

        ``self.gpus_per_node`` is the NVLink-domain size here. Rank numbers are block-local (add
        ``rank_offset`` for global).
        Example (2 domains of 8, ep_group_size=8): domain 0 = [0-7], domain 1 = [8-15].
        """
        domain_size = self.gpus_per_node
        self._create_ep_groups(
            global_rank,
            num_groups=self.num_nodes * node_local_groups_per_domain(domain_size, self.ep_group_size),
            my_coords=node_local_rank_and_group(global_rank - self.rank_offset, domain_size, self.ep_group_size),
            group_ranks=lambda idx: node_local_group_ranks(idx, domain_size, self.ep_group_size),
            replica_ranks=lambda r: node_local_replica_ranks(r, self.world_size, domain_size, self.ep_group_size),
        )

    def _create_cross_node_groups(self, global_rank: int):
        """Create EP groups spanning NVLink domains (cross-node EP) using the column-block layout.

        Each EP group spans every domain with a contiguous device block per domain, so DeepEP's
        intranode P2P kernel gets contiguous IPC peers. ``self.gpus_per_node`` = NVLink domain size.
        Rank numbers are block-local (add ``rank_offset`` for global).

        Example (world_size=16, 2 domains of 8, ep_group_size=8)::

            EP group 0: [0, 1, 2, 3, 8, 9, 10, 11]
            EP group 1: [4, 5, 6, 7, 12, 13, 14, 15]
        """
        num_groups, _, _ = cross_node_layout(self.world_size, self.ep_group_size, self.gpus_per_node)
        self._create_ep_groups(
            global_rank,
            num_groups=num_groups,
            my_coords=cross_node_rank_and_group(
                global_rank - self.rank_offset, self.world_size, self.ep_group_size, self.gpus_per_node
            ),
            group_ranks=lambda idx: cross_node_group_ranks(
                idx, self.world_size, self.ep_group_size, self.gpus_per_node
            ),
            replica_ranks=lambda r: cross_node_replica_ranks(
                r, self.world_size, self.ep_group_size, self.gpus_per_node
            ),
        )

    def _rank_block_bases(self) -> range:
        """First global rank of every ``world_size``-wide rank block in the job (one per pipeline stage)."""
        return range(0, get_global_world_size(), self.world_size)

    def _iter_all_ep_group_ranks(self):
        """Yield ``(block_base, group_idx, global_ranks)`` for EVERY EP group in the job.

        ``dist.new_group`` is a collective over the default process group, so each rank must take part
        in creating the EP groups of the OTHER pipeline stages too (keeping only its own). Creating
        just its own block's groups would give the stages different ``new_group`` call sequences and
        hang the job. The block-major order is identical on every rank.
        """
        for base in self._rank_block_bases():
            for group_idx in range(self.num_ep_groups):
                yield base, group_idx, [base + r for r in self._block_group_ranks(group_idx)]

    def _create_ep_groups(self, global_rank: int, *, num_groups: int, my_coords, group_ranks, replica_ranks):
        """Create the EP token-routing groups and expert-replica grad-sync groups for one layout.

        ``my_coords`` = this rank's ``(ep_rank, ep_group_idx)``; ``group_ranks`` / ``replica_ranks``
        are the layout's rank-list functions in BLOCK-LOCAL rank space (group index → EP group members;
        ep-rank position → cross-group replica members), lifted to global ranks per block. All ranks
        create ALL groups of ALL blocks in the same fixed order, keeping only their own.
        """
        self.num_ep_groups = num_groups
        self.ep_rank, self.ep_group_idx = my_coords
        self._block_group_ranks = group_ranks

        self.process_group = None
        my_ep_group_ranks = None
        if self.ep_group_size == 1:
            # One EP group PER RANK: every group is a singleton that can carry no collective, and the
            # dispatcher is in no-op mode anyway (``ep_size <= 1``). Creating them is world_size
            # rendezvous nothing can use — 512 of them at 512 ranks, on the DEFAULT grouped-GEMM MoE
            # path. The replica group below is the one that matters here.
            my_ep_group_ranks = [global_rank]
        else:
            for base, group_idx, ep_group_ranks in self._iter_all_ep_group_ranks():
                group = dist.new_group(ep_group_ranks, timeout=get_nccl_timeout())
                if base == self.rank_offset and group_idx == self.ep_group_idx:
                    self.process_group = group
                    my_ep_group_ranks = ep_group_ranks

        # Fail fast: a wrong process_group would silently hang the first all-to-all.
        if my_ep_group_ranks is None or global_rank not in my_ep_group_ranks:
            raise RuntimeError(
                f"EP group selection inconsistency: global_rank={global_rank} not in its "
                f"selected EP group {my_ep_group_ranks} (ep_group_idx={self.ep_group_idx}, "
                f"ep_rank={self.ep_rank}, rank_offset={self.rank_offset}, "
                f"NVLink domain size={self.gpus_per_node}). "
                "This indicates a rank-math/layout mismatch."
            )

        self._my_ep_group_ranks = my_ep_group_ranks

        self.expert_replica_group = None
        self.expert_replica_ranks = []
        self.needs_expert_grad_sync = self.num_ep_groups > 1

        if self.needs_expert_grad_sync:
            # Per rank block: the same ep_rank in another PP stage owns different layers.
            for base in self._rank_block_bases():
                for target_ep_rank in range(self.ep_group_size):
                    replicas = [base + r for r in replica_ranks(target_ep_rank)]
                    group = dist.new_group(replicas, timeout=get_nccl_timeout())
                    if base == self.rank_offset and target_ep_rank == self.ep_rank:
                        self.expert_replica_group = group
                        self.expert_replica_ranks = replicas
            if self.expert_replica_group is None or global_rank not in self.expert_replica_ranks:
                # Every rank belongs to exactly one (block base, ep_rank) pair, so the loop above
                # always matches — unless the rank math and the replica layout disagree. Leaving the
                # group None would give the deferred sweep nothing to reduce over, and the replicas
                # would silently diverge.
                raise RuntimeError(
                    f"EP expert gradient sync is required ({self.num_ep_groups} EP groups) but rank "
                    f"{global_rank} is not in the expert-replica set it was assigned "
                    f"({self.expert_replica_ranks}) — the EP group layout and the replica-set "
                    f"construction disagree. The deferred sweep would reduce over a group this rank "
                    f"is not a member of, and expert replicas would silently diverge."
                )
        else:
            self.expert_replica_ranks = [global_rank]

    def _create_expert_tp_groups(self, global_rank: int):
        """Create expert TP groups and sub-EP (dispatch) groups.

        Dispatch (sub-EP) groups back DeepEP's NVLink all-to-all buffer, so they MUST be contiguous rank
        ranges (the intranode P2P kernel fails on strided sets). Each EP group is laid out as
        ``expert_tp_size`` contiguous dispatch chunks of ``ep_size`` ranks; expert-TP groups (plain NCCL)
        take one rank per chunk and reduce in token space outside the dispatch->combine span.

        Example: 8 GPUs, ep_size=4, expert_tp_size=2, ep_group_size=8 (node-local):
          EP group: [0,1,2,3,4,5,6,7]
          Dispatch (sub-EP) groups: [0,1,2,3], [4,5,6,7]    (contiguous — DeepEP)
          Expert TP groups: [0,4], [1,5], [2,6], [3,7]      (strided — NCCL)
        """
        expert_tp_size = self.expert_tp_size
        ep_size = self.ep_size

        self.expert_tp_group = None
        # Left None at ep_size == 1 (pure ETP): each dispatch group would be a singleton carrying no
        # collective — world_size rendezvous nothing can use. Both consumers already no-op there: the
        # dispatcher short-circuits on ``ep_size <= 1`` and the gather is identity at world_size 1.
        self.dispatch_ep_group = None
        # Shared leaf with ParallelismConfig: ETP partners share dispatch_ep_rank → identical batches.
        self.dispatch_ep_rank, self.expert_tp_rank = etp_dispatch_coords(
            self.ep_rank, ep_size, expert_tp_size, self.node_local
        )

        if self.node_local:
            for _, _, group_ranks in self._iter_all_ep_group_ranks():
                for d in range(ep_size):
                    tp_ranks = [group_ranks[d + ep_size * t] for t in range(expert_tp_size)]
                    group = dist.new_group(tp_ranks, timeout=get_nccl_timeout())
                    if global_rank in tp_ranks:
                        self.expert_tp_group = group
            if ep_size > 1:
                for _, _, group_ranks in self._iter_all_ep_group_ranks():
                    for t in range(expert_tp_size):
                        sub_ep_ranks = group_ranks[t * ep_size : (t + 1) * ep_size]
                        group = dist.new_group(sub_ep_ranks, timeout=get_nccl_timeout())
                        if global_rank in sub_ep_ranks:
                            self.dispatch_ep_group = group
        else:
            # CROSS-NODE EP + node-local ETP: each domain's contiguous block IS one ETP group (keeps
            # its all-reduce on NVLink); the dispatch group strides across domains over Gin.
            for _, _, group_ranks in self._iter_all_ep_group_ranks():
                domains = [r // self.gpus_per_node for r in group_ranks]
                domains_spanned = len(set(domains))
                members_per_domain = len(group_ranks) // domains_spanned
                reject_cross_node_etp_shape(
                    expert_tp_size,
                    members_per_domain,
                    ep_size,
                    domains_spanned,
                    remedy=(
                        f"EP group {group_ranks}. Use ep_scope='node' for node-local EP+ETP, or set "
                        f"expert_tp_size={members_per_domain}."
                    ),
                )
                for c in range(ep_size):
                    etp_ranks = group_ranks[c * expert_tp_size : (c + 1) * expert_tp_size]
                    if len({r // self.gpus_per_node for r in etp_ranks}) > 1:
                        raise ValueError(
                            f"Expert TP group {etp_ranks} spans multiple nodes (cross-node EP+ETP). "
                            f"This should not happen for a one-ETP-group-per-domain shape — report a bug."
                        )
            for _, _, group_ranks in self._iter_all_ep_group_ranks():
                for c in range(ep_size):
                    etp_ranks = group_ranks[c * expert_tp_size : (c + 1) * expert_tp_size]
                    group = dist.new_group(etp_ranks, timeout=get_nccl_timeout())
                    if global_rank in etp_ranks:
                        self.expert_tp_group = group
            # Dispatch (sub-EP) group = same intra-domain position across all domains (cross-node).
            if ep_size > 1:
                for _, _, group_ranks in self._iter_all_ep_group_ranks():
                    for p in range(expert_tp_size):
                        disp_ranks = [group_ranks[p + expert_tp_size * c] for c in range(ep_size)]
                        group = dist.new_group(disp_ranks, timeout=get_nccl_timeout())
                        if global_rank in disp_ranks:
                            self.dispatch_ep_group = group

        if is_global_main_process():
            logger.info(
                f"  Expert TP groups created: expert_tp_rank={self.expert_tp_rank}, "
                f"dispatch_ep_rank={self.dispatch_ep_rank}, "
                f"sub-EP size (ep_size)={self.ep_size}"
            )

    def _log_process_groups(self, global_rank: int):
        """Log process group information (summary on rank 0, debug on others)."""
        scope_str = "node-local" if self.node_local else "cross-node"
        if is_global_main_process():
            logger.info(f"EP Process Groups created ({scope_str}):")
            logger.info(f"  - Number of EP groups: {self.num_ep_groups}")
            logger.info(f"  - EP group ranks (rank 0): {self._my_ep_group_ranks}")
            logger.info(f"  - Expert replica ranks (rank 0): {self.expert_replica_ranks}")
            logger.info(f"  - Needs expert grad sync: {self.needs_expert_grad_sync}")
        domain_id = global_rank // self.gpus_per_node if self.gpus_per_node > 0 else 0
        logger.debug(
            f"EP rank {global_rank}: EP group idx={self.ep_group_idx}, "
            f"EP rank={self.ep_rank}, NVLink domain={domain_id}, "
            f"group ranks={self._my_ep_group_ranks}"
        )

    def finalize_expert_assignment(self, num_experts: int):
        """Finalize configuration after detecting ``num_experts``.

        Assignment uses ``dispatch_ep_rank`` and ``ep_size`` so ranks in the same expert-TP group get the
        same expert range (shared sharded weights). ``ep_size`` must divide ``num_experts``: DeepEP
        dispatch maps tokens to ranks by a UNIFORM expert→rank division, so an uneven per-rank
        assignment could not be honored — tokens would reach the wrong rank's experts or be silently
        dropped by the local valid-index filter.
        """
        if num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by ep_size ({self.ep_size}): DeepEP "
                f"dispatch assumes a uniform expert→rank division, so an uneven assignment would "
                f"route tokens to the wrong experts or silently drop them. Choose an ep_size that "
                f"divides {num_experts}."
            )

        self.num_experts = num_experts
        self.experts_per_rank = num_experts // self.ep_size
        self.expert_start_idx = self.dispatch_ep_rank * self.experts_per_rank
        self.expert_end_idx = self.expert_start_idx + self.experts_per_rank

        if is_global_main_process():
            global_rank = get_global_rank()
            logger.info(
                f"EP Config finalized: ep_size={self.ep_size} distribution, ep_group_size={self.ep_group_size}, {num_experts} experts"
            )
            if self.expert_tp_size > 1:
                logger.info(f"  - Expert TP: size={self.expert_tp_size}, ep_size (distribution)={self.ep_size}")
                logger.info(
                    f"  - Global rank {global_rank}, dispatch_ep_rank {self.dispatch_ep_rank}, "
                    f"expert_tp_rank {self.expert_tp_rank}: {self.experts_per_rank} experts "
                    f"(indices {self.expert_start_idx}-{self.expert_end_idx - 1})"
                )
            else:
                logger.info(
                    f"  - Global rank {global_rank}, EP rank {self.ep_rank}: {self.experts_per_rank} experts (indices {self.expert_start_idx}-{self.expert_end_idx - 1})"
                )

            logger.info("Expert and Data Parallelism work orthogonally:")
            logger.info(f"  - {self.world_size} ranks total, {self.num_ep_groups} EP groups")
            logger.info(f"  - Each EP group has {self.ep_group_size} ranks, {self.ep_size}-way expert distribution")
            logger.info(f"  - All {self.world_size} ranks participate in gradient sync (DP)")


def get_num_experts(model_config) -> int:
    """Routed-expert count from a HuggingFace model config, via the shared field registry.

    The same :func:`get_first_router_field` probe over ``ROUTER_EXPERT_COUNT_FIELDS`` that the MoE
    detection, the load metrics and ``ParallelismConfig``'s expert-divisibility gate use — same
    ``text_config`` descent, so a family cannot be MoE for one consumer and dense for another. Only
    the raise is this caller's own: EP cannot place experts it cannot count.
    """
    num_experts = get_first_router_field(model_config, ROUTER_EXPERT_COUNT_FIELDS)
    if not num_experts:
        raise ValueError(
            f"Could not determine the routed-expert count from {type(model_config).__name__}: none of "
            f"{ROUTER_EXPERT_COUNT_FIELDS} is set on it or on its text config."
        )
    return num_experts
