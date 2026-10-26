"""Expert Parallelism configuration: the EP topology, its process groups and the expert assignment.

:class:`EPConfig` creates the groups for token routing, expert-replica gradient sync and expert TP;
:data:`EPBufferBackend` names the DeepEP buffer backends a run selects from.
"""

from __future__ import annotations

import logging
from typing import Literal

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
from src.env import nccl_timeout
from src.hardware import supports_grouped_mm
from src.models.moe.expert_lora_spec import ExpertLoraSpec, reject_expert_lora_with_expert_tp
from src.models.moe_balancing import ROUTER_EXPERT_COUNT_FIELDS, get_first_router_field
from src.runtime.ranks import get_global_rank, get_global_world_size, get_local_world_size, is_global_main_process

logger = logging.getLogger(__name__)

# The DeepEP buffer backend: "elastic" (V2 ElasticBuffer, what "auto" resolves to) or "legacy" (V1, intranode).
EPBufferBackend = Literal["auto", "elastic", "legacy"]


class EPConfig:
    """The Expert Parallelism topology (Megatron-LM style), its process groups and the expert assignment.

    EP is orthogonal to DP: experts receive tokens from any batch. ``node_local`` keeps each EP group
    within one NVLink domain; otherwise groups span domains over RDMA.
    """

    def __init__(
        self,
        ep_size: int,
        *,
        ep_group_size: int = 0,
        stage_world_size: int | None = None,
        stage_base_rank: int = 0,
        node_local: bool = True,
        nvlink_domain_size: int | None = None,
        fp32_router: bool = False,
        fp32_experts: bool = False,
        expert_tp_size: int = 1,
        use_grouped_gemm: bool = True,
        fp32_grad_reduce: bool = False,
        expert_lora: ExpertLoraSpec | None = None,
        experts_fsdp_managed: bool,
        ep_buffer_backend: EPBufferBackend = "auto",
    ):
        """``ep_size`` distributes the experts; ``ep_group_size`` (``ep_size * expert_tp_size`` when 0) is
        the full EP process group, and ``expert_tp_size`` shards each expert FFN across it.

        ``stage_world_size`` (default: the world) is the width of one rank block, a pipeline stage, and
        ``stage_base_rank`` its first global rank: the layout math runs in stage-local rank space and every
        membership list is lifted back to global ranks. ``nvlink_domain_size`` is the locality unit, a
        whole rack on NVL72. ``experts_fsdp_managed`` is ``ParallelismConfig``'s verdict that FSDP2 shards
        the replicated experts, its reduce-scatter their only sync. ``fp32_grad_reduce`` reduces
        router/expert grads in fp32 with bf16 storage.
        """
        self.ep_size = ep_size
        self.expert_tp_size = expert_tp_size
        self.ep_group_size = ep_group_size if ep_group_size > 0 else ep_size * expert_tp_size
        self.node_local = node_local
        self.fp32_router = fp32_router
        self.fp32_experts = fp32_experts
        self.fp32_grad_reduce = fp32_grad_reduce
        self.use_grouped_gemm = use_grouped_gemm
        # The LoRA gather/load paths assume adapters are never ETP-sharded; lifting it needs a
        # per-projection A/B shard split there.
        if expert_lora is not None and expert_tp_size > 1:
            reject_expert_lora_with_expert_tp()
        self.expert_lora = expert_lora
        self.experts_fsdp_managed = experts_fsdp_managed
        self.ep_buffer_backend = ep_buffer_backend
        self.stage_world_size = get_global_world_size() if stage_world_size is None else stage_world_size
        self.stage_base_rank = stage_base_rank
        if nvlink_domain_size is None:
            nvlink_domain_size = get_local_world_size()
            logger.warning(
                "EPConfig built without an explicit domain size — assuming LOCAL_WORLD_SIZE "
                f"({nvlink_domain_size}). On a multi-node NVLink domain (NVL72) that is the wrong unit; "
                "pass ParallelismConfig.nvlink_domain_size."
            )
        self.nvlink_domain_size = nvlink_domain_size
        self.num_domains = self.stage_world_size // nvlink_domain_size if nvlink_domain_size > 0 else 1
        # Set by finalize_expert_assignment once the model's expert count is known.
        self.num_experts = None
        self.experts_per_rank = None
        self.expert_start_idx = None
        self.expert_end_idx = None
        # This rank's groups as resolved without a process group (rank 0 of each, which CPU tests plan
        # with); _create_process_groups replaces them. dispatch_ep_group stays None at ep_size == 1 (a no-op
        # dispatcher); dp_scope_group None is the world group, replaced per stage under PP.
        self.process_group = None
        self.dispatch_ep_group = None
        self.expert_tp_group = None
        self.expert_replica_group = None
        self.dp_scope_group = None
        self.ep_rank = 0
        self.ep_group_idx = 0
        self.num_ep_groups = 1
        self.expert_tp_rank = 0
        self.dispatch_ep_rank = 0
        self.expert_replica_ranks = [0]
        self.needs_expert_grad_sync = False
        self.is_deferred_dp = False
        self.num_rank_blocks = 1
        # The layout's group-index -> stage-local members function, and this rank's EP group members.
        self._block_group_ranks = None
        self._my_ep_group_ranks = None

        # Same predicates ParallelismConfig gates the run with, so a directly built EPConfig cannot
        # accept a shape the run-level config rejects; only the remedy is worded for this API.
        if node_local:
            reject_node_local_ep_group(
                self.ep_group_size,
                self.nvlink_domain_size,
                ep_size=ep_size,
                expert_tp_size=expert_tp_size,
                remedy="Use node_local=False for cross-node EP.",
            )
        else:
            reject_cross_node_ep_group(
                self.ep_group_size,
                self.stage_world_size,
                scope_desc="world size",
                ep_size=ep_size,
                expert_tp_size=expert_tp_size,
            )

        self._create_process_groups()

        # Every cross-replica topology defers, not just the cross-node one: a post-accumulate hook
        # fires only where the expert weight accumulated a grad, and a rank whose dispatch delivered
        # no tokens for a layer never enters the collective its replicas wait in. ``is_deferred_dp``
        # is a subset of ``needs_expert_grad_sync``, so it adds no term. PP defers too: per-backward
        # hooks would fire on every microbatch and re-scale already-synced grads (not idempotent).
        self.defer_grad_sync = (
            self.needs_expert_grad_sync or self.num_rank_blocks > 1
        ) and not self.experts_fsdp_managed

        scope_str = "node-local" if node_local else "cross-node"
        if is_global_main_process():
            logger.info(
                f"EP Config ({scope_str}): ep_size={ep_size}, ep_group_size={self.ep_group_size}, "
                f"world_size={self.stage_world_size}"
            )
            logger.info(f"  NVLink domains={self.num_domains}, GPUs/domain={self.nvlink_domain_size}")
            if expert_tp_size > 1:
                logger.info(f"  Expert TP: size={expert_tp_size}, ep_size (distribution)={ep_size}")
            logger.info(f"  Gradient sync: {self.stage_world_size} ranks")  # non-EP FSDP2 grad sync spans all ranks
            logger.info("EP and DP work orthogonally - experts can receive tokens from any batch")

    @property
    def uses_backward_grad_hooks(self) -> bool:
        """Whether the EP layer syncs router and expert grads with in-backward hooks: neither FSDP2 nor
        the deferred post-backward sweep does it."""
        return not self.experts_fsdp_managed and not self.defer_grad_sync

    @property
    def keeps_fp32_experts(self) -> bool:
        """Whether the expert weights are kept as fp32 masters; under FSDP2 its bf16 policy casts them."""
        return self.fp32_experts and not self.experts_fsdp_managed

    @property
    def spans_domains(self) -> bool:
        """Whether an EP group spans NVLink domains, dispatching over Gin (RDMA) rather than NVLink."""
        return self.num_domains > 1 and not self.node_local

    @property
    def grouped_gemm_enabled(self) -> bool:
        """Whether the experts compute with grouped GEMM: requested, and supported by this GPU."""
        return self.use_grouped_gemm and supports_grouped_mm()

    def _create_process_groups(self):
        """Create EP token-routing groups and expert-replica grad-sync groups.

        Every rank must call ``dist.new_group`` for all groups of every pipeline stage, not just its
        own (:meth:`_iter_all_ep_group_ranks`), in the same order.

        Without an initialized process group the single-process values ``__init__`` declares stand; a
        real run cannot reach that branch, since torchrun initializes the default group first.
        """
        if not dist.is_initialized():
            logger.warning("Distributed not initialized - using single process mode")
            return

        global_rank = get_global_rank()
        # stage_world_size is one rank block's width (a pipeline stage), so >1 blocks means running under PP.
        self.num_rank_blocks = get_global_world_size() // self.stage_world_size

        if self.node_local:
            self._create_node_local_groups(global_rank)
        else:
            self._create_cross_node_groups(global_rank)

        # Narrower than ``defer_grad_sync``: this flag marks the multi-domain multi-group case where
        # non-expert DP additionally shards over the EP group. Attention TP cannot reach it;
        # ParallelismConfig._validate_tp rejects that shape.
        self.is_deferred_dp = (
            self.num_ep_groups > 1 and self.num_domains > 1 and self.expert_tp_size == 1 and self.ep_group_size > 1
        )

        if self.expert_tp_size > 1:
            self._create_expert_tp_groups(global_rank)
        else:
            self.dispatch_ep_group = self.process_group
            self.dispatch_ep_rank = self.ep_rank

        # Under PP the world group would blend stages holding different layers, so each rank block gets
        # its own DP-average domain.
        if self.num_rank_blocks > 1:
            for base in self._rank_block_bases():
                block_ranks = list(range(base, base + self.stage_world_size))
                group = dist.new_group(block_ranks, timeout=nccl_timeout())
                if base == self.stage_base_rank:
                    self.dp_scope_group = group
            if self.dp_scope_group is None:
                raise RuntimeError(
                    f"DP-scope group selection inconsistency: stage_base_rank={self.stage_base_rank} matched "
                    f"no rank-block base in {list(self._rank_block_bases())} "
                    f"(stage_world_size={self.stage_world_size}). This indicates a rank-block/offset mismatch."
                )

        self._log_process_groups(global_rank)

    def _create_node_local_groups(self, global_rank: int):
        """Create EP groups within each NVLink domain (node-local EP) — contiguous rank blocks.

        Rank numbers are stage-local (add ``stage_base_rank`` for global).
        Example (2 domains of 8, ep_group_size=8): domain 0 = [0-7], domain 1 = [8-15].
        """
        domain_size = self.nvlink_domain_size
        self._create_ep_groups(
            global_rank,
            num_groups=self.num_domains * node_local_groups_per_domain(domain_size, self.ep_group_size),
            my_coords=node_local_rank_and_group(global_rank - self.stage_base_rank, domain_size, self.ep_group_size),
            group_ranks=lambda idx: node_local_group_ranks(idx, domain_size, self.ep_group_size),
            replica_ranks=lambda r: node_local_replica_ranks(
                r, self.stage_world_size, domain_size, self.ep_group_size
            ),
        )

    def _create_cross_node_groups(self, global_rank: int):
        """Create EP groups spanning NVLink domains (cross-node EP) using the column-block layout.

        Each EP group spans every domain with a contiguous device block per domain, so DeepEP's
        intranode P2P kernel gets contiguous IPC peers. Rank numbers are stage-local (add
        ``stage_base_rank`` for global).

        Example (stage_world_size=16, 2 domains of 8, ep_group_size=8)::

            EP group 0: [0, 1, 2, 3, 8, 9, 10, 11]
            EP group 1: [4, 5, 6, 7, 12, 13, 14, 15]
        """
        num_groups, _, _ = cross_node_layout(self.stage_world_size, self.ep_group_size, self.nvlink_domain_size)
        self._create_ep_groups(
            global_rank,
            num_groups=num_groups,
            my_coords=cross_node_rank_and_group(
                global_rank - self.stage_base_rank, self.stage_world_size, self.ep_group_size, self.nvlink_domain_size
            ),
            group_ranks=lambda idx: cross_node_group_ranks(
                idx, self.stage_world_size, self.ep_group_size, self.nvlink_domain_size
            ),
            replica_ranks=lambda r: cross_node_replica_ranks(
                r, self.stage_world_size, self.ep_group_size, self.nvlink_domain_size
            ),
        )

    def _rank_block_bases(self) -> range:
        """First global rank of every ``stage_world_size``-wide rank block in the job (one per pipeline stage)."""
        return range(0, get_global_world_size(), self.stage_world_size)

    def _iter_all_ep_group_ranks(self):
        """Yield ``(block_base, group_idx, global_ranks)`` for every EP group in the job.

        ``dist.new_group`` is a collective over the default process group, so each rank must take part
        in creating the EP groups of the other pipeline stages too, keeping only its own; creating
        just its own block's groups gives the stages different ``new_group`` call sequences and hangs
        the job. The block-major order is identical on every rank.
        """
        for base in self._rank_block_bases():
            for group_idx in range(self.num_ep_groups):
                yield base, group_idx, [base + r for r in self._block_group_ranks(group_idx)]

    def _create_ep_groups(self, global_rank: int, *, num_groups: int, my_coords, group_ranks, replica_ranks):
        """Create the EP token-routing groups and expert-replica grad-sync groups for one layout.

        ``my_coords`` = this rank's ``(ep_rank, ep_group_idx)``; ``group_ranks`` / ``replica_ranks``
        are the layout's rank-list functions in block-local rank space (group index → EP group members;
        ep-rank position → cross-group replica members), lifted to global ranks per block. All ranks
        create every group of every block in the same fixed order, keeping only their own.
        """
        self.num_ep_groups = num_groups
        self.ep_rank, self.ep_group_idx = my_coords
        self._block_group_ranks = group_ranks

        my_ep_group_ranks = None
        if self.ep_group_size == 1:
            # One EP group per rank: every group is a singleton that can carry no collective, and the
            # dispatcher is in no-op mode anyway (``ep_size <= 1``). Creating them costs world_size
            # rendezvous nothing can use, on the default grouped-GEMM MoE path. Only the replica
            # group below matters here.
            my_ep_group_ranks = [global_rank]
        else:
            for base, group_idx, ep_group_ranks in self._iter_all_ep_group_ranks():
                group = dist.new_group(ep_group_ranks, timeout=nccl_timeout())
                if base == self.stage_base_rank and group_idx == self.ep_group_idx:
                    self.process_group = group
                    my_ep_group_ranks = ep_group_ranks

        # A wrong process_group hangs the first all-to-all, so check membership here.
        if my_ep_group_ranks is None or global_rank not in my_ep_group_ranks:
            raise RuntimeError(
                f"EP group selection inconsistency: global_rank={global_rank} not in its "
                f"selected EP group {my_ep_group_ranks} (ep_group_idx={self.ep_group_idx}, "
                f"ep_rank={self.ep_rank}, stage_base_rank={self.stage_base_rank}, "
                f"NVLink domain size={self.nvlink_domain_size}). "
                "This indicates a rank-math/layout mismatch."
            )

        self._my_ep_group_ranks = my_ep_group_ranks

        self.expert_replica_ranks = []
        self.needs_expert_grad_sync = self.num_ep_groups > 1

        if self.needs_expert_grad_sync:
            # Per rank block: the same ep_rank in another PP stage holds different layers.
            for base in self._rank_block_bases():
                for target_ep_rank in range(self.ep_group_size):
                    replicas = [base + r for r in replica_ranks(target_ep_rank)]
                    group = dist.new_group(replicas, timeout=nccl_timeout())
                    if base == self.stage_base_rank and target_ep_rank == self.ep_rank:
                        self.expert_replica_group = group
                        self.expert_replica_ranks = replicas
            if self.expert_replica_group is None or global_rank not in self.expert_replica_ranks:
                # Every rank belongs to exactly one (block base, ep_rank) pair, so the loop above
                # matches unless the rank math and the replica layout disagree. A None group leaves
                # the deferred sweep nothing to reduce over and the replicas diverge.
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

        Dispatch (sub-EP) groups back DeepEP's NVLink all-to-all buffer, so they must be contiguous rank
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

        # Shared leaf with ParallelismConfig: ETP partners share dispatch_ep_rank → identical batches.
        self.dispatch_ep_rank, self.expert_tp_rank = etp_dispatch_coords(
            self.ep_rank, ep_size, expert_tp_size, self.node_local
        )

        if self.node_local:
            for _, _, group_ranks in self._iter_all_ep_group_ranks():
                for d in range(ep_size):
                    tp_ranks = [group_ranks[d + ep_size * t] for t in range(expert_tp_size)]
                    group = dist.new_group(tp_ranks, timeout=nccl_timeout())
                    if global_rank in tp_ranks:
                        self.expert_tp_group = group
            if ep_size > 1:
                for _, _, group_ranks in self._iter_all_ep_group_ranks():
                    for t in range(expert_tp_size):
                        sub_ep_ranks = group_ranks[t * ep_size : (t + 1) * ep_size]
                        group = dist.new_group(sub_ep_ranks, timeout=nccl_timeout())
                        if global_rank in sub_ep_ranks:
                            self.dispatch_ep_group = group
        else:
            # Cross-node EP + node-local ETP: each domain's contiguous block is one ETP group, keeping
            # its all-reduce on NVLink; the dispatch group strides across domains over Gin.
            for _, _, group_ranks in self._iter_all_ep_group_ranks():
                domains = [r // self.nvlink_domain_size for r in group_ranks]
                domains_spanned = len(set(domains))
                members_per_domain = len(group_ranks) // domains_spanned
                reject_cross_node_etp_shape(
                    expert_tp_size,
                    members_per_domain,
                    ep_size,
                    domains_spanned,
                    remedy=(
                        f"EP group {group_ranks}. Use ep_scope='node' for node-local EP+ETP, or set "
                        f"expert_tensor_parallel_size={members_per_domain}."
                    ),
                )
                for c in range(ep_size):
                    etp_ranks = group_ranks[c * expert_tp_size : (c + 1) * expert_tp_size]
                    if len({r // self.nvlink_domain_size for r in etp_ranks}) > 1:
                        raise ValueError(
                            f"Expert TP group {etp_ranks} spans multiple nodes (cross-node EP+ETP). "
                            f"This should not happen for a one-ETP-group-per-domain shape — report a bug."
                        )
            for _, _, group_ranks in self._iter_all_ep_group_ranks():
                for c in range(ep_size):
                    etp_ranks = group_ranks[c * expert_tp_size : (c + 1) * expert_tp_size]
                    group = dist.new_group(etp_ranks, timeout=nccl_timeout())
                    if global_rank in etp_ranks:
                        self.expert_tp_group = group
            # Dispatch (sub-EP) group = same intra-domain position across all domains (cross-node).
            if ep_size > 1:
                for _, _, group_ranks in self._iter_all_ep_group_ranks():
                    for p in range(expert_tp_size):
                        disp_ranks = [group_ranks[p + expert_tp_size * c] for c in range(ep_size)]
                        group = dist.new_group(disp_ranks, timeout=nccl_timeout())
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
        domain_id = global_rank // self.nvlink_domain_size if self.nvlink_domain_size > 0 else 0
        logger.debug(
            f"EP rank {global_rank}: EP group idx={self.ep_group_idx}, "
            f"EP rank={self.ep_rank}, NVLink domain={domain_id}, "
            f"group ranks={self._my_ep_group_ranks}"
        )

    def finalize_expert_assignment(self, num_experts: int):
        """Finalize configuration after detecting ``num_experts``.

        Assignment uses ``dispatch_ep_rank`` and ``ep_size`` so ranks in the same expert-TP group get
        the same expert range (shared sharded weights). ``ep_size`` must divide ``num_experts``:
        DeepEP dispatch maps tokens to ranks by a uniform expert→rank division, so an uneven per-rank
        assignment routes tokens to the wrong rank's experts or drops them at the local valid-index
        filter.
        """
        if num_experts % self.ep_size != 0:
            raise ValueError(
                f"num_experts ({num_experts}) must be divisible by expert_parallel_size ({self.ep_size}): "
                f"DeepEP dispatch assumes a uniform expert→rank division, so an uneven assignment would route "
                f"tokens to the wrong experts or silently drop them. Choose an expert_parallel_size that "
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
            logger.info(f"  - {self.stage_world_size} ranks total, {self.num_ep_groups} EP groups")
            logger.info(f"  - Each EP group has {self.ep_group_size} ranks, {self.ep_size}-way expert distribution")
            logger.info(f"  - All {self.stage_world_size} ranks participate in gradient sync (DP)")


def get_num_experts(model_config) -> int:
    """Routed-expert count from a HuggingFace model config, via the shared field registry.

    Uses the same :func:`get_first_router_field` probe over ``ROUTER_EXPERT_COUNT_FIELDS``, with the
    same ``text_config`` descent, as MoE detection, the load metrics and ``ParallelismConfig``'s
    expert-divisibility gate, so a family cannot be MoE for one consumer and dense for another.
    Raises when the count cannot be read.
    """
    num_experts = get_first_router_field(model_config, ROUTER_EXPERT_COUNT_FIELDS)
    if not num_experts:
        raise ValueError(
            f"Could not determine the routed-expert count from {type(model_config).__name__}: none of "
            f"{ROUTER_EXPERT_COUNT_FIELDS} is set on it or on its text config."
        )
    return num_experts
