"""Expert Parallelism configuration: the EP topology, its process groups and the expert assignment.

:class:`EPConfig` builds the groups for token routing, expert-replica gradient sync and expert TP;
:data:`EPBufferBackend` names the DeepEP buffer backends a run selects from.
"""

from __future__ import annotations

import logging
from typing import Literal

import torch.distributed as dist

from src.distributed.group_layout import (
    CrossNodeLayout,
    NodeLocalLayout,
    ep_layout,
    reject_cross_node_ep_group,
    reject_cross_node_etp_shape,
    reject_node_local_ep_group,
)
from src.hardware import supports_grouped_mm
from src.models.moe.expert_lora_spec import ExpertLoraSpec, reject_expert_lora_with_expert_tp
from src.runtime.collectives import build_groups
from src.runtime.ranks import get_global_rank, get_global_world_size, has_process_group, is_global_main_process

logger = logging.getLogger(__name__)

# The DeepEP buffer backend: "elastic" (V2 ElasticBuffer, what "auto" resolves to) or "legacy" (V1, intranode).
EPBufferBackend = Literal["auto", "elastic", "legacy"]


def _build_member_group(rank_lists: list[list[int]], rank: int, what: str) -> dist.ProcessGroup:
    """COLLECTIVE: build every group in ``rank_lists`` and return the one holding ``rank``."""
    mine = next((index for index, ranks in enumerate(rank_lists) if rank in ranks), -1)
    return build_groups(rank_lists, mine=mine, what=what)


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
        stage_world_size: int,
        stage_base_rank: int = 0,
        node_local: bool = True,
        nvlink_domain_size: int,
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

        ``stage_world_size`` is one rank block's width, a pipeline stage (the world without PP), and
        ``stage_base_rank`` its first global rank: the layout math runs in stage-local ranks.
        ``nvlink_domain_size`` is the locality unit, a whole rack on NVL72. ``experts_fsdp_managed`` is
        ``ParallelismConfig``'s verdict that FSDP2's reduce-scatter is the replicated experts' only sync.
        ``fp32_grad_reduce`` reduces router/expert grads in fp32 with bf16 storage.
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
        if stage_base_rank % stage_world_size != 0:
            raise ValueError(
                f"stage_base_rank ({stage_base_rank}) must be a multiple of stage_world_size "
                f"({stage_world_size}): each rank block starts at its first global rank."
            )
        self.stage_world_size = stage_world_size
        self.stage_base_rank = stage_base_rank
        self.nvlink_domain_size = nvlink_domain_size
        self.num_domains = stage_world_size // nvlink_domain_size
        # Set by finalize_expert_assignment once the model's expert count is known.
        self.num_experts = None
        self.experts_per_rank = None
        self.expert_start_idx = None
        self.expert_end_idx = None
        # This rank's groups as resolved without a process group (rank 0 of each, which CPU tests plan
        # with); _build_process_groups replaces them. dispatch_ep_group stays None at ep_size == 1 (a no-op
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

        self._build_process_groups()

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

    def _build_process_groups(self):
        """COLLECTIVE over the default group: build the EP, expert-replica, expert-TP and DP-scope groups
        of every rank block, in one order on every rank, keeping this rank's own.

        Without a process group the single-process values ``__init__`` declares stand.
        """
        if not has_process_group():
            logger.warning("Distributed not initialized - using single process mode")
            return

        global_rank = get_global_rank()
        # More than one rank block means pipeline stages.
        self.num_rank_blocks = get_global_world_size() // self.stage_world_size
        block = self.stage_base_rank // self.stage_world_size
        layout = ep_layout(
            world_size=self.stage_world_size,
            group_size=self.ep_group_size,
            domain_size=self.nvlink_domain_size,
            node_local=self.node_local,
        )
        self._build_ep_groups(global_rank, layout, block)

        # Narrower than ``defer_grad_sync``: this flag marks the multi-domain multi-group case where
        # non-expert DP additionally shards over the EP group. Attention TP cannot reach it;
        # parallelism_rules._reject_tp rejects that shape.
        self.is_deferred_dp = (
            self.num_ep_groups > 1 and self.num_domains > 1 and self.expert_tp_size == 1 and self.ep_group_size > 1
        )

        if self.expert_tp_size > 1:
            self._build_expert_tp_groups(global_rank, layout)
        else:
            self.dispatch_ep_group = self.process_group
            self.dispatch_ep_rank = self.ep_rank

        # Under PP the world group would blend stages holding different layers, so each rank block gets
        # its own DP-average domain.
        if self.num_rank_blocks > 1:
            self.dp_scope_group = build_groups(
                [list(range(base, base + self.stage_world_size)) for base in self._rank_block_bases()],
                mine=block,
                what="DP-scope group",
            )

        self._log_process_groups(global_rank)

    def _build_ep_groups(self, global_rank: int, layout: NodeLocalLayout | CrossNodeLayout, block: int):
        """COLLECTIVE: build the EP token-routing groups and, with more than one per stage, the
        expert-replica grad-sync groups; ``block`` is this rank's rank-block index."""
        self.num_ep_groups = layout.num_groups
        self.ep_rank, self.ep_group_idx = layout.rank_and_group(global_rank - self.stage_base_rank)

        if self.ep_group_size == 1:
            # One EP group per rank: every group is a singleton that can carry no collective, and the
            # dispatcher is in no-op mode anyway (``ep_size <= 1``). Creating them costs world_size
            # rendezvous nothing can use, on the default grouped-GEMM MoE path. Only the replica
            # group below matters here.
            self._my_ep_group_ranks = [global_rank]
        else:
            ep_groups = self._job_ep_groups(layout)
            mine = block * self.num_ep_groups + self.ep_group_idx
            self.process_group = build_groups(ep_groups, mine=mine, what="EP group")
            self._my_ep_group_ranks = ep_groups[mine]
            # A wrong process_group hangs the first all-to-all, so check membership here.
            if global_rank not in self._my_ep_group_ranks:
                raise RuntimeError(
                    f"EP group selection inconsistency: global_rank={global_rank} not in its "
                    f"selected EP group {self._my_ep_group_ranks} (ep_group_idx={self.ep_group_idx}, "
                    f"ep_rank={self.ep_rank}, stage_base_rank={self.stage_base_rank}, "
                    f"NVLink domain size={self.nvlink_domain_size}). "
                    "This indicates a rank-math/layout mismatch."
                )

        self.needs_expert_grad_sync = self.num_ep_groups > 1
        if not self.needs_expert_grad_sync:
            self.expert_replica_ranks = [global_rank]
            return
        # Per rank block: the same ep_rank in another PP stage holds different layers.
        replica_lists = [
            [base + rank for rank in layout.replica_ranks(target_ep_rank)]
            for base in self._rank_block_bases()
            for target_ep_rank in range(self.ep_group_size)
        ]
        mine = block * self.ep_group_size + self.ep_rank
        self.expert_replica_group = build_groups(replica_lists, mine=mine, what="expert-replica group")
        self.expert_replica_ranks = replica_lists[mine]
        if global_rank not in self.expert_replica_ranks:
            # A rank outside its replica set leaves the deferred sweep reducing over a group it is not
            # in, and the replicas diverge.
            raise RuntimeError(
                f"EP expert gradient sync is required ({self.num_ep_groups} EP groups) but rank "
                f"{global_rank} is not in the expert-replica set it was assigned "
                f"({self.expert_replica_ranks}) — the EP group layout and the replica-set "
                f"construction disagree. The deferred sweep would reduce over a group this rank "
                f"is not a member of, and expert replicas would silently diverge."
            )

    def _rank_block_bases(self) -> range:
        """First global rank of every ``stage_world_size``-wide rank block in the job (one per pipeline stage)."""
        return range(0, get_global_world_size(), self.stage_world_size)

    def _job_ep_groups(self, layout: NodeLocalLayout | CrossNodeLayout) -> list[list[int]]:
        """Every EP group of the job in global ranks, block-major: each rank builds the other pipeline
        stages' groups too, or the stages' ``new_group`` sequences differ and the job hangs."""
        return [
            [base + rank for rank in layout.group_ranks(group_idx)]
            for base in self._rank_block_bases()
            for group_idx in range(layout.num_groups)
        ]

    def _build_expert_tp_groups(self, global_rank: int, layout: NodeLocalLayout | CrossNodeLayout):
        """COLLECTIVE: build the expert-TP groups and, at ``ep_size > 1``, the dispatch (sub-EP) groups.

        Dispatch groups back DeepEP's NVLink buffer, whose intranode P2P kernel needs contiguous ranks;
        expert-TP groups (plain NCCL) reduce in token space outside dispatch->combine. Node-local::

            EP group [0..7], ep_size=4, expert_tp_size=2
            dispatch [0,1,2,3] [4,5,6,7]         (contiguous: DeepEP)
            expert TP [0,4] [1,5] [2,6] [3,7]    (strided: NCCL)
        """
        expert_tp_size = self.expert_tp_size
        ep_size = self.ep_size

        # Shared leaf with ParallelismConfig: ETP partners share dispatch_ep_rank → identical batches.
        self.dispatch_ep_rank, self.expert_tp_rank = layout.etp_partition(self.ep_rank, ep_size, expert_tp_size)

        ep_groups = self._job_ep_groups(layout)
        if self.node_local:
            tp_lists = [[g[d + ep_size * t] for t in range(expert_tp_size)] for g in ep_groups for d in range(ep_size)]
            dispatch_lists = [g[t * ep_size : (t + 1) * ep_size] for g in ep_groups for t in range(expert_tp_size)]
        else:
            # Cross-node EP + node-local ETP: each domain's contiguous block is one ETP group, keeping
            # its all-reduce on NVLink; the dispatch group strides across domains over Gin.
            for group_ranks in ep_groups:
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
            tp_lists = [g[c * expert_tp_size : (c + 1) * expert_tp_size] for g in ep_groups for c in range(ep_size)]
            dispatch_lists = [
                [g[p + expert_tp_size * c] for c in range(ep_size)] for g in ep_groups for p in range(expert_tp_size)
            ]
        self.expert_tp_group = _build_member_group(tp_lists, global_rank, "expert-TP group")
        if ep_size > 1:
            self.dispatch_ep_group = _build_member_group(dispatch_lists, global_rank, "dispatch group")

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
        domain_id = global_rank // self.nvlink_domain_size
        logger.debug(
            f"EP rank {global_rank}: EP group idx={self.ep_group_idx}, "
            f"EP rank={self.ep_rank}, NVLink domain={domain_id}, "
            f"group ranks={self._my_ep_group_ranks}"
        )
