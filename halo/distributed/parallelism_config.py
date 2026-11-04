"""The job's axis sizes, resolved topology and rank coordinates. EP is orthogonal to DP; PP divides the world into
stages and TP/CP/expert-TP reduce ``data_parallel_size`` within each."""

import logging
from dataclasses import dataclass, field
from typing import Optional, get_args

from halo.distributed import group_layout
from halo.distributed.context_parallel.config import CPConfig
from halo.distributed.expert_parallel.config import EPBufferBackend, EPConfig
from halo.distributed.nvlink import (
    requested_nvlink_domain_size,
    validate_nvlink_domain_against_fabric,
    verify_mnnvl_prerequisites,
)
from halo.distributed.parallelism_rules import (
    AXES,
    PP_DEFAULT_SCHEDULE,
    EPScope,
    PPSchedule,
    reject_invalid_composition,
    reject_unsupported_axes,
)
from halo.kernels.lowp.spec import LowpPrecision, LowpSpec
from halo.models.moe.expert_lora_spec import ExpertLoraSpec
from halo.runtime.consensus import refusal_text, reject_across_ranks, reject_divergent_settings
from halo.runtime.ranks import get_global_rank, get_global_world_size, get_local_world_size, is_global_main_process

logger = logging.getLogger(__name__)


@dataclass
class ParallelismConfig:
    """Parallelism sizes and ranks; builds the EPConfig/CPConfig that create the process groups."""

    world_size: int = field(default=0)
    gpus_per_node: int = field(default=0)
    # Locality unit of every node-local group; 0 = NVLINK_DOMAIN_SIZE, else gpus_per_node.
    nvlink_domain_size: int = field(default=0)

    ep_size: int = 1
    cp_size: int = 1
    tp_size: int = 1
    expert_tp_size: int = 1  # expert FFN TP: MoE-only, node-local

    # The outermost axis and the only one meant to cross NVLink domains: pp_size contiguous rank blocks.
    pp_size: int = 1
    pp_schedule: PPSchedule = PP_DEFAULT_SCHEDULE
    pp_microbatches: int = 0  # 0 = gradient_accumulation_steps
    pp_split: list[int] | None = None  # per-stage decoder-layer counts; None = the head-weighted default

    ep_scope: EPScope = "auto"  # resolved to node/global from ep_group_size vs the NVLink domain

    ep_fp32_router: bool = False
    ep_fp32_experts: bool = False
    fp32_non_ep_params: bool = False  # fp32 non-expert masters; compute stays bf16 via autocast
    fp32_grad_reduce: bool = False
    bf16_optimizer: bool | None = None  # AdamWBF16; None = on under bf16 with a default optim, off under DDP
    use_grouped_gemm: bool = True  # at ep_size=1 this alone applies the EP layers

    lowp: LowpSpec = field(default_factory=LowpSpec)

    max_concurrent_loading: int | None = None  # per node; None = half the node width capped at 4, 0 = all
    ep_lazy_loading: bool = True  # every EP path; TP-only MoE ignores it, PP always loads stage-lazily
    ep_buffer_backend: EPBufferBackend = "auto"  # "auto" == "elastic" (DeepEP V2); "legacy" = V1 CUDA-IPC

    # The rows one MoE forward carries per device and the max_length the dispatch-ceiling gate checks;
    # 0 rows = no budget, 0 length = context window.
    ep_rows_per_device: int = 0
    ep_declared_max_length: int = 0

    fsdp_reshard_after_forward: bool = False  # True = FULL_SHARD, False = SHARD_GRAD_OP
    # False keeps params unsharded across a grad-accum window instead of re-gathering every microstep.
    fsdp_reshard_after_backward: bool = True
    # True reduce-scatters once per grad-accum window, holding an unsharded gradient copy across it.
    fsdp_defer_grad_sync: bool = False
    # At ep_group_size == 1, FSDP-shard the replicated experts so their reduce-scatter is the grad sync.
    fsdp_shard_ep1_experts: bool = True
    use_hsdp: bool = False  # shard within an NVLink domain, replicate across domains
    expert_lora: Optional[ExpertLoraSpec] = None
    merge_expert_lora_on_save: bool = False

    num_nodes: int = field(default=0, init=False)
    num_nvlink_domains: int = field(default=0, init=False)  # per pipeline stage
    ep_group_size: int = field(default=0, init=False)  # ep_size * expert_tp_size
    data_parallel_size: int = field(default=0, init=False)

    # Pipeline-stage coordinates; at pp_size=1 the stage is the whole world.
    stage_world_size: int = field(default=0, init=False)
    pp_rank: int = field(default=0, init=False)
    stage_base_rank: int = field(default=0, init=False)
    stage_local_rank: int = field(default=0, init=False)

    global_rank: int = field(default=0, init=False)

    _ep_config: Optional["EPConfig"] = field(default=None, init=False, repr=False)
    _cp_config: Optional["CPConfig"] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """COLLECTIVE. Resolve and validate, then raise one world-uniform ValueError on any rank's failure. Two inputs
        are per-node (``gpus_per_node``, ``NVLINK_DOMAIN_SIZE``), so a local raise would leave other ranks in the fabric
        collectives until the watchdog fires."""
        try:
            self._resolve_and_validate()
        except Exception as exc:  # any type, or the raising rank would skip the gather its peers wait in
            local_reason = refusal_text(exc)
        else:
            local_reason = None
        reject_across_ranks(local_reason, "parallelism config", exc_type=ValueError)

        # The MNNVL gate runs a collective only above nvlink_domain_size > gpus_per_node, so a per-node
        # drift across that threshold would split the world at it.
        reject_divergent_settings(
            {"nvlink_domain_size": self.nvlink_domain_size, "gpus_per_node": self.gpus_per_node},
            "NVLink topology",
            "They are the locality unit every group's rank math divides by, and they decide whether a "
            "rank runs the Multi-Node NVLink prerequisite collective at all.",
        )

        verify_mnnvl_prerequisites(self.nvlink_domain_size, self.gpus_per_node)
        validate_nvlink_domain_against_fabric(self.nvlink_domain_size, self.world_size, self.gpus_per_node)

        if is_global_main_process():
            logger.info(self.summary())

    def _resolve_and_validate(self) -> None:
        """Fill the computed fields and apply every rank-local rule. Collective-free: it runs inside the gather."""
        if self.world_size == 0:
            self.world_size = get_global_world_size()
        if self.gpus_per_node == 0:
            self.gpus_per_node = get_local_world_size()
            # Device count can exceed world_size on a sub-node job; clamp so domain divisibility holds.
            self.gpus_per_node = min(self.gpus_per_node, self.world_size)

        self._validate_field_values()
        # Before any rank math, so an unsupported combination is not reported as a divisibility error.
        reject_unsupported_axes(self)
        self._resolve_nvlink_domain()
        self._resolve_topology()
        reject_invalid_composition(self)
        self._validate_lowp()

    def _validate_field_values(self) -> None:
        for axis in AXES:
            size = getattr(self, axis.size_field)
            if size < 1:
                raise ValueError(f"{axis.flag} must be >= 1 (1 = disabled), got {size}")
        if self.pp_schedule not in get_args(PPSchedule):
            raise ValueError(f"pipeline_schedule must be one of {get_args(PPSchedule)}, got {self.pp_schedule!r}")
        if self.pp_microbatches < 0:
            raise ValueError(f"pipeline_microbatches must be >= 0 (0 = auto), got {self.pp_microbatches}")
        if self.ep_scope not in get_args(EPScope):
            raise ValueError(f"ep_scope must be one of {get_args(EPScope)}, got {self.ep_scope!r}")
        if self.ep_buffer_backend not in get_args(EPBufferBackend):
            raise ValueError(
                f"ep_buffer_backend must be one of {get_args(EPBufferBackend)}, got {self.ep_buffer_backend!r}"
            )
        if self.max_concurrent_loading is not None and self.max_concurrent_loading < 0:
            raise ValueError(
                f"max_concurrent_loading must be >= 0 (0 = all ranks in parallel, null = derive from "
                f"the node width), got {self.max_concurrent_loading}"
            )

    def _resolve_nvlink_domain(self) -> None:
        """Resolve ``nvlink_domain_size`` (0: ``NVLINK_DOMAIN_SIZE``, else the node) into whole nodes of the world."""
        # First: the domain math below is in whole nodes and would misreport an unequal node size.
        if self.gpus_per_node > 0 and self.world_size % self.gpus_per_node != 0:
            raise ValueError(
                f"world_size ({self.world_size}) must be divisible by gpus_per_node "
                f"({self.gpus_per_node}). Heterogeneous / under-provisioned node sizes are "
                f"not supported (floor division would truncate num_nodes and orphan trailing "
                f"ranks). Use uniform node sizes, or pass an explicit gpus_per_node."
            )

        if self.gpus_per_node <= 0:
            raise ValueError(
                f"gpus_per_node resolved to {self.gpus_per_node}, but every domain rule below divides "
                f"by it. It comes from LOCAL_WORLD_SIZE / SLURM_NTASKS_PER_NODE / the visible device "
                f"count — set one of them, or pass gpus_per_node explicitly."
            )
        if self.nvlink_domain_size < 0:
            # A negative multiple passes Python's modulo in both divisibility gates below.
            raise ValueError(
                f"nvlink_domain_size must be positive (0 = resolve from NVLINK_DOMAIN_SIZE / "
                f"gpus_per_node), got {self.nvlink_domain_size}."
            )
        if self.nvlink_domain_size == 0:
            self.nvlink_domain_size = requested_nvlink_domain_size() or self.gpus_per_node
        # A rack-wide domain must never be silently shrunk onto a multi-node sub-job (may span racks).
        if self.nvlink_domain_size > self.world_size and self.world_size > self.gpus_per_node:
            raise ValueError(
                f"NVLINK_DOMAIN_SIZE ({self.nvlink_domain_size}) exceeds this job's world_size "
                f"({self.world_size}) on a multi-node job. If all {self.world_size} GPUs share one "
                f"NVLink domain (one rack), set NVLINK_DOMAIN_SIZE={self.world_size} for this job; "
                f"if they span racks, unset it (per-node NVLink only)."
            )
        if self.nvlink_domain_size > self.world_size:
            # Single-node only; the summary shows only the clamped value.
            if is_global_main_process():
                logger.info(
                    f"nvlink_domain_size ({self.nvlink_domain_size}) exceeds this job's world_size; "
                    f"clamping to {self.world_size}."
                )
            self.nvlink_domain_size = self.world_size
        if self.nvlink_domain_size % self.gpus_per_node != 0:
            raise ValueError(
                f"nvlink_domain_size ({self.nvlink_domain_size}) must be a multiple of gpus_per_node "
                f"({self.gpus_per_node}). A domain SMALLER than a node (a box whose GPUs sit on "
                f"separate NVLink islands) is not representable: every node-local group is a "
                f"contiguous rank block sized in whole nodes, so the rank math would hand out groups "
                f"that straddle the island boundary. Unset NVLINK_DOMAIN_SIZE, or run one process "
                f"group per island."
            )
        if self.world_size % self.nvlink_domain_size != 0:
            raise ValueError(
                f"world_size ({self.world_size}) must be divisible by nvlink_domain_size "
                f"({self.nvlink_domain_size}); a partial trailing NVLink domain would truncate "
                f"num_nvlink_domains (floor division) and orphan the trailing ranks from every "
                f"node-local EP/CP group."
            )

    def _resolve_topology(self) -> None:
        # PP carves the world first: every other mode's group math uses stage_world_size, not world_size.
        if self.world_size % self.pp_size != 0:
            raise ValueError(
                f"world_size ({self.world_size}) must be divisible by pipeline_parallel_size ({self.pp_size}): "
                f"pipeline stages are equal contiguous rank blocks."
            )
        self.stage_world_size = self.world_size // self.pp_size

        self.ep_group_size = self.ep_size * self.expert_tp_size

        if self.ep_scope == "auto":
            self.ep_scope = "node" if self.ep_group_size <= self.nvlink_domain_size else "global"
        elif self.ep_group_size == 1:
            # A one-rank group spans no domain; 'global' would send the dense run through cross-node rank math.
            self.ep_scope = "node"

        self.num_nodes = self.world_size // self.gpus_per_node
        self.num_nvlink_domains = self.stage_world_size // self.nvlink_domain_size

        self.data_parallel_size = self.stage_world_size // max(self.tp_size, self.cp_size, self.expert_tp_size)

        self.global_rank = get_global_rank()
        self.pp_rank = self.global_rank // self.stage_world_size
        self.stage_base_rank = self.pp_rank * self.stage_world_size
        self.stage_local_rank = self.global_rank % self.stage_world_size

    def _validate_lowp(self):
        self.lowp.validate()
        if not self.lowp.enabled and self.lowp.non_default_shape_fields() and is_global_main_process():
            enabled = ", ".join(repr(p) for p in get_args(LowpPrecision) if p != "bf16")
            logger.warning(
                "lowp_* tuning options are set but lowp_precision='bf16' (low precision is OFF) — "
                f"keep_blocks/apply_* have no effect until lowp_precision is one of {enabled}."
            )

    @property
    def experts_fsdp_managed(self) -> bool:
        """FSDP syncs the replicated experts (``ep_group_size == 1``), so the EP layer's grad hooks must not."""
        return self.fsdp_shard_ep1_experts and self.ep_group_size == 1

    @property
    def fsdp_reduces_in_fp32(self) -> bool:
        """Whether FSDP2 reduces gradients in fp32: on request, or for fp32 non-expert masters."""
        return self.fp32_grad_reduce or self.fp32_non_ep_params

    def pp_microbatch_count(self, gradient_accumulation_steps: int) -> int:
        """``pipeline_microbatches``, else the accumulation, raised to a full pipeline under 1F1B."""
        count = self.pp_microbatches or gradient_accumulation_steps
        return max(count, self.pp_size) if self.pp_schedule == "1f1b" else count

    @property
    def active_axes(self) -> frozenset[str]:
        return frozenset(a.name for a in AXES if getattr(self, a.size_field) > 1)

    @property
    def is_racy_single_domain_multigroup_ep(self) -> bool:
        """Several >2-rank DeepEP dispatch groups on one NVLink domain, whose combine barriers race FSDP2's DP-wide
        collectives (``legacy`` deadlocks, ``elastic`` faults on peer memory). Keyed on ``ep_group_size``, since EP+ETP
        filling the domain (ep4+etp2 on 8) runs clean."""
        return self.num_nvlink_domains == 1 and self.ep_size > 2 and self.nvlink_domain_size > self.ep_group_size

    @property
    def num_ep_groups(self) -> int:
        """EP groups per stage, counted off the layout ``EPConfig`` builds its groups from."""
        return self.ep_layout.num_groups

    @property
    def ep_layout(self) -> group_layout.NodeLocalLayout | group_layout.CrossNodeLayout:
        """This stage's EP group layout, in stage-local rank space."""
        return group_layout.ep_layout(
            world_size=self.stage_world_size,
            group_size=self.ep_group_size,
            domain_size=self.nvlink_domain_size,
            node_local=self.ep_scope == "node",
        )

    @property
    def has_ep_group(self) -> bool:
        """An expert-distribution group exists: EP, expert TP, or both."""
        return self.ep_group_size > 1

    @property
    def needs_ep_layers(self) -> bool:
        """EP layers apply: an EP group, or grouped GEMM alone (applied on MoE models only)."""
        return self.ep_group_size > 1 or self.use_grouped_gemm

    @property
    def is_cp_mode(self) -> bool:
        return self.cp_size > 1

    @property
    def is_tp_mode(self) -> bool:
        return self.tp_size > 1

    @property
    def is_expert_tp_mode(self) -> bool:
        return self.expert_tp_size > 1

    @property
    def is_pp_mode(self) -> bool:
        return self.pp_size > 1

    @property
    def is_first_pp_stage(self) -> bool:
        """This rank holds the pipeline's first stage (owns the embedding and consumes input_ids)."""
        return self.pp_rank == 0

    @property
    def is_last_pp_stage(self) -> bool:
        """This rank holds the pipeline's last stage (owns the head and computes the loss)."""
        return self.pp_rank == self.pp_size - 1

    def pp_stage_rank_blocks(self) -> list[list[int]]:
        return [list(range(s * self.stage_world_size, (s + 1) * self.stage_world_size)) for s in range(self.pp_size)]

    def pp_chain_ranks(self) -> list[int]:
        """The ranks at this rank's intra-stage position in every stage, in stage order: one batch's P2P chain."""
        return [block[self.stage_local_rank] for block in self.pp_stage_rank_blocks()]

    @property
    def has_custom_parallelism(self) -> bool:
        """An axis whose mesh only the toolkit's torchrun path builds (``use_grouped_gemm`` is not one)."""
        return bool(self.active_axes)

    @property
    def is_hsdp(self) -> bool:
        """HSDP's 2-D (dp_replicate, dp_shard) mesh is active: only across more than one NVLink domain."""
        return self.use_hsdp and self.num_nvlink_domains > 1

    @property
    def dp_shard_size(self) -> int:
        """FSDP shard-group width: one NVLink domain under HSDP, else the whole DP world."""
        return self.nvlink_domain_size if self.is_hsdp else self.stage_world_size

    @property
    def dp_replicate_size(self) -> int:
        return self.num_nvlink_domains if self.is_hsdp else 1

    @property
    def non_dp_replication_factor(self) -> int:
        """How many ranks see the same batch; a world-gathered token counter divides by it."""
        return self.world_size // self.data_parallel_size

    @property
    def requires_rdma(self) -> bool:
        """The EP group spans NVLink domains (not OS nodes: an NVL72 rack-wide group stays on MNNVL)."""
        return self.has_ep_group and self.ep_scope == "global" and self.num_nvlink_domains > 1

    @property
    def mode_string(self) -> str | None:
        parts = []
        if self.is_pp_mode:
            parts.append("pp")
        if self.has_ep_group:
            parts.append("ep")
        if self.is_cp_mode:
            parts.append("cp")
        if self.is_tp_mode:
            parts.append("tp")
        if self.is_expert_tp_mode:
            parts.append("expert-tp")
        if not parts and self.use_grouped_gemm:
            parts.append("gmm")
        return "-".join(parts) if parts else None

    def get_data_parallel_rank(self) -> int:
        """The batch this rank processes, shared by its TP/CP/expert-TP partners and its pipeline chain."""
        if self.expert_tp_size > 1:
            # ETP partners share dispatch_ep_rank, so the expert-TP SUM adds matching tokens.
            layout = self.ep_layout
            ep_rank, ep_group_idx = layout.rank_and_group(self.stage_local_rank)
            dispatch_ep_rank, _ = layout.etp_partition(ep_rank, self.ep_size, self.expert_tp_size)
            return dispatch_ep_rank * layout.num_groups + ep_group_idx
        return self.stage_local_rank // max(self.tp_size, self.cp_size)

    def ep_config(self) -> "EPConfig":
        """COLLECTIVE on first call, which builds every EP process group."""
        if self._ep_config is None:
            self._ep_config = EPConfig(
                ep_size=self.ep_size,
                # EP lives inside one pipeline stage: tile stage_world_size from this stage's base.
                stage_world_size=self.stage_world_size,
                stage_base_rank=self.stage_base_rank,
                node_local=self.ep_scope == "node",
                nvlink_domain_size=self.nvlink_domain_size,
                # The router is skipped by the fp32_non_ep_params upcast; bf16 there would trip
                # FSDP2's uniform-dtype check against the fp32 dense params.
                fp32_router=self.ep_fp32_router or self.fp32_non_ep_params,
                fp32_experts=self.ep_fp32_experts,
                expert_tp_size=self.expert_tp_size,
                use_grouped_gemm=self.use_grouped_gemm,
                fp32_grad_reduce=self.fp32_grad_reduce,
                expert_lora=self.expert_lora,
                experts_fsdp_managed=self.experts_fsdp_managed,
                ep_buffer_backend=self.ep_buffer_backend,
            )
        return self._ep_config

    def cp_config(self) -> "CPConfig":
        """COLLECTIVE on first call, which builds every CP process group."""
        if self._cp_config is None:
            self._cp_config = CPConfig(
                self.cp_size,
                stage_world_size=self.stage_world_size,
                stage_base_rank=self.stage_base_rank,
                nvlink_domain_size=self.nvlink_domain_size,
            )
        return self._cp_config

    def summary(self) -> str:
        parts = []
        if self.is_pp_mode:
            parts.append(f"PP={self.pp_size} ({self.pp_schedule})")
        if self.ep_size > 1:
            parts.append(f"EP={self.ep_size} ({self.ep_scope})")
        if self.is_cp_mode:
            parts.append(f"CP={self.cp_size}")
        if self.is_tp_mode:
            parts.append(f"TP={self.tp_size}")
        if self.is_expert_tp_mode:
            parts.append(f"ExpertTP={self.expert_tp_size}")

        mode_str = " + ".join(parts) if parts else "DATA PARALLEL"

        lines = [
            "=" * 60,
            f"PARALLELISM: {mode_str}",
            f"  World: {self.world_size} ({self.num_nodes} nodes × {self.gpus_per_node} GPUs)",
            f"  Data parallel: {self.data_parallel_size} distinct batches",
            f"  Gradient sync: {self.stage_world_size} ranks",
        ]

        if self.is_pp_mode:
            lines.append(
                f"  Pipeline: {self.pp_size} stages × {self.stage_world_size} ranks "
                f"(this rank = stage {self.pp_rank}; boundaries on NVLink-domain edges)"
            )

        if self.nvlink_domain_size != self.gpus_per_node:
            lines.append(
                f"  NVLink domain: {self.nvlink_domain_size} GPUs "
                f"({self.num_nvlink_domains} domain(s)) — Multi-Node NVLink (NVL72/MNNVL)"
            )
        if self.has_ep_group:
            lines.append(f"  EP groups: {self.num_ep_groups}")
        if self.is_expert_tp_mode:
            lines.append(
                f"  Expert TP: {self.expert_tp_size} (ep_size={self.ep_size}, ep_group_size={self.ep_group_size})"
            )
        if self.is_cp_mode:
            lines.append(f"  CP groups: {self.data_parallel_size}")
        if self.is_hsdp:
            lines.append(
                f"  HSDP: shard {self.dp_shard_size}-way within domain, "
                f"replicate {self.dp_replicate_size}-way across domains"
            )
        if self.requires_rdma:
            lines.append("  EP spans NVLink domains — requires RDMA")

        lines.append("=" * 60)
        return "\n".join(lines)
