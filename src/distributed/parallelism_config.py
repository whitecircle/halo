"""Unified parallelism configuration: EP/CP/TP/expert-TP/DP sizes and ranks, validated at config time.

EP is orthogonal to DP (all-to-all routing); only TP/CP/expert_tp reduce data_parallel_size.
ep_scope "node" keeps EP within one NVLink domain, "global" spans domains over RDMA.
"""

import logging
import os
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, Optional, get_args

from src.distributed.context_parallel.config import CPConfig
from src.distributed.expert_parallel.config import (
    EPConfig,
    ep_dispatch_capacity,
    padded_wire_hidden,
    reject_expert_lora_with_expert_tp,
    reject_legacy_backend_topology,
    reject_oversized_dispatch,
)
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
from src.distributed.nvlink import (
    check_mnnvl_prerequisites,
    get_nvlink_domain_size,
    validate_nvlink_domain_against_fabric,
)
from src.distributed.runtime import (
    get_global_rank,
    get_global_world_size,
    get_local_world_size,
    is_global_main_process,
    reject_across_ranks,
    reject_divergent_settings,
)
from src.distributed.tensor_parallel.parallelize_attention import validate_tp_head_divisibility
from src.env import is_accelerate_launch
from src.models.loading.config_levels import text_config
from src.models.loading.tokenizer_setup import context_window_from_config
from src.models.moe_balancing import (
    ROUTER_EXPERT_COUNT_FIELDS,
    get_first_router_field,
    resolve_expert_ffn_shard_width,
    resolve_router_topk,
)

if TYPE_CHECKING:
    from .expert_parallel.config import ExpertLoraSpec

logger = logging.getLogger(__name__)

# The Literal type doubles as the validation table for each enum-like field.
PPSchedule = Literal["1f1b", "gpipe"]
PP_SCHEDULES: tuple[str, ...] = get_args(PPSchedule)
PP_DEFAULT_SCHEDULE: PPSchedule = "1f1b"

EPScope = Literal["auto", "node", "global"]
EP_SCOPES: tuple[str, ...] = get_args(EPScope)
EPBufferBackend = Literal["auto", "elastic", "legacy"]
EP_BUFFER_BACKENDS: tuple[str, ...] = get_args(EPBufferBackend)
LowpPrecision = Literal["bf16", "fp8", "fp4", "mxfp4"]
LOWP_PRECISIONS: tuple[str, ...] = get_args(LowpPrecision)

# Axis → CLI flag, so rejection messages quote the flag the user typed rather than the field name.
AXIS_FLAGS: dict[str, str] = {
    "pp": "pipeline_parallel_size",
    "ep": "expert_parallel_size",
    "etp": "expert_tensor_parallel_size",
    "tp": "tensor_parallel_size",
    "cp": "context_parallel_size",
}

# Allowlist of combinable model-sharding axes (FSDP is not an axis — it shards the leftover DP
# width). Unlisted combinations are rejected rather than running unvalidated.
SUPPORTED_AXIS_SETS: frozenset[frozenset[str]] = frozenset(
    map(
        frozenset,
        (
            (),  # plain data parallelism (FSDP2 / DDP)
            ("ep",),
            ("etp",),  # pure ETP (ep_size == 1): expert FFN sharded, experts replicated
            ("tp",),
            ("cp",),
            ("pp",),
            ("ep", "tp"),
            ("ep", "cp"),
            ("ep", "etp"),
            ("pp", "ep"),
            ("pp", "etp"),
        ),
    )
)

# Explanation per rejected combination; a missing entry only shortens the message.
AXIS_SET_MECHANISMS: dict[frozenset[str], str] = {
    frozenset({"tp", "cp"}): (
        "TP and CP would partition the same ranks twice: both groups are contiguous rank blocks, so "
        "at tp_size == cp_size a rank's TP partners ARE its CP partners, required to hold the same "
        "tokens (TP) and different sequence chunks (CP) at once. Ulysses also redistributes attention "
        "over HEADS, of which TP has already left this rank only num_attention_heads / tp_size, while "
        "validate_model_for_ulysses checks divisibility against the config's full head count; and "
        "data_parallel_size takes max(tp_size, cp_size), which counts one of the two axes only. "
        "Use EP+TP or EP+CP."
    ),
    frozenset({"etp", "cp"}): (
        "expert-TP partners hold shards of one expert and must see the SAME tokens — ReduceFromExpertTP "
        "sums their outputs element-wise in token space — but CP hands each rank a different sequence "
        "chunk, so that sum would add unrelated tokens. get_data_parallel_rank keys on the expert-TP "
        "layout alone, so it also stops agreeing with data_parallel_size once cp_size exceeds "
        "expert_tp_size. Drop one of the two."
    ),
    frozenset({"tp", "etp"}): (
        "attention TP and expert TP would shard the same ranks along two different axes. Use EP+TP "
        "(attention sharded, experts distributed) or pure ETP (expert FFN sharded)."
    ),
    frozenset({"pp", "tp"}): (
        "PP+TP needs at least 2 nodes to launch at all (a stage must be whole NVLink domains), and "
        "the composition was never exercised on real multi-node hardware. It also excludes the "
        "dense families most likely to want it: transformers reconciles a "
        "'replicated_with_grad_allreduce' plan entry (Qwen3 q_norm/k_norm) with a per-backward "
        "all-reduce hook, which the pipeline schedule disables on non-final microbatches, so the "
        "hook re-reduces accumulated history. Use PP+EP, or TP inside a single stage-less job."
    ),
    frozenset({"pp", "cp"}): (
        "the pipeline loss normalizer is already stage-wide and carries no cancelling x cp_size "
        "factor, so every gradient would come out cp_size x too small with no error raised. "
        "Use PP without CP, or CP without PP."
    ),
    frozenset({"pp", "ep", "etp"}): (
        "ReduceFromExpertTP.backward is model math, not gradient sync, so it CANNOT be deferred the "
        "way the DP sweep defers its all-reduce: it runs inside every microbatch backward. At "
        "ep_size > 1 that strided expert-TP all-reduce interleaves with the DeepEP combine the same "
        "backward is inside, and the two orderings are not guaranteed to agree across ranks. At "
        "ep_size == 1 (PP+ETP) the dispatch group is width 1, so there is no combine to interleave "
        "with and the composition is safe. Use PP+EP or PP+ETP, not all three."
    ),
    frozenset({"pp", "ep", "tp"}): (
        "the deferred cross-replica expert sweep needs FSDP to shard non-expert params over the EP "
        "group (a 1-D ep-sized mesh), while EP+TP shards them over the 2-D (dp, tp) mesh — the two "
        "contracts cannot both hold. EP already shards the experts, so TP would add only attention. "
        "Use PP+EP."
    ),
    frozenset({"pp", "ep", "cp"}): (
        "under PP the expert gradients are synced by the deferred post-backward sweep, whose divisor "
        "counts every rank of the stage as a distinct DP replica. CP ranks are not — they hold "
        "sequence shards of the SAME batch — and the pipeline loss carries no cancelling x cp_size "
        "factor, so every expert gradient would come out cp_size x too small. Use PP+EP without CP."
    ),
    frozenset({"ep", "tp", "etp"}): (
        "attention TP and expert TP cannot both shard the EP group. Use EP+TP or EP+ETP."
    ),
}


def _divisors_up_to(value: int, ceiling: int) -> list[int]:
    """Divisors of ``value`` in ``2..ceiling`` — the sizes a rejection may suggest.

    1 is excluded (it means no expert parallelism). Suggestions are arithmetic about the model only;
    filtering them against the topology gates is the caller's job.
    """
    return [d for d in range(2, min(value, ceiling) + 1) if value % d == 0]


def _render_axis_set(axes: frozenset[str]) -> str:
    """Axis set rendered for messages: 'PP + EP', or 'plain data parallelism' when empty."""
    ordered = [a for a in AXIS_FLAGS if a in axes]
    return " + ".join(a.upper() for a in ordered) if ordered else "plain data parallelism"


@dataclass
class ParallelismConfig:
    """Unified parallelism sizes/ranks; creates specialized EPConfig/CPConfig for process groups.

    ep_group_size = ep_size * expert_tp_size (full EP process-group width). EP is orthogonal to DP.
    """

    world_size: int = field(default=0)
    gpus_per_node: int = field(default=0)

    # Locality unit for "node-local" TP/CP/EP; auto == gpus_per_node. On GB200/GB300 NVL72 set
    # NVLINK_DOMAIN_SIZE to the rack-wide MNNVL fabric (gpus_per_node stays the OS-node unit).
    nvlink_domain_size: int = field(default=0)

    ep_size: int = 1
    cp_size: int = 1
    tp_size: int = 1
    expert_tp_size: int = 1  # Expert FFN TP (independent of tp_size, MoE-only, node-local)

    # Outermost dimension, the only one meant to cross NVLink domains: the world splits into pp_size
    # contiguous rank blocks and every other mode runs unchanged inside one block (only P2P uses RDMA).
    pp_size: int = 1
    pp_schedule: PPSchedule = PP_DEFAULT_SCHEDULE
    pp_microbatches: int = 0  # 0 = resolved by the trainer from gradient_accumulation_steps
    # Manual per-stage decoder-layer counts (must sum to the model's layer count). None = the
    # head-weighted default, shrinking the last stage's budget by the lm_head's layer-equivalent cost.
    pp_split: list[int] | None = None

    # "auto" resolves below to node/global from ep_group_size vs the NVLink domain; same default as
    # DistributedArguments, so a hand-built config behaves like a YAML-built one.
    ep_scope: EPScope = "auto"

    ep_fp32_router: bool = False
    ep_fp32_experts: bool = False

    # Store non-expert params in FP32 for stable optimizer updates (compute stays BF16 via autocast).
    fp32_non_ep_params: bool = False

    # Reduce grads in fp32 with bf16 params (bf16 sums lose precision); fp32_non_ep_params implies this.
    fp32_grad_reduce: bool = False

    # Tri-state AdamWBF16 switch. None = auto (on under bf16 + a default optim, off under DDP).
    bf16_optimizer: bool | None = None

    # F.grouped_mm for expert matmuls on SM90+; at ep_size=1 also applies EP wrappers (no inter-rank comm).
    use_grouped_gemm: bool = True

    # Low-precision GEMM compute, masters + checkpoint stay bf16: 'fp8'=mxfp8, 'fp4'=nvfp4 (most
    # accurate), 'mxfp4'=fast fp4. keep_{first,last}_blocks pin end blocks to bf16.
    lowp_precision: LowpPrecision = "bf16"
    lowp_apply_dense_mlp: bool = True
    lowp_apply_moe_experts: bool = True
    lowp_keep_first_blocks: int = 0
    lowp_keep_last_blocks: int = 0

    # Max ranks loading simultaneously per node. None = derive from the node width (half of it, capped
    # at 4); 1 = sequential (lowest CPU-RAM peak), 0 = all parallel. An explicit value is used verbatim
    # (see resolve_load_concurrency).
    max_concurrent_loading: int | None = None

    # True: every EP path (EP, EP+CP, EP+TP, pure ETP) loads lazily from safetensors (meta init +
    # per-rank I/O). TP-only MoE ignores it and uses from_pretrained + patching; PP always loads
    # stage-lazily and rejects a checkpoint the lazy loader cannot read.
    ep_lazy_loading: bool = True

    # DeepEP transport: "auto"=="elastic" (ElasticBuffer/NCCL Gin, cross-node); "legacy" = V1 CUDA-IPC.
    ep_buffer_backend: EPBufferBackend = "auto"

    # The run's per-rank shape for the dispatch-ceiling gate below, derived from the training config by
    # ``parallelism_config_from_args``: rows one MoE forward carries per device (rows_per_forward x
    # per_device_train_batch_size) and the config's max_length. Length 0 means ``max_length: null``,
    # resolved by the gate against the model's context window; 0 rows declares no budget.
    ep_rows_per_device: int = 0
    ep_declared_max_length: int = 0

    # True = reshard after forward (FULL_SHARD, lower peak memory); False = SHARD_GRAD_OP (faster).
    fsdp_reshard_after_forward: bool = False

    # False keeps params unsharded across a grad-accum window's microsteps (set_reshard_after_backward,
    # toggled back on for its last backward so the optimizer still reads sharded params carrying grads).
    # FSDP2 otherwise reshards after each microstep's backward and re-all-gathers the full model on the
    # next — once per gradient_accumulation_step, and costly when NCCL is forced onto sockets
    # (rollout_backend=sglang). Costs one unsharded param copy per GPU; plain-DP torchrun path only.
    fsdp_reshard_after_backward: bool = True

    # ep_size==1 only: True shards the replicated experts via FSDP reduce-scatter (grad-equivalent,
    # frees DP-scaling memory); RL-safe — the vLLM weight-sync gather materializes shards first.
    fsdp_shard_ep1_experts: bool = True

    # Shard non-expert params within each NVLink domain, replicate across domains. Pure DP or CP only
    # (EP / TP / ETP are rejected — see _validate_hsdp); no-op on a single domain.
    use_hsdp: bool = False

    # Disable accelerate's bf16→fp32 logits wrapper — OOMs on long seqs; the model is bf16 native.
    fp32_output_conversion: bool = False

    # Native grouped-LoRA on EP experts; assigned before load_distributed_model. None = no expert LoRA.
    expert_lora: Optional["ExpertLoraSpec"] = None

    # On save, fold the grouped-LoRA delta into the base experts instead of a standalone adapter.
    merge_expert_lora_on_save: bool = False

    num_nodes: int = field(default=0, init=False)
    num_nvlink_domains: int = field(default=0, init=False)  # domains per pipeline stage
    ep_group_size: int = field(default=0, init=False)  # ep_size * expert_tp_size
    data_parallel_size: int = field(default=0, init=False)

    # Pipeline-stage coordinates; at pp_size=1 they reduce to the whole world (formulas unchanged).
    stage_world_size: int = field(default=0, init=False)
    pp_rank: int = field(default=0, init=False)
    stage_base_rank: int = field(default=0, init=False)
    stage_local_rank: int = field(default=0, init=False)

    global_rank: int = field(default=0, init=False)

    _ep_config: Optional["EPConfig"] = field(default=None, init=False, repr=False)
    _cp_config: Optional["CPConfig"] = field(default=None, init=False, repr=False)

    def __post_init__(self):
        """Resolve the computed fields and validate, reporting the verdict world-uniformly.

        Every rule in :meth:`_resolve_and_validate` is rank-local, while ``gpus_per_node`` and
        ``nvlink_domain_size`` are per-node inputs: a drift that trips a divisibility rule raises on
        one node's ranks alone, leaving the rest blocked in the fabric collective below.
        """
        try:
            self._resolve_and_validate()
        except Exception as exc:  # the reason is re-raised world-uniformly below
            # Any exception, not just ValueError: one escaping this block would skip the gather below
            # and leave every peer blocked in it. The type is kept in the message across the re-raise.
            local_reason = str(exc) if isinstance(exc, ValueError) else f"{type(exc).__name__}: {exc}"
        else:
            local_reason = None
        reject_across_ranks(local_reason, "parallelism config", exc_type=ValueError)

        # Both inputs to the MNNVL gate below are per-node, and that gate runs a collective only above
        # ``nvlink_domain_size > gpus_per_node``: a drift straddling that threshold splits the world at
        # it, and no rank-local divisibility check catches it. Rejected before either collective.
        reject_divergent_settings(
            {"nvlink_domain_size": self.nvlink_domain_size, "gpus_per_node": self.gpus_per_node},
            "NVLink topology",
            "They are the locality unit every group's rank math divides by, and they decide whether a "
            "rank runs the Multi-Node NVLink prerequisite collective at all.",
        )

        check_mnnvl_prerequisites(self.nvlink_domain_size, self.gpus_per_node)
        validate_nvlink_domain_against_fabric(self.nvlink_domain_size, self.world_size, self.gpus_per_node)

        if is_global_main_process():
            logger.info(self.summary())

    def _resolve_and_validate(self) -> None:
        """Fill the computed fields and apply every rank-local rule. Raises ``ValueError``.

        Must stay collective-free: :meth:`__post_init__` runs it inside the uniformity gather, so a
        rule added here is reported world-uniformly.
        """
        if self.world_size == 0:
            self.world_size = get_global_world_size()
        if self.gpus_per_node == 0:
            self.gpus_per_node = get_local_world_size()
            # Device count can exceed world_size on a sub-node job; clamp so domain divisibility holds.
            self.gpus_per_node = min(self.gpus_per_node, self.world_size)

        for _name, _val in (
            ("ep_size", self.ep_size),
            ("tp_size", self.tp_size),
            ("cp_size", self.cp_size),
            ("expert_tp_size", self.expert_tp_size),
            ("pp_size", self.pp_size),
        ):
            if _val < 1:
                raise ValueError(f"{_name} must be >= 1 (1 = disabled), got {_val}")
        if self.pp_schedule not in PP_SCHEDULES:
            raise ValueError(f"pp_schedule must be one of {PP_SCHEDULES}, got {self.pp_schedule!r}")
        if self.pp_microbatches < 0:
            raise ValueError(f"pp_microbatches must be >= 0 (0 = auto), got {self.pp_microbatches}")
        if self.ep_scope not in EP_SCOPES:
            raise ValueError(f"ep_scope must be one of {EP_SCOPES}, got {self.ep_scope!r}")
        # The dispatcher's own check runs after the whole model has loaded and returns early at
        # ep_size <= 1, so a hand-built config's typo would not surface there.
        if self.ep_buffer_backend not in EP_BUFFER_BACKENDS:
            raise ValueError(f"ep_buffer_backend must be one of {EP_BUFFER_BACKENDS}, got {self.ep_buffer_backend!r}")
        if self.max_concurrent_loading is not None and self.max_concurrent_loading < 0:
            raise ValueError(
                f"max_concurrent_loading must be >= 0 (0 = all ranks in parallel, null = derive from "
                f"the node width), got {self.max_concurrent_loading}"
            )

        # Before any rank math: a combination with no validated composition is rejected as
        # unsupported rather than as a divisibility error. Reads only the five axis sizes above.
        self._validate_capability_matrix()

        # Before the domain math, which is expressed in whole nodes: an unequal node size would make
        # every domain rule below report a domain problem for what is really a node-count problem.
        self._validate_node_topology()

        if self.gpus_per_node <= 0:
            raise ValueError(
                f"gpus_per_node resolved to {self.gpus_per_node}, but every domain rule below divides "
                f"by it. It comes from LOCAL_WORLD_SIZE / SLURM_NTASKS_PER_NODE / the visible device "
                f"count — set one of them, or pass gpus_per_node explicitly."
            )
        if self.nvlink_domain_size < 0:
            # Python modulo would pass a negative multiple through both divisibility gates below,
            # leaving num_nvlink_domains negative.
            raise ValueError(
                f"nvlink_domain_size must be positive (0 = resolve from NVLINK_DOMAIN_SIZE / "
                f"gpus_per_node), got {self.nvlink_domain_size}."
            )
        if self.nvlink_domain_size == 0:
            self.nvlink_domain_size = get_nvlink_domain_size(self.gpus_per_node)
        # A rack-wide domain must not be shrunk onto a multi-node sub-job, which may span racks.
        if self.nvlink_domain_size > self.world_size and self.world_size > self.gpus_per_node:
            raise ValueError(
                f"NVLINK_DOMAIN_SIZE ({self.nvlink_domain_size}) exceeds this job's world_size "
                f"({self.world_size}) on a multi-node job. If all {self.world_size} GPUs share one "
                f"NVLink domain (one rack), set NVLINK_DOMAIN_SIZE={self.world_size} for this job; "
                f"if they span racks, unset it (per-node NVLink only)."
            )
        if self.nvlink_domain_size > self.world_size:
            # Single-node only (the multi-node case raised above), and legal — a rack-wide
            # NVLINK_DOMAIN_SIZE on a small job. The summary prints only the clamped value, so this
            # log is the record of the adjustment.
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

        # PP carves the world first: every other mode's group math uses stage_world_size, not world_size.
        if self.world_size % self.pp_size != 0:
            raise ValueError(
                f"world_size ({self.world_size}) must be divisible by pp_size ({self.pp_size}): "
                f"pipeline stages are equal contiguous rank blocks."
            )
        self.stage_world_size = self.world_size // self.pp_size

        self.ep_group_size = self.ep_size * self.expert_tp_size

        if self.ep_scope == "auto":
            self.ep_scope = "node" if self.ep_group_size <= self.nvlink_domain_size else "global"
        elif self.ep_group_size == 1:
            # A one-rank group spans no domain, so an explicit 'global' (copied off an EP config onto
            # a dense run) describes nothing, and ``_validate_ep_group`` skips it at this size — the
            # cross-node rank math would then raise about expert parallelism the run does not use.
            self.ep_scope = "node"

        self.num_nodes = self.world_size // self.gpus_per_node  # gpus_per_node > 0, checked above
        self.num_nvlink_domains = self.stage_world_size // self.nvlink_domain_size

        # DP = distinct batches: only TP/CP/expert_tp reduce it within a stage; EP stays orthogonal.
        dp_divisor = max(self.tp_size, self.cp_size, self.expert_tp_size)
        if dp_divisor > 1:
            self.data_parallel_size = self.stage_world_size // dp_divisor
        else:
            self.data_parallel_size = self.stage_world_size

        self.global_rank = get_global_rank()
        self.pp_rank = self.global_rank // self.stage_world_size
        self.stage_base_rank = self.pp_rank * self.stage_world_size
        self.stage_local_rank = self.global_rank % self.stage_world_size

        self._validate()

    def _validate(self):
        """Validate the parallelism configuration (one sub-validator per topology concern).

        ``_validate_node_topology`` and ``_validate_capability_matrix`` run earlier, from
        ``__post_init__``: the domain math divides by ``gpus_per_node``, and an unsupported axis set
        must be rejected before any rank math reports a divisibility error.
        """
        self._validate_pipeline_parallel()
        self._validate_ep_group()
        self._validate_cp_locality()
        self._validate_tp()
        self._validate_expert_tp()
        self._validate_ep_cp()
        self._validate_single_domain_multigroup_ep()
        self._validate_ep_buffer_backend()
        self._validate_hsdp()
        self._validate_fsdp_settings()
        self._validate_lowp()

    @property
    def experts_fsdp_managed(self) -> bool:
        """Mirrors :attr:`EPConfig.experts_fsdp_managed` for call sites that run before (or without)
        an ``EPConfig``. Both must agree: the EP layer's grad hooks and FSDP's reduce-scatter each
        assume the other is not syncing the experts."""
        return self.fsdp_shard_ep1_experts and self.ep_group_size == 1

    @property
    def _is_multi_domain_multi_group_ep(self) -> bool:
        """More than one EP dispatch group laid across more than one NVLink domain — the shape the TP
        and ETP composition validators reject, each for its own gradient-sync reason."""
        return self.ep_size > 1 and self.num_nvlink_domains > 1 and self.stage_world_size // self.ep_group_size > 1

    def _axis_sizes(self) -> dict[str, int]:
        """This config's size per :data:`AXIS_FLAGS` axis."""
        return {
            "pp": self.pp_size,
            "ep": self.ep_size,
            "etp": self.expert_tp_size,
            "tp": self.tp_size,
            "cp": self.cp_size,
        }

    @property
    def active_axes(self) -> frozenset[str]:
        """The model-sharding axes this config turns on, derived from the sizes."""
        return frozenset(axis for axis, size in self._axis_sizes().items() if size > 1)

    def _validate_capability_matrix(self):
        """Reject any axis combination outside :data:`SUPPORTED_AXIS_SETS`.

        Called from ``__post_init__`` ahead of every rank-math check, so an unsupported combination
        is reported as such rather than as a divisibility error about a shape that cannot run.
        """
        axes = self.active_axes
        if axes in SUPPORTED_AXIS_SETS:
            return
        mechanism = AXIS_SET_MECHANISMS.get(axes) or (
            "this combination has no validated composition — no equivalence gate has ever compared "
            "its gradients against an unsplit reference, so enabling it would train silently "
            "different gradients."
        )
        sizes = self._axis_sizes()
        typed = ", ".join(f"{AXIS_FLAGS[a]}={sizes[a]}" for a in AXIS_FLAGS if a in axes)
        supported = ", ".join(sorted(_render_axis_set(s) for s in SUPPORTED_AXIS_SETS))
        raise ValueError(
            f"{_render_axis_set(axes)} is not a supported parallelism combination ({typed}): "
            f"{mechanism} Supported combinations: {supported} — each composes with FSDP data "
            f"parallelism over whatever world width is left."
        )

    def _validate_node_topology(self):
        """world_size must be an exact multiple of gpus_per_node (else floor division orphans trailing ranks)."""
        if self.gpus_per_node > 0 and self.world_size % self.gpus_per_node != 0:
            raise ValueError(
                f"world_size ({self.world_size}) must be divisible by gpus_per_node "
                f"({self.gpus_per_node}). Heterogeneous / under-provisioned node sizes are "
                f"not supported (floor division would truncate num_nodes and orphan trailing "
                f"ranks). Use uniform node sizes, or pass an explicit gpus_per_node."
            )

    def _validate_pipeline_parallel(self):
        """PP stages must be whole NVLink domains, and PP must not be combined with modes whose
        collectives or loss aggregation it would break.

        Stage boundaries on domain boundaries keep every EP/CP/TP/ETP group (all domain-local rank
        blocks) inside one stage, so no intra-stage collective straddles a pipeline boundary and only
        PP's point-to-point activations cross RDMA.
        """
        if self.pp_size == 1:
            # Nothing reads the PP-only knobs at pp_size == 1, so a set value would have no effect.
            for name, value, default in (
                ("pp_split", self.pp_split, None),
                ("pp_microbatches", self.pp_microbatches, 0),
                ("pp_schedule", self.pp_schedule, PP_DEFAULT_SCHEDULE),
            ):
                if value != default:
                    raise ValueError(
                        f"{name}={value!r} is only meaningful with pipeline_parallel_size > 1; "
                        f"at pp_size=1 nothing reads it. Remove it or set pipeline_parallel_size."
                    )
            return

        if self.expert_lora is not None:
            raise ValueError(
                "Expert LoRA (native grouped adapters) is not supported under pipeline parallelism: "
                "the adapter save/merge paths (save_ep_lora_adapters, merge_expert_lora_on_save) "
                "write flat global expert names with no stage→global remap, so a PP save would "
                "silently record stage-local layer indices. Train expert LoRA without PP."
            )

        if self.pp_split is not None:
            if len(self.pp_split) != self.pp_size:
                raise ValueError(
                    f"pp_split has {len(self.pp_split)} entries but pipeline_parallel_size={self.pp_size}: "
                    f"one decoder-layer count per stage is required."
                )
            if min(self.pp_split) < 1:
                raise ValueError(f"pp_split entries must be >= 1 decoder layer, got {self.pp_split}.")

        if self.stage_world_size % self.nvlink_domain_size != 0:
            raise ValueError(
                f"Each pipeline stage owns {self.stage_world_size} ranks (world_size "
                f"{self.world_size} / pp_size {self.pp_size}), which is not a multiple of the NVLink "
                f"domain ({self.nvlink_domain_size}). Stage boundaries must fall on NVLink-domain "
                f"boundaries so EP/TP/ETP/CP groups stay inside one stage and only PP's "
                f"point-to-point activations cross RDMA. Choose pp_size dividing "
                f"{self.world_size // self.nvlink_domain_size} (the domain count), or set "
                f"NVLINK_DOMAIN_SIZE to the real NVLink partition size."
            )

        if self.stage_world_size == 1:
            # setup_fsdp2_for_dp skips wrapping at dp <= 1, so the runtime FSDP contract assert would
            # report a missing wrap instead of the stage width that caused it.
            raise ValueError(
                f"pp_size={self.pp_size} leaves each pipeline stage with a single rank "
                f"(world_size={self.world_size}): unsharded one-rank stages are not supported yet "
                f"(the stage FSDP wrap and its grad-reduction contract assume dp >= 2 per stage). "
                f"Use a smaller pp_size, or add data-parallel width per stage."
            )

        if self.needs_ep_wrappers and self.ep_group_size == 1 and not self.experts_fsdp_managed:
            raise ValueError(
                f"PP with fsdp_shard_ep1_experts=False (pp_size={self.pp_size}, ep_size=1) is not "
                f"supported: a MoE stage would then hold plain replicated expert tensors while a "
                f"fully-dense stage of the same pipeline runs the plain FSDP clip — the two stages "
                f"would issue different collective programs and deadlock at the gradient norm. Keep "
                f"fsdp_shard_ep1_experts=True (the default) so experts are stage-mesh DTensors."
            )

        if self.use_hsdp:
            # The shape is coherent whenever a stage holds >1 domain (its domains hold the same
            # layers, so replicating across them is ordinary HSDP); the mesh is what is missing.
            raise ValueError(
                f"PP + HSDP is not supported (pp_size={self.pp_size}, stage_world_size="
                f"{self.stage_world_size}, {self.num_nvlink_domains} NVLink domain(s) per stage). "
                f"HSDP's 2-D (dp_replicate, dp_shard) mesh is built by init_device_mesh over the whole "
                f"world, and cannot be restricted to a stage's rank block (create_dp_mesh takes a "
                f"process group on the 1-D path only) — a stage-sized 2-D mesh would silently be made "
                f"of the FIRST stage's ranks on every rank. Enabling it needs the pipeline as an outer "
                f"mesh dimension: init_device_mesh((pp, dp_replicate, dp_shard))[dp_replicate, "
                f"dp_shard]. At one domain per stage HSDP is a no-op anyway (dp_replicate_size counts "
                f"a stage's domains). Use plain FSDP inside each stage (use_hsdp=False)."
            )

        if (self.fp32_grad_reduce or self.fp32_non_ep_params) and is_global_main_process():
            # torch's schedule disables FSDP gradient sync for the whole microbatch loop
            # (set_requires_gradient_sync(False)) and reduce-scatters once per optimizer step, so FSDP2
            # accumulates into an unsharded grad buffer at reduce_dtype — 4 B/param in fp32, double the
            # bf16 accumulator and not shrinking with DP width. Without PP the knob changes only the
            # reduction.
            logger.warning(
                "fp32 gradient reduction under pipeline parallelism doubles the PERSISTENT gradient "
                "buffer: the schedule defers FSDP's reduce-scatter to the optimizer step, so each "
                "stage holds a full unsharded fp32 grad (4 B/param) for the whole microbatch loop "
                "instead of a bf16 one. Budget ~2xP_stage (params) + 4xP_stage (grads) per rank, or "
                "leave fp32_grad_reduce/fp32_non_ep_params off under PP."
            )

        if self.fsdp_reshard_after_forward:
            # The two settings agree numerically; the composition is untested at the trainer level
            # (clip / checkpoint / EP) and pays an all-gather the schedule does not need.
            raise ValueError(
                f"PP + fsdp_reshard_after_forward=True (FULL_SHARD / ZeRO-3) is not enabled "
                f"(pp_size={self.pp_size}): torch's pipeline schedule already pins each stage "
                f"unsharded across the backward (set_reshard_after_backward(False)) and drives FSDP's "
                f"post-backward itself, so resharding after every FORWARD only re-fetches parameters "
                f"the next microbatch immediately needs — and no equivalence gate covers the "
                f"composition through the trainer. Use fsdp_reshard_after_forward=False "
                f"(SHARD_GRAD_OP / ZeRO-2), with activation checkpointing for the memory."
            )

        if self.lowp_precision != "bf16":
            raise ValueError(
                f"PP + lowp_precision={self.lowp_precision!r} is not supported (pp_size={self.pp_size}). "
                f"No equivalence gate has ever compared PP low-precision gradients against an unsplit "
                f"reference, and the composition has a known silent failure: "
                f"apply_mixed_precision_compute derives each block's index from its MODULE NAME and "
                f"keeps lowp_keep_first_blocks / lowp_keep_last_blocks at the ends of that numbering, "
                f"but a stage's layers are re-based to 0 — so every stage would keep its OWN first and "
                f"last blocks in bf16 while the network's true ends get low precision, with nothing "
                f"raised. bf16 is the production default; train the low-precision recipe without PP."
            )

    def _validate_ep_group(self):
        """The full EP process group (ep_group_size = ep_size * expert_tp_size) must fit within and
        divide its scope — the NVLink domain for ep_scope='node', the world for ep_scope='global'."""
        if self.ep_group_size > 1:
            if self.ep_scope == "node":
                reject_node_local_ep_group(
                    self.ep_group_size,
                    self.nvlink_domain_size,
                    ep_size=self.ep_size,
                    expert_tp_size=self.expert_tp_size,
                    remedy="Use ep_scope='global' for EP spanning NVLink domains (RDMA).",
                )
            else:  # global scope
                # "global" EP spans one pipeline stage, so the bound is stage_world_size (equal to
                # world_size at pp_size == 1). The message names the unit, or an ep16+pp2 job reads
                # "cannot exceed world size (8)" on 16 GPUs.
                scope_desc = "world size" if self.pp_size == 1 else f"stage world size (world / pp{self.pp_size})"
                reject_cross_node_ep_group(
                    self.ep_group_size,
                    self.stage_world_size,
                    scope_desc=scope_desc,
                    ep_size=self.ep_size,
                    expert_tp_size=self.expert_tp_size,
                )
                # Cross-node EP groups must tile the world as equal contiguous per-domain blocks, else
                # DeepEP's intranode IPC peers are strided; cross_node_layout raises if they cannot.
                cross_node_layout(self.stage_world_size, self.ep_group_size, self.nvlink_domain_size)

    def _validate_cp_locality(self):
        """CP must be NVLink-local (Ulysses all-to-all is bandwidth-heavy) and divide the domain."""
        if self.cp_size > 1:
            if self.cp_size > self.nvlink_domain_size:
                raise ValueError(
                    f"CP size ({self.cp_size}) cannot exceed the NVLink domain ({self.nvlink_domain_size}). "
                    f"CP must stay on NVLink for efficient Ulysses attention."
                )
            if self.nvlink_domain_size % self.cp_size != 0:
                raise ValueError(f"CP size ({self.cp_size}) must divide the NVLink domain ({self.nvlink_domain_size})")

    def _validate_tp(self):
        """TP must divide world; under EP+TP, ep_size must be a multiple of tp_size so each EP group
        spans whole TP groups (else double-counted grads and misrouted tokens)."""
        if self.tp_size > 1:
            if self.tp_size > self.stage_world_size:
                raise ValueError(f"TP size ({self.tp_size}) cannot exceed world size ({self.stage_world_size})")
            if self.stage_world_size % self.tp_size != 0:
                raise ValueError(f"TP size ({self.tp_size}) must divide world size ({self.stage_world_size})")
            if self.ep_size > 1 and self.ep_size % self.tp_size != 0:
                raise ValueError(
                    f"EP+TP requires ep_size ({self.ep_size}) to be a multiple of tp_size "
                    f"({self.tp_size}) so each EP group spans whole TP groups. Use ep_size==tp_size "
                    f"(node-local EP+TP) or k*tp_size (cross-node EP)."
                )
            # TP groups are contiguous rank blocks and must stay inside one NVLink domain.
            if self.nvlink_domain_size % self.tp_size != 0:
                raise ValueError(
                    f"tp_size ({self.tp_size}) must divide the NVLink domain "
                    f"({self.nvlink_domain_size}): TP groups are contiguous rank blocks, so a "
                    f"non-dividing tp_size makes some TP groups straddle a domain boundary and "
                    f"every attention all-reduce crosses RDMA. Use tp_size that divides "
                    f"{self.nvlink_domain_size}, with DP across domains."
                )
            # Multi-domain multi-group EP+TP: the deferred-DP sweep assumes FSDP shards over the EP
            # group, but EP+TP shards over the (dp, tp) mesh — the average would mix dp shards.
            if self._is_multi_domain_multi_group_ep:
                raise ValueError(
                    f"Multi-domain multi-group EP+TP is not supported: ep_size={self.ep_size} with "
                    f"tp_size={self.tp_size} on {self.stage_world_size} ranks forms "
                    f"{self.stage_world_size // self.ep_group_size} EP groups across "
                    f"{self.num_nvlink_domains} NVLink domains, and the cross-replica gradient "
                    "average is incompatible with the EP+TP (dp, tp) FSDP mesh. Use a SINGLE EP "
                    f"group (ep_size={self.stage_world_size} with ep_scope='global'), drop TP (node-local "
                    "EP uses the deferred cross-replica sync), or use EP+ETP."
                )

    def _validate_expert_tp(self):
        """Expert-TP topology (MoE-only, independent of tp_size): NVLink-local, and node-local ETP
        groups even under cross-node EP. Which axes may accompany it is the capability matrix's rule."""
        if self.expert_tp_size > 1:
            # EPConfig enforces the same contract at group construction, i.e. only after the whole
            # checkpoint has downloaded.
            if self.expert_lora is not None:
                reject_expert_lora_with_expert_tp()
            if self.nvlink_domain_size % self.expert_tp_size != 0:
                raise ValueError(
                    f"expert_tp_size ({self.expert_tp_size}) must divide the NVLink domain ({self.nvlink_domain_size}). "
                    f"Expert TP groups must stay on NVLink for efficient all-reduce."
                )
            # Multi-domain multi-group EP+ETP: expert-TP keeps is_deferred_dp off, so the non-expert
            # FSDP2 reduce-scatter stays DP-wide while the combine spans one narrower dispatch group.
            if self._is_multi_domain_multi_group_ep:
                raise ValueError(
                    f"EP+Expert-TP with multiple dispatch groups across NVLink domains is not "
                    f"supported (ep_group_size={self.ep_group_size}, world={self.stage_world_size}): "
                    f"expert-TP keeps the deferred cross-replica DP path off, so FSDP2's DP-wide "
                    f"reduce-scatter would race the narrower DeepEP combine across domains. Use a "
                    f"single EP group (ep_size*expert_tp_size == world_size) or drop expert_tp_size to 1."
                )
            # Global-scope EP+ETP must form one ETP group per NVLink domain, keeping the ETP
            # all-reduce on NVLink. Not gated on multi-domain: on one domain the rule degenerates to
            # pure ETP and EPConfig raises there too, only after the whole model has loaded.
            if self.ep_scope == "global":
                _, members_per_domain, num_domains = cross_node_layout(
                    self.stage_world_size, self.ep_group_size, self.nvlink_domain_size
                )
                reject_cross_node_etp_shape(
                    self.expert_tp_size,
                    members_per_domain,
                    self.ep_size,
                    num_domains,
                    remedy=(
                        f"Set expert_tp_size={members_per_domain} with ep_size={num_domains}, or use "
                        f"NVLink-local EP (ep_scope='node')."
                    ),
                )

    def _validate_ep_cp(self):
        """EP+CP requires node-local EP: the all-to-all dispatch must stay orthogonal to CP's sequence
        partition; cross-domain global EP mismatches EP/CP membership (NCCL hang / corrupt routing)."""
        if self.ep_group_size > 1 and self.cp_size > 1:
            if self.ep_scope == "global":
                raise ValueError(
                    f"EP+CP requires node-local EP (ep_scope='node' with ep_group_size == nvlink_domain_size); "
                    f"cross-NVLink-domain EP (ep_scope='global') is incompatible with CP. "
                    f"Got ep_group_size={self.ep_group_size}, nvlink_domain_size={self.nvlink_domain_size}, "
                    f"cp_size={self.cp_size}."
                )
            if self.ep_scope == "node" and self.ep_group_size != self.nvlink_domain_size:
                raise ValueError(
                    f"Node-local EP+CP requires ep_group_size=nvlink_domain_size.\n"
                    f"Got ep_group_size={self.ep_group_size} (ep_size={self.ep_size} * expert_tp_size={self.expert_tp_size}), "
                    f"nvlink_domain_size={self.nvlink_domain_size}"
                )

    @property
    def is_racy_single_domain_multigroup_ep(self) -> bool:
        """Multiple >2-rank DeepEP dispatch groups whose combine barrier races an FSDP2 collective of
        different membership. Shared by the config-time validator and the trainer-side guard.

        Both backends fail, with different symptoms: ``legacy`` (V1 ``Buffer``) deadlocks around step
        2; the ``elastic`` default (V2 over NCCL Gin) faults with ``CUDA error: Invalid access of peer
        GPU memory over nvlink``. ``CUDA_DEVICE_MAX_CONNECTIONS=1`` does not cover it.

        ``num_nvlink_domains == 1`` stands for "FSDP2 does not share the EP group's membership":
        :attr:`EPConfig.is_deferred_dp` engages only above one domain, and it is what makes
        ``_apply_ep_aware_dp_fsdp2`` shard the non-expert params over the EP group instead of the whole
        DP world. On one domain the reduce-scatter spans every rank while the combine spans a subset.

        Keyed on ``ep_group_size``, not ``ep_size``, so EP+ETP passes: ``_create_expert_tp_groups``
        splits an ``ep_group_size``-wide group into ``expert_tp_size`` dispatch groups of ``ep_size``,
        and ep4+etp2 on one 8-GPU domain runs clean where bare ep4 faults.
        """
        return self.num_nvlink_domains == 1 and self.ep_size > 2 and self.nvlink_domain_size > self.ep_group_size

    @property
    def racy_ep_topology_message(self) -> str:
        """Rejection message for the racy-EP topology, shared by the config gate and trainer guard."""
        return (
            f"ep_size={self.ep_size} on a single {self.nvlink_domain_size}-GPU NVLink domain forms "
            f"{self.nvlink_domain_size // self.ep_size} concurrent >2-rank DeepEP dispatch "
            f"groups (ep_group_size={self.ep_group_size}), whose combine barriers race FSDP2's "
            f"DP-wide collectives. Measured on an 8-GPU node: the legacy buffer deadlocks, the elastic "
            f"default faults with 'Invalid access of peer GPU memory over nvlink' — both with and "
            f"without gradient checkpointing. Use a SINGLE dispatch group per domain: ep_size=2, or "
            f"raise ep_size*expert_tp_size to the domain ({self.nvlink_domain_size}) with ETP, or "
            f"shrink the job to {self.ep_group_size} GPUs. Sizing ep_size itself to the domain works "
            f"only where the model has that many experts per rank to give (rarely on a "
            f"{self.nvlink_domain_size}-wide rack); attention TP leaves ep_group_size unchanged, so "
            f"EP+TP lands back on this same rejection."
        )

    def _validate_ep_buffer_backend(self):
        """Reject at config time a ``legacy`` (DeepEP V1) backend this topology cannot drive.

        The rules live with the V1 limits they read
        (:func:`~src.distributed.expert_parallel.config.reject_legacy_backend_topology`); the
        dispatcher applies the same function to a hand-built ``EPConfig`` inside
        ``EPMoELayerBase.__init__``, after the checkpoint download and the model load on every rank.
        """
        if self.ep_buffer_backend != "legacy" or self.ep_size <= 1:
            return
        # The dispatch group is what rides the buffer; under expert-TP it is narrower than the EP group.
        reject_legacy_backend_topology(
            self.ep_size,
            is_cross_node=self.num_nodes > 1 and not self.is_node_local_ep,
            gpus_per_node=self.gpus_per_node,
            scope=f"ep_scope={self.ep_scope!r} over {self.num_nodes} nodes",
        )

    def _validate_single_domain_multigroup_ep(self):
        """Reject single-domain multi-group EP with >2-rank dispatch groups, before any model loading.

        Covers pure EP and EP+TP alike (attention TP leaves ``ep_group_size`` untouched); EP+ETP is
        exempt — see :attr:`is_racy_single_domain_multigroup_ep`."""
        if self.is_racy_single_domain_multigroup_ep:
            raise ValueError(self.racy_ep_topology_message)

    def _validate_hsdp(self):
        """HSDP shards non-expert params within an NVLink domain, replicates across domains. Pure DP
        or CP only: TP / Expert-TP build their own (dp, tp) mesh, and EP's grad sync must share the
        combine's group membership — both are rejected below."""
        if not self.use_hsdp:
            return
        if self.tp_size > 1 or self.expert_tp_size > 1:
            raise ValueError(
                f"use_hsdp=True is not supported with TP (tp_size={self.tp_size}) or Expert-TP "
                f"(expert_tp_size={self.expert_tp_size}); those modes build their own (dp, tp) device "
                f"mesh. Use HSDP on the standard DP path (pure DP or CP)."
            )
        if self.ep_size > 1:
            # EP already shards over the EP group; HSDP would be a no-op or race the combine.
            raise ValueError(
                f"use_hsdp=True is not supported with EP (ep_size={self.ep_size}). Multi-group EP "
                f"already shards over the EP group (deferred cross-replica sync); single-group EP "
                f"must use 1D FSDP so backward collectives share the combine's membership."
            )
        if self.num_nvlink_domains <= 1 and is_global_main_process():
            logger.warning(
                "use_hsdp=True but there is only one NVLink domain (single-node job) — nothing to "
                "replicate across, so FSDP full-shards over the node exactly as 1D. HSDP engages "
                "automatically once the job spans multiple domains."
            )

    def _validate_fsdp_settings(self):
        """fsdp_reshard_after_forward (ZeRO-3) re-gathers params in backward. Allowed only where a plain
        all-gather works (pure DP, CP, ep_size==1 MoE); rejected under EP (races the combine) and TP+DP
        (all-gather on TP-sharded DTensor params has no sharding strategy)."""
        if not self.fsdp_reshard_after_backward and self.fsdp_reshard_after_forward:
            raise ValueError(
                "fsdp_reshard_after_backward=False contradicts fsdp_reshard_after_forward=True: "
                "FULL_SHARD exists to reshard for memory, keeping params unsharded across the "
                "grad-accum window defeats it. Use SHARD_GRAD_OP (fsdp_reshard_after_forward=False) "
                "with the backward reshard off."
            )
        if not self.fsdp_reshard_after_backward and (self.tp_size > 1 or self.pp_size > 1):
            raise ValueError(
                f"fsdp_reshard_after_backward=False is only wired through the plain-DP/CP/EP torchrun "
                f"path (tp_size={self.tp_size}, pp_size={self.pp_size}): the TP setup shards through "
                f"its own fully_shard calls and PP already pins params unsharded per stage. Remove "
                f"the flag for those modes."
            )
        if not self.fsdp_shard_ep1_experts and (self.tp_size > 1 or self.cp_size > 1):
            # The TP and CP setup paths FSDP-shard ep1 experts unconditionally (their fully_shard
            # replaces the params), so the flag would have no effect.
            raise ValueError(
                f"fsdp_shard_ep1_experts=False is not honored under TP or CP "
                f"(tp_size={self.tp_size}, cp_size={self.cp_size}): those paths FSDP-shard the "
                f"replicated experts unconditionally. Remove the flag (sharded experts are "
                f"grad-equivalent and throughput-neutral), or use pure DP for the full replicated "
                f"expert copy."
            )
        if not self.fsdp_reshard_after_forward:
            return
        if self.is_ep_mode:
            raise ValueError(
                f"fsdp_reshard_after_forward=True (FULL_SHARD / ZeRO-3) is not supported where an "
                f"expert-distribution group exists (ep_size={self.ep_size}, "
                f"expert_tp_size={self.expert_tp_size}, ep_group_size={self.ep_group_size}): its "
                f"backward-pass all-gather can race the DeepEP combine, and pure ETP shares that "
                f"path. Full-shard is supported where ep_group_size==1 (pure DP, CP, and ep_size==1 "
                f"MoE without expert TP). Otherwise reduce peak memory with activation "
                f"checkpointing instead, or set fsdp_reshard_after_forward=False "
                f"(SHARD_GRAD_OP / ZeRO-2)."
            )
        if self.is_tp_mode and self.data_parallel_size > 1:
            raise ValueError(
                f"fsdp_reshard_after_forward=True (FULL_SHARD / ZeRO-3) is not supported with Tensor "
                f"Parallelism + data parallelism (tp_size={self.tp_size}, data_parallel_size="
                f"{self.data_parallel_size}): FSDP2's backward re-gather issues a plain c10d all-gather on "
                f"the TP-sharded DTensor params, which has no registered DTensor sharding strategy "
                f"(NotImplementedError mid-step). Use fsdp_reshard_after_forward=False (SHARD_GRAD_OP / "
                f"ZeRO-2), which keeps params gathered between forward and backward, or use_hsdp for replica "
                f"memory savings."
            )

    def _validate_lowp(self):
        """Low-precision compute validation: guard enums/ranges and reject no-op/contradictory combos."""
        if self.lowp_precision not in LOWP_PRECISIONS:
            raise ValueError(
                f"lowp_precision must be one of {', '.join(repr(p) for p in LOWP_PRECISIONS)} "
                f"('bf16' = off), got {self.lowp_precision!r}"
            )
        if self.lowp_keep_first_blocks < 0 or self.lowp_keep_last_blocks < 0:
            raise ValueError(
                f"lowp_keep_first_blocks / lowp_keep_last_blocks must be >= 0, got "
                f"{self.lowp_keep_first_blocks} / {self.lowp_keep_last_blocks}"
            )
        if self.lowp_precision != "bf16":
            if not (self.lowp_apply_dense_mlp or self.lowp_apply_moe_experts):
                raise ValueError(
                    f"lowp_precision={self.lowp_precision!r} is set but both lowp_apply_dense_mlp "
                    f"and lowp_apply_moe_experts are False — low precision would apply to nothing. "
                    f"Enable at least one, or set lowp_precision='bf16'."
                )
        elif (
            self.lowp_keep_first_blocks
            or self.lowp_keep_last_blocks
            or not self.lowp_apply_dense_mlp
            or not self.lowp_apply_moe_experts
        ) and is_global_main_process():
            enabled = ", ".join(repr(p) for p in LOWP_PRECISIONS if p != "bf16")
            logger.warning(
                "lowp_* tuning options are set but lowp_precision='bf16' (low precision is OFF) — "
                f"keep_blocks/apply_* have no effect until lowp_precision is one of {enabled}."
            )

    def validate_against_model_config(self, model_config) -> None:
        """Reject expert-sharding shapes this model cannot take, before any weight is read.

        ``ParallelismConfig`` is otherwise model-blind, so the expert-divisibility rules are reached
        only once the shapes are in hand — ``EPConfig.finalize_expert_assignment`` after the EP process
        groups exist, and the expert-TP split during layer construction — minutes and several
        collectives into a job. The arithmetic needs nothing but ``config.json``.
        """
        if self.ep_size > 1:
            num_experts = get_first_router_field(model_config, ROUTER_EXPERT_COUNT_FIELDS)
            if num_experts and num_experts % self.ep_size != 0:
                raise ValueError(
                    f"expert_parallel_size={self.ep_size} does not divide this model's "
                    f"{num_experts} routed experts: DeepEP dispatch maps tokens to ranks by a uniform "
                    f"expert-to-rank division, so an uneven assignment would route tokens to the wrong "
                    f"experts or drop them. ep_sizes dividing {num_experts} that fit this job: "
                    f"{', '.join(str(e) for e in _divisors_up_to(num_experts, self.stage_world_size))} "
                    f"(the topology gates narrow this further — a single-domain job takes 2 or the "
                    f"domain width)."
                )
        if self.expert_tp_size > 1:
            width = resolve_expert_ffn_shard_width(model_config)
            if width and width % self.expert_tp_size != 0:
                raise ValueError(
                    f"expert_tensor_parallel_size={self.expert_tp_size} does not divide this model's "
                    f"expert FFN width ({width}): expert-TP shards that dimension, so the shards would "
                    f"be ragged. Use an expert_tp_size dividing {width}."
                )
        if (
            self.fp32_non_ep_params
            and self.experts_fsdp_managed
            and get_first_router_field(model_config, ROUTER_EXPERT_COUNT_FIELDS)
        ):
            # Config-time and rank-symmetric: under PP a hybrid stack can leave one stage with no MoE
            # layer, so the equivalent check on the trainer's per-rank module walk would raise on MoE
            # stages while the dense stages walked into the wrap and hung the job.
            remedy = (
                "drop fp32_non_ep_params (fsdp_shard_ep1_experts=false is refused under PP/TP/CP)"
                if self.pp_size > 1 or self.tp_size > 1 or self.cp_size > 1
                else "set fsdp_shard_ep1_experts: false (see agent-docs/models/bailing.md#ling-30) or drop fp32_non_ep_params"
            )
            raise ValueError(
                f"fp32_non_ep_params=True cannot combine with fsdp_shard_ep1_experts=True at "
                f"ep_size=1 on a MoE model: dense params go fp32 while the FSDP-managed experts "
                f"stay bf16 in the same shard group, and FSDP2 asserts 'uniform original parameter "
                f"dtype' at the first forward. Remedy: {remedy}."
            )
        # Dense TP goes through HF's tp_plan="auto", which checks no head count of its own, so
        # without this it fails on the first forward's reshape, after the full load.
        validate_tp_head_divisibility(text_config(model_config), self.tp_size)
        self._validate_ep_token_budget(model_config)

    def _validate_ep_token_budget(self, model_config) -> None:
        """Reject a per-rank token budget the first MoE dispatch would reject, before any weight is read.

        The dispatcher sizes its arena from the all-reduced max tokens/rank of the EP group, aligned by
        :func:`ep_dispatch_capacity`, and rejects a capacity past DeepEP's 32-bit wire index or past the
        validated cross-node Gin ceiling. That max is knowable here: every rank of a group carries the
        same declared per-device shape, so the group max is ``per_device_train_batch_size × max_length``.
        Left to the dispatcher, the same verdict lands only after the full load, and for a corpus whose
        early batches are short only at the step that first reaches ``max_length``.

        ``max_length: null`` resolves here against ``config.json`` (:func:`context_window_from_config`):
        the "use the model's own limit" spelling, and therefore the largest budget a run can declare.

        Keyed on ``ep_size``, not ``is_ep_mode``: pure ETP folds into the latter but never reaches the
        transport. The inter-node question is the NVLink domain's, not the OS node's
        (:attr:`requires_rdma`), so an NVL72 rack-wide group stays on MNNVL and is not Gin-bound.
        """
        if self.ep_size <= 1 or self.ep_rows_per_device <= 0:
            return
        text = text_config(model_config)
        hidden = getattr(text, "hidden_size", 0)
        num_topk = resolve_router_topk(text)
        if not hidden or not num_topk:
            return
        max_length = self.ep_declared_max_length or context_window_from_config(model_config)
        if not max_length:
            # max_length: null on a config that states no window — nothing to judge against.
            return
        # CP is the one axis that shrinks a rank's token count (it holds one sequence chunk). EP is
        # orthogonal to DP, TP replicates the batch, and a PP microbatch is a fraction of it — so
        # dividing by none of them keeps this an upper bound.
        budget = self.ep_rows_per_device * max_length
        tokens_per_rank = budget // self.cp_size
        capacity = ep_dispatch_capacity(tokens_per_rank)
        try:
            reject_oversized_dispatch(
                capacity,
                num_topk=num_topk,
                padded_hidden=padded_wire_hidden(hidden),
                is_inter_node=self.requires_rdma,
            )
        except ValueError as exc:
            cp_note = f" ({budget} / cp_size={self.cp_size})" if self.cp_size > 1 else ""
            length_note = "" if self.ep_declared_max_length else " (max_length: null → the model's context window)"
            raise ValueError(
                f"Lower max_length or per_device_train_batch_size — or, for the cross-node Gin ceiling "
                f"only, raise HALO_DEEPEP_GIN_MAX_TOKENS_PER_RANK. This run's declared token budget "
                f"cannot be dispatched: per_device_train_batch_size × max_length={max_length}"
                f"{length_note} gives {tokens_per_rank} tokens/rank"
                f"{cp_note}, sized to a {capacity}-token DeepEP arena on an EP group of "
                f"{self.ep_group_size} rank(s) (ep_scope={self.ep_scope}, nvlink_domain_size="
                f"{self.nvlink_domain_size}, "
                f"{'cross-domain — Gin/RDMA' if self.requires_rdma else 'within one NVLink domain'}). "
                f"{exc}"
            ) from exc

    @property
    def num_ep_groups(self) -> int:
        """Number of EP groups, computed per scope from the same layout ``EPConfig`` builds its groups
        from, so the count cannot drift from the groups that exist."""
        if self.ep_group_size <= 1:
            # Degenerate layout: one singleton group per rank, matching what
            # ``node_local_groups_per_domain(domain, 1) * domains`` gives ``EPConfig``. Own branch
            # because ``cross_node_layout`` cannot express a group that spans no domain evenly.
            return self.stage_world_size
        if self.ep_scope == "node":
            per_domain = node_local_groups_per_domain(self.nvlink_domain_size, self.ep_group_size)
            return per_domain * self.num_nvlink_domains
        return cross_node_layout(self.stage_world_size, self.ep_group_size, self.nvlink_domain_size)[0]

    @property
    def is_ep_mode(self) -> bool:
        """EP enabled (EP or Expert TP or both)."""
        return self.ep_group_size > 1

    @property
    def needs_ep_wrappers(self) -> bool:
        """Whether EP MoE layer wrappers apply. True for EP distribution or grouped-GEMM-only mode
        (the dispatcher gates the latter on the model being MoE via config_has_experts())."""
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
        """This rank holds the pipeline's first stage (the embedding; consumes input_ids)."""
        return self.pp_rank == 0

    @property
    def is_last_pp_stage(self) -> bool:
        """This rank holds the pipeline's last stage (the head; computes the loss)."""
        return self.pp_rank == self.pp_size - 1

    def get_pp_group_ranks(self) -> list[int]:
        """Global ranks of this rank's pipeline chain, in stage order.

        One chain = the ranks holding the same intra-stage position across every stage; they share a DP
        coordinate and therefore a batch. Members are ``stage_world_size`` apart, so on the intended
        placement each is on a different NVLink domain and only their P2P activations use RDMA."""
        return [self.stage_local_rank + s * self.stage_world_size for s in range(self.pp_size)]

    @property
    def is_ep_tp_mode(self) -> bool:
        return self.ep_group_size > 1 and self.tp_size > 1

    @property
    def is_ep_cp_mode(self) -> bool:
        return self.ep_group_size > 1 and self.cp_size > 1

    @property
    def is_hsdp(self) -> bool:
        """HSDP active (2D dp_replicate × dp_shard mesh); only meaningful across >1 NVLink domain."""
        return self.use_hsdp and self.num_nvlink_domains > 1

    @property
    def dp_shard_size(self) -> int:
        """FSDP shard-group width: one NVLink domain under HSDP, else the whole DP world."""
        return self.nvlink_domain_size if self.is_hsdp else self.stage_world_size

    @property
    def dp_replicate_size(self) -> int:
        """HSDP replica groups = one per NVLink domain (1 when HSDP off)."""
        return self.num_nvlink_domains if self.is_hsdp else 1

    @property
    def non_dp_replication_factor(self) -> int:
        """How many ranks see the same batch: ``world_size // data_parallel_size``.

        Equals ``pp_size * max(tp_size, cp_size, expert_tp_size)`` — the stage split times the
        non-DP divisor — since both divisions are validated exact and EP is orthogonal to DP. Token
        counters gathered across the whole world must divide by it or they multiply-count.
        """
        return self.world_size // self.data_parallel_size

    @property
    def is_node_local_ep(self) -> bool:
        return self.ep_scope == "node"

    @property
    def requires_rdma(self) -> bool:
        """True when the EP group spans >1 NVLink domain (keyed on NVLink domain, not OS node, so an
        NVL72 cross-OS-node group within one rack stays on MNNVL)."""
        return self.is_ep_mode and self.ep_scope == "global" and self.num_nvlink_domains > 1

    @property
    def mode_string(self) -> str | None:
        """Short parallelism-mode string (e.g. 'ep', 'tp', 'ep-tp')."""
        parts = []
        if self.is_pp_mode:
            parts.append("pp")
        if self.is_ep_mode:
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

    def _ep_rank_and_group(self) -> tuple[int, int]:
        """``(rank-in-EP-group, EP-group-index)`` for this rank, per the configured EP scope
        (node-local groups use NVLink-domain coordinates; cross-node uses the column-block layout).

        Coordinates are *stage-local*: EP groups never straddle a pipeline stage, so the group index
        counts groups within this rank's stage (identical numbering in every stage)."""
        if self.ep_scope == "node":
            return node_local_rank_and_group(self.stage_local_rank, self.nvlink_domain_size, self.ep_group_size)
        return cross_node_rank_and_group(
            self.stage_local_rank, self.stage_world_size, self.ep_group_size, self.nvlink_domain_size
        )

    def get_ep_rank(self) -> int:
        """Rank within the EP group (node-local groups use NVLink-domain coordinates)."""
        return self._ep_rank_and_group()[0]

    def get_ep_group_idx(self) -> int:
        """EP group index this rank belongs to."""
        return self._ep_rank_and_group()[1]

    def get_cp_rank(self) -> int:
        """Rank within the CP group."""
        return node_local_rank_and_group(self.stage_local_rank, self.nvlink_domain_size, self.cp_size)[0]

    def get_cp_group_idx(self) -> int:
        """CP group index (= DP rank for data loading)."""
        return node_local_rank_and_group(self.stage_local_rank, self.nvlink_domain_size, self.cp_size)[1]

    def get_data_parallel_rank(self) -> int:
        """Data parallel rank (which batch this rank processes). Ranks in the same TP/CP/expert_tp
        group share a batch; expert-TP partners key on dispatch_ep_rank, not floor-division.

        Stage-local: every rank of one pipeline chain (same stage_local_rank, different pp_rank)
        returns the same value, so the whole chain consumes the same batch — stage 0 reads input_ids,
        the last stage reads labels."""
        if self.expert_tp_size > 1:
            # ETP partners share dispatch_ep_rank (same formula EPConfig builds its groups from) →
            # identical DP batches → matching shapes in the ReduceFromExpertTP all_reduce.
            dispatch_ep_rank, _ = etp_dispatch_coords(
                self.get_ep_rank(), self.ep_size, self.expert_tp_size, self.is_node_local_ep
            )
            return dispatch_ep_rank * self.num_ep_groups + self.get_ep_group_idx()

        dp_divisor = max(self.tp_size, self.cp_size)
        if dp_divisor > 1:
            return self.stage_local_rank // dp_divisor
        return self.get_cp_group_idx()

    def _to_global_ranks(self, stage_local_ranks: list[int]) -> list[int]:
        """Lift stage-local rank numbers into global rank space (identity at ``pp_size == 1``)."""
        return [r + self.stage_base_rank for r in stage_local_ranks]

    def get_ep_group_ranks(self) -> list[int]:
        """Global ranks in this rank's EP group (node-local groups are contiguous within one domain).

        Also a test seam, with :meth:`get_cp_group_ranks`, :meth:`get_expert_replica_ranks` and
        :meth:`get_cp_rank`: the CPU tests check ``EPConfig``/``CPConfig`` membership against these.
        """
        if self.ep_scope == "node":
            local = node_local_group_ranks(self.get_ep_group_idx(), self.nvlink_domain_size, self.ep_group_size)
        else:
            local = cross_node_group_ranks(
                self.get_ep_group_idx(), self.stage_world_size, self.ep_group_size, self.nvlink_domain_size
            )
        return self._to_global_ranks(local)

    def get_cp_group_ranks(self) -> list[int]:
        """Global ranks in this rank's CP group."""
        return self._to_global_ranks(
            node_local_group_ranks(self.get_cp_group_idx(), self.nvlink_domain_size, self.cp_size)
        )

    def get_expert_replica_ranks(self) -> list[int]:
        """Ranks holding the same experts (for gradient sync across EP groups).

        Confined to this rank's pipeline stage: under PP, the same ep_rank in a *different* stage holds
        a different set of layers, so averaging across stages would corrupt gradients."""
        if self.ep_group_size <= 1:
            # Every rank is a singleton EP group holding the full expert set, so the stage's rank block
            # is one replica set — the single group ``EPConfig`` builds here. Not routed through the
            # layout helpers: the cross-node one cannot take a group of 1.
            return self._to_global_ranks(list(range(self.stage_world_size)))
        if self.num_ep_groups <= 1:
            return [self.global_rank]

        ep_rank = self.get_ep_rank()

        if self.ep_scope == "node":
            local = node_local_replica_ranks(
                ep_rank, self.stage_world_size, self.nvlink_domain_size, self.ep_group_size
            )
        else:
            local = cross_node_replica_ranks(
                ep_rank, self.stage_world_size, self.ep_group_size, self.nvlink_domain_size
            )
        return self._to_global_ranks(local)

    def create_ep_config(self) -> "EPConfig":
        """Create (and cache) the EPConfig that builds the EP process groups."""
        if self._ep_config is None:
            self._ep_config = EPConfig(
                ep_size=self.ep_size,
                ep_group_size=self.ep_group_size,
                # EP lives inside one pipeline stage: tile stage_world_size from this stage's base.
                world_size=self.stage_world_size,
                rank_offset=self.stage_base_rank,
                node_local=self.is_node_local_ep,
                # Locality unit = NVLink domain; EPConfig's "node" abstraction maps onto it.
                gpus_per_node=self.nvlink_domain_size,
                # The router is skipped by the fp32_non_ep_params upcast; bf16 there would trip
                # FSDP2's uniform-dtype check against the fp32 dense params.
                fp32_router=self.ep_fp32_router or self.fp32_non_ep_params,
                fp32_experts=self.ep_fp32_experts,
                expert_tp_size=self.expert_tp_size,
                use_grouped_gemm=self.use_grouped_gemm,
                fp32_grad_reduce=self.fp32_grad_reduce,
                expert_lora=self.expert_lora,
                fsdp_shard_ep1_experts=self.fsdp_shard_ep1_experts,
                ep_buffer_backend=self.ep_buffer_backend,
            )
        return self._ep_config

    def create_cp_config(self) -> "CPConfig":
        """Create (and cache) the CPConfig that builds the CP process groups."""
        if self._cp_config is None:
            self._cp_config = CPConfig(
                cp_size=self.cp_size,
                world_size=self.stage_world_size,
                rank_offset=self.stage_base_rank,
                gpus_per_node=self.nvlink_domain_size,
            )
        return self._cp_config

    def summary(self) -> str:
        """Summary string of the parallelism configuration."""
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
        if self.is_ep_mode:
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
        if not self.fp32_output_conversion:
            lines.append("  FP32 output conversion: disabled (bf16 native)")

        lines.append("=" * 60)
        return "\n".join(lines)


def accelerate_launch_rejection(pc: ParallelismConfig) -> str | None:
    """Why this config cannot run under ``accelerate launch``, or ``None`` when it can.

    ``accelerate launch`` performs the FSDP/DDP wrapping for every ``distributed_type``, so the custom
    EP/CP/TP/PP meshes need ``torchrun``. Shared by the trainer's ``_validate_parallelism_modes`` and
    the loader's ``_validate_launch_method_for_parallelism`` so both report the same recipe.
    """
    if not (pc.is_ep_mode or pc.is_cp_mode or pc.is_tp_mode or pc.is_pp_mode) or not is_accelerate_launch():
        return None

    return (
        "EP/CP/TP/PP modes are not compatible with 'accelerate launch' "
        "(any distributed_type: FSDP and MULTI_GPU/DDP alike).\n"
        "\n"
        "Detected accelerate launcher env: "
        f"ACCELERATE_MIXED_PRECISION={os.environ.get('ACCELERATE_MIXED_PRECISION')}, "
        f"ACCELERATE_USE_FSDP={os.environ.get('ACCELERATE_USE_FSDP')}\n"
        f"Active parallelism: EP={pc.ep_size}, CP={pc.cp_size}, TP={pc.tp_size}, PP={pc.pp_size}\n"
        "\n"
        "For EP/CP/TP/PP training, use 'torchrun' instead:\n"
        "\n"
        "  # Single node\n"
        "  torchrun --nproc_per_node=8 scripts/training/sft.py \\\n"
        "      --expert_parallel_size=8\n"
        "\n"
        "  # Multi-node\n"
        "  torchrun --nnodes=2 --node_rank=0 --nproc_per_node=8 \\\n"
        "      --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \\\n"
        "      scripts/training/sft.py \\\n"
        "      --expert_parallel_size=8\n"
        "\n"
        "For standard data parallelism (no EP/CP/TP/PP), use 'torchrun' too:\n"
        "\n"
        "  torchrun --nproc_per_node=<N> scripts/training/sft.py <config>\n"
        "\n"
        "See agent-docs/parallelism/multi-node.md and agent-docs/parallelism/data-parallelism.md for details."
    )
