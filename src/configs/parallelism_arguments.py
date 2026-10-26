"""The parallelism flags a script parses (``ParallelismArguments``) and the ``ParallelismConfig`` they build."""

from dataclasses import dataclass, field, fields
from typing import TYPE_CHECKING

from src.config_validation import ValidatedConfig, present, require_int
from src.distributed.expert_parallel.config import EPBufferBackend
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.parallelism_rules import AXES, EPScope, PPSchedule, reject_cross_domain_determinism
from src.kernels.lowp.spec import LowpPrecision, LowpSpec
from src.models.context_window import is_bounded_length

if TYPE_CHECKING:
    from src.models.moe.expert_lora_spec import ExpertLoraSpec


@dataclass
class ParallelismArguments(ValidatedConfig):
    """The YAML/CLI spelling of every ``ParallelismConfig`` knob, defaulting to the config's own value.

    ``ParallelismConfig`` validates ranges and combinations; this class refuses only a size that is not an int.
    """

    expert_parallel_size: int = field(
        default=ParallelismConfig.ep_size,
        metadata={
            "help": "Expert-parallel size: the ranks one DeepEP all-to-all spans, each holding num_experts / "
            "expert_parallel_size experts (> 1 enables EP on a MoE model). The EP group of expert_parallel_size × "
            "expert_tensor_parallel_size ranks must divide the NVLink domain when node-local, or world_size when "
            "cross-node. See agent-docs/parallelism/expert-parallelism.md#ep-grouping."
        },
    )
    ep_scope: EPScope = field(
        default=ParallelismConfig.ep_scope,
        metadata={
            "help": "Where EP groups sit: 'node' keeps each inside one NVLink domain, 'global' spans domains over "
            "RDMA, and 'auto' picks 'node' when the EP group (expert_parallel_size × expert_tensor_parallel_size) "
            "fits the domain. See agent-docs/parallelism/expert-parallelism.md#ep-grouping."
        },
    )
    merge_expert_lora_on_save: bool = field(
        default=ParallelismConfig.merge_expert_lora_on_save,
        metadata={
            "help": "Fold the native expert LoRA delta, and any attention adapters, into the base on save so each "
            "checkpoint is a merged, servable HF model; the frozen base itself never changes. Training checkpoints "
            "also keep the unmerged adapter in resume_adapter/ for resume, and false writes a standalone adapter "
            "file instead. See agent-docs/reference/checkpoints.md#merge-on-save-checkpoints."
        },
    )
    fp32_router: bool = field(
        default=ParallelismConfig.ep_fp32_router,
        metadata={
            "help": "Keep EP router weights as FP32 masters with an FP32 routing matmul, except at ep_group_size == 1 "
            "under fsdp_shard_ep1_experts, where FSDP2's BF16 param policy owns the compute dtype. "
            "fp32_non_ep_params implies it; Gemma 4 ignores it with a warning. "
            "See agent-docs/parallelism/expert-parallelism.md#precision-control."
        },
    )
    fp32_experts: bool = field(
        default=ParallelismConfig.ep_fp32_experts,
        metadata={
            "help": "Keep EP expert weights as FP32 masters; compute stays BF16 under autocast. Skipped with a "
            "warning at ep_group_size == 1 under fsdp_shard_ep1_experts, where FSDP2's BF16 param policy casts the "
            "replicated experts anyway: set fsdp_shard_ep1_experts: false for FP32 experts there. "
            "See agent-docs/parallelism/expert-parallelism.md#precision-control."
        },
    )
    fp32_non_ep_params: bool = field(
        default=ParallelismConfig.fp32_non_ep_params,
        metadata={
            "help": "Keep non-expert (dense) parameters as FP32 masters for stable optimizer updates; compute stays "
            "BF16 under autocast. See agent-docs/reference/configuration-reference.md#parallelismconfig."
        },
    )
    fp32_grad_reduce: bool = field(
        default=ParallelismConfig.fp32_grad_reduce,
        metadata={
            "help": "Reduce gradients across ranks in FP32 while parameters stay BF16 (the FSDP2 reduce dtype and "
            "the EP, TP and QLoRA reductions), at about twice the bandwidth on that collective. Independent of "
            "fp32_non_ep_params, which implies it for the FSDP2 reduce dtype only. "
            "See agent-docs/optimization/bf16-optimizer.md#master-weight-and-grad-reduce-options."
        },
    )
    bf16_optimizer: bool | None = field(
        default=ParallelismConfig.bf16_optimizer,
        metadata={
            "help": "Use AdamWBF16 (BF16 master weights with stochastic rounding, 6 B/param); unset = auto, on under "
            "bf16 with the default AdamW and off under replicated DDP. False builds the stock AdamW over the "
            "parameters as loaded, BF16 under bf16. See agent-docs/optimization/bf16-optimizer.md#usage."
        },
    )
    use_grouped_gemm: bool = field(
        default=ParallelismConfig.use_grouped_gemm,
        metadata={
            "help": "Run MoE expert matmuls through F.grouped_mm on SM90+ hardware, wrapping the MoE layers even at "
            "expert_parallel_size 1, so an MoE model under accelerate launch is rejected at load: launch with "
            "torchrun or set this false. "
            "See agent-docs/optimization/grouped-gemm.md#standalone-grouped-gemm-mode."
        },
    )
    lowp_precision: LowpPrecision = field(
        default=LowpSpec.precision,
        metadata={
            "help": "Low-precision GEMM operands for dense MLP and MoE expert matmuls over BF16/FP32 masters, "
            "parameters and checkpoints unchanged: 'bf16' (off), 'fp8' (mxfp8), 'fp4' (nvfp4) or 'mxfp4'. SFT only, "
            "and every format runs slower than bf16. "
            "See agent-docs/optimization/low-precision-moe-kernels.md#usage."
        },
    )
    lowp_apply_dense_mlp: bool = field(
        default=LowpSpec.apply_dense_mlp,
        metadata={
            "help": "Apply lowp_precision to dense MLP projections (gate/up/down). Set False to keep "
            "dense MLPs in bf16 (e.g. low precision on MoE experts only)."
        },
    )
    lowp_apply_moe_experts: bool = field(
        default=LowpSpec.apply_moe_experts,
        metadata={
            "help": "Apply lowp_precision to MoE expert grouped GEMMs; false keeps the experts in bf16. DeepGEMM's "
            "native kernel is opt-in through HALO_DEEPGEMM_NATIVE=1 and never auto-routed. "
            "See agent-docs/optimization/low-precision-moe-kernels.md#deepgemm-the-native-fp8fp4-path."
        },
    )
    lowp_keep_first_blocks: int = field(
        default=LowpSpec.keep_first_blocks,
        metadata={
            "help": "Leading transformer blocks kept in bf16 under lowp_precision (the NVFP4 recipe: the first "
            "blocks are the most precision-sensitive)."
        },
    )
    lowp_keep_last_blocks: int = field(
        default=LowpSpec.keep_last_blocks,
        metadata={
            "help": "Trailing transformer blocks kept in bf16 under lowp_precision (the NVFP4 recipe keeps the last "
            "~8)."
        },
    )
    max_concurrent_loading: int | None = field(
        default=ParallelismConfig.max_concurrent_loading,
        metadata={
            "help": "Most ranks per node loading the model at once; unset derives half the node width, capped at 4. "
            "An explicit value is used verbatim: 1 loads sequentially on CPU-RAM-constrained hosts, 0 all at once. "
            "See agent-docs/parallelism/multi-node.md#model-loading."
        },
    )
    ep_lazy_loading: bool = field(
        default=ParallelismConfig.ep_lazy_loading,
        metadata={
            "help": "Load an EP model by lazy safetensors slicing, each rank reading only its own expert slice "
            "(meta init, low CPU RAM); false loads with from_pretrained and patches EP afterwards. "
            "See agent-docs/parallelism/expert-parallelism.md#model-loading."
        },
    )
    ep_buffer_backend: EPBufferBackend = field(
        default=ParallelismConfig.ep_buffer_backend,
        metadata={
            "help": "DeepEP all-to-all transport: 'auto'/'elastic' is the ElasticBuffer over NCCL Gin (cross-node "
            "capable), 'legacy' the CUDA-IPC Buffer (numerically identical, intranode only). Pick 'legacy' for "
            "long-context ep8, where elastic at about 64k tokens/rank or more deadlocks its combine against FSDP2. "
            "See agent-docs/infrastructure/deepep.md#transport-backend."
        },
    )
    context_parallel_size: int = field(
        default=ParallelismConfig.cp_size,
        metadata={
            "help": "Context-parallel (Ulysses) size: each sequence is split across this many ranks of one NVLink "
            "domain. Not compatible with tensor parallelism or the padding-free collator. "
            "See agent-docs/parallelism/context-parallelism.md#requirements."
        },
    )
    tensor_parallel_size: int = field(
        default=ParallelismConfig.tp_size,
        metadata={
            "help": "Tensor-parallel size for attention, embeddings and lm_head (> 1 enables TP); it must divide "
            "world_size, and FSDP2 data-parallelizes across the TP groups. Not compatible with context parallelism."
        },
    )
    expert_tensor_parallel_size: int = field(
        default=ParallelismConfig.expert_tp_size,
        metadata={
            "help": "Expert-FFN tensor-parallel size: shards each expert's gate_up_proj and down_proj across this "
            "many ranks of one NVLink domain, cutting per-rank expert memory by the same factor. Combines with "
            "expert_parallel_size (the EP group is their product), not with tensor or context parallelism. "
            "See agent-docs/parallelism/expert-tensor-parallelism.md#validation-rules."
        },
    )
    pipeline_parallel_size: int = field(
        default=ParallelismConfig.pp_size,
        metadata={
            "help": "Pipeline-parallel size: splits the decoder layers into contiguous stages, each a whole number "
            "of NVLink domains, and divides data_parallel_size by it. Needs fixed input shapes and a trainer that "
            "supports PP; every other script rejects a value > 1. "
            "See agent-docs/parallelism/pipeline-parallelism.md#launching."
        },
    )
    pipeline_microbatches: int = field(
        default=ParallelismConfig.pp_microbatches,
        metadata={
            "help": "Microbatches per optimizer step under pipeline parallelism, the pipeline's own gradient "
            "accumulation; 0 = gradient_accumulation_steps, raised to pipeline_parallel_size where 1f1b needs it. "
            "per_device_train_batch_size must be divisible by it, and a non-default value is rejected without PP."
        },
    )
    pipeline_schedule: PPSchedule = field(
        default=ParallelismConfig.pp_schedule,
        metadata={
            "help": "Pipeline schedule: '1f1b' (lowest activation memory, needs pipeline_microbatches >= "
            "pipeline_parallel_size) or 'gpipe' (all forwards then all backwards, no microbatch-count constraint). "
            "A non-default value is rejected without PP."
        },
    )
    pipeline_split: list[int] | None = field(
        default=ParallelismConfig.pp_split,
        metadata={
            "help": "Per-stage decoder-layer counts under pipeline parallelism, like [19, 17] for 36 layers at two "
            "stages, summing to the layer count with boundaries on the layer_types period. Unset = the "
            "head-weighted split, which shrinks the last stage by the lm_head's layer-equivalent cost; a value is "
            "rejected without PP."
        },
    )
    nvlink_domain_size: int = field(
        default=ParallelismConfig.nvlink_domain_size,
        metadata={
            "help": "GPUs one NVLink domain spans, the locality unit of node-local EP/CP/TP/ETP groups; 0 reads "
            "NVLINK_DOMAIN_SIZE, else gpus_per_node. Set the rack's partition size (72 on GB200/GB300 NVL72) to "
            "group NVLink-wide across OS nodes. "
            "See agent-docs/parallelism/multi-node.md#gb200gb300-nvl72-multi-node-nvlink."
        },
    )
    fsdp_reshard_after_forward: bool = field(
        default=ParallelismConfig.fsdp_reshard_after_forward,
        metadata={
            "help": "False (ZeRO-2) keeps FSDP2 params gathered from forward to backward, faster at a higher peak "
            "memory; true (ZeRO-3) reshards them after forward. Torchrun only, and rejected with an "
            "expert-distribution group (ep_group_size > 1), TP at data_parallel_size > 1, or PP. "
            "See agent-docs/parallelism/data-parallelism.md#zero-2-vs-zero-3-reshard_after_forward."
        },
    )
    fsdp_reshard_after_backward: bool = field(
        default=ParallelismConfig.fsdp_reshard_after_backward,
        metadata={
            "help": "False keeps FSDP2 params unsharded across a gradient-accumulation window's microsteps, one "
            "re-all-gather per optimizer step instead of one per microstep, at unchanged peak memory. Torchrun "
            "DP/CP/EP only; rejected with fsdp_reshard_after_forward, TP or PP. "
            "See agent-docs/parallelism/data-parallelism.md#zero-2-vs-zero-3-reshard_after_forward."
        },
    )
    fsdp_defer_grad_sync: bool = field(
        default=ParallelismConfig.fsdp_defer_grad_sync,
        metadata={
            "help": "Skip FSDP2's gradient reduce-scatter on a gradient-accumulation window's microsteps 1..n-1, "
            "reducing once per optimizer step at the cost of one unsharded gradient copy per GPU held across the "
            "window. Torchrun FSDP2 only; rejected under PP, TP at data_parallel_size 1, QLoRA and "
            "fsdp_reshard_after_forward. "
            "See agent-docs/parallelism/data-parallelism.md#deferred-gradient-reduce-fsdp_defer_grad_sync."
        },
    )
    fsdp_shard_ep1_experts: bool = field(
        default=ParallelismConfig.fsdp_shard_ep1_experts,
        metadata={
            "help": "At ep_group_size == 1, where the experts are truly replicated, FSDP-shard them so their "
            "reduce-scatter is the sole gradient sync, freeing memory that grows with DP at unchanged gradients. "
            "False keeps a full expert copy per rank; no effect at ep_group_size > 1, pure ETP included. "
            "See agent-docs/parallelism/data-parallelism.md#ep1-expert-sharding."
        },
    )
    use_hsdp: bool = field(
        default=ParallelismConfig.use_hsdp,
        metadata={
            "help": "Hybrid sharded data parallel on the pure DP/CP path: shard within each NVLink domain and "
            "replicate across domains, so only one gradient all-reduce crosses RDMA per step. Trades a per-domain "
            "parameter replica for inter-node bandwidth; rejected with EP, TP or Expert-TP, and a no-op on one "
            "domain. See agent-docs/parallelism/data-parallelism.md#hsdp-hybrid-sharded-data-parallel."
        },
    )
    init_from_scratch: bool = field(
        default=False,
        metadata={
            "help": "Build the model from model_name_or_path's config with freshly initialized random weights, "
            "reusing its tokenizer, instead of loading the pretrained weights. Dense data-parallel/FSDP2 only; pair "
            "it with max_steps and a warmup-plus-decay schedule. "
            "See agent-docs/training-methods/pretraining.md#random-initialization."
        },
    )
    fp32_output_conversion: bool = field(
        default=ParallelismConfig.fp32_output_conversion,
        metadata={
            "help": "Keep accelerate's wrapper that converts model outputs (the logits) to fp32. Off by default: "
            "the upcast can OOM on long sequences, and a bf16 model needs no fp32 outputs."
        },
    )

    def _validate_fields(self) -> None:
        super()._validate_fields()
        # ParallelismConfig's range checks read a bool as 0 or 1 and pass a fraction into the rank math.
        require_int(
            type(self).__name__,
            **present(
                expert_parallel_size=self.expert_parallel_size,
                expert_tensor_parallel_size=self.expert_tensor_parallel_size,
                context_parallel_size=self.context_parallel_size,
                tensor_parallel_size=self.tensor_parallel_size,
                pipeline_parallel_size=self.pipeline_parallel_size,
                pipeline_microbatches=self.pipeline_microbatches,
                nvlink_domain_size=self.nvlink_domain_size,
                lowp_keep_first_blocks=self.lowp_keep_first_blocks,
                lowp_keep_last_blocks=self.lowp_keep_last_blocks,
                max_concurrent_loading=self.max_concurrent_loading,
            ),
        )


def same_name_arg_forwards(dist_args) -> tuple[str, ...]:
    """``ParallelismConfig`` fields ``dist_args`` declares under the same name, in field order.

    The by-name half of :func:`parallelism_config_from_args`' wiring: derived from the two
    declarations rather than listed, so a knob added to both is forwarded automatically. Fields the
    builder renames (``expert_parallel_size`` → ``ep_size``) or derives do not appear here.
    """
    return tuple(f.name for f in fields(ParallelismConfig) if f.init and hasattr(dist_args, f.name))


def parallelism_config_from_args(
    dist_args,
    *,
    trainer_cls,
    training_config=None,
    supports_init_from_scratch: bool = False,
    expert_lora: "ExpertLoraSpec | None" = None,
) -> ParallelismConfig:
    """The :class:`ParallelismConfig` parsed ``ParallelismArguments`` describe, for ``trainer_cls``'s script.

    Refuses here, before any model loads, what the trainer would refuse only after: a CP/PP size its
    ``SUPPORTS_CP`` / ``PP_REFUSAL`` gate rejects, ``lowp_*`` knobs without ``SUPPORTS_LOWP_PRECISION`` (SFT only),
    ``init_from_scratch`` without ``supports_init_from_scratch``, and ``training_config.full_determinism`` on an
    EP group spanning NVLink domains. ``training_config`` also declares the per-rank token budget the DeepEP
    dispatch-ceiling gate judges (no ``max_length`` field leaves it off). ``expert_lora`` must be passed here, not
    assigned onto the result: only construction runs its PP and expert-TP refusals, and ``ep_config()``
    caches the ``EPConfig`` it builds.
    """
    lowp = LowpSpec(**{f.name: getattr(dist_args, f"lowp_{f.name}") for f in fields(LowpSpec)})
    if not trainer_cls.SUPPORTS_LOWP_PRECISION:
        if lowp.precision != "bf16":
            raise ValueError(
                f"lowp_precision={lowp.precision!r} is only validated for SFT training; this "
                "trainer runs in bf16. Remove lowp_precision from the config (or use the SFT script)."
            )
        # A config setting the other lowp_* keys outside SFT would parse cleanly and have no effect.
        ignored = lowp.non_default_shape_fields()
        if ignored:
            raise ValueError(
                f"{', '.join(ignored)} apply only to SFT training and would be silently dropped here. "
                "Remove them from the config, or run the SFT script."
            )

    # Only SFT threads init_from_scratch into the loader; elsewhere the run would fine-tune pretrained weights.
    if not supports_init_from_scratch and dist_args.init_from_scratch:
        raise ValueError(
            "init_from_scratch is only supported by the SFT script (scripts/training/sft.py); "
            "this script would silently ignore it and train pretrained weights."
        )

    if not trainer_cls.SUPPORTS_CP and dist_args.context_parallel_size > 1:
        raise ValueError(
            f"{trainer_cls.__name__} does not support Context Parallelism (CP). Use Expert Parallelism "
            "(EP) and/or Tensor Parallelism (TP) instead, or remove --context_parallel_size."
        )

    if trainer_cls.PP_REFUSAL is not None and dist_args.pipeline_parallel_size > 1:
        raise ValueError(
            f"{trainer_cls.__name__} does not support Pipeline Parallelism (PP). Use Expert Parallelism "
            "(EP) and/or Tensor Parallelism (TP) instead, or remove --pipeline_parallel_size."
        )

    # Fields the two dataclasses spell identically forward themselves; only the renames and the
    # derivations stay explicit.
    kwargs = {name: getattr(dist_args, name) for name in same_name_arg_forwards(dist_args)}
    kwargs.update({axis.size_field: getattr(dist_args, axis.flag) for axis in AXES})
    kwargs.update(
        pp_schedule=dist_args.pipeline_schedule,
        pp_microbatches=dist_args.pipeline_microbatches,
        pp_split=dist_args.pipeline_split,
        ep_fp32_router=dist_args.fp32_router,
        ep_fp32_experts=dist_args.fp32_experts,
        expert_lora=expert_lora,
        lowp=lowp,
        # The two factors stay separate because only one of them is knowable here: a config with no
        # max_length field at all (generation-shaped budgets) declares neither rows nor budget, while
        # ``max_length: null`` declares its rows and leaves the length to the gate's context-window
        # resolution unless a bounded prompt + completion budget caps each row.
        ep_rows_per_device=(
            trainer_cls.FORWARD_ROWS_PER_EXAMPLE * int(getattr(training_config, "per_device_train_batch_size", 0) or 0)
            if hasattr(training_config, "max_length")
            else 0
        ),
        ep_declared_max_length=declared_row_length(training_config),
    )
    config = ParallelismConfig(**kwargs)
    reject_cross_domain_determinism(config, bool(getattr(training_config, "full_determinism", False)))
    return config


def declared_row_length(training_config) -> int:
    """The token cap a config puts on one training row, or 0 when it declares none.

    ``max_length`` where set; otherwise a bounded prompt + completion budget, which the trainers that
    carry both (offline GRPO) truncate each row to. 0 leaves the EP capacity gate to the model's
    context window.
    """
    max_length = getattr(training_config, "max_length", None)
    if is_bounded_length(max_length):
        return int(max_length)
    prompt = getattr(training_config, "max_prompt_length", None)
    completion = getattr(training_config, "max_completion_length", None)
    if is_bounded_length(prompt) and is_bounded_length(completion):
        return int(prompt) + int(completion)
    return 0
