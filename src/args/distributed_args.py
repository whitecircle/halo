"""Combined distributed training arguments for EP, CP, and TP parallelism."""

from dataclasses import dataclass, field

from src.checkpoint.format import DEFAULT_MAX_SHARD_SIZE
from src.config_validation import RangeValidatedConfig, present, require_int
from src.distributed.parallelism_config import EPBufferBackend, EPScope, PPSchedule
from src.kernels.lowp.spec import LowpPrecision


@dataclass
class DistributedArguments(RangeValidatedConfig):
    """
    Combined arguments for all parallelism modes: EP, CP, and TP.

    ParallelismConfig validates ranges and combinations and raises on incompatible ones
    (e.g. TP+CP); this class refuses only a value that is not an int at all. See
    agent-docs/parallelism/ for the support matrix.
    """

    expert_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Expert-parallel size: the ranks one DeepEP all-to-all spans, each holding num_experts / "
            "expert_parallel_size experts (> 1 enables EP on a MoE model). The EP group of expert_parallel_size × "
            "expert_tensor_parallel_size ranks must divide the NVLink domain when node-local, or world_size when "
            "cross-node. See agent-docs/parallelism/expert-parallelism.md#ep-grouping."
        },
    )

    ep_scope: EPScope = field(
        default="auto",
        metadata={
            "help": "Where EP groups sit: 'node' keeps each inside one NVLink domain, 'global' spans domains over "
            "RDMA, and 'auto' picks 'node' when the EP group (expert_parallel_size × expert_tensor_parallel_size) "
            "fits the domain. See agent-docs/parallelism/expert-parallelism.md#ep-grouping."
        },
    )

    save_sharded_ep: bool = field(
        default=False,
        metadata={
            "help": "Save EP checkpoints as per-rank expert shards written in parallel instead of the gathered HF "
            "checkpoint from_pretrained loads. Nothing loads the shards until "
            "scripts/after_training/merge_ep_shards.py merges them, without optimizer state, so pair it with "
            "save_only_model. See agent-docs/reference/checkpoints.md#expert-parallelism-ep-eptp-epcp."
        },
    )

    save_max_shard_size: str = field(
        default=DEFAULT_MAX_SHARD_SIZE,
        metadata={
            "help": "Largest safetensors shard the gathered EP/TP/FSDP2/CP saves write, like '5GB'; HF's own "
            "save_pretrained keeps its default. A sharded EP save writes one file per rank regardless, and "
            "merge_ep_shards.py sizes its output by its own --max_shard_size."
        },
    )

    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": "Allow starting a run in a non-empty output_dir. Off by default so a fresh run "
            "cannot silently interleave its checkpoints with an existing one's."
        },
    )

    allow_optimizer_warm_restart: bool = field(
        default=False,
        metadata={
            "help": "On resume, continue with a freshly initialized optimizer (weights, step and LR schedule still "
            "resume) when restoring this topology's optimizer shards fails on a rank, or the checkpoint holds an "
            "interrupted save's optimizer state. Off, both raise on every rank, since a warm restart resets the "
            "optimizer moments mid-run. "
            "See agent-docs/reference/checkpoints.md#warm-restart-vs-exact-resume-torchrun."
        },
    )

    merge_expert_lora_on_save: bool = field(
        default=False,
        metadata={
            "help": "Fold the native expert LoRA delta, and any attention adapters, into the base on save so each "
            "checkpoint is a merged, servable HF model; the frozen base itself never changes. Training checkpoints "
            "also keep the unmerged adapter in resume_adapter/ for resume, and false writes a standalone adapter "
            "file instead. See agent-docs/reference/checkpoints.md#merge-on-save-checkpoints."
        },
    )

    fp32_router: bool = field(
        default=False,
        metadata={
            "help": "Keep EP router weights as FP32 masters with an FP32 routing matmul, except at ep_group_size == 1 "
            "under fsdp_shard_ep1_experts, where FSDP2's BF16 param policy owns the compute dtype. "
            "fp32_non_ep_params implies it; Gemma 4 ignores it with a warning. "
            "See agent-docs/parallelism/expert-parallelism.md#precision-control."
        },
    )

    fp32_experts: bool = field(
        default=False,
        metadata={
            "help": "Keep EP expert weights as FP32 masters; compute stays BF16 under autocast. Skipped with a "
            "warning at ep_group_size == 1 under fsdp_shard_ep1_experts, where FSDP2's BF16 param policy casts the "
            "replicated experts anyway: set fsdp_shard_ep1_experts: false for FP32 experts there. "
            "See agent-docs/parallelism/expert-parallelism.md#precision-control."
        },
    )

    fp32_non_ep_params: bool = field(
        default=False,
        metadata={
            "help": "Keep non-expert (dense) parameters as FP32 masters for stable optimizer updates; compute stays "
            "BF16 under autocast. See agent-docs/reference/configuration-reference.md#parallelismconfig."
        },
    )

    fp32_grad_reduce: bool = field(
        default=False,
        metadata={
            "help": "Reduce gradients across ranks in FP32 while parameters stay BF16 (the FSDP2 reduce dtype and "
            "the EP, TP and QLoRA reductions), at about twice the bandwidth on that collective. Independent of "
            "fp32_non_ep_params, which implies it for the FSDP2 reduce dtype only. "
            "See agent-docs/optimization/bf16-optimizer.md#master-weight-and-grad-reduce-options."
        },
    )

    bf16_optimizer: bool | None = field(
        default=None,
        metadata={
            "help": "Use AdamWBF16 (BF16 master weights with stochastic rounding, 6 B/param); unset = auto, on under "
            "bf16 with the default AdamW and off under replicated DDP. False builds the stock AdamW over the "
            "parameters as loaded, BF16 under bf16. See agent-docs/optimization/bf16-optimizer.md#usage."
        },
    )

    use_grouped_gemm: bool = field(
        default=True,
        metadata={
            "help": "Run MoE expert matmuls through F.grouped_mm on SM90+ hardware, wrapping the MoE layers even at "
            "expert_parallel_size 1, so an MoE model under accelerate launch is rejected at load: launch with "
            "torchrun or set this false. "
            "See agent-docs/optimization/grouped-gemm.md#standalone-grouped-gemm-mode."
        },
    )

    lowp_precision: LowpPrecision = field(
        default="bf16",
        metadata={
            "help": "Low-precision GEMM operands for dense MLP and MoE expert matmuls over BF16/FP32 masters, "
            "parameters and checkpoints unchanged: 'bf16' (off), 'fp8' (mxfp8), 'fp4' (nvfp4) or 'mxfp4'. SFT only, "
            "and every format runs slower than bf16. "
            "See agent-docs/optimization/low-precision-moe-kernels.md#usage."
        },
    )

    lowp_apply_dense_mlp: bool = field(
        default=True,
        metadata={
            "help": "Apply lowp_precision to dense MLP projections (gate/up/down). Set False to keep "
            "dense MLPs in bf16 (e.g. low precision on MoE experts only)."
        },
    )

    lowp_apply_moe_experts: bool = field(
        default=True,
        metadata={
            "help": "Apply lowp_precision to MoE expert grouped GEMMs; false keeps the experts in bf16. DeepGEMM's "
            "native kernel is opt-in through HALO_DEEPGEMM_NATIVE=1 and never auto-routed. "
            "See agent-docs/optimization/low-precision-moe-kernels.md#deepgemm-the-native-fp8fp4-path."
        },
    )

    lowp_keep_first_blocks: int = field(
        default=0,
        metadata={
            "help": "Leading transformer blocks kept in bf16 under lowp_precision (the NVFP4 recipe: the first "
            "blocks are the most precision-sensitive)."
        },
    )

    lowp_keep_last_blocks: int = field(
        default=0,
        metadata={
            "help": "Trailing transformer blocks kept in bf16 under lowp_precision (the NVFP4 recipe keeps the last "
            "~8)."
        },
    )

    max_concurrent_loading: int | None = field(
        default=None,
        metadata={
            "help": "Most ranks per node loading the model at once; unset derives half the node width, capped at 4. "
            "An explicit value is used verbatim: 1 loads sequentially on CPU-RAM-constrained hosts, 0 all at once. "
            "See agent-docs/parallelism/multi-node.md#model-loading."
        },
    )

    ep_lazy_loading: bool = field(
        default=True,
        metadata={
            "help": "Load an EP model by lazy safetensors slicing, each rank reading only its own expert slice "
            "(meta init, low CPU RAM); false loads with from_pretrained and patches EP afterwards. "
            "See agent-docs/parallelism/expert-parallelism.md#model-loading."
        },
    )

    ep_buffer_backend: EPBufferBackend = field(
        default="auto",
        metadata={
            "help": "DeepEP all-to-all transport: 'auto'/'elastic' is the ElasticBuffer over NCCL Gin (cross-node "
            "capable), 'legacy' the CUDA-IPC Buffer (numerically identical, intranode only). Pick 'legacy' for "
            "long-context ep8, where elastic at about 64k tokens/rank or more deadlocks its combine against FSDP2. "
            "See agent-docs/infrastructure/deepep.md#transport-backend."
        },
    )

    context_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Context-parallel (Ulysses) size: each sequence is split across this many ranks of one NVLink "
            "domain. Not compatible with tensor parallelism or the padding-free collator. "
            "See agent-docs/parallelism/context-parallelism.md#requirements."
        },
    )

    tensor_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Tensor-parallel size for attention, embeddings and lm_head (> 1 enables TP); it must divide "
            "world_size, and FSDP2 data-parallelizes across the TP groups. Not compatible with context parallelism."
        },
    )

    expert_tensor_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Expert-FFN tensor-parallel size: shards each expert's gate_up_proj and down_proj across this "
            "many ranks of one NVLink domain, cutting per-rank expert memory by the same factor. Combines with "
            "expert_parallel_size (the EP group is their product), not with tensor or context parallelism. "
            "See agent-docs/parallelism/expert-tensor-parallelism.md#validation-rules."
        },
    )

    pipeline_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Pipeline-parallel size: splits the decoder layers into contiguous stages, each a whole number "
            "of NVLink domains, and divides data_parallel_size by it. Needs fixed input shapes and a trainer that "
            "supports PP; every other script rejects a value > 1. "
            "See agent-docs/parallelism/pipeline-parallelism.md#launching."
        },
    )

    pipeline_microbatches: int = field(
        default=0,
        metadata={
            "help": "Microbatches per optimizer step under pipeline parallelism, the pipeline's own gradient "
            "accumulation; 0 = gradient_accumulation_steps, raised to pipeline_parallel_size where 1f1b needs it. "
            "per_device_train_batch_size must be divisible by it, and a non-default value is rejected without PP."
        },
    )

    pipeline_schedule: PPSchedule = field(
        default="1f1b",
        metadata={
            "help": "Pipeline schedule: '1f1b' (lowest activation memory, needs pipeline_microbatches >= "
            "pipeline_parallel_size) or 'gpipe' (all forwards then all backwards, no microbatch-count constraint). "
            "A non-default value is rejected without PP."
        },
    )

    pipeline_split: list[int] | None = field(
        default=None,
        metadata={
            "help": "Per-stage decoder-layer counts under pipeline parallelism, like [19, 17] for 36 layers at two "
            "stages, summing to the layer count with boundaries on the layer_types period. Unset = the "
            "head-weighted split, which shrinks the last stage by the lm_head's layer-equivalent cost; a value is "
            "rejected without PP."
        },
    )

    nvlink_domain_size: int = field(
        default=0,
        metadata={
            "help": "GPUs one NVLink domain spans, the locality unit of node-local EP/CP/TP/ETP groups; 0 reads "
            "NVLINK_DOMAIN_SIZE, else gpus_per_node. Set the rack's partition size (72 on GB200/GB300 NVL72) to "
            "group NVLink-wide across OS nodes. "
            "See agent-docs/parallelism/multi-node.md#gb200gb300-nvl72-multi-node-nvlink."
        },
    )

    fsdp_reshard_after_forward: bool = field(
        default=False,
        metadata={
            "help": "False (ZeRO-2) keeps FSDP2 params gathered from forward to backward, faster at a higher peak "
            "memory; true (ZeRO-3) reshards them after forward. Torchrun only, and rejected with an "
            "expert-distribution group (ep_group_size > 1), TP at data_parallel_size > 1, or PP. "
            "See agent-docs/parallelism/data-parallelism.md#zero-2-vs-zero-3-reshard_after_forward."
        },
    )
    fsdp_reshard_after_backward: bool = field(
        default=True,
        metadata={
            "help": "False keeps FSDP2 params unsharded across a gradient-accumulation window's microsteps, one "
            "re-all-gather per optimizer step instead of one per microstep, at unchanged peak memory. Torchrun "
            "DP/CP/EP only; rejected with fsdp_reshard_after_forward, TP or PP. "
            "See agent-docs/parallelism/data-parallelism.md#zero-2-vs-zero-3-reshard_after_forward."
        },
    )
    fsdp_defer_grad_sync: bool = field(
        default=False,
        metadata={
            "help": "Skip FSDP2's gradient reduce-scatter on a gradient-accumulation window's microsteps 1..n-1, "
            "reducing once per optimizer step at the cost of one unsharded gradient copy per GPU held across the "
            "window. Torchrun FSDP2 only; rejected under PP, TP at data_parallel_size 1, QLoRA and "
            "fsdp_reshard_after_forward. "
            "See agent-docs/parallelism/data-parallelism.md#deferred-gradient-reduce-fsdp_defer_grad_sync."
        },
    )
    fsdp_shard_ep1_experts: bool = field(
        default=True,
        metadata={
            "help": "At ep_group_size == 1, where the experts are truly replicated, FSDP-shard them so their "
            "reduce-scatter is the sole gradient sync, freeing memory that grows with DP at unchanged gradients. "
            "False keeps a full expert copy per rank; no effect at ep_group_size > 1, pure ETP included. "
            "See agent-docs/parallelism/data-parallelism.md#ep1-expert-sharding."
        },
    )
    use_hsdp: bool = field(
        default=False,
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
    reset_sinks: bool = field(
        default=True,
        metadata={
            "help": "GptOss only: true neutralizes the attention sinks for fine-tuning; false keeps the pretrained "
            "sinks live and frozen, the on-policy RL setting, which needs a sink-carrying attention implementation "
            "and no context parallelism. See agent-docs/models/gpt-oss.md#attention-sinks."
        },
    )
    train_sinks: bool = field(
        default=False,
        metadata={
            "help": "GptOss only: train the live sinks (requires reset_sinks: false) under full fine-tuning with "
            "flash_attention_4 or eager. Refused with flex_attention, FA3, adapter runs and weight-sync RL. "
            "See agent-docs/models/gpt-oss.md#attention-sinks."
        },
    )

    text_only_model: bool = field(
        default=False,
        metadata={
            "help": "Load a multimodal checkpoint through its text-only CausalLM class (Qwen3.5/3.6: "
            "Qwen3_5MoeForCausalLM), dropping the vision tower and any MTP tail from the build and the export. For "
            "text-only datasets; it also restores the output_router_logits path moe_balancing: aux_loss needs. "
            "See agent-docs/reference/configuration-reference.md#distributedarguments-model--from-scratch."
        },
    )

    fp32_output_conversion: bool = field(
        default=False,
        metadata={
            "help": "Keep accelerate's wrapper that converts model outputs (the logits) to fp32. Off by default: "
            "the upcast can OOM on long sequences, and a bf16 model needs no fp32 outputs."
        },
    )

    def __post_init__(self):
        self._validate_ranges()

    def _validate_ranges(self) -> None:
        super()._validate_ranges()
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
