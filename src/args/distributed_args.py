"""Combined distributed training arguments for EP, CP, and TP parallelism."""

from dataclasses import dataclass, field
from typing import Literal

from src.checkpoint.format import DEFAULT_MAX_SHARD_SIZE


@dataclass
class DistributedArguments:
    """
    Combined arguments for all parallelism modes: EP, CP, and TP.

    ParallelismConfig validates combinations and raises on incompatible ones
    (e.g. TP+CP). See agent-docs/parallelism/ for the support matrix.
    """

    expert_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Expert distribution size — how many distinct expert subsets to create. "
            "Full EP group size = expert_parallel_size * expert_tensor_parallel_size. "
            "For node-local EP, ep_group_size must divide the NVLink domain "
            "(nvlink_domain_size; = gpus_per_node on a standard node, = the rack via "
            "NVLINK_DOMAIN_SIZE on NVL72). For cross-node EP, ep_group_size must divide "
            "world_size. Number of experts will be auto-detected from model."
        },
    )

    ep_scope: Literal["auto", "node", "global"] = field(
        default="auto",
        metadata={
            "help": "EP scope for multi-node training. "
            "'auto': node-local if ep_group_size <= the NVLink domain (nvlink_domain_size; "
            "= gpus_per_node normally, = the rack on NVL72), else cross-node "
            "(ep_group_size = expert_parallel_size * expert_tensor_parallel_size). "
            "'node': EP groups within one NVLink domain (NVLink only, fastest). "
            "'global': EP groups span NVLink domains (uses RDMA, more distribution)."
        },
    )

    save_sharded_ep: bool = field(
        default=False,
        metadata={
            "help": "Save EP checkpoints as per-rank expert shards (every rank writes its own slice "
            "in parallel) instead of the gathered HF-standard checkpoint. Buys write bandwidth at the "
            "price of an artifact nothing loads: scripts/after_training/merge_ep_shards.py must run "
            "before resume or serving, and the merge carries no optimizer state — pair it with "
            "save_only_model. If False (default), each layer's experts are gathered and streamed to "
            "one writer, producing a checkpoint from_pretrained loads as it stands."
        },
    )

    save_max_shard_size: str = field(
        default=DEFAULT_MAX_SHARD_SIZE,
        metadata={
            "help": "Max size of one safetensors shard the distributed save paths write (e.g. '5GB'). "
            "The gathered EP/TP/FSDP2 writers read it; HF's own save_pretrained uses its own default. "
            "A sharded EP save writes one file per rank by design, so it applies to the merged "
            "artifact merge_ep_shards.py produces, not to the shards themselves."
        },
    )

    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": "Allow starting a run in a non-empty output_dir. Off by default so a fresh run "
            "cannot silently interleave its checkpoints with an existing one's."
        },
    )

    merge_expert_lora_on_save: bool = field(
        default=False,
        metadata={
            "help": "When training native grouped-LoRA on EP experts, fold the LoRA delta into the "
            "base on save so the checkpoint is a fully-merged, servable HF model. Covers both "
            "halves of a mixed run: the expert deltas inside each family's gather, and any "
            "attention adapters via merge_adapter held over the write (undone afterwards, so "
            "training continues). If False (default), a standalone adapter file is written and the "
            "frozen base is left unchanged (reload pairs the adapter with the base model)."
        },
    )

    fp32_router: bool = field(
        default=False,
        metadata={
            "help": "Store EP router weights in FP32 for stable optimizer updates. The gate matmul "
            "then runs in FP32 with autocast explicitly disabled (the router input is upcast), so "
            "routing decisions are computed at full precision; the rest of the layer stays BF16. "
            "fp32_non_ep_params turns this on regardless (its upcast skips the router, and a BF16 "
            "router beside FP32 dense params would trip FSDP2's uniform-dtype check), so leaving "
            "this False does not keep the router in BF16 there. "
            "No effect on the gate input at ep_group_size==1 with fsdp_shard_ep1_experts, where FSDP2 "
            "manages the router and its BF16 param policy owns the compute dtype (the weights are "
            "still stored FP32). Gemma4 ignores it entirely and warns: its router is a sibling module "
            "outside the EP wrapper."
        },
    )

    fp32_experts: bool = field(
        default=False,
        metadata={
            "help": "Store EP expert weights in FP32 for stable optimizer updates. Compute stays in "
            "BF16 via autocast. Skipped with a warning at ep_group_size==1 with fsdp_shard_ep1_experts "
            "— FSDP2 manages the replicated experts and its BF16 param policy casts them anyway, so "
            "the upcast would only mix FP32 params into a BF16 fully_shard group. For FP32 experts in "
            "that state set fsdp_shard_ep1_experts: false (fp32_non_ep_params is rejected there — its "
            "upcast skips the experts FSDP manages)."
        },
    )

    fp32_non_ep_params: bool = field(
        default=False,
        metadata={
            "help": "Store non-expert (dense) parameters in FP32 for stable optimizer updates. "
            "Compute stays in BF16 via autocast. Useful for mixed precision training stability "
            "when training non-MoE parts of the model."
        },
    )

    fp32_grad_reduce: bool = field(
        default=False,
        metadata={
            "help": "Reduce gradients across ranks in FP32 (FSDP2 reduce_dtype) even when params "
            "are stored in BF16. Removes the ~10^4x precision loss of summing many bf16 "
            "gradients; matches TorchTitan/Megatron/DeepSpeed defaults. Independent of "
            "fp32_non_ep_params (which also moves storage to fp32): set this alone to keep "
            "BF16 master weights + AdamWBF16 (6 B/param) with fp32 grad reduction. Costs ~2x "
            "bandwidth on the reduce-scatter collective only. fp32_non_ep_params implies it for "
            "the FSDP2 reduce dtype ONLY — the EP, TP and QLoRA reductions read this knob "
            "directly, so set it explicitly to cover them."
        },
    )

    bf16_optimizer: bool | None = field(
        default=None,
        metadata={
            "help": "Use AdamWBF16 (bf16 master weights + stochastic rounding, 6 B/param) instead of the "
            "standard optimizer. None (default) = auto: on when bf16 is set and the optimizer is the "
            "default AdamW, off under replicated DDP (accelerate MULTI_GPU). True forces it on (e.g. to "
            "opt in under DDP); False forces full fp32 master weights — rejected only where the run "
            "mixes plain-tensor experts with FSDP2 DTensors (any expert distribution with "
            "ep_group_size>1 — EP or pure ETP — or ep_size==1 with fsdp_shard_ep1_experts=false), "
            "since fused AdamW cannot span the two; allowed on dense runs and on ep_size==1 MoE with "
            "the default FSDP-sharded experts. "
            "See agent-docs/optimization/bf16-optimizer.md."
        },
    )

    use_grouped_gemm: bool = field(
        default=True,
        metadata={
            "help": "Enable F.grouped_mm for batched expert matmuls on SM90+ hardware (default: True). "
            "When ep_size > 1, controls grouped GEMM inside EP layers. When ep_size = 1, also "
            "applies EP wrappers for grouped GEMM without distribution. Set False to disable."
        },
    )

    lowp_precision: Literal["bf16", "fp8", "fp4", "mxfp4"] = field(
        default="bf16",
        metadata={
            "help": "Low-precision matmul compute (mixed precision: bf16/fp32 master weights with "
            "low-precision GEMM operands — parameters and checkpoints are unchanged). One of "
            "'bf16' (default, off), 'fp8' (mxfp8), 'fp4' (nvfp4, most accurate) or 'mxfp4' "
            "(fast fp4). Applies to dense MLP projections "
            "(gate/up/down) and MoE expert grouped GEMMs; attention, embeddings, lm_head and "
            "norms stay bf16. See agent-docs/optimization/low-precision-moe-kernels.md."
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
            "help": "Apply lowp_precision to MoE expert grouped GEMMs. Set False to keep experts in "
            "bf16. The grouped path is the fake-quant reference and stays bf16-speed at "
            "fine-grained MoE expert shapes; DeepGEMM's native fp8/fp4 kernel is opt-in via "
            "HALO_DEEPGEMM_NATIVE=1 (plus a roofline/shape gate) and is never auto-routed. See "
            "agent-docs/optimization/low-precision-moe-kernels.md."
        },
    )

    lowp_keep_first_blocks: int = field(
        default=0,
        metadata={
            "help": "Leading transformer blocks kept in bf16 under lowp_precision (NVFP4 recipe — the "
            "first blocks are the most precision-sensitive). Default 0."
        },
    )

    lowp_keep_last_blocks: int = field(
        default=0,
        metadata={
            "help": "Trailing transformer blocks kept in bf16 under lowp_precision (the NVFP4 recipe "
            "keeps the last ~8). Default 0."
        },
    )

    max_concurrent_loading: int | None = field(
        default=None,
        metadata={
            "help": "Max ranks loading model simultaneously per node. Unset (default) derives it "
            "from the node width — half of it, capped at 4 — so a 4-GPU tray throttles to 2 instead "
            "of admitting every rank. Set 1 for fully sequential loading on CPU-RAM-constrained "
            "machines, 0 for all parallel; any explicit value is used verbatim."
        },
    )

    ep_lazy_loading: bool = field(
        default=True,
        metadata={
            "help": "Expert-parallel model load path. True (default): lazy safetensors slicing from disk "
            "(meta init, low CPU RAM). False: standard from_pretrained then EP patch "
            "(higher CPU RAM, no meta-device init)."
        },
    )

    ep_buffer_backend: Literal["auto", "elastic", "legacy"] = field(
        default="auto",
        metadata={
            "help": "DeepEP transport backend for the EP all-to-all. 'auto' (default) == 'elastic': "
            "ElasticBuffer over NCCL Gin — cross-node capable (AWS EFA) and forwards arbitrary sequence "
            "length (gpt-oss-20b ep8 to 65536). The only guarded limit is DeepEP's 32-bit wire index "
            "(~175k tokens/rank, far beyond any training sequence). 'legacy': the original deep_ep.Buffer "
            "(CUDA IPC P2P over NVLink) — numerically identical, intranode / node-local only; pick it for "
            "an IBGDA InfiniBand fabric or to A/B the transport. Legacy cross-node forces NVSHMEM IBGDA, "
            "so it is rejected for a cross-node group on AWS EFA (no IBGDA). See agent-docs/infrastructure/deepep.md."
        },
    )

    context_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Context parallel size for sequence splitting. "
            "When > 1, sequences are split across cp_size ranks. "
            "CP is always NVLink-local (must be <= nvlink_domain_size; = gpus_per_node on a "
            "standard node, = the rack via NVLINK_DOMAIN_SIZE on NVL72). "
            "NOTE: CP is NOT compatible with padding-free collator or TP."
        },
    )

    tensor_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Tensor parallel size for attention/embeddings/lm_head. Set to >1 to enable TP. "
            "Must divide world size evenly. "
            "If world_size > tp_size, FSDP2 is used for data parallelism. "
            "NOTE: TP is NOT compatible with CP."
        },
    )

    expert_tensor_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Expert FFN tensor parallel size (independent of tensor_parallel_size). "
            "Shards expert FFN weights (gate_up_proj, down_proj) across expert_tp_size ranks "
            "within NVLink-local groups, reducing per-rank expert memory by expert_tp_size. "
            "Must divide the NVLink domain (nvlink_domain_size; = gpus_per_node normally, "
            "= the rack on NVL72). "
            "Full EP group size = expert_parallel_size * expert_tensor_parallel_size. "
            "Combines with EP (EP+ETP, node-local) but not with attention "
            "tensor_parallel_size > 1, and not with context_parallel_size > 1."
        },
    )

    pipeline_parallel_size: int = field(
        default=1,
        metadata={
            "help": "Pipeline parallel size — splits the model's decoder layers into contiguous stages. "
            "PP is the OUTERMOST parallelism dimension and the only one designed to cross NVLink "
            "domains: stage boundaries must fall on NVLink-domain boundaries (world_size / "
            "pipeline_parallel_size must be a multiple of nvlink_domain_size), so only point-to-point "
            "boundary activations traverse EFA/IB while EP/TP/ETP/CP/FSDP stay domain-local inside each "
            "stage. Reduces data_parallel_size by pipeline_parallel_size. Requires fixed input shapes "
            "(padding-free collators are rejected; packing is supported at one packed row per "
            "microbatch) and a trainer that declares _supports_pp. "
            "See agent-docs/parallelism/pipeline-parallelism.md."
        },
    )

    pipeline_microbatches: int = field(
        default=0,
        metadata={
            "help": "Microbatches per optimizer step under pipeline parallelism (the pipeline's own "
            "gradient accumulation). 0 (default) = auto: gradient_accumulation_steps, raised to "
            "pipeline_parallel_size when the 1f1b schedule requires it. per_device_train_batch_size "
            "must be divisible by this."
        },
    )

    pipeline_schedule: Literal["1f1b", "gpipe"] = field(
        default="1f1b",
        metadata={
            "help": "Pipeline schedule: '1f1b' (default, one-forward-one-backward — lowest activation "
            "memory, requires pipeline_microbatches >= pipeline_parallel_size) or 'gpipe' (all forwards "
            "then all backwards — higher activation memory, no microbatch-count constraint)."
        },
    )

    pipeline_split: list[int] | None = field(
        default=None,
        metadata={
            "help": "Manual per-stage decoder-layer counts under pipeline parallelism, e.g. [19, 17] "
            "for a 36-layer model at pipeline_parallel_size=2. Must sum to the model's layer count, "
            "with stage boundaries on the model's layer_types period. Default (null) is the "
            "head-weighted split: the last stage's layer budget shrinks by the lm_head's "
            "layer-equivalent cost, so the stages' real compute balances."
        },
    )

    nvlink_domain_size: int = field(
        default=0,
        metadata={
            "help": "Number of GPUs reachable over NVLink — the locality unit for node-local "
            "EP/CP/TP/ETP grouping. 0 (default) = auto: reads the NVLINK_DOMAIN_SIZE env var, "
            "else falls back to gpus_per_node (one OS node, the standard ≤8-GPU cluster case). "
            "On GB200/GB300 NVL72 set this (or NVLINK_DOMAIN_SIZE) to the rack's NVLink "
            "partition size (e.g. 72) to run EP/CP/TP NVLink-wide across OS nodes. The "
            "NVLINK_DOMAIN_SIZE env var applies to every entrypoint; this arg is the YAML/CLI "
            "equivalent for every training script."
        },
    )

    fsdp_reshard_after_forward: bool = field(
        default=False,
        metadata={
            "help": "FSDP2 resharding mode for torchrun data parallelism. "
            "False (default): SHARD_GRAD_OP behavior — params stay unsharded between "
            "forward and backward. Faster throughput, higher peak memory. "
            "True: FULL_SHARD behavior — params resharded after forward. "
            "Lower peak memory, slightly slower. "
            "Only affects torchrun modes (not accelerate launch). "
            "Rejected whenever an expert-distribution group exists (ep_group_size>1 — ep_size>1 OR "
            "expert_tp_size>1: the backward re-gather can race the DeepEP combine, and pure ETP "
            "shares that path), under TP with data_parallel_size>1 (the backward re-gather issues a "
            "plain all-gather on TP-sharded DTensor params), and under PP; allowed on pure DP, CP, "
            "and ep_size==1 MoE without expert TP."
        },
    )
    fsdp_reshard_after_backward: bool = field(
        default=True,
        metadata={
            "help": "False keeps FSDP2 params unsharded across a gradient-accumulation window's "
            "microsteps (torch set_reshard_after_backward, re-armed for the window's last backward): "
            "FSDP2 otherwise reshards after every microstep's backward and re-all-gathers the full "
            "model on the next microstep — wasted collectives when nothing changed, and ~15s x "
            "gradient_accumulation_steps per optimizer step when NCCL is forced onto sockets "
            "(rollout_backend: sglang). Leaves one re-gather per optimizer step instead of one per "
            "microstep (so it does nothing at gradient_accumulation_steps=1), and costs one full "
            "unsharded bf16 param copy per GPU for the whole run. Plain-DP/CP/EP torchrun path only; "
            "rejected with fsdp_reshard_after_forward=True, TP, or PP."
        },
    )
    fsdp_shard_ep1_experts: bool = field(
        default=True,
        metadata={
            "help": "At ep_group_size==1 only (ep_size * expert_tp_size, i.e. the experts are truly "
            "replicated): FSDP-shard the MoE experts (reduce-scatter is their sole gradient sync; the "
            "EP layer skips its own hooks). True (default): frees memory that grows with DP "
            "(gpt-oss-20b -19%/-37% at 2/8 GPU), throughput-neutral, grad-equivalent. False: every DP "
            "rank keeps a full copy (max throughput, e.g. dense-ep1 b1 SFT). No effect when "
            "ep_group_size>1 — that includes pure ETP (ep_size==1, expert_tp_size>1), where the "
            "experts are sharded across the ETP group rather than replicated. RL-safe (the vLLM "
            "weight-sync gather materializes the shards with full_tensor before reshaping) under both "
            "ZeRO2 and ZeRO3."
        },
    )
    use_hsdp: bool = field(
        default=False,
        metadata={
            "help": "Hybrid Sharded Data Parallel for the standard DP path (pure DP or CP; EP, TP "
            "and Expert-TP are rejected). False (default): 1D FSDP full-shards non-expert params "
            "across every DP rank — "
            "every all-gather / reduce-scatter crosses the inter-node fabric. True: shard within "
            "each NVLink domain and replicate across domains, so the bandwidth-heavy shard "
            "collectives stay on NVLink and only one gradient all-reduce crosses RDMA per step. "
            "Shard width and replica count are derived from the topology automatically. Trades "
            "memory (a per-domain param replica) for inter-node bandwidth; a no-op on a single "
            "node."
        },
    )

    init_from_scratch: bool = field(
        default=False,
        metadata={
            "help": "Train from scratch (random init) instead of fine-tuning pretrained weights. "
            "When True, the model is built from the architecture/config of model_name_or_path "
            "with FRESHLY-INITIALIZED random weights (transformers _init_weights), reusing that "
            "id's tokenizer and config — no checkpoint weights are loaded. Pretraining is "
            "step-based: set max_steps and a warmup+decay LR schedule. Supported for dense "
            "data-parallel / FSDP2 training; EP/TP/CP from-scratch is not yet supported "
            "(see agent-docs/training-methods/pretraining.md)."
        },
    )
    reset_sinks: bool = field(
        default=True,
        metadata={
            "help": "GptOss only (ignored elsewhere). True (default): neutralize the attention sinks for "
            "fine-tuning — filled with dtype min and frozen (zero softmax mass), or removed under "
            "flash_attention_2. False: keep the pretrained sinks live and FROZEN, the on-policy RL "
            "setting (the trainer scores with the sinks the rollout engine serves). Live sinks accept only "
            "a sink-carrying attention implementation — flash_attention_4 on Blackwell, flex_attention or "
            "eager on Hopper; a sink-dropping one (FA2/SDPA) is rejected, since dropping the sink column "
            "shifts every logprob by nats. Live sinks are refused under Context Parallelism (its kernels "
            "never see the column)."
        },
    )
    train_sinks: bool = field(
        default=False,
        metadata={
            "help": "GptOss only: keep the live sinks TRAINABLE (default: False; requires reset_sinks: false, "
            "the contradiction is rejected). Full fine-tuning under flash_attention_4 (the fused kernel "
            "emits no sink gradient, so grad-requiring sinks run through an exact sink-less + "
            "sigmoid(lse - sink) rescale) or eager (native gradient); flex_attention and FA3 are rejected "
            "(no usable sink gradient). Refused by adapter (PEFT) runs and by weight-sync RL."
        },
    )

    text_only_model: bool = field(
        default=False,
        metadata={
            "help": "Load a multimodal checkpoint through its text-only CausalLM class instead of the "
            "vision-bearing wrapper (Qwen3.5/3.6: Qwen3_5MoeForCausalLM vs ...ForConditionalGeneration). "
            "Drops the vision tower and any MTP tail from the build and restores the config-honored "
            "output_router_logits path, so moe_balancing: aux_loss works where the wrapper's forward "
            "never consults the flag. For text-only datasets; a VLM dataset needs the wrapper."
        },
    )

    fp32_output_conversion: bool = field(
        default=False,
        metadata={
            "help": "Keep accelerate's fp32 output conversion wrapper (default: False = off). "
            "The wrapper converts model outputs (logits) from bf16 to fp32, "
            "which can cause OOM for long sequences (e.g., 25k tokens × 200k vocab = 18+ GiB). "
            "Safe to leave off when the model is loaded in bf16 — forward/backward run in bf16 anyway."
        },
    )
