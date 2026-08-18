#!/usr/bin/env python
"""
MoE model SFT benchmark with Expert Parallelism (EP).

Measures Model FLOPS Utilization (MFU), Sparse MFU (S-MFU), achieved TFLOPS,
and tokens/sec for MoE models using DistributedSFTTrainer with EfficiencyCallback
and Expert Parallelism via DeepEP all-to-all routing.

Usage:
    # EP=2 with default model (gpt-oss-20b), seq_len=4096, 10 steps
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/benchmark_sft_ep.py --ep 2 --seq 4096 --steps 10

    # Short sequence benchmark
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/benchmark_sft_ep.py --ep 2 --seq 2048 --steps 5

    # Long sequence benchmark
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/benchmark_sft_ep.py --ep 2 --seq 16384 --steps 20 --warmup 3

    # 8 GPUs with full EP
    torchrun --nproc_per_node=8 \
        tests/gpu/profiling/benchmark_sft_ep.py --ep 8 --seq 4096 --steps 10

    # Custom model path
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/benchmark_sft_ep.py --ep 2 --model_path /path/to/moe/model --seq 4096

    # Disable Liger kernels
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/benchmark_sft_ep.py --ep 2 --seq 4096 --no_liger
"""

import argparse
import os
import sys

import torch
import torch.distributed as dist
from accelerate import PartialState
from peft import LoraConfig
from transformers import AutoTokenizer
from trl import SFTConfig

from src.callbacks.efficiency import EfficiencyCallback
from src.callbacks.profiler import TorchProfilerCallback
from src.diagnostics.performance_monitor import get_performance_monitor
from src.distributed.expert_parallel.base_layer import EPMoELayerBase
from src.distributed.expert_parallel.config import ExpertLoraSpec
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import LOWP_PRECISIONS, ParallelismConfig
from src.distributed.runtime import barrier
from src.env import torch_trace_dir
from src.trainers.sft import DistributedSFTTrainer
from tests.common.benchmark_args import create_benchmark_parser, pp_topology_kwargs
from tests.common.datasets import create_benchmark_dataset
from tests.common.distributed import (
    cleanup_dirs,
    ensure_model_downloaded,
    init_distributed,
    setup_cache_dirs,
    teardown_distributed,
)
from tests.common.models import MODEL_CONFIGS
from tests.common.reporting import emit_benchmark, format_benchmark_report
from tests.common.utils import cleanup_memory, gpu_mem_gb, log

# Main Benchmark


def _liger_config_pairs(spec: str) -> dict[str, bool]:
    """Parse ``key=true,key=false`` into the liger_kernel_config dict the loader takes."""
    config = {}
    for pair in spec.split(","):
        key, _, value = pair.partition("=")
        if value.lower() not in ("true", "false"):
            raise argparse.ArgumentTypeError(f"--liger_config expects key=true|false, got {pair!r}")
        config[key.strip()] = value.lower() == "true"
    return config


def main() -> int:
    failed = False
    parser = create_benchmark_parser(
        description="MoE SFT benchmark with Expert Parallelism",
    )
    parser.add_argument(
        "--fused_linear_ce", action="store_true", help="Use FusedLinearCrossEntropy instead of CrossEntropy"
    )
    parser.add_argument(
        "--liger_config",
        type=_liger_config_pairs,
        default=None,
        help="Explicit liger_kernel_config entries, e.g. 'geglu=true,swiglu=true'. The only way to "
        "measure what a parallelism force-off costs: swiglu/geglu yield to an explicit request, so "
        "this re-enables the fused MLP kernel that EP disables by default.",
    )
    parser.add_argument("--lora_r", type=int, default=0, help="LoRA rank (0 = full fine-tuning)")
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_torch_fused",
        help="Transformers optimizer name (for example adamw_torch_fused or sgd)",
    )
    parser.add_argument(
        "--lora_experts",
        action="store_true",
        help="Target the MoE expert FFNs (gate/up/down) with native grouped EP-LoRA (no attention adapters)",
    )
    parser.add_argument(
        "--lora_all_linear",
        action="store_true",
        help="Target attention (q/k/v/o, PEFT) AND the expert FFNs (native grouped EP-LoRA)",
    )
    parser.add_argument("--padding_free", action="store_true", help="Use padding-free Flash Attention 2 collator")
    parser.add_argument("--packing", action="store_true", help="Use sequence packing collator")
    parser.add_argument(
        "--no_gc", action="store_true", help="Disable gradient checkpointing (more memory, no GC recompute pass)"
    )
    parser.add_argument(
        "--no_grouped_gemm",
        action="store_true",
        help="Disable F.grouped_mm for expert compute (per-expert loop instead)",
    )
    parser.add_argument(
        "--fp32_non_ep_params",
        action="store_true",
        help="Partial fp32: store non-EP params (attention/router/lm_head/norms) in fp32 master "
        "(12 B/param), EP experts stay bf16. Compute stays bf16 via autocast.",
    )
    parser.add_argument(
        "--fp32_experts",
        action="store_true",
        help="Store EP expert weights in fp32 master (compute stays bf16 via autocast). Combine with "
        "--fp32_non_ep_params for a full-fp32-master MoE under the DTensor-aware AdamWBF16 path.",
    )
    parser.add_argument(
        "--fp32_router",
        action="store_true",
        help="Store EP router weights in fp32 master (compute stays bf16 via autocast).",
    )
    parser.add_argument(
        "--fp32_grad_reduce",
        action="store_true",
        help="Reduce gradients across ranks in fp32 while keeping bf16 master weights (6 B/param).",
    )
    parser.add_argument(
        "--fp32_optimizer",
        action="store_true",
        help="Full fp32 master weights (disable AdamWBF16 — standard fp32-state AdamW, 12 B/param).",
    )
    parser.add_argument(
        "--fp32",
        action="store_true",
        help="True fp32 training: load the model in fp32 and disable bf16 autocast (compute runs in "
        "fp32, not just storage). MFU is then measured against the fp32 Tensor Core peak. Forces "
        "attn_implementation=sdpa when unset (FA4/FA3 are bf16/fp16-only). Refused under EP — "
        "DeepEP's dispatch buffer is sized for 2-byte tokens — so pair it with --ep 1.",
    )
    parser.add_argument(
        "--lowp_precision",
        type=str,
        default="bf16",
        choices=list(LOWP_PRECISIONS),
        help="Low-precision GEMM compute for the dense MLP and MoE experts; master weights and "
        "checkpoints stay bf16. 'bf16' = off, 'fp8' = mxfp8, 'fp4' = nvfp4, 'mxfp4' = fast fp4.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Attach torch.profiler (kernel-level breakdown for bottleneck analysis); "
        "writes Chrome trace + *.top_ops.txt to --profile_dir on rank 0. "
        "NOTE: CUPTI does not complete even one step when FlashAttention-4 "
        "(CuTe DSL JIT) is active on a multi-GPU EP run — the first step crawls. "
        "For a kernel trace on this stack, add --attn_implementation sdpa, drop to "
        "1 GPU, or profile a smaller model; the roofline probe "
        "(benchmark_roofline.py) is the better tool for the MoE memory/compute "
        "bound question.",
    )
    parser.add_argument(
        "--profile_dir",
        type=str,
        default=torch_trace_dir(),
        help="Output dir for torch.profiler artifacts",
    )
    parser.add_argument(
        "--comm_profile",
        action="store_true",
        help="Attribute EP step time to DeepEP all-to-all (dispatch+combine) vs "
        "local expert compute via the PerformanceMonitor's per-phase CUDA "
        "timers. Serialized/synchronized — measures the comm FRACTION, not "
        "throughput. Prints a dispatch/compute/combine breakdown on rank 0.",
    )
    parser.add_argument(
        "--ep_scope",
        type=str,
        default="node",
        choices=["auto", "node", "global"],
        help="EP group scope: 'node' (NVLink-local), 'global' (cross-NVLink-domain, needs RDMA), "
        "or 'auto'. Use 'global' for cross-node EP spanning nodes.",
    )
    parser.add_argument("--cp", type=int, default=1, help="Context parallel size (EP+CP, node-local).")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size (attention, node-local).")
    parser.add_argument(
        "--etp", type=int, default=1, help="Expert tensor parallel size (expert FFN sharding, node-local)."
    )
    parser.add_argument(
        "--hsdp",
        action="store_true",
        help="Hybrid Sharded Data Parallel: shard non-expert params within the NVLink domain, "
        "replicate across domains (one gradient all-reduce crosses RDMA per step).",
    )
    parser.add_argument(
        "--ep_buffer_backend",
        type=str,
        default="auto",
        choices=["auto", "elastic", "legacy"],
        help="DeepEP transport: elastic (= auto, V2 ElasticBuffer, streams arbitrary length) / legacy "
        "(V1 CUDA-IPC, intranode, numerically identical).",
    )
    parser.add_argument(
        "--no_fsdp_shard_ep1_experts",
        action="store_true",
        help="EP=1 only: replicate the experts instead of FSDP-sharding them. Sharding is the "
        "production default (and required under PP); replication is the higher-memory variant.",
    )
    parser.add_argument(
        "--zero3",
        action="store_true",
        help="FSDP2 ZeRO-3 (reshard_after_forward=True) — params resharded after forward, lowest memory. "
        "Default off = ZeRO-2 (reshard_after_forward=False), params kept gathered fwd->bwd.",
    )
    args = parser.parse_args()

    # Must be set BEFORE the EP layers are constructed (in load_distributed_model), which read
    # it at __init__ to decide whether to wrap dispatch/compute/combine with CUDA-event timers.
    if args.comm_profile:
        os.environ["HALO_EP_PERF_PROFILE"] = "1"

    # --- Distributed setup ---
    rank, world_size, local_rank = init_distributed()
    PartialState()

    # --- Resolve model config ---
    model_cfg = MODEL_CONFIGS[args.model]
    model_name = args.model_path or model_cfg["hf_name"]
    full_params = model_cfg["full_params"]
    num_experts = model_cfg["num_experts"]
    top_k = model_cfg["top_k"]
    revision = model_cfg.get("revision")
    ep_size = args.ep

    # --- Resolve LoRA targeting ---
    # lora_r>0 enables LoRA. Default targets attention only; --lora_experts targets only the MoE
    # expert FFNs via native grouped EP-LoRA; --lora_all_linear targets both. Expert adapters are
    # built inside each EP layer at construction (set on parallelism_config below, before load);
    # attention adapters go through the standard PEFT path.
    lora_r = args.lora_r
    lora_on = lora_r > 0
    want_experts = lora_on and (args.lora_experts or args.lora_all_linear)
    want_attn = lora_on and (args.lora_all_linear or not args.lora_experts)
    if lora_on:
        targets = "+".join([t for t, on in (("attn", want_attn), ("experts", want_experts)) if on])
        peft_str = f"LoRA r={lora_r} ({targets})"
    else:
        peft_str = "Full FT"

    log(f"\n{'=' * 70}")
    log(f"  MoE SFT Benchmark (EP={ep_size}, {peft_str})")
    log(f"  Model: {model_name} ({full_params / 1e9:.1f}B params, {num_experts} experts, top_k={top_k})")
    log(f"  Sequence length: {args.seq}, Steps: {args.steps}, Warmup: {args.warmup}")
    log(f"  World size: {world_size}, EP size: {ep_size}")
    log(f"  Batch size: {args.batch_size}")
    log(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
    collator_mode = "padding_free" if args.padding_free else ("packing" if args.packing else "standard")
    log(f"  Attention: {args.attn_implementation}, Liger: {not args.no_liger}")
    log(f"  Collator: {collator_mode}")
    log(f"{'=' * 70}")

    if world_size < ep_size:
        log(f"\nERROR: EP size ({ep_size}) exceeds world size ({world_size})")
        dist.destroy_process_group()
        return 1

    # Isolate HF cache
    output_dir, cache_dir = setup_cache_dirs("bench_ep", rank)

    try:
        # --- Ensure model is downloaded ---
        log("\n[1/5] Ensuring model is downloaded...")
        ensure_model_downloaded(model_name, rank, revision=revision)

        # --- Load tokenizer ---
        log("\n[2/5] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, revision=revision)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # --- Create dataset ---
        log(f"\n[3/5] Creating dummy dataset ({args.num_samples} samples, seq={args.seq})...")
        train_dataset = create_benchmark_dataset(tokenizer, args.seq, args.num_samples)
        log(f"  Dataset created: {len(train_dataset)} samples")

        # --- Load model with EP ---
        log(f"\n[4/5] Loading model with EP={ep_size}...")
        log(f"  GPU memory before load: {gpu_mem_gb():.1f}GB")

        parallelism_config = ParallelismConfig(
            ep_size=ep_size,
            cp_size=args.cp,
            tp_size=args.tp,
            expert_tp_size=args.etp,
            ep_scope=args.ep_scope,
            pp_size=args.pp,
            pp_microbatches=args.pp_microbatches,
            **pp_topology_kwargs(args.pp, world_size),
            use_hsdp=args.hsdp,
            use_grouped_gemm=not args.no_grouped_gemm,
            fp32_non_ep_params=args.fp32_non_ep_params,
            fp32_grad_reduce=args.fp32_grad_reduce,
            ep_fp32_experts=args.fp32_experts,
            ep_fp32_router=args.fp32_router,
            ep_buffer_backend=args.ep_buffer_backend,
            fsdp_shard_ep1_experts=not args.no_fsdp_shard_ep1_experts,
            fsdp_reshard_after_forward=args.zero3,
            lowp_precision=args.lowp_precision,
        )
        # Native grouped EP-LoRA on the expert FFNs — each EP layer reads this at construction.
        if want_experts:
            parallelism_config.expert_lora = ExpertLoraSpec(
                r=lora_r,
                alpha=2 * lora_r,
                dropout=0.0,
                projections=frozenset({"gate", "up", "down"}),
            )
        if args.fp32:
            precision = "fp32-compute (no autocast)"
        elif args.fp32_optimizer:
            precision = "fp32-master"
        elif args.fp32_non_ep_params and args.fp32_experts:
            precision = "full-fp32-master(AdamWBF16)"
        elif args.fp32_non_ep_params or args.fp32_experts:
            precision = "partial-fp32"
        else:
            precision = "bf16-master"
        log(
            f"  Parallelism: {parallelism_config.mode_string} (grouped_gemm={not args.no_grouped_gemm}, "
            f"master={precision}, lowp={args.lowp_precision}, ZeRO-{3 if args.zero3 else 2})"
        )

        liger_kernel_config = None
        if getattr(args, "fused_linear_ce", False):
            liger_kernel_config = {
                "cross_entropy": False,
                "fused_linear_cross_entropy": True,
            }
        if args.liger_config:
            liger_kernel_config = {**(liger_kernel_config or {}), **args.liger_config}

        # True fp32 training (--fp32): fp32 params + no autocast. FA4/FA3 are bf16/fp16-only,
        # so fall back to SDPA (which supports fp32) unless the user pinned an attn impl.
        attn_impl = args.attn_implementation
        if args.fp32 and attn_impl is None:
            attn_impl = "sdpa"
        model, _ = load_distributed_model(
            model_name_or_path=model_name,
            parallelism_config=parallelism_config,
            dtype=torch.float32 if args.fp32 else torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=attn_impl,
            use_liger_kernel=not args.no_liger,
            liger_kernel_config=liger_kernel_config,
            revision=revision,
        )

        param_count = sum(p.numel() for p in model.parameters())
        log(f"  Model loaded: {param_count / 1e9:.2f}B parameters (local)")
        log(f"  GPU memory after load: {gpu_mem_gb():.1f}GB")

        # --- Adapter run: freeze base, then restore the trainable adapters ---
        # Expert adapters are plain grouped parameters built at load; PEFT (attention) is applied
        # by the trainer below. Freeze everything, re-enable the native expert adapters here, and
        # re-enable them once more after trainer init (get_peft_model re-freezes the base).
        peft_config = None
        if want_attn:
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=2 * lora_r,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            )

        def _reenable_expert_lora(m):
            for module in m.modules():
                if isinstance(module, EPMoELayerBase):
                    for attr in getattr(module, "_expert_lora_attrs", ()):
                        getattr(module, f"{attr}_lora_A").requires_grad_(True)
                        getattr(module, f"{attr}_lora_B").requires_grad_(True)

        if lora_on:
            for p in model.parameters():
                p.requires_grad = False
            if want_experts:
                _reenable_expert_lora(model)

        # --- Setup EfficiencyCallback ---
        # Pass the REAL cp/tp/etp sizes so per-GPU tokens/s and cluster throughput come from the actual
        # parallel layout rather than the token-count sanity-clamp fallback: under CP the Trainer counts
        # the full (pre-split) sequence, so the callback divides per-GPU tokens by cp_size; under TP/ETP
        # the dp_actual denominator needs the group size. Matches the production callback-wiring path.
        efficiency_cb = EfficiencyCallback(
            parallelism_config,
            n_warmup_steps=args.warmup,
            num_full_model_params=full_params,
        )

        # Global batch arithmetic (printed so baseline-vs-PP comparisons can assert equal
        # tokens/step): dp = world / (pp x max(tp, cp, etp)); EP is orthogonal to DP and a whole
        # pipeline chain is one stream.
        dp_size = parallelism_config.data_parallel_size
        global_seqs = dp_size * args.batch_size * args.grad_accum
        log(
            f"  Global batch: dp={dp_size} x bs={args.batch_size} x grad_accum={args.grad_accum} "
            f"= {global_seqs} seqs/step ({global_seqs * args.seq:,} tokens/step)"
        )

        # --- SFT Config ---
        sft_config = SFTConfig(
            output_dir=output_dir,
            max_steps=args.steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=2e-5,
            optim=args.optim,
            bf16=not args.fp32,
            gradient_checkpointing=not args.no_gc,
            # Pass the real flag and let the mixin defer it across TRL's __init__ then restore it
            # (_init_distributed_config / _setup_distributed_modes), exactly as production does.
            # Hardcoding False here left nothing to restore, so TRL's entropy guard stayed off and
            # compute_loss dereferenced outputs.logits — None on every family whose Liger patch
            # fuses the head into the loss (GLM-4 MoE Lite, Zaya default fused_linear_cross_entropy).
            use_liger_kernel=not args.no_liger,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            max_length=args.seq,
            padding_free=args.padding_free,
            packing=args.packing,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            ddp_find_unused_parameters=True,
            fsdp="",
            include_num_input_tokens_seen=True,
        )

        # FusedLinearCE: logits are None, need use_liger_kernel=True for TRL's
        # entropy guard (skips logits access). The mixin defers the flag to prevent
        # TRL from re-applying Liger to EP-wrapped modules.
        if liger_kernel_config and liger_kernel_config.get("fused_linear_cross_entropy"):
            sft_config.use_liger_kernel = True
            sft_config.liger_kernel_config = liger_kernel_config

        # --- Callbacks (+ optional torch.profiler for kernel-level bottleneck analysis) ---
        callbacks = [efficiency_cb]
        if args.profile:
            # Skip warmup + 1, then record 3 steady-state steps. The active window must fit
            # inside --steps; profiling needs steps >= warmup + 5.
            callbacks.append(
                TorchProfilerCallback(
                    output_dir=args.profile_dir,
                    wait=max(args.warmup, 1) + 1,
                    warmup=1,
                    active=3,
                    ranks="0",
                    # CUDA-kernel-only trace: with_stack / profile_memory / record_shapes and the
                    # CPU op activity each add heavy per-op instrumentation that starves an 8-rank
                    # EP run (GPUs idle, first step crawls). Disabling them keeps the top-ops kernel
                    # table (the bottleneck signal) while letting steps run at near-normal speed.
                    record_shapes=False,
                    profile_memory=False,
                    with_stack=False,
                    cpu_activity=False,
                    label=f"ep{ep_size}_s{args.seq}{'_nogc' if args.no_gc else ''}",
                )
            )
            log(f"  Profiler ON → {args.profile_dir} (Chrome trace + *.top_ops.txt on rank 0)")

        # --- Create trainer ---
        log("\n[5/5] Starting benchmark training...")
        trainer_kwargs = {
            "model": model,
            "args": sft_config,
            "train_dataset": train_dataset,
            "processing_class": tokenizer,
            "parallelism_config": parallelism_config,
            "callbacks": callbacks,
            # False forces standard fp32-state AdamW (full fp32 master); None keeps the bf16=True auto-AdamWBF16 default
            "bf16_optimizer": False if args.fp32_optimizer else None,
        }
        if args.pp > 1:
            # No balancing callback runs in this benchmark, but the default 'auto' resolves to the
            # aux-loss strategy for several families and the PP split rejects that (a stage severs
            # the HF aux-loss path). 'none' is compute-identical here.
            trainer_kwargs["moe_balancing"] = "none"
        if peft_config is not None:
            trainer_kwargs["peft_config"] = peft_config
        trainer = DistributedSFTTrainer(**trainer_kwargs)

        # get_peft_model (attention PEFT) re-freezes the base, including the native expert adapters;
        # restore them so the expert FFNs actually train.
        if want_experts:
            _reenable_expert_lora(trainer.model)

        if lora_on:
            trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in trainer.model.parameters())
            log(f"  Trainable: {trainable / 1e6:.0f}M / {total / 1e6:.0f}M ({100 * trainable / max(total, 1):.2f}%)")

        barrier()
        train_result = trainer.train()

        # --- Report ---
        log(f"\n{'=' * 70}")
        log(f"  BENCHMARK RESULTS (EP={ep_size}, {peft_str})")
        log(f"{'=' * 70}")
        log(f"  Training loss: {train_result.training_loss:.4f}")
        log(f"  Steps completed: {train_result.global_step}")
        log("\n" + format_benchmark_report(efficiency_cb))
        pp_tag = f"_pp{args.pp}" if args.pp > 1 else ""
        emit_benchmark(f"sft_ep_{args.model}_ep{ep_size}{pp_tag}_s{args.seq}", efficiency_cb)

        if args.comm_profile and rank == 0:
            stats = get_performance_monitor().stats
            disp = stats["ep.dispatch"].avg_time if "ep.dispatch" in stats else 0.0
            comp = stats["ep.expert_compute"].avg_time if "ep.expert_compute" in stats else 0.0
            comb = stats["ep.combine"].avg_time if "ep.combine" in stats else 0.0
            total = disp + comp + comb
            if total > 0:
                comm = disp + comb
                log("\n  --- EP step attribution (per MoE layer, serialized CUDA-event timing) ---")
                log(f"  dispatch (all-to-all) : {disp * 1e3:7.3f} ms  ({100 * disp / total:4.1f}%)")
                log(f"  expert compute (GEMM) : {comp * 1e3:7.3f} ms  ({100 * comp / total:4.1f}%)")
                log(f"  combine  (all-to-all) : {comb * 1e3:7.3f} ms  ({100 * comb / total:4.1f}%)")
                log(
                    f"  => communication      : {comm * 1e3:7.3f} ms  ({100 * comm / total:4.1f}%)  "
                    f"(seq={args.seq}, ep={ep_size})"
                )
        log(f"{'=' * 70}\n")

    except Exception as e:
        failed = True
        log(f"\nBENCHMARK FAILED: {e}")
        if rank == 0:
            import traceback

            traceback.print_exc()

    finally:
        if "trainer" in locals() and hasattr(trainer, "cleanup_ep"):
            trainer.cleanup_ep()
        cleanup_memory()
        cleanup_dirs(output_dir, cache_dir)
        barrier()
        teardown_distributed()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
