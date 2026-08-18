#!/usr/bin/env python
"""
Dense model SFT MFU/TFLOPS benchmark.

Measures Model FLOPS Utilization (MFU), achieved TFLOPS, and tokens/sec
for dense (non-MoE) models using DistributedSFTTrainer with EfficiencyCallback.

Supports modes: FSDP (default), TP, LoRA, QLoRA.

Usage:
    # FSDP mode, default model (Qwen3-8B), seq_len=4096, 10 steps
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/benchmark_sft_dense.py --seq 4096 --steps 10

    # TP mode (tp=2) with Qwen3-8B
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/benchmark_sft_dense.py --seq 4096 --steps 10 --tp 2

    # LoRA r=64, attention-only targets
    torchrun --nproc_per_node=1 \
        tests/gpu/profiling/benchmark_sft_dense.py --model qwen3-8b --seq 16384 --lora_r 64

    # LoRA r=64, all linear targets
    torchrun --nproc_per_node=1 \
        tests/gpu/profiling/benchmark_sft_dense.py --model qwen3-8b --seq 16384 --lora_r 64 --lora_all_linear

    # QLoRA r=64 (4-bit base model), all linear targets
    torchrun --nproc_per_node=1 \
        tests/gpu/profiling/benchmark_sft_dense.py --model qwen3-8b --seq 16384 --lora_r 64 --lora_all_linear --qlora

    # FusedLinearCrossEntropy with QLoRA
    torchrun --nproc_per_node=1 \
        tests/gpu/profiling/benchmark_sft_dense.py --model qwen3-8b --seq 32768 --lora_r 32 --lora_all_linear --qlora --fused_linear_ce

    # Disable Liger kernels
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/benchmark_sft_dense.py --seq 4096 --no_liger
"""

import sys

import torch
from accelerate import PartialState
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig

from src.callbacks.efficiency import EfficiencyCallback
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.kernels.liger.orchestrator import apply_liger_kernel_for_direct_loading
from src.trainers.sft import DistributedSFTTrainer
from tests.common.benchmark_args import create_benchmark_parser, pp_topology_kwargs, resolve_benchmark_attn
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


def main() -> int:
    failed = False
    parser = create_benchmark_parser(
        description="Dense model SFT MFU/TFLOPS benchmark",
        require_ep=False,
    )
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel size")
    parser.add_argument(
        "--fused_linear_ce", action="store_true", help="Use FusedLinearCrossEntropy instead of CrossEntropy"
    )
    parser.add_argument("--no_grad_checkpoint", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument("--padding_free", action="store_true", help="Use padding-free Flash Attention 2 collator")
    parser.add_argument("--packing", action="store_true", help="Use sequence packing collator")
    parser.add_argument("--lora_r", type=int, default=0, help="LoRA rank (0 = full fine-tuning)")
    parser.add_argument(
        "--lora_all_linear", action="store_true", help="Target all linear layers (default: attention only)"
    )
    parser.add_argument("--qlora", action="store_true", help="Enable 4-bit QLoRA (requires --lora_r > 0)")
    args = parser.parse_args()

    # --- Distributed setup ---
    rank, world_size, local_rank = init_distributed()
    PartialState()

    # --- Resolve model config ---
    model_cfg = MODEL_CONFIGS[args.model]
    model_name = args.model_path or model_cfg["hf_name"]
    full_params = model_cfg["full_params"]
    tp_size = args.tp
    pp_size = args.pp

    lora_r = args.lora_r
    lora_all_linear = args.lora_all_linear
    qlora = args.qlora

    mode_str = f"TP={tp_size}" if tp_size > 1 else "FSDP"
    if pp_size > 1:
        mode_str = f"PP={pp_size} x {mode_str}"
    if lora_r > 0:
        targets_str = "all_linear" if lora_all_linear else "attn_only"
        peft_str = f"QLoRA r={lora_r} {targets_str}" if qlora else f"LoRA r={lora_r} {targets_str}"
    else:
        peft_str = "Full FT"

    log(f"\n{'=' * 70}")
    log(f"  Dense SFT Benchmark ({mode_str}, {peft_str})")
    log(f"  Model: {model_name} ({full_params / 1e9:.1f}B params)")
    log(f"  Sequence length: {args.seq}, Steps: {args.steps}, Warmup: {args.warmup}")
    log(f"  World size: {world_size}, Batch size: {args.batch_size}")
    log(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
    gc_enabled = not getattr(args, "no_grad_checkpoint", False)
    collator_mode = "padding_free" if args.padding_free else ("packing" if args.packing else "standard")
    log(f"  Attention: {args.attn_implementation}, Liger: {not args.no_liger}, GradCheckpoint: {gc_enabled}")
    log(f"  Collator: {collator_mode}, PEFT: {peft_str}")
    log(f"{'=' * 70}")

    # Isolate HF cache
    output_dir, cache_dir = setup_cache_dirs("bench_dense", rank)

    try:
        # --- Ensure model is downloaded ---
        log("\n[1/5] Ensuring model is downloaded...")
        ensure_model_downloaded(model_name, rank)

        # --- Load tokenizer ---
        log("\n[2/5] Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # --- Create dataset ---
        log(f"\n[3/5] Creating dummy dataset ({args.num_samples} samples, seq={args.seq})...")
        train_dataset = create_benchmark_dataset(tokenizer, args.seq, args.num_samples)
        log(f"  Dataset created: {len(train_dataset)} samples")

        # --- Build Liger config ---
        use_liger = not args.no_liger
        liger_kernel_config = None
        if getattr(args, "fused_linear_ce", False):
            liger_kernel_config = {
                "cross_entropy": False,
                "fused_linear_cross_entropy": True,
            }

        # --- SFT Config (built before model loading for Liger pre-application) ---
        sft_config = SFTConfig(
            output_dir=output_dir,
            max_steps=args.steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=2e-5,
            bf16=True,
            gradient_checkpointing=not getattr(args, "no_grad_checkpoint", False),
            use_liger_kernel=use_liger,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            max_length=args.seq,
            padding_free=args.padding_free,
            packing=args.packing,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            ddp_find_unused_parameters=False,
            include_num_input_tokens_seen=True,
        )
        if liger_kernel_config:
            sft_config.liger_kernel_config = liger_kernel_config

        # --- Build PEFT config ---
        peft_config = None
        if lora_r > 0:
            attn_targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
            all_linear_targets = attn_targets + ["gate_proj", "up_proj", "down_proj"]
            target_modules = all_linear_targets if lora_all_linear else attn_targets
            peft_config = LoraConfig(
                r=lora_r,
                lora_alpha=2 * lora_r,
                lora_dropout=0.0,
                bias="none",
                task_type="CAUSAL_LM",
                target_modules=target_modules,
            )

        # --- Load model ---
        log(f"\n[4/5] Loading model ({mode_str}, {peft_str})...")
        log(f"  GPU memory before load: {gpu_mem_gb():.1f}GB")

        pp_kwargs = {
            "pp_size": pp_size,
            "pp_microbatches": args.pp_microbatches,
            **pp_topology_kwargs(pp_size, world_size),
        }
        if tp_size > 1:
            parallelism_config = ParallelismConfig(tp_size=tp_size, **pp_kwargs)
            model, _ = load_distributed_model(
                model_name_or_path=model_name,
                parallelism_config=parallelism_config,
                dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation=args.attn_implementation,
                use_liger_kernel=use_liger,
                liger_kernel_config=liger_kernel_config,
            )
            sft_config.use_liger_kernel = False  # Applied via load_distributed_model
        else:
            parallelism_config = ParallelismConfig(**pp_kwargs)
            # Apply Liger BEFORE model loading so RMSNorm/SwiGLU patches take effect
            apply_liger_kernel_for_direct_loading(model_name, sft_config, trust_remote_code=True)

            load_kwargs = {
                "dtype": torch.bfloat16,
                "trust_remote_code": True,
                "attn_implementation": resolve_benchmark_attn(model_name, args.attn_implementation),
            }
            if qlora:
                load_kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.bfloat16,
                    bnb_4bit_use_double_quant=True,
                )
            model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)

        # FusedLinearCE: logits are None, need use_liger_kernel=True for TRL's
        # entropy guard (skips logits access). The mixin defers the flag to
        # prevent TRL from re-applying Liger to EP-wrapped modules.
        if liger_kernel_config and liger_kernel_config.get("fused_linear_cross_entropy"):
            sft_config.use_liger_kernel = True
            sft_config.liger_kernel_config = liger_kernel_config

        # Under PP the Liger class patches are already applied (direct loading above); TRL's
        # instance re-application would run on the pipeline stage module, so defer the flag —
        # the same handling as the TP branch.
        if pp_size > 1:
            sft_config.use_liger_kernel = False

        param_count = sum(p.numel() for p in model.parameters())
        log(f"  Model loaded: {param_count / 1e9:.2f}B parameters")
        if peft_config:
            log(f"  LoRA: r={lora_r}, targets={'all_linear' if lora_all_linear else 'attn_only'}, QLoRA={qlora}")
        log(f"  GPU memory after load: {gpu_mem_gb():.1f}GB")

        # --- Setup EfficiencyCallback ---
        efficiency_cb = EfficiencyCallback(
            parallelism_config,
            n_warmup_steps=args.warmup,
            num_full_model_params=full_params,
        )

        # Global batch arithmetic (printed so baseline-vs-PP comparisons can assert equal
        # tokens/step): dp = world / (pp x max(tp, cp, etp)); a whole pipeline chain is one stream.
        dp_size = parallelism_config.data_parallel_size
        global_seqs = dp_size * args.batch_size * args.grad_accum
        log(
            f"  Global batch: dp={dp_size} x bs={args.batch_size} x grad_accum={args.grad_accum} "
            f"= {global_seqs} seqs/step ({global_seqs * args.seq:,} tokens/step)"
        )

        # --- Create trainer ---
        log("\n[5/5] Starting benchmark training...")
        trainer_kwargs = {
            "model": model,
            "args": sft_config,
            "train_dataset": train_dataset,
            "processing_class": tokenizer,
            "parallelism_config": parallelism_config,
            "callbacks": [efficiency_cb],
        }
        if peft_config:
            trainer_kwargs["peft_config"] = peft_config

        trainer = DistributedSFTTrainer(**trainer_kwargs)

        # Report trainable params after trainer init (PEFT wrapping happens there)
        trainable = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in trainer.model.parameters())
        log(f"  Trainable: {trainable / 1e6:.0f}M / {total / 1e6:.0f}M ({100 * trainable / total:.1f}%)")

        barrier()
        train_result = trainer.train()

        # --- Report ---
        log(f"\n{'=' * 70}")
        log(f"  BENCHMARK RESULTS ({mode_str}, {peft_str})")
        log(f"{'=' * 70}")
        log(f"  Training loss: {train_result.training_loss:.4f}")
        log(f"  Steps completed: {train_result.global_step}")
        log(f"  Trainable: {trainable / 1e6:.0f}M / {total / 1e6:.0f}M ({100 * trainable / total:.1f}%)")
        log("\n" + format_benchmark_report(efficiency_cb))
        pp_tag = f"_pp{pp_size}" if pp_size > 1 else ""
        emit_benchmark(f"sft_dense_{args.model}_tp{tp_size}{pp_tag}_s{args.seq}", efficiency_cb)
        log(f"{'=' * 70}\n")

    except Exception as e:
        failed = True
        log(f"\nBENCHMARK FAILED: {e}")
        if rank == 0:
            import traceback

            traceback.print_exc()

    finally:
        cleanup_memory()
        cleanup_dirs(output_dir, cache_dir)
        barrier()
        teardown_distributed()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
