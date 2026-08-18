#!/usr/bin/env python
"""
MoE model SFT benchmark with Expert Parallelism + Context Parallelism (EP+CP).

Measures Model FLOPS Utilization (MFU), Sparse MFU (S-MFU), achieved TFLOPS,
and tokens/sec for MoE models using DistributedSFTTrainer with EfficiencyCallback,
Expert Parallelism (DeepEP all-to-all), and Context Parallelism (Ulysses attention).

EP+CP mode: experts are distributed across GPUs via EP, and sequences are split
across GPUs within CP groups via Ulysses attention. EP is orthogonal to data
parallelism -- only CP reduces the effective DP size.

Note: seq_len must be divisible by cp_size (Ulysses splits sequences evenly).

Usage:
    # EP=2 + CP=2 (orthogonal), seq_len=4096, 10 steps
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/benchmark_sft_ep_cp.py --ep 2 --cp 2 --seq 4096 --steps 10

    # Long sequences benefit most from CP
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/benchmark_sft_ep_cp.py --ep 2 --cp 2 --seq 16384 --steps 10

    # 8 GPUs, EP=8, CP=8 (orthogonal mode)
    torchrun --nproc_per_node=8 \
        tests/gpu/profiling/benchmark_sft_ep_cp.py --ep 8 --cp 8 --seq 32768 --steps 10

    # Quick test with fewer steps
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/benchmark_sft_ep_cp.py --ep 2 --cp 2 --seq 4096 --steps 5 --warmup 1

    # Custom model path
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/benchmark_sft_ep_cp.py --ep 2 --cp 2 \
        --model_path /path/to/moe/model --seq 8192

    # Disable Liger kernels
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/benchmark_sft_ep_cp.py --ep 2 --cp 2 --seq 4096 --no_liger
"""

import sys

import torch
import torch.distributed as dist
from accelerate import PartialState
from transformers import AutoTokenizer
from trl import SFTConfig

from src.callbacks.efficiency import EfficiencyCallback
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.sft import DistributedSFTTrainer
from tests.common.benchmark_args import create_benchmark_parser
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
        description="MoE SFT benchmark with EP + Context Parallelism",
    )
    parser.add_argument("--cp", type=int, required=True, help="Context parallel size")
    parser.add_argument("--packing", action="store_true", help="Use sequence packing collator")
    args = parser.parse_args()

    # --- Distributed setup ---
    rank, world_size, local_rank = init_distributed()
    PartialState()

    # --- Resolve model config ---
    model_cfg = MODEL_CONFIGS[args.model]
    model_name = args.model_path or model_cfg["hf_name"]
    full_params = model_cfg["full_params"]
    num_experts = model_cfg["num_experts"]
    top_k = model_cfg["top_k"]
    ep_size = args.ep
    cp_size = args.cp

    # Ensure seq_len is divisible by cp_size
    seq_len = args.seq
    if seq_len % cp_size != 0:
        seq_len = (seq_len // cp_size) * cp_size
        log(f"  Adjusted seq_len to {seq_len} (divisible by cp_size={cp_size})")

    log(f"\n{'=' * 70}")
    log(f"  MoE SFT Benchmark (EP={ep_size} + CP={cp_size})")
    log(f"  Model: {model_name} ({full_params / 1e9:.1f}B params, {num_experts} experts, top_k={top_k})")
    log(f"  Sequence length: {seq_len}, Steps: {args.steps}, Warmup: {args.warmup}")
    log(f"  World size: {world_size}, EP size: {ep_size}, CP size: {cp_size}")
    log(f"  Effective per-GPU seq: {seq_len // cp_size} tokens")
    log(f"  Batch size: {args.batch_size}")
    log(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
    log(f"  Attention: {args.attn_implementation}, Liger: {not args.no_liger}")
    log(f"{'=' * 70}")

    if world_size < max(ep_size, cp_size):
        log(f"\nERROR: World size ({world_size}) must be >= max(ep={ep_size}, cp={cp_size})")
        dist.destroy_process_group()
        return 1

    # Isolate HF cache
    output_dir, cache_dir = setup_cache_dirs("bench_epcp", rank)

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
        log(f"\n[3/5] Creating dummy dataset ({args.num_samples} samples, seq={seq_len})...")
        train_dataset = create_benchmark_dataset(tokenizer, seq_len, args.num_samples)
        log(f"  Dataset created: {len(train_dataset)} samples")

        # --- Load model with EP+CP ---
        log(f"\n[4/5] Loading model with EP={ep_size}, CP={cp_size}...")
        log(f"  GPU memory before load: {gpu_mem_gb():.1f}GB")

        parallelism_config = ParallelismConfig(ep_size=ep_size, cp_size=cp_size)
        log(f"  Parallelism: {parallelism_config.mode_string}")
        log(f"  Data parallel size: {parallelism_config.data_parallel_size}")

        model, _ = load_distributed_model(
            model_name_or_path=model_name,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=args.attn_implementation,
            use_liger_kernel=not args.no_liger,
        )

        param_count = sum(p.numel() for p in model.parameters())
        log(f"  Model loaded: {param_count / 1e9:.2f}B parameters (local)")
        log(f"  GPU memory after load: {gpu_mem_gb():.1f}GB")

        # --- Setup EfficiencyCallback ---
        efficiency_cb = EfficiencyCallback(
            parallelism_config,
            n_warmup_steps=args.warmup,
            num_full_model_params=full_params,
        )

        # --- SFT Config ---
        sft_config = SFTConfig(
            output_dir=output_dir,
            max_steps=args.steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=2e-5,
            bf16=True,
            gradient_checkpointing=True,
            use_liger_kernel=False,  # Applied via load_distributed_model
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            max_length=seq_len,
            packing=args.packing,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            ddp_find_unused_parameters=True,
            fsdp="",
            include_num_input_tokens_seen=True,
        )

        # --- Create trainer ---
        log("\n[5/5] Starting benchmark training...")
        trainer = DistributedSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
            callbacks=[efficiency_cb],
        )

        barrier()
        train_result = trainer.train()

        # --- Report ---
        log(f"\n{'=' * 70}")
        log(f"  BENCHMARK RESULTS (EP={ep_size} + CP={cp_size})")
        log(f"{'=' * 70}")
        log(f"  Training loss: {train_result.training_loss:.4f}")
        log(f"  Steps completed: {train_result.global_step}")
        log("\n" + format_benchmark_report(efficiency_cb))
        emit_benchmark(f"sft_ep_cp_{args.model}_ep{ep_size}_cp{cp_size}_s{seq_len}", efficiency_cb)
        log(f"  Context parallel: each GPU processes {seq_len // cp_size} of {seq_len} tokens")
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
