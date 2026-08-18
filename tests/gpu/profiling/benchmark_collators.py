#!/usr/bin/env python
"""
Collator comparison benchmark: Standard vs Packing vs Padding-Free.

Uses variable-length data where avg sequence length << max_length to expose
padding waste. With fixed-length data all collators perform identically;
this benchmark demonstrates the throughput advantage of packing and
padding-free approaches on realistic variable-length training data.

Usage:
    # Default: Qwen3-30B-A3B EP=2, max_length=4096, avg_ratio=0.25
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/benchmark_collators.py --ep 2

    # Custom avg_ratio (0.1 = avg 10% of max_length → 90% padding waste)
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/benchmark_collators.py --ep 2 --avg_ratio 0.1

    # Single collator mode
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/benchmark_collators.py --ep 2 --mode packing

    # Dense model (FSDP, no EP)
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/benchmark_collators.py --model qwen3-8b --seq 4096
"""

import gc
import sys
import time
import traceback

import torch
import torch.distributed as dist
from accelerate import PartialState
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig

from src.callbacks.efficiency import EfficiencyCallback
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.kernels.liger.orchestrator import apply_liger_kernel_for_direct_loading
from src.trainers.sft import DistributedSFTTrainer
from tests.common.benchmark_args import create_benchmark_parser, resolve_benchmark_attn
from tests.common.datasets import create_variable_length_sft_dataset
from tests.common.distributed import (
    ensure_model_downloaded,
    init_distributed,
    teardown_distributed,
)
from tests.common.models import MODEL_CONFIGS
from tests.common.reporting import emit_benchmark, format_benchmark_report
from tests.common.utils import gpu_mem_gb, log

# Constants

ALL_MODES = ["standard", "packing", "padding_free"]


# Benchmark Logic


def run_collator_mode(
    mode: str,
    args,
    model_cfg: dict,
    tokenizer,
    dataset,
    rank: int,
) -> dict | None:
    """Run a single collator benchmark mode.

    Loads the model fresh, configures the collator per mode, trains for
    the specified steps, and returns metrics from EfficiencyCallback.
    """
    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"  COLLATOR: {mode}")
        print(f"{'=' * 60}")

    model_name = args.model_path or model_cfg["hf_name"]
    ep_size = getattr(args, "ep", 1)
    use_liger = not args.no_liger
    output_dir = f"/tmp/bench_collator_{mode}_{ep_size}_{args.seq}"

    # --- Load model ---
    if ep_size > 1:
        parallelism_config = ParallelismConfig(ep_size=ep_size)
        model, _ = load_distributed_model(
            model_name_or_path=model_name,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=args.attn_implementation,
            use_liger_kernel=use_liger,
        )
    else:
        parallelism_config = ParallelismConfig()
        sft_config_tmp = SFTConfig(output_dir=output_dir, use_liger_kernel=use_liger)
        apply_liger_kernel_for_direct_loading(model_name, sft_config_tmp, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=resolve_benchmark_attn(model_name, args.attn_implementation),
        )

    if rank == 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  Model loaded: {trainable / 1e9:.2f}B params, {gpu_mem_gb():.1f} GB")

    # --- SFT Config ---
    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        max_steps=args.steps,
        learning_rate=2e-5,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        report_to=[],
        # padding_free without packing forbids an enforced max_length in current TRL;
        # the synthetic inputs are already generated <= args.seq, so dropping the cap is safe.
        max_length=None if mode == "padding_free" else args.seq,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=ep_size > 1,
        fsdp="" if ep_size > 1 else None,
        # Count REAL tokens (attention_mask.sum), not the padded tensor's numel.
        # A bare True counts inputs[input_ids].numel() (pads included), which credits
        # `standard` padding-as-throughput while `padding_free` counts only real tokens —
        # inverting the collator comparison. "non_padding" makes the metric mode-comparable.
        include_num_input_tokens_seen="non_padding",
        # Collator mode
        padding_free=(mode == "padding_free"),
        packing=(mode == "packing"),
        use_liger_kernel=False,
    )

    # --- Efficiency Callback ---
    efficiency_cb = EfficiencyCallback(
        parallelism_config,
        n_warmup_steps=args.warmup,
        num_full_model_params=model_cfg["full_params"],
    )

    # --- Create Trainer ---
    trainer = DistributedSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[efficiency_cb],
        parallelism_config=parallelism_config,
    )

    # --- Train ---
    barrier()
    torch.cuda.reset_peak_memory_stats()

    train_start = time.perf_counter()
    trainer.train()
    train_elapsed = time.perf_counter() - train_start

    # --- Collect metrics ---
    results = None
    if rank == 0:
        results = {
            "mode": mode,
            "avg_mfu_percent": efficiency_cb.mfu.avg_mfu_percent,
            "avg_tflops": efficiency_cb.mfu.avg_tflops_per_sec,
            "avg_tps_gpu": efficiency_cb.tps.avg_tokens_per_second,
            "avg_tps_cluster": efficiency_cb.tps.avg_cluster_tokens_per_second,
            "avg_step_time_seconds": efficiency_cb.time.avg_step_time_seconds,
            "peak_memory_gb": efficiency_cb.memory.peak_allocated_gb,
            "total_time_sec": train_elapsed,
        }

        print(f"\n  --- {mode} Results ---")
        log("\n" + format_benchmark_report(efficiency_cb))
        emit_benchmark(f"collator_{mode}_{args.model}_ep{ep_size}_s{args.seq}", efficiency_cb)
        print(f"  Total Time:   {results['total_time_sec']:.1f}s")

    # --- Cleanup ---
    if hasattr(trainer, "cleanup_ep"):
        trainer.cleanup_ep()
    barrier()
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    barrier()

    return results


def print_summary(all_results: list[dict], args, model_cfg: dict, avg_tokens: int):
    """Print comparison table of all collator modes."""
    model_name = args.model_path or model_cfg["hf_name"]
    ep_size = getattr(args, "ep", 1)
    padding_pct = (1.0 - args.avg_ratio) * 100

    print(f"\n{'=' * 80}")
    print("  COLLATOR COMPARISON")
    print(f"  Model: {model_name}")
    print(
        f"  EP={ep_size}, max_length={args.seq}, avg_tokens≈{avg_tokens} "
        f"({args.avg_ratio:.0%} of max → {padding_pct:.0f}% padding waste)"
    )
    print(f"{'=' * 80}")

    print(f"\n  {'Collator':<14} {'MFU%':>6} {'TFLOPS':>7} {'TPS/GPU':>8} {'Step(s)':>8} {'Mem GB':>7}")
    print(f"  {'-' * 14} {'-' * 6} {'-' * 7} {'-' * 8} {'-' * 8} {'-' * 7}")

    for r in all_results:
        print(
            f"  {r['mode']:<14} {r['avg_mfu_percent']:>5.1f}% {r['avg_tflops']:>7.1f} "
            f"{r['avg_tps_gpu']:>7.0f}  {r['avg_step_time_seconds']:>7.3f}s "
            f"{r['peak_memory_gb']:>6.1f}"
        )

    # Speedup vs standard
    std = next((r for r in all_results if r["mode"] == "standard"), None)
    if std and std["avg_tps_gpu"] > 0 and len(all_results) > 1:
        print("\n  Throughput vs Standard (tokens/sec per GPU):")
        for r in all_results:
            if r["mode"] == "standard":
                continue
            speedup = r["avg_tps_gpu"] / std["avg_tps_gpu"]
            print(f"    {r['mode']:<14}: {speedup:.1f}x")

    print(f"\n{'=' * 80}")


# Main


def main():
    parser = create_benchmark_parser(
        description="Collator comparison: Standard vs Packing vs Padding-Free",
        require_ep=False,
    )
    parser.add_argument(
        "--ep",
        type=int,
        default=1,
        help="Expert parallel size (default: 1 = FSDP only)",
    )
    parser.add_argument(
        "--avg_ratio",
        type=float,
        default=0.25,
        help="Average sequence length as fraction of max_length (default: 0.25)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=ALL_MODES,
        help="Run a single mode instead of all three",
    )
    args = parser.parse_args()

    # --- Distributed setup ---
    rank, world_size, local_rank = init_distributed()
    PartialState()

    model_cfg = MODEL_CONFIGS[args.model]
    model_name = args.model_path or model_cfg["hf_name"]
    ep_size = args.ep
    avg_tokens = int(args.seq * args.avg_ratio)
    padding_pct = (1.0 - args.avg_ratio) * 100

    if rank == 0:
        print(f"\n{'#' * 70}")
        print("  Collator Comparison Benchmark")
        print(f"  Model: {model_name} ({model_cfg['full_params'] / 1e9:.1f}B params)")
        print(f"  EP={ep_size}, max_length={args.seq}, batch_size={args.batch_size}")
        print(f"  Avg tokens/sample: ~{avg_tokens} ({args.avg_ratio:.0%} of max_length)")
        print(f"  Padding waste (standard): ~{padding_pct:.0f}%")
        print(f"  Steps={args.steps}, Warmup={args.warmup}")
        print(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
        print(f"{'#' * 70}")

    if ep_size > 1 and world_size < ep_size:
        if rank == 0:
            print(f"\nERROR: Need at least {ep_size} GPUs, got {world_size}")
        dist.destroy_process_group()
        return 1

    # --- Ensure model is cached ---
    ensure_model_downloaded(model_name, rank)

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Create variable-length dataset ---
    dataset = create_variable_length_sft_dataset(
        tokenizer=tokenizer,
        max_length=args.seq,
        num_samples=args.num_samples,
        avg_ratio=args.avg_ratio,
    )

    if rank == 0:
        # Show actual token distribution
        sample_lengths = []
        for i in range(min(20, len(dataset))):
            toks = len(tokenizer.encode(dataset[i]["text"]))
            sample_lengths.append(toks)
        actual_avg = sum(sample_lengths) / len(sample_lengths)
        print(f"  Dataset: {len(dataset)} samples, actual avg ≈ {actual_avg:.0f} tokens")
        print(
            f"  Sample lengths (first 20): min={min(sample_lengths)}, max={max(sample_lengths)}, avg={actual_avg:.0f}"
        )

    # --- Determine modes to run ---
    modes = [args.mode] if args.mode else ALL_MODES
    if rank == 0:
        print(f"\n  Modes to benchmark: {modes}")

    # --- Run benchmarks ---
    all_results = []
    failed = False
    for mode in modes:
        try:
            result = run_collator_mode(mode, args, model_cfg, tokenizer, dataset, rank)
            if result is not None:
                all_results.append(result)
        except Exception as e:
            failed = True
            if rank == 0:
                print(f"\n  ERROR in mode '{mode}': {e}")
                traceback.print_exc()
            gc.collect()
            torch.cuda.empty_cache()
            barrier()
            continue

    # --- Print summary ---
    if rank == 0 and all_results:
        print_summary(all_results, args, model_cfg, avg_tokens)

    # On rank 0 the per-mode results gate success; on other ranks fall back to
    # whether any mode raised (results are only collected on rank 0).
    if rank == 0 and not all_results:
        failed = True

    # --- Cleanup ---
    barrier()
    teardown_distributed()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
