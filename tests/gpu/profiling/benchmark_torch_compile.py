#!/usr/bin/env python
"""
Benchmark: Liger kernels x torch.compile 2x2 matrix on GptOss-20B MoE.

Tests four optimization combinations for SFT training with Expert Parallelism
on a MoE model, measuring MFU, TFLOPS, throughput, and memory for each:

    neither       - No Liger, no compile (true baseline)
    liger_only    - Liger kernels enabled
    compile_only  - torch.compile enabled
    liger_compile - Both Liger + compile

Usage:
    # Run all 4 modes
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/benchmark_torch_compile.py --ep 2 --seq 4096 --steps 10

    # Run a single mode
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/benchmark_torch_compile.py --ep 2 --seq 4096 --steps 10 \
        --mode liger_compile

    # Custom warmup steps
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/benchmark_torch_compile.py --ep 2 --seq 4096 --steps 12 \
        --warmup 3

Requirements:
    - 2x GPUs with >=80 GB memory each
    - DeepEP installed
    - Model: unsloth/gpt-oss-20b-BF16 (auto-downloaded)
"""

import argparse
import gc
import random
import sys
import time
import traceback

import torch
import torch.distributed as dist
from accelerate import PartialState
from datasets import Dataset
from transformers import AutoTokenizer

from src.callbacks.efficiency import EfficiencyCallback
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.sft import DistributedSFTTrainer
from tests.common.distributed import (
    ensure_model_downloaded,
    init_distributed,
    teardown_distributed,
)
from tests.common.models import DEFAULT_MODEL, MODEL_CONFIGS
from tests.common.reporting import emit_benchmark, format_benchmark_report

# Constants

ALL_MODES = ["neither", "liger_only", "compile_only", "liger_compile"]

SEED = 42
NUM_SAMPLES = 64
BATCH_SIZE = 1
LEARNING_RATE = 2e-5


# Dataset


def create_synthetic_sft_dataset(
    tokenizer,
    seq_len: int,
    num_samples: int = NUM_SAMPLES,
    seed: int = SEED,
) -> Dataset:
    """Create synthetic math SFT dataset with chat-templated text.

    Generates single-turn math conversations, applies the tokenizer's chat
    template, and returns {"text": ...} records. Completions are padded with
    filler text to approximate the target sequence length.

    Args:
        tokenizer: HuggingFace tokenizer for chat template application.
        seq_len: Target sequence length.
        num_samples: Number of samples to generate.
        seed: Random seed for reproducibility.

    Returns:
        Dataset with "text" column containing chat-templated conversations.
    """
    random.seed(seed)

    templates = [
        {
            "instruction": "What is {a} + {b}?",
            "response": "The answer is {result}. To calculate this, I added {a} and {b} together.",
            "op": "+",
        },
        {
            "instruction": "Calculate {a} * {b}.",
            "response": "{a} times {b} equals {result}. This is found by multiplying the two numbers.",
            "op": "*",
        },
        {
            "instruction": "What is {a} - {b}?",
            "response": "The result of {a} minus {b} is {result}.",
            "op": "-",
        },
        {
            "instruction": "What is the sum of {a} and {b}?",
            "response": "The sum of {a} and {b} is {result}.",
            "op": "+",
        },
    ]

    # Estimate tokens per word for filler padding
    filler_sentence = " The quick brown fox jumps over the lazy dog."
    filler_tokens_estimate = len(tokenizer.encode(filler_sentence))
    filler_words = len(filler_sentence.split())
    tokens_per_word = filler_tokens_estimate / max(filler_words, 1)

    data = []
    for _ in range(num_samples):
        template = random.choice(templates)
        a = random.randint(1, 100)
        b = random.randint(1, 100)

        if template["op"] == "+":
            result = a + b
        elif template["op"] == "*":
            result = a * b
        else:
            result = a - b

        instruction = template["instruction"].format(a=a, b=b, result=result)
        response = template["response"].format(a=a, b=b, result=result)

        # Pad response to approximate target seq_len
        base_messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": response},
        ]
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            base_text = tokenizer.apply_chat_template(
                base_messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            base_text = f"User: {instruction}\nAssistant: {response}"

        base_tokens = len(tokenizer.encode(base_text))
        remaining_tokens = max(0, seq_len - base_tokens - 20)
        filler_word_count = int(remaining_tokens / max(tokens_per_word, 1))
        filler = (filler_sentence * ((filler_word_count // filler_words) + 1))[: filler_word_count * 6]

        padded_response = response + filler
        messages = [
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": padded_response},
        ]

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )
        else:
            text = f"User: {instruction}\nAssistant: {padded_response}"

        data.append({"text": text})

    return Dataset.from_list(data)


# Benchmark Logic


def run_benchmark_mode(
    mode: str,
    args: argparse.Namespace,
    tokenizer,
    dataset: Dataset,
    rank: int,
    local_rank: int,
) -> dict | None:
    """Run a single benchmark mode.

    Loads the model fresh, configures Liger/compile per mode, trains for
    the specified number of steps, and returns metrics from EfficiencyCallback.

    Args:
        mode: One of "neither", "liger_only", "compile_only", "liger_compile".
        args: Parsed CLI arguments.
        tokenizer: HuggingFace tokenizer.
        dataset: Training dataset.
        rank: Global rank.
        local_rank: Local rank.

    Returns:
        Dict with benchmark metrics, or None on failure.
    """
    use_liger = mode in ("liger_only", "liger_compile")
    use_compile = mode in ("compile_only", "liger_compile")

    if rank == 0:
        print(f"\n{'=' * 60}")
        print(f"  MODE: {mode}")
        print(f"  Liger: {'ON' if use_liger else 'OFF'}, Compile: {'ON' if use_compile else 'OFF'}")
        print(f"{'=' * 60}")

    # --- Load model with EP (Liger applied pre-loading via load_distributed_model) ---
    model_cfg = MODEL_CONFIGS[args.model]
    model_name = args.model_path or model_cfg["hf_name"]
    parallelism_config = ParallelismConfig(ep_size=args.ep)

    model, _ = load_distributed_model(
        model_name_or_path=model_name,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        use_liger_kernel=use_liger,
    )

    if rank == 0:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        mem_gb = torch.cuda.memory_allocated() / 1e9
        print(f"  Model loaded: {trainable / 1e9:.2f}B params, {mem_gb:.1f} GB")

    # --- SFT Config ---
    output_dir = f"/tmp/bench_compile_{mode}_{args.ep}_{args.seq}"

    from trl import SFTConfig

    sft_config = SFTConfig(
        output_dir=output_dir,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        max_steps=args.steps,
        learning_rate=LEARNING_RATE,
        logging_steps=1,
        save_strategy="no",
        bf16=True,
        gradient_checkpointing=True,
        dataloader_pin_memory=False,
        report_to=[],
        include_num_input_tokens_seen=True,
        max_length=args.seq,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=True,
        fsdp="",
        # torch.compile settings (applied by DistributedTrainerMixin after FSDP)
        torch_compile=use_compile,
        torch_compile_backend="inductor",
        torch_compile_mode="reduce-overhead",
    )
    # Disable TRL's internal Liger re-application
    sft_config.use_liger_kernel = False

    # --- Efficiency Callback ---
    efficiency_callback = EfficiencyCallback(
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
        callbacks=[efficiency_callback],
        parallelism_config=parallelism_config,
    )

    if rank == 0:
        print(f"  Trainer created: EP={args.ep}, DP={parallelism_config.data_parallel_size}")
        print(f"  Training for {args.steps} steps (warmup={args.warmup})...")

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
            "liger": use_liger,
            "compile": use_compile,
            "avg_mfu_percent": efficiency_callback.mfu.avg_mfu_percent,
            "avg_smfu_percent": efficiency_callback.smfu.avg_smfu_percent,
            "avg_tflops": efficiency_callback.mfu.avg_tflops_per_sec,
            "avg_active_tflops": efficiency_callback.smfu.avg_smfu_tflops_per_sec,
            "avg_tps_gpu": efficiency_callback.tps.avg_tokens_per_second,
            "avg_tps_cluster": efficiency_callback.tps.avg_cluster_tokens_per_second,
            "avg_step_time_seconds": efficiency_callback.time.avg_step_time_seconds,
            "peak_memory_gb": efficiency_callback.memory.peak_allocated_gb,
            "local_params_b": efficiency_callback.mfu.local_params / 1e9,
            "total_time_sec": train_elapsed,
            "gpu_model": efficiency_callback.mfu.gpu_model,
        }

        print(f"\n  --- {mode} Results ---")
        print("\n" + format_benchmark_report(efficiency_callback))
        emit_benchmark(f"compile_{mode}_{args.model}_ep{args.ep}_s{args.seq}", efficiency_callback)
        print(f"  Total Time:     {results['total_time_sec']:.1f}s")

    # --- Cleanup ---
    if hasattr(trainer, "cleanup_ep"):
        trainer.cleanup_ep()
    barrier()
    del trainer
    del model
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Allow GPU memory to settle between modes
    barrier()

    return results


# Results Summary


def print_summary(all_results: list[dict], args: argparse.Namespace):
    """Print comparison table of all benchmark modes.

    Args:
        all_results: List of result dicts from each mode.
        args: Parsed CLI arguments.
    """
    model_cfg = MODEL_CONFIGS[args.model]
    model_name = args.model_path or model_cfg["hf_name"]
    print(f"\n{'=' * 80}")
    print("  SUMMARY: Liger x Compile Benchmark")
    print(f"  Model: {model_name}, EP={args.ep}, SeqLen={args.seq}, Steps={args.steps}")
    if all_results:
        print(f"  GPU: {all_results[0].get('gpu_model', 'Unknown')}")
    print(f"{'=' * 80}")

    # Header
    print(
        f"\n  {'Mode':<16} {'Liger':>6} {'Compile':>8} {'MFU%':>6} {'TFLOPS':>7} "
        f"{'Mem GB':>7} {'Step(s)':>8} {'TPS/GPU':>8}"
    )
    print(f"  {'-' * 16} {'-' * 6} {'-' * 8} {'-' * 6} {'-' * 7} {'-' * 7} {'-' * 8} {'-' * 8}")

    baseline_mfu = None
    baseline_mem = None
    baseline_step = None

    for r in all_results:
        liger_str = "ON" if r["liger"] else "OFF"
        compile_str = "ON" if r["compile"] else "OFF"

        if r["mode"] == "neither":
            baseline_mfu = r["avg_mfu_percent"]
            baseline_mem = r["peak_memory_gb"]
            baseline_step = r["avg_step_time_seconds"]

        print(
            f"  {r['mode']:<16} {liger_str:>6} {compile_str:>8} "
            f"{r['avg_mfu_percent']:>5.1f}% {r['avg_tflops']:>7.1f} "
            f"{r['peak_memory_gb']:>6.1f}  {r['avg_step_time_seconds']:>7.2f}s "
            f"{r['avg_tps_gpu']:>7.0f}"
        )

    # Relative improvements vs baseline
    if baseline_mfu is not None and baseline_mfu > 0 and len(all_results) > 1:
        print(f"\n  Relative to baseline ({all_results[0]['mode']}):")
        print(f"  {'Mode':<16} {'MFU delta':>10} {'Mem delta':>10} {'Step delta':>11}")
        print(f"  {'-' * 16} {'-' * 10} {'-' * 10} {'-' * 11}")

        for r in all_results[1:]:
            mfu_delta = r["avg_mfu_percent"] - baseline_mfu
            mfu_pct = (mfu_delta / baseline_mfu) * 100 if baseline_mfu > 0 else 0
            mem_delta = r["peak_memory_gb"] - baseline_mem if baseline_mem else 0
            step_delta_pct = (
                ((r["avg_step_time_seconds"] - baseline_step) / baseline_step) * 100
                if baseline_step and baseline_step > 0
                else 0
            )

            print(
                f"  {r['mode']:<16} {mfu_delta:>+5.1f}% ({mfu_pct:>+4.0f}%) "
                f"{mem_delta:>+6.1f} GB  {step_delta_pct:>+6.0f}% step"
            )

    print(f"\n{'=' * 80}")


# Main


def main():
    parser = argparse.ArgumentParser(
        description="Liger x torch.compile 2x2 benchmark for MoE models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, choices=list(MODEL_CONFIGS.keys()), help="Model config key"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Override model path (instead of using MODEL_CONFIGS)"
    )
    parser.add_argument("--ep", type=int, default=2, help="Expert parallel size (default: 2)")
    parser.add_argument("--seq", type=int, default=8192, help="Sequence length (default: 8192)")
    parser.add_argument("--steps", type=int, default=10, help="Number of training steps per mode (default: 10)")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup steps excluded from metrics (default: 2)")
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=ALL_MODES,
        help="Run a single mode instead of all four (default: run all)",
    )
    args = parser.parse_args()

    # --- Distributed setup ---
    rank, world_size, local_rank = init_distributed()
    PartialState()

    model_cfg = MODEL_CONFIGS[args.model]
    model_name = args.model_path or model_cfg["hf_name"]

    if rank == 0:
        print(f"\n{'#' * 70}")
        print("  Liger x torch.compile Benchmark")
        print(f"  Model: {model_name} ({model_cfg['full_params'] / 1e9:.1f}B params)")
        print(f"  EP={args.ep}, SeqLen={args.seq}, Steps={args.steps}, Warmup={args.warmup}")
        print(f"  World size: {world_size}")
        print(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
        print(f"  GPU memory: {torch.cuda.get_device_properties(local_rank).total_memory / 1e9:.0f} GB")
        print(f"  PyTorch: {torch.__version__}")
        print(f"{'#' * 70}")

    if world_size < args.ep:
        if rank == 0:
            print(f"\nERROR: Need at least {args.ep} GPUs, got {world_size}")
        dist.destroy_process_group()
        return 1

    # --- Ensure model is cached ---
    if rank == 0:
        print("\nEnsuring model is downloaded...")
    ensure_model_downloaded(model_name, rank)

    # --- Load tokenizer ---
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # --- Create dataset (shared across all modes) ---
    dataset = create_synthetic_sft_dataset(tokenizer, args.seq)
    if rank == 0:
        print(f"Dataset created: {len(dataset)} samples, target seq_len={args.seq}")
        sample_tokens = len(tokenizer.encode(dataset[0]["text"]))
        print(f"Sample token count: {sample_tokens}")

    # --- Determine modes to run ---
    modes = [args.mode] if args.mode else ALL_MODES

    if rank == 0:
        print(f"\nModes to benchmark: {modes}")

    # --- Run benchmarks ---
    all_results = []
    failed = False
    for mode in modes:
        try:
            result = run_benchmark_mode(mode, args, tokenizer, dataset, rank, local_rank)
            if result is not None:
                all_results.append(result)
        except Exception as e:
            failed = True
            if rank == 0:
                print(f"\n  ERROR in mode '{mode}': {e}")
                traceback.print_exc()
            # Clean up and continue to next mode
            gc.collect()
            torch.cuda.empty_cache()
            barrier()
            continue

    # --- Print summary ---
    if rank == 0 and len(all_results) > 0:
        print_summary(all_results, args)

    # On rank 0 a run with no successful modes is a failure; other ranks only
    # collect results on rank 0, so they gate on whether any mode raised.
    if rank == 0 and not all_results:
        failed = True

    # --- Cleanup ---
    barrier()
    teardown_distributed()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
