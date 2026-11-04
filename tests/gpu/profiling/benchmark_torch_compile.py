#!/usr/bin/env python
"""One cell of the Liger x torch.compile 2x2 matrix for EP MoE SFT.

Liger patches model classes process-wide, so each process runs one mode; loop over them in the shell. A compiled cell
recompiles when a rank first meets a second sequence length, so keep --warmup past it. Qwen3-30B-A3B at seq 16384 peaks
at 127-173 GiB per GPU.
"""

import argparse
import random
import sys
import time
import traceback

import torch
from accelerate import PartialState
from datasets import Dataset
from transformers import AutoTokenizer
from trl import SFTConfig

from halo.callbacks.efficiency import EfficiencyCallback
from halo.distributed.parallelism_config import ParallelismConfig
from halo.log import configure_training_logging
from halo.model_loading.parallel.loader import load_distributed_model
from halo.runtime.collectives import barrier
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.distributed import (
    cleanup_dirs,
    ensure_model_downloaded,
    init_distributed,
    setup_cache_dirs,
    teardown_distributed,
)
from tests.common.memory import cleanup_memory
from tests.common.models import DEFAULT_MODEL, MODEL_CONFIGS
from tests.common.rank_log import log
from tests.common.reporting import emit_benchmark, format_benchmark_report

# mode -> (Liger, torch.compile)
MODES = {
    "neither": (False, False),
    "liger_only": (True, False),
    "compile_only": (False, True),
    "liger_compile": (True, True),
}

SEED = 42
NUM_SAMPLES = 64
BATCH_SIZE = 1
LEARNING_RATE = 2e-5


def create_synthetic_sft_dataset(
    tokenizer,
    seq_len: int,
    num_samples: int = NUM_SAMPLES,
    seed: int = SEED,
) -> Dataset:
    """Single-turn math conversations, chat-templated, with completions padded by filler toward ``seq_len``."""
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


def run_benchmark(
    args: argparse.Namespace, model_name: str, full_params: float, tokenizer, dataset: Dataset, output_dir: str
) -> None:
    use_liger, use_compile = MODES[args.mode]
    cell = f"{args.mode}_{args.compile_mode}" if use_compile else args.mode
    log(f"  Cell: {cell} (Liger {'ON' if use_liger else 'OFF'})")
    parallelism_config = ParallelismConfig(ep_size=args.ep)

    model, _ = load_distributed_model(
        model_name_or_path=model_name,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        use_liger_kernel=use_liger,
    )
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"  Model loaded: {trainable / 1e9:.2f}B params, {torch.cuda.memory_allocated() / 1e9:.1f} GB")

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
        logging_nan_inf_filter=False,
        include_num_input_tokens_seen=True,
        max_length=args.seq,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
        ddp_find_unused_parameters=True,
        # HF turns torch_compile on whenever a backend or mode is set, so an eager mode passes neither; the mixin
        # compiles after FSDP wrapping.
        **(
            {"torch_compile": True, "torch_compile_backend": "inductor", "torch_compile_mode": args.compile_mode}
            if use_compile
            else {}
        ),
    )
    # Disables TRL's Liger re-application.
    sft_config.use_liger_kernel = False

    efficiency_callback = EfficiencyCallback(
        parallelism_config,
        n_warmup_steps=args.warmup,
        num_full_model_params=full_params,
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        processing_class=tokenizer,
        callbacks=[efficiency_callback],
        parallelism_config=parallelism_config,
    )
    log(f"  Trainer created: EP={args.ep}, DP={parallelism_config.data_parallel_size}")
    log(f"  Training for {args.steps} steps (warmup={args.warmup})...")

    barrier()
    torch.cuda.reset_peak_memory_stats()
    train_start = time.perf_counter()
    trainer.train()
    train_elapsed = time.perf_counter() - train_start

    log(f"\n  --- {cell} Results ---")
    log("\n" + format_benchmark_report(efficiency_callback))
    emit_benchmark(f"compile_{cell}_{args.model}_ep{args.ep}_s{args.seq}", efficiency_callback)
    log(f"  Total Time:     {train_elapsed:.1f}s")
    trainer.cleanup_ep()


def main() -> int:
    configure_training_logging()
    parser = argparse.ArgumentParser(
        description="One cell of the Liger x torch.compile 2x2 benchmark for MoE models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, choices=list(MODEL_CONFIGS.keys()), help="Model config key"
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="Override model path (instead of using MODEL_CONFIGS)"
    )
    parser.add_argument("--mode", type=str, required=True, choices=list(MODES), help="Liger x compile cell to run")
    parser.add_argument(
        "--compile_mode",
        type=str,
        default="default",
        choices=["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"],
        help="torch.compile mode for the compiled cells (default: default, the trainer's own fallback)",
    )
    parser.add_argument("--ep", type=int, default=2, help="Expert parallel size (default: 2)")
    parser.add_argument("--seq", type=int, default=8192, help="Sequence length (default: 8192)")
    parser.add_argument("--steps", type=int, default=20, help="Number of training steps (default: 20)")
    parser.add_argument("--warmup", type=int, default=5, help="Warmup steps excluded from metrics (default: 5)")
    args = parser.parse_args()

    rank, world_size, local_rank = init_distributed()
    PartialState()

    model_cfg = MODEL_CONFIGS[args.model]
    model_name = args.model_path or model_cfg["hf_name"]

    log(f"\n{'#' * 70}")
    log("  Liger x torch.compile Benchmark")
    log(f"  Model: {model_name} ({model_cfg['full_params'] / 1e9:.1f}B params)")
    log(f"  EP={args.ep}, SeqLen={args.seq}, Steps={args.steps}, Warmup={args.warmup}")
    log(f"  World size: {world_size}")
    log(f"  GPU: {torch.cuda.get_device_name(local_rank)}")
    log(f"  PyTorch: {torch.__version__}")
    log(f"{'#' * 70}")

    if world_size < args.ep:
        log(f"\nERROR: Need at least {args.ep} GPUs, got {world_size}")
        teardown_distributed()
        return 1

    failed = False
    output_dir, cache_dir = setup_cache_dirs("bench_compile", rank)
    try:
        ensure_model_downloaded(model_name, rank)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        dataset = create_synthetic_sft_dataset(tokenizer, args.seq)
        log(f"Dataset created: {len(dataset)} samples, target seq_len={args.seq}")
        log(f"Sample token count: {len(tokenizer.encode(dataset[0]['text']))}")

        run_benchmark(args, model_name, model_cfg["full_params"], tokenizer, dataset, output_dir)
    except Exception as e:
        failed = True
        log(f"\nBENCHMARK FAILED: {e}")
        if rank == 0:
            traceback.print_exc()
    finally:
        cleanup_memory()
        cleanup_dirs(output_dir, cache_dir)
        barrier()
        teardown_distributed()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
