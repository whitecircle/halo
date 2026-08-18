#!/usr/bin/env python
"""
SMPO MFU benchmark with Expert Parallelism using SmoothMarginPOTrainer.

Benchmarks Smooth Margin Preference Optimization training on a GptOss-20B MoE model
with expert parallelism. Measures MFU, S-MFU, tokens per second, and peak memory
using EfficiencyCallback.

Usage:
    # EP=2 with 4096 sequence length, 10 steps
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/benchmark_smpo_ep.py --ep 2 --seq 4096 --steps 10

    # EP=2 with 8192 sequence length, custom warmup
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/benchmark_smpo_ep.py --ep 2 --seq 8192 --steps 12 --warmup 3

    # Custom model path with sdpa attention
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/benchmark_smpo_ep.py --ep 2 --seq 4096 --steps 10 \
        --model_path /path/to/local/model --attn_implementation sdpa
"""

import gc
import random
import sys

import torch
import torch.distributed as dist
from accelerate import PartialState
from datasets import Dataset
from transformers import AutoTokenizer

from src.callbacks.efficiency import EfficiencyCallback
from src.configs.smpo_config import SmoothMarginPOConfig
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.preference.smpo import SmoothMarginPOTrainer
from tests.common.benchmark_args import create_benchmark_parser
from tests.common.distributed import (
    ensure_model_downloaded,
    init_distributed,
)
from tests.common.models import MODEL_CONFIGS
from tests.common.reporting import emit_benchmark, format_benchmark_report
from tests.common.utils import log

# Dataset Creation


def create_smpo_dataset(
    tokenizer,
    seq_len: int,
    num_samples: int = 100,
    seed: int = 42,
) -> Dataset:
    """Create a dummy SMPO preference dataset with prompt/chosen/rejected pairs.

    Each sample generates a chat-templated prompt with chosen (correct) and
    rejected (incorrect) completions. Completions are padded with filler text
    to approximate the target sequence length.

    Args:
        tokenizer: HuggingFace tokenizer for chat template application.
        seq_len: Target total sequence length (prompt + completion).
        num_samples: Number of preference pairs to generate.
        seed: Random seed for reproducibility.

    Returns:
        Dataset with columns: prompt, chosen, rejected.
    """
    random.seed(seed)

    prompt_len = seq_len // 3
    completion_len = seq_len - prompt_len

    templates = [
        {
            "q": "What is {a} + {b}?",
            "chosen": "The answer is {correct}. Adding {a} and {b} gives {correct}.",
            "rejected": "The answer is {incorrect}. That is {a} plus {b}.",
        },
        {
            "q": "Calculate {a} * {b}.",
            "chosen": "{a} times {b} equals {correct}.",
            "rejected": "{a} times {b} equals {incorrect}.",
        },
        {
            "q": "What is {a} - {b}?",
            "chosen": "The result of {a} minus {b} is {correct}.",
            "rejected": "The result of {a} minus {b} is {incorrect}.",
        },
        {
            "q": "If you have {a} apples and get {b} more, how many total?",
            "chosen": "You would have {correct} apples in total.",
            "rejected": "You would have {incorrect} apples in total.",
        },
        {
            "q": "Solve: {a} + {b} = ?",
            "chosen": "The solution is {correct}. I added the two numbers together.",
            "rejected": "The solution is {incorrect}. I think that is right.",
        },
    ]

    data = []
    for i in range(num_samples):
        template = random.choice(templates)
        a = random.randint(1, 100)
        b = random.randint(1, 100)

        if "+" in template["q"] or "apples" in template["q"] or "Solve" in template["q"]:
            correct = a + b
        elif "*" in template["q"]:
            correct = a * b
        else:
            correct = a - b

        offset = random.randint(1, 10) * random.choice([-1, 1])
        incorrect = correct + offset

        q = template["q"].format(a=a, b=b)
        messages = [{"role": "user", "content": q}]

        # Apply chat template to create the prompt
        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"User: {q}\nAssistant:"

        # Build chosen and rejected with filler to approximate seq_len
        chosen_base = template["chosen"].format(
            a=a,
            b=b,
            correct=correct,
            incorrect=incorrect,
        )
        rejected_base = template["rejected"].format(
            a=a,
            b=b,
            correct=correct,
            incorrect=incorrect,
        )

        chosen_filler = " " + "good response " * (completion_len // 15)
        rejected_filler = " " + "bad response " * (completion_len // 14)

        data.append(
            {
                "prompt": prompt,
                "chosen": chosen_base + chosen_filler,
                "rejected": rejected_base + rejected_filler,
            }
        )

    return Dataset.from_list(data)


# Main Benchmark


def main() -> int:
    failed = False
    trainer = None
    model = None
    parser = create_benchmark_parser(
        description="SMPO MFU benchmark with Expert Parallelism",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        choices=["adamw_torch", "adamw_bnb_8bit"],
        help="Optimizer",
    )
    args = parser.parse_args()

    model_config = MODEL_CONFIGS[args.model]

    # --- Distributed Setup ---
    rank, world_size, local_rank = init_distributed()
    PartialState()

    seq_len = args.seq
    output_dir = f"/tmp/smpo_ep_benchmark_{args.ep}_{seq_len}"

    try:
        if rank == 0:
            print(f"\n{'=' * 60}")
            print("SMPO EP BENCHMARK with SmoothMarginPOTrainer")
            print(f"Model: {args.model}, EP={args.ep}, SeqLen={args.seq}")
            print(
                f"Full params: {model_config['full_params'] / 1e9:.1f}B, "
                f"Experts: {model_config['num_experts']}, top_k={model_config['top_k']}"
            )
            print(f"Optimizer: {args.optim}, Steps: {args.steps}, Warmup: {args.warmup}")
            print(f"Attention: {args.attn_implementation}")
            print(f"GPU: {torch.cuda.get_device_name(local_rank)}")
            if args.model_path:
                print(f"Model path override: {args.model_path}")
            print(f"{'=' * 60}")

        model_name = args.model_path or model_config["hf_name"]

        # --- Ensure model is cached (download on rank 0 first) ---
        ensure_model_downloaded(model_name, rank)

        # --- Load tokenizer ---
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        # --- Load model with EP ---
        if rank == 0:
            print("\n--- Loading model with Expert Parallelism ---")

        parallelism_config = ParallelismConfig(ep_size=args.ep)

        model, _ = load_distributed_model(
            model_name_or_path=model_name,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=args.attn_implementation,
            use_liger_kernel=not args.no_liger,
        )

        if rank == 0:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            mem_gb = torch.cuda.memory_allocated() / 1e9
            print(f"Model loaded: {model_name}")
            print(f"Trainable params: {trainable / 1e9:.2f}B")
            print(f"GPU memory after model load: {mem_gb:.1f}GB")

        # --- Create dataset ---
        prompt_len = seq_len // 3

        dataset = create_smpo_dataset(tokenizer, seq_len, num_samples=100, seed=42)
        if rank == 0:
            print(f"Dataset created: {len(dataset)} samples")

        # --- SMPO Config ---
        smpo_config = SmoothMarginPOConfig(
            output_dir=output_dir,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            max_steps=args.steps,
            learning_rate=1e-5,
            optim=args.optim,
            logging_steps=1,
            save_strategy="no",
            bf16=True,
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to=[],
            include_num_input_tokens_seen=True,
            ddp_find_unused_parameters=True,
            # SMPO specific
            max_length=seq_len,
            max_prompt_length=prompt_len,
            target_margin=0.5,
            loss_type="sigmoid",
            use_margin_schedule=False,
            chosen_sft_ratio=0.0,
        )
        # Disable TRL re-application of Liger (we applied via load_distributed_model)
        smpo_config.use_liger_kernel = False

        # --- Efficiency Callback ---
        efficiency_callback = EfficiencyCallback(
            parallelism_config,
            n_warmup_steps=args.warmup,
            num_full_model_params=model_config["full_params"],
        )

        # --- Create Trainer ---
        trainer = SmoothMarginPOTrainer(
            model=model,
            args=smpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[efficiency_callback],
            parallelism_config=parallelism_config,
        )

        if rank == 0:
            print(f"Trainer created: EP={args.ep}, DP={parallelism_config.data_parallel_size}")

        # --- Train ---
        barrier()
        torch.cuda.reset_peak_memory_stats()

        if rank == 0:
            print(f"\n--- Training for {args.steps} steps (warmup={args.warmup}) ---")

        trainer.train()

        # --- Print Results ---
        if rank == 0:
            log("\n" + format_benchmark_report(efficiency_callback))
            emit_benchmark(f"smpo_ep_{args.model}_ep{args.ep}_s{seq_len}", efficiency_callback)

    except Exception as e:
        failed = True
        log(f"\nBENCHMARK FAILED: {e}")
        if rank == 0:
            import traceback

            traceback.print_exc()

    finally:
        # --- Cleanup ---
        if trainer is not None and hasattr(trainer, "cleanup_ep"):
            trainer.cleanup_ep()
        barrier()
        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()

        if dist.is_initialized():
            dist.destroy_process_group()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
