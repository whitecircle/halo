#!/usr/bin/env python
"""
Offline GRPO MFU benchmark with Expert Parallelism using OfflineGRPOTrainer.

Benchmarks Offline Group Relative Policy Optimization training on a GptOss-20B MoE
model with expert parallelism. Each sample has multiple completions with pre-computed
rewards. Measures MFU, S-MFU, tokens per second, and peak memory using
EfficiencyCallback.

Note: CP is NOT supported for Offline GRPO due to the logits_to_keep optimization.

Usage:
    # EP=2 with 4096 sequence length, 10 steps
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/benchmark_offline_grpo_ep.py --ep 2 --seq 4096 --steps 10

    # EP=2 with 8 generations per prompt
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/benchmark_offline_grpo_ep.py --ep 2 --seq 4096 --steps 10 \
        --num_generations 8

    # Custom model path
    torchrun --nproc_per_node=2 \
        tests/gpu/profiling/benchmark_offline_grpo_ep.py --ep 2 --seq 4096 --steps 10 \
        --model_path /path/to/local/model
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
from src.configs.offline_grpo_config import OfflineGRPOConfig
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.grpo.offline import OfflineGRPOTrainer
from tests.common.benchmark_args import create_benchmark_parser
from tests.common.distributed import (
    ensure_model_downloaded,
    init_distributed,
)
from tests.common.models import MODEL_CONFIGS
from tests.common.reporting import emit_benchmark, format_benchmark_report
from tests.common.utils import log

# Dataset Creation


def create_offline_grpo_dataset(
    tokenizer,
    seq_len: int,
    num_samples: int = 100,
    num_generations: int = 4,
    seed: int = 42,
) -> Dataset:
    """Create a dummy Offline GRPO dataset with grouped generations and rewards.

    Each sample has a prompt with multiple completions and pre-computed rewards.
    Format: {"prompt": str, "completions": List[str], "rewards": List[float]}

    Completions are padded with filler text to approximate the target sequence
    length distribution across prompt and completion.

    Args:
        tokenizer: HuggingFace tokenizer for chat template application.
        seq_len: Target total sequence length (prompt + completion).
        num_samples: Number of unique prompts.
        num_generations: Number of completions per prompt.
        seed: Random seed for reproducibility.

    Returns:
        Dataset with columns: prompt, completions, rewards.
    """
    random.seed(seed)

    seq_len // 3
    completion_len = seq_len * 2 // 3

    templates = [
        {
            "q": "What is {a} + {b}?",
            "answers": [
                ("The answer is {correct}. Adding {a} and {b} gives {correct}.", 1.0),
                ("I think {a} + {b} = {correct}. Let me verify: yes, that is correct.", 0.7),
                ("The answer might be {incorrect}. Actually wait, it should be {correct}.", 0.3),
                ("The answer is {incorrect}. {a} plus {b} equals {incorrect}.", -0.5),
            ],
        },
        {
            "q": "Calculate {a} * {b}.",
            "answers": [
                ("{a} times {b} equals {correct}. This is the product.", 1.0),
                ("The product is {correct}.", 0.5),
                ("Let me compute: {a} * {b} = {incorrect}. Hmm, actually {correct}.", 0.2),
                ("{a} times {b} is {incorrect}.", -0.5),
            ],
        },
        {
            "q": "What is {a} - {b}?",
            "answers": [
                ("The result of {a} minus {b} is {correct}.", 1.0),
                ("{a} - {b} = {correct}. Subtraction gives this result.", 0.6),
                ("I believe the answer is {incorrect}. No wait, it is {correct}.", 0.1),
                ("The difference is {incorrect}.", -0.5),
            ],
        },
        {
            "q": "If you have {a} apples and get {b} more, how many total?",
            "answers": [
                ("You would have {correct} apples in total.", 1.0),
                ("Total apples: {a} + {b} = {correct}.", 0.7),
                ("After getting more apples, you have {incorrect}. Wait, {correct}.", 0.2),
                ("You would have {incorrect} apples.", -0.5),
            ],
        },
    ]

    data = []
    for i in range(num_samples):
        template = random.choice(templates)
        a = random.randint(1, 100)
        b = random.randint(1, 100)

        if "+" in template["q"] or "apples" in template["q"]:
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

        # Build completions with varying quality
        completions = []
        rewards = []
        base_answers = template["answers"]

        for gen_idx in range(num_generations):
            # Cycle through template answers if num_generations > len(answers)
            answer_template, base_reward = base_answers[gen_idx % len(base_answers)]

            answer = answer_template.format(
                a=a,
                b=b,
                correct=correct,
                incorrect=incorrect,
            )

            # Add filler text to approximate target completion length
            filler = " " + "detailed explanation " * (completion_len // 22)
            completions.append(answer + filler)

            # Add small noise to rewards for variation
            reward = base_reward + random.uniform(-0.1, 0.1)
            rewards.append(round(reward, 3))

        data.append(
            {
                "prompt": prompt,
                "completions": completions,
                "rewards": rewards,
            }
        )

    return Dataset.from_list(data)


# Main Benchmark


def main() -> int:
    failed = False
    trainer = None
    model = None
    parser = create_benchmark_parser(
        description="Offline GRPO MFU benchmark with Expert Parallelism",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=4,
        help="Number of completions per prompt",
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

    try:
        if rank == 0:
            print(f"\n{'=' * 60}")
            print("OFFLINE GRPO EP BENCHMARK with OfflineGRPOTrainer")
            print(f"Model: {args.model}, EP={args.ep}, SeqLen={args.seq}")
            print(
                f"Full params: {model_config['full_params'] / 1e9:.1f}B, "
                f"Experts: {model_config['num_experts']}, top_k={model_config['top_k']}"
            )
            print(f"Generations per prompt: {args.num_generations}")
            print(f"Optimizer: {args.optim}, Steps: {args.steps}, Warmup: {args.warmup}")
            print(f"Attention: {args.attn_implementation}")
            print("Note: CP is NOT supported for Offline GRPO")
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
        seq_len = args.seq
        prompt_len = seq_len // 3
        completion_len = seq_len * 2 // 3

        dataset = create_offline_grpo_dataset(
            tokenizer,
            seq_len,
            num_samples=100,
            num_generations=args.num_generations,
            seed=42,
        )
        if rank == 0:
            print(
                f"Dataset created: {len(dataset)} prompts x "
                f"{args.num_generations} generations = "
                f"{len(dataset) * args.num_generations} total examples"
            )

        # --- Offline GRPO Config ---
        output_dir = f"/tmp/offline_grpo_ep_benchmark_{args.ep}_{seq_len}"

        grpo_config = OfflineGRPOConfig(
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
            # GRPO specific
            max_prompt_length=prompt_len,
            max_completion_length=completion_len,
            loss_type="bnpo",
            advantage_method="z_norm",
        )
        # Disable TRL re-application of Liger (we applied via load_distributed_model)
        grpo_config.use_liger_kernel = False

        # --- Efficiency Callback ---
        efficiency_callback = EfficiencyCallback(
            parallelism_config,
            n_warmup_steps=args.warmup,
            num_full_model_params=model_config["full_params"],
        )

        # --- Create Trainer ---
        trainer = OfflineGRPOTrainer(
            model=model,
            args=grpo_config,
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
            emit_benchmark(
                f"offline_grpo_ep_{args.model}_ep{args.ep}_s{seq_len}",
                efficiency_callback,
            )

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
