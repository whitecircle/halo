#!/usr/bin/env python
"""SMPO throughput of ``SMPOTrainer`` on an EP (or EP+CP) MoE run: MFU, S-MFU, tok/s and peak memory.

torchrun --nproc_per_node=8 tests/gpu/profiling/benchmark_smpo_ep.py --ep 8 --cp 2 --seq 4096 --steps 10
"""

import gc
import random
import sys
import traceback

import torch
import torch.distributed as dist
from accelerate import PartialState
from datasets import Dataset
from transformers import AutoTokenizer

from halo.callbacks.efficiency import EfficiencyCallback
from halo.configs.preference import SMPOConfig
from halo.distributed.parallelism_config import ParallelismConfig
from halo.log import configure_training_logging
from halo.model_loading.parallel.loader import load_distributed_model
from halo.runtime.collectives import barrier
from halo.trainers.preference.smpo import SMPOTrainer
from tests.common.benchmark_args import axis_tags, create_benchmark_parser
from tests.common.distributed import (
    cleanup_dirs,
    ensure_model_downloaded,
    init_distributed,
    setup_cache_dirs,
)
from tests.common.models import MODEL_CONFIGS
from tests.common.rank_log import log
from tests.common.reporting import emit_benchmark, format_benchmark_report


def create_smpo_dataset(
    tokenizer,
    seq_len: int,
    num_samples: int = 100,
    seed: int = 42,
) -> Dataset:
    """``num_samples`` chat-templated arithmetic prompts with a correct (chosen) and a wrong (rejected)
    answer, each padded with filler text toward ``seq_len``."""
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

        if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"User: {q}\nAssistant:"

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


def main() -> int:
    configure_training_logging()
    failed = False
    trainer = None
    model = None
    parser = create_benchmark_parser(
        description="SMPO MFU benchmark with Expert Parallelism",
    )
    parser.add_argument("--cp", type=int, default=1, help="Context parallel size (EP+CP, node-local).")
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw_torch",
        choices=["adamw_torch", "adamw_bnb_8bit"],
        help="Optimizer",
    )
    args = parser.parse_args()

    model_config = MODEL_CONFIGS[args.model]

    rank, world_size, local_rank = init_distributed()
    PartialState()

    seq_len = args.seq
    output_dir, cache_dir = setup_cache_dirs("bench_smpo_ep", rank)

    try:
        if rank == 0:
            print(f"\n{'=' * 60}")
            print("SMPO EP BENCHMARK with SMPOTrainer")
            print(f"Model: {args.model}, EP={args.ep}, CP={args.cp}, SeqLen={args.seq}")
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

        ensure_model_downloaded(model_name, rank)

        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        if rank == 0:
            print("\n--- Loading model with Expert Parallelism ---")

        parallelism_config = ParallelismConfig(ep_size=args.ep, cp_size=args.cp)

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

        prompt_len = seq_len // 3

        dataset = create_smpo_dataset(tokenizer, seq_len, num_samples=100, seed=42)
        if rank == 0:
            print(f"Dataset created: {len(dataset)} samples")

        smpo_config = SMPOConfig(
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
            logging_nan_inf_filter=False,
            include_num_input_tokens_seen=True,
            ddp_find_unused_parameters=True,
            max_length=seq_len,
            max_prompt_length=prompt_len,
            target_margin=0.5,
            loss_type="sigmoid",
            use_margin_schedule=False,
            chosen_sft_ratio=0.0,
        )
        # Disable TRL re-application of Liger (we applied via load_distributed_model)
        smpo_config.use_liger_kernel = False

        efficiency_callback = EfficiencyCallback(
            parallelism_config,
            n_warmup_steps=args.warmup,
            num_full_model_params=model_config["full_params"],
        )

        trainer = SMPOTrainer(
            model=model,
            args=smpo_config,
            train_dataset=dataset,
            processing_class=tokenizer,
            callbacks=[efficiency_callback],
            parallelism_config=parallelism_config,
        )

        if rank == 0:
            print(f"Trainer created: EP={args.ep}, CP={args.cp}, DP={parallelism_config.data_parallel_size}")

        barrier()
        torch.cuda.reset_peak_memory_stats()

        if rank == 0:
            print(f"\n--- Training for {args.steps} steps (warmup={args.warmup}) ---")

        trainer.train()

        if rank == 0:
            log("\n" + format_benchmark_report(efficiency_callback))
            layout = "_".join([f"ep{args.ep}", *axis_tags(cp=args.cp)])
            emit_benchmark(f"smpo_ep_{args.model}_{layout}_s{seq_len}", efficiency_callback)

    except Exception as e:
        failed = True
        log(f"\nBENCHMARK FAILED: {e}")
        if rank == 0:
            traceback.print_exc()

    finally:
        if trainer is not None and hasattr(trainer, "cleanup_ep"):
            trainer.cleanup_ep()
        barrier()
        del trainer
        del model
        gc.collect()
        torch.cuda.empty_cache()
        cleanup_dirs(output_dir, cache_dir)

        if dist.is_initialized():
            dist.destroy_process_group()

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
