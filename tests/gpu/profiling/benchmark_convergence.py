#!/usr/bin/env python
"""
Convergence / loss-tracking comparison: stock TRL vs Halo (EP / dense).

Trains the same model (gpt-oss-20b) on the same seeded synthetic data, same LR,
same global batch (DP=8), for a longer run, and dumps the per-step loss curve so
Halo's optimizations (Expert Parallelism, grouped GEMM, AdamWBF16 with stochastic
rounding) can be shown to preserve training dynamics — not just speed/memory.

  --ep 0   -> stock trl.SFTTrainer + transformers FSDP + fp32-AdamW  (baseline)
  --ep 1   -> Halo DistributedSFTTrainer, dense FSDP2 (+ AdamWBF16, grouped GEMM)
  --ep N>1 -> Halo with Expert Parallelism size N

Loss curve is printed as parseable `CONV_LOSS step=<i> loss=<x> lr=<y>` lines
(rank 0), plus the EfficiencyCallback headline. Same seed (42) + DistributedSampler
=> identical data order across frameworks, so curves are directly comparable.

Usage (8x B300):
    torchrun --nproc_per_node=8 \
        tests/gpu/profiling/benchmark_convergence.py --ep 0 --seq 2048 --batch_size 2 --steps 200
"""

import sys

import torch
from accelerate import PartialState
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.callbacks.efficiency import EfficiencyCallback
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.sft import DistributedSFTTrainer
from tests.common.benchmark_args import create_benchmark_parser, resolve_benchmark_attn
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
from tests.common.utils import cleanup_memory, log

_LAYER_CLS = {"gpt-oss-20b": "GptOssDecoderLayer", "qwen3-30b-a3b": "Qwen3MoeDecoderLayer"}


def main() -> int:
    failed = False
    parser = create_benchmark_parser(description="Convergence comparison (TRL vs Halo)", require_ep=False)
    parser.add_argument("--ep", type=int, default=0, help="0=stock TRL, 1=Halo dense, N>1=Halo EP size N")
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rank, world_size, local_rank = init_distributed()
    PartialState()
    model_cfg = MODEL_CONFIGS[args.model]
    model_name = args.model_path or model_cfg["hf_name"]
    full_params = model_cfg["full_params"]
    use_liger = not args.no_liger
    is_halo = args.ep >= 1
    tag = "TRL-stock" if not is_halo else (f"Halo-EP{args.ep}" if args.ep > 1 else "Halo-dense")

    log(f"\n{'=' * 70}\n  CONVERGENCE: {tag}  ({model_name})")
    log(f"  seq={args.seq} bs={args.batch_size} steps={args.steps} lr={args.lr} seed={args.seed} liger={use_liger}")
    log(f"  global batch = {world_size * args.batch_size} (DP={world_size}); GC on; FA4")
    log(f"{'=' * 70}")

    output_dir, cache_dir = setup_cache_dirs("bench_conv", rank)
    try:
        ensure_model_downloaded(model_name, rank)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        train_dataset = create_benchmark_dataset(tokenizer, args.seq, args.num_samples)

        common = {
            "output_dir": output_dir,
            "max_steps": args.steps,
            "per_device_train_batch_size": args.batch_size,
            "gradient_accumulation_steps": 1,
            "learning_rate": args.lr,
            "lr_scheduler_type": "constant",
            "warmup_steps": 0,
            "bf16": True,
            "gradient_checkpointing": True,
            "logging_steps": 1,
            "save_strategy": "no",
            "report_to": "none",
            "max_length": args.seq,
            "seed": args.seed,
            "data_seed": args.seed,
            "dataloader_drop_last": True,
            "dataloader_num_workers": 0,
            "include_num_input_tokens_seen": True,
        }
        # One config for both arms — the stock-TRL arm distributes nothing (args.ep == 0 => ep_size
        # 1), the Halo arm distributes experts — because the callback needs it in both.
        parallelism_config = ParallelismConfig(ep_size=max(args.ep, 1), use_grouped_gemm=True)
        efficiency_cb = EfficiencyCallback(
            parallelism_config,
            n_warmup_steps=args.warmup,
            num_full_model_params=full_params,
        )

        if not is_halo:
            sft_config = SFTConfig(
                **common,
                optim="adamw_torch_fused",
                use_liger_kernel=use_liger,
                gradient_checkpointing_kwargs={"use_reentrant": False},
                fsdp="full_shard auto_wrap",
                fsdp_config={
                    "transformer_layer_cls_to_wrap": [_LAYER_CLS.get(args.model, "GptOssDecoderLayer")],
                    "fsdp_version": 2,
                },
            )
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation=resolve_benchmark_attn(model_name, args.attn_implementation),
                use_cache=False,
            )
            trainer = SFTTrainer(
                model=model,
                args=sft_config,
                train_dataset=train_dataset,
                processing_class=tokenizer,
                callbacks=[efficiency_cb],
            )
        else:
            model, _ = load_distributed_model(
                model_name_or_path=model_name,
                parallelism_config=parallelism_config,
                dtype=torch.bfloat16,
                trust_remote_code=True,
                attn_implementation=args.attn_implementation,
                use_liger_kernel=use_liger,
            )
            sft_config = SFTConfig(**common, use_liger_kernel=False, fsdp="", ddp_find_unused_parameters=True)
            trainer = DistributedSFTTrainer(
                model=model,
                args=sft_config,
                train_dataset=train_dataset,
                processing_class=tokenizer,
                parallelism_config=parallelism_config,
                callbacks=[efficiency_cb],
            )

        barrier()
        trainer.train()

        if rank == 0:
            log(f"\n--- LOSS CURVE ({tag}) ---")
            for e in trainer.state.log_history:
                if "loss" in e:
                    log(
                        f"CONV_LOSS step={int(e.get('step', 0))} loss={e['loss']:.6f} lr={e.get('learning_rate', args.lr):.2e}"
                    )
        log("\n" + format_benchmark_report(efficiency_cb))
        emit_benchmark(f"conv_{tag}_{args.model}_s{args.seq}_b{args.batch_size}", efficiency_cb)
    except Exception as e:
        failed = True
        log(f"\nCONVERGENCE FAILED: {type(e).__name__}: {e}")
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
