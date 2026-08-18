#!/usr/bin/env python
"""
Stock TRL/transformers SFT baseline — the "what you get without Halo" reference.

Runs the upstream `trl.SFTTrainer` on an MoE (default gpt-oss-20b) with
transformers-native FSDP, the stock fp32-state optimizer, transformers' own
grouped-GEMM expert kernel, FA4 + Liger. Shares the SAME synthetic dataset and
the SAME `EfficiencyCallback` as the Halo benchmarks, so tokens/s/GPU and peak
memory are measured identically and are directly comparable against
`benchmark_sft_ep.py` / `benchmark_sft_dense.py`.

Differences vs the Halo path (the things this baseline deliberately does NOT have):
  - no Expert Parallelism (experts replicated, sharded only by FSDP — every rank computes all experts)
  - stock optimizer (adamw_torch_fused, fp32 moments) instead of AdamWBF16/SR
  - transformers-native FSDP instead of Halo's FSDP2 + EP wrapper

Expert compute is transformers' OWN grouped GEMM (`torch.nn.functional.grouped_mm`, the default
`experts_implementation` for gpt-oss) — the same grouped-GEMM CLASS as Halo, NOT an
eager loop. So this is a grouped-vs-grouped baseline on the expert kernel; Halo's lead comes from EP
token distribution + AdamWBF16 + FSDP2, not from "grouped GEMM vs loop". `--experts_impl eager` forces
the slow per-expert loop for the loop-vs-grouped contrast.

Usage (8x B300):
    torchrun --nproc_per_node=8 \
        tests/gpu/profiling/benchmark_trl_baseline.py --seq 4096 --batch_size 1
    # --no_grad_checkpoint disables GC; --no_liger disables Liger;
    # --fsdp_version 1 for the legacy sharding backend.
"""

import sys

import torch
from torch.distributed.optim import ZeroRedundancyOptimizer
from transformers import AutoConfig, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from src.callbacks.efficiency import EfficiencyCallback
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.models.loading.model_preparation import resolve_auto_model_class
from src.optimizers.flash_adamw import create_flash_adamw_optimizer
from src.optimizers.registry import register_custom_optimizers
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

# Decoder-layer class to wrap per architecture for FSDP auto-wrap.
_LAYER_CLS = {
    "gpt-oss-20b": "GptOssDecoderLayer",
    "qwen3-30b-a3b": "Qwen3MoeDecoderLayer",
    "qwen3.5-35b-a3b": "Qwen3_5MoeDecoderLayer",
    "qwen3-8b": "Qwen3DecoderLayer",
    "qwen3-4b": "Qwen3DecoderLayer",
    "gemma4-26b-a4b": "Gemma4TextDecoderLayer",
    "mistral3-119b-moe": "Mistral4DecoderLayer",
}


class FlashAdamWSFTTrainer(SFTTrainer):
    """Stock TRL trainer with Halo's explicit FlashAdamW OOM fallback."""

    def create_optimizer(self):
        if self.args.optim != "flash_adamw":
            return super().create_optimizer()
        if self.optimizer is None:
            self.optimizer = create_flash_adamw_optimizer(
                self.model,
                lr=self.args.learning_rate,
                betas=(self.args.adam_beta1, self.args.adam_beta2),
                eps=self.args.adam_epsilon,
                weight_decay=self.args.weight_decay,
                master_weight_bits=24,
                decay_parameters=set(self.get_decay_parameter_names(self.model)),
            )
            # Accelerate snapshots the optimizer before the first step when preparing FSDP2.
            # FlashAdamW's state_dict expects an entry for every parameter even before its
            # lazy state initialization, unlike torch.optim.AdamW.
            for group in self.optimizer.param_groups:
                for param in group["params"]:
                    self.optimizer.state[param] = {}
        return self.optimizer


def main() -> int:
    failed = False
    parser = create_benchmark_parser(
        description="Stock TRL/transformers SFT baseline (no Halo optimizations)",
        require_ep=False,
    )
    parser.add_argument("--no_grad_checkpoint", action="store_true", help="Disable gradient checkpointing")
    parser.add_argument(
        "--fsdp_version",
        type=int,
        default=2,
        choices=[1, 2],
        help="Deprecated / ignored — this baseline is FSDP2-only (kept so older invocations still parse).",
    )
    parser.add_argument(
        "--fsdp_sharding",
        type=str,
        default="full_shard",
        choices=["full_shard", "shard_grad_op"],
        help="FSDP2 reshard_after_forward: full_shard=True (ZeRO-3, params resharded after fwd); "
        "shard_grad_op=False (ZeRO-2, params kept gathered fwd->bwd, only grads+optim sharded)",
    )
    parser.add_argument(
        "--zero1_ddp",
        action="store_true",
        help="True ZeRO-1: DDP + ZeroRedundancyOptimizer (optimizer state sharded; params+grads replicated). "
        "FSDP2 cannot express ZeRO-1. Overrides --fsdp_sharding.",
    )
    parser.add_argument("--optim", type=str, default="adamw_torch_fused", help="stock optimizer name")
    parser.add_argument(
        "--experts_impl",
        type=str,
        default="grouped_mm",
        choices=["grouped_mm", "eager", "batched_mm"],
        help="transformers experts_implementation (from_pretrained kwarg): grouped_mm (default — "
        "torch.nn.functional.grouped_mm, the fast stock path) / eager (per-expert loop) / batched_mm.",
    )
    args = parser.parse_args()
    register_custom_optimizers()

    rank, world_size, local_rank = init_distributed()

    model_cfg = MODEL_CONFIGS[args.model]
    model_name = args.model_path or model_cfg["hf_name"]
    full_params = model_cfg["full_params"]
    num_experts = model_cfg["num_experts"]
    gc_enabled = not args.no_grad_checkpoint
    use_liger = not args.no_liger
    layer_cls = _LAYER_CLS.get(args.model, "GptOssDecoderLayer")

    log(f"\n{'=' * 70}")
    log("  STOCK TRL/transformers SFT baseline (no Halo optimizations)")
    log(f"  Model: {model_name} ({full_params / 1e9:.1f}B, {num_experts} experts)")
    log(f"  seq={args.seq} bs={args.batch_size} steps={args.steps} warmup={args.warmup}")
    reshard_after_fwd = args.fsdp_sharding == "full_shard"  # True = ZeRO-3, False = ZeRO-2 (FSDP2)
    zero = (
        "ZeRO-1 (DDP + ZeroRedundancyOptimizer)" if args.zero1_ddp else ("ZeRO-3" if reshard_after_fwd else "ZeRO-2")
    )
    shard_desc = "DDP (no FSDP)" if args.zero1_ddp else f"FSDP2 (reshard_after_forward={reshard_after_fwd})"
    log(f"  World: {world_size}  {shard_desc} = {zero}, wrap {layer_cls}")
    log(f"  Attn: {args.attn_implementation or 'flash_attention_4'}  Liger: {use_liger}  GC: {gc_enabled}")
    optimizer_desc = "FlashAdamW OOM fallback" if args.optim == "flash_adamw" else "stock fp32-moment AdamW"
    log(f"  Optimizer: {args.optim} ({optimizer_desc})")
    log(f"  Experts: experts_implementation={args.experts_impl} (transformers-native)")
    log(f"{'=' * 70}")

    output_dir, cache_dir = setup_cache_dirs("bench_trl_baseline", rank)

    try:
        ensure_model_downloaded(model_name, rank)
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_dataset = create_benchmark_dataset(tokenizer, args.seq, args.num_samples)

        # ZeRO-1 takes the DDP path (no FSDP); the FSDP2 path sets ZeRO-2 vs ZeRO-3 by the reshard
        # bool (transformers reads it from fsdp_config under the version=2 key — the string-derived
        # default is unreliable). FSDP2 only, no v1.
        fsdp_str = "" if args.zero1_ddp else "full_shard auto_wrap"
        fsdp_cfg = (
            None
            if args.zero1_ddp
            else {
                "version": 2,
                "reshard_after_forward": reshard_after_fwd,
                "transformer_layer_cls_to_wrap": [layer_cls],
            }
        )

        sft_config = SFTConfig(
            output_dir=output_dir,
            max_steps=args.steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=1,
            learning_rate=2e-5,
            bf16=True,
            optim=args.optim,
            gradient_checkpointing=gc_enabled,
            gradient_checkpointing_kwargs={"use_reentrant": False} if gc_enabled else None,
            use_liger_kernel=use_liger,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            max_length=args.seq,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            include_num_input_tokens_seen=True,
            # transformers-native FSDP2 (empty => DDP for ZeRO-1); reshard bool sets ZeRO-2 vs ZeRO-3.
            fsdp=fsdp_str,
            fsdp_config=fsdp_cfg,
        )

        log(f"\n[load] {model_name} | mem before {gpu_mem_gb():.1f}GB")
        model_config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        model_config.use_cache = False
        model_config.experts_implementation = args.experts_impl
        text_config = getattr(model_config, "text_config", None)
        if text_config is not None:
            text_config.use_cache = False
            text_config.experts_implementation = args.experts_impl
        model_class = resolve_auto_model_class(model_config)
        model = model_class.from_pretrained(
            model_name,
            config=model_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation=args.attn_implementation or "flash_attention_4",
        )
        log(f"  loaded {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B | mem {gpu_mem_gb():.1f}GB")

        # The baseline shards nothing: a default config is exactly world-wide data parallelism,
        # which is the topology the callback's throughput math needs.
        efficiency_cb = EfficiencyCallback(
            ParallelismConfig(),
            n_warmup_steps=args.warmup,
            num_full_model_params=full_params,
        )

        # ZeRO-1: shard only the optimizer state across ranks (params+grads stay replicated under DDP).
        optimizers = (None, None)
        if args.zero1_ddp:
            zero_opt = ZeroRedundancyOptimizer(
                [p for p in model.parameters() if p.requires_grad],
                optimizer_class=torch.optim.AdamW,
                lr=2e-5,
                fused=True,
            )
            optimizers = (zero_opt, None)

        trainer_class = FlashAdamWSFTTrainer if args.optim == "flash_adamw" else SFTTrainer
        trainer = trainer_class(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            callbacks=[efficiency_cb],
            optimizers=optimizers,
        )

        barrier()
        result = trainer.train()

        log(f"\n{'=' * 70}\n  TRL BASELINE RESULTS  (loss {result.training_loss:.4f})")
        log("\n" + format_benchmark_report(efficiency_cb))
        emit_benchmark(f"trl_baseline_{args.model}_s{args.seq}_b{args.batch_size}", efficiency_cb)
        log(f"{'=' * 70}\n")

    except Exception as e:
        failed = True
        log(f"\nBENCHMARK FAILED: {type(e).__name__}: {e}")
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
