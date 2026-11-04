#!/usr/bin/env python
"""GptOss-20B trains at EP=2, TP=2 on FlexAttention, whose sinks shard across TP ranks; SDPA would silently drop them.
Compares no loss against a reference."""

import torch
from transformers import AutoTokenizer
from trl import SFTConfig

from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.parallel_shape import parallel_shape_checks
from tests.common.rank_log import log
from tests.common.training_checks import training_run_checks

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 2
TP_SIZE = 2
NUM_TRAIN_SAMPLES = 32
NUM_EVAL_SAMPLES = 8
MAX_SEQ_LENGTH = 4096
NUM_TRAIN_STEPS = 5
BATCH_SIZE = 1
LEARNING_RATE = 2e-5
SEED = 42

EXPECTED_TOTAL_HEADS = 64  # GptOss-20B's attention heads.
EXPECTED_LOCAL_HEADS = EXPECTED_TOTAL_HEADS // TP_SIZE


@gpu_test_main(min_world_size=TP_SIZE, prefix="sft_ep_tp_flex_test")
def run(ctx):
    log(f"\n{'#' * 70}")
    log(f"  SFT EP+TP FlexAttention Test (EP={EP_SIZE}, TP={TP_SIZE})")
    log(f"  World: {ctx.world_size}, Model: {MODEL_NAME}")
    log(f"  GPU: {torch.cuda.get_device_name(ctx.local_rank)}")
    log("  Attention: flex_attention (with sinks support)")
    log(f"{'#' * 70}")

    log("\nEnsuring model is downloaded...")
    ensure_model_downloaded(MODEL_NAME, ctx.rank)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("\n--- Creating synthetic datasets ---")
    train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)
    eval_dataset = create_sft_dataset(NUM_EVAL_SAMPLES, tokenizer, seed=SEED + 100)
    log(f"Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

    log(f"\n--- Loading model with EP={EP_SIZE}, TP={TP_SIZE}, flex_attention ---")
    parallelism_config = ParallelismConfig(ep_size=EP_SIZE, tp_size=TP_SIZE)
    log(f"Config: {parallelism_config.summary()}")

    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flex_attention",
        use_liger_kernel=True,
    )
    log(f"GPU mem after load: {torch.cuda.memory_allocated() / 1e9:.1f}GB")
    log(f"Params: {sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    log("\n--- Validating attention implementation ---")
    attn_impl = getattr(model.config, "_attn_implementation", None)
    log(f"Model attn_implementation: {attn_impl}")

    log("\n--- Validating sinks sharding ---")
    checks = {}

    sinks_sizes = []
    sinks_values = []
    for layer_idx, layer in enumerate(model.model.layers):
        attn = layer.self_attn
        if hasattr(attn, "sinks"):
            sinks = attn.sinks
            sinks_sizes.append(sinks.shape[0])
            sinks_values.append(sinks.data.clone())
            if layer_idx < 3:
                log(
                    f"  Layer {layer_idx}: sinks shape={sinks.shape}, "
                    f"requires_grad={sinks.requires_grad}, "
                    f"min={sinks.min().item():.4f}, max={sinks.max().item():.4f}"
                )

    if sinks_sizes:
        all_correct_size = all(s == EXPECTED_LOCAL_HEADS for s in sinks_sizes)
        checks["sinks_sharded"] = all_correct_size
        log(
            f"  Sinks sharded to {EXPECTED_LOCAL_HEADS} per rank: "
            f"{'PASS' if all_correct_size else 'FAIL'} "
            f"(sizes: {set(sinks_sizes)})"
        )

        # They were reset to dtype.min.
        any_nonzero = any(v.abs().max().item() > 0 for v in sinks_values)
        checks["sinks_nonzero"] = any_nonzero
        log(f"  Sinks have values (not all zeros): {'PASS' if any_nonzero else 'FAIL'}")
    else:
        log("  WARNING: No sinks found in model!")
        checks["sinks_sharded"] = False
        checks["sinks_nonzero"] = False

    sft_config = SFTConfig(
        output_dir=ctx.output_dir,
        max_steps=NUM_TRAIN_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,  # Already applied in load_distributed_model.
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        dataloader_num_workers=0,
    )

    log("\n--- Creating DistributedSFTTrainer ---")
    trainer = DistributedSFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )
    ctx.on_teardown(trainer.cleanup_ep)
    checks |= parallel_shape_checks(model, parallelism_config)

    log(f"\n--- Training ({NUM_TRAIN_STEPS} steps with FlexAttention) ---")
    train_result = trainer.train()
    checks |= training_run_checks(train_result, trainer, NUM_TRAIN_STEPS, grad_norms=True)
    return {"checks": checks}


if __name__ == "__main__":
    run()
