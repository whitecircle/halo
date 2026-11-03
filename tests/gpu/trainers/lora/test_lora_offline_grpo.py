#!/usr/bin/env python
"""LoRA, QLoRA and native expert LoRA on the Offline GRPO trainer through the production adapter path, with an adapter
checkpoint."""

import argparse
import math

from halo.configs.grpo_offline import OfflineGRPOConfig
from halo.distributed.expert_parallel.moe_layer.expert_lora import model_has_expert_lora
from halo.runtime.collectives import barrier
from halo.trainers.grpo.offline.trainer import OfflineGRPOTrainer
from tests.common.adapters import (
    check_adapters_moved,
    check_only_adapters_trainable,
    load_peft_model,
    parallelism_config_for,
    save_and_check_adapter_checkpoint,
    snapshot_adapters,
    unwrap,
)
from tests.common.datasets import create_offline_grpo_dataset
from tests.common.harness import gpu_test_main
from tests.common.rank_log import log

MAX_STEPS = 3
NUM_TRAIN_SAMPLES = 32
NUM_COMPLETIONS = 4
SEED = 42


def run(ctx) -> dict:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="lora", choices=["lora", "qlora", "lora_ep", "expert_lora", "lora_etp"])
    parser.add_argument("--ep", type=int, default=2)
    args, _ = parser.parse_known_args()
    mode, ep = args.mode, args.ep

    log(f"\n{'=' * 70}\n  Offline GRPO adapter test — mode={mode}, world={ctx.world_size}, ep={ep}\n{'=' * 70}")
    checks: dict[str, bool] = {}
    parallelism_config = parallelism_config_for(mode, ep)
    # gpt-oss needs flex_attention under EP; dense qwen uses the auto default.
    attn = "flex_attention" if parallelism_config.has_ep_group else None
    model, tokenizer, peft_config = load_peft_model(mode, parallelism_config, attn_implementation=attn)
    expert_lora = model_has_expert_lora(model)
    log(f"  expert_lora_active={expert_lora}, peft_config={'set' if peft_config else 'None'}")

    config = OfflineGRPOConfig(
        output_dir=ctx.output_dir,
        max_steps=MAX_STEPS,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=1e-3,  # Large so adapters move in a few steps.
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=False,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_prompt_length=512,
        max_completion_length=512,
        dataloader_drop_last=True,
    )
    trainer = OfflineGRPOTrainer(
        model=model,
        args=config,
        train_dataset=create_offline_grpo_dataset(
            tokenizer, NUM_TRAIN_SAMPLES, seed=SEED, num_completions=NUM_COMPLETIONS
        ),
        processing_class=tokenizer,
        peft_config=peft_config,
        parallelism_config=parallelism_config,
    )
    ctx.on_teardown(trainer.cleanup_ep)
    # The trainer's get_peft_model applies attention LoRA, so the adapters only exist after construction.
    wrapped = unwrap(trainer.model)
    ok, detail = check_only_adapters_trainable(wrapped)
    checks["only_adapters_trainable"] = ok
    log(f"  [trainable] {detail}")
    adapters_before = snapshot_adapters(wrapped, expert_lora=expert_lora)

    barrier()
    result = trainer.train()
    checks["loss_finite"] = math.isfinite(result.training_loss)
    checks["steps_done"] = result.global_step == MAX_STEPS
    log(f"  training_loss={result.training_loss:.5f}, steps={result.global_step}")

    after_model = unwrap(trainer.model)
    adapters_after = snapshot_adapters(after_model, expert_lora=expert_lora)
    ok, detail = check_adapters_moved(adapters_before, adapters_after)
    checks["adapters_moved"] = ok
    log(f"  [moved] {detail}")

    ok, detail = save_and_check_adapter_checkpoint(trainer, ctx.output_dir, ctx.rank, expert_lora=expert_lora)
    checks["adapter_checkpoint"] = ok
    log(f"  [checkpoint] {detail}")

    # Only rank 0 reads the checkpoint back; share its verdict so every rank exits alike.
    return {"checks": ctx.broadcast_checks(checks)}


main = gpu_test_main(min_world_size=2, prefix="lora_offline_grpo")(run)

if __name__ == "__main__":
    main()
