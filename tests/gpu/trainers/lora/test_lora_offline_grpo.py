#!/usr/bin/env python
"""LoRA / QLoRA / native-expert-LoRA training on the Offline GRPO trainer (+ adapter checkpoint).

One file, five adapter modes (selected by ``--mode``); each drives the production adapter path
(``split_expert_lora_targets`` → ``load_distributed_model`` → ``setup_peft_model``) into
``OfflineGRPOTrainer`` and asserts, each of which FAILS when the wiring breaks:

  - only adapters are trainable (frozen base, incl. the 4-bit QLoRA weights);
  - training is finite and the adapters actually move (zero-init lora_B becomes non-zero);
  - ``trainer.save_model`` writes a round-trippable adapter-only checkpoint.

Modes / parallelism:
    lora        — attention PEFT LoRA, dense Qwen3-0.6B, FSDP2 (no parallelism args)
    qlora       — attention PEFT LoRA on a 4-bit base, dense Qwen3-0.6B, FSDP2
    lora_ep     — attention PEFT LoRA, GptOss-20B MoE, EP=2
    expert_lora — native grouped LoRA on the MoE expert FFNs, GptOss-20B, EP=2
    lora_etp    — attention PEFT LoRA, GptOss-20B MoE, ep_size=1 + expert_tp_size=2 (pure ETP)

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_offline_grpo.py --mode lora
    ... --mode qlora | --mode lora_ep | --mode expert_lora | --mode lora_etp
"""

import argparse
import math
import sys
import traceback

from accelerate import PartialState

from src.configs.offline_grpo_config import OfflineGRPOConfig
from src.distributed.runtime import barrier
from src.trainers.grpo.offline import OfflineGRPOTrainer
from tests.common.datasets import create_offline_grpo_dataset
from tests.common.distributed import cleanup_dirs, init_distributed, setup_cache_dirs, teardown_distributed
from tests.common.peft_helpers import (
    assert_adapter_checkpoint,
    assert_adapters_moved,
    assert_only_adapters_trainable,
    is_expert_lora_active,
    load_peft_model,
    parallelism_config_for,
    snapshot_adapters,
    unwrap,
)
from tests.common.utils import cleanup_memory, log

MAX_STEPS = 3
NUM_TRAIN_SAMPLES = 32
NUM_COMPLETIONS = 4
SEED = 42


def run(mode: str, ep: int) -> bool:
    rank, world_size, local_rank = init_distributed()
    PartialState()
    output_dir, cache_dir = setup_cache_dirs(f"lora_offline_grpo_{mode}", rank)

    log(f"\n{'=' * 70}\n  Offline GRPO adapter test — mode={mode}, world={world_size}, ep={ep}\n{'=' * 70}")
    checks: dict[str, bool] = {}
    try:
        parallelism_config = parallelism_config_for(mode, ep)
        # gpt-oss needs flex_attention under EP; dense qwen uses the auto default.
        attn = "flex_attention" if parallelism_config.is_ep_mode else None
        model, tokenizer, peft_config = load_peft_model(mode, parallelism_config, attn_implementation=attn)
        expert_lora = is_expert_lora_active(model)
        log(f"  expert_lora_active={expert_lora}, peft_config={'set' if peft_config else 'None'}")

        config = OfflineGRPOConfig(
            output_dir=output_dir,
            max_steps=MAX_STEPS,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=1e-3,  # large so adapters move in a few steps
            bf16=True,
            gradient_checkpointing=True,
            use_liger_kernel=False,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            max_prompt_length=512,
            max_completion_length=512,
            dataloader_drop_last=True,
            fsdp="",
            ddp_find_unused_parameters=parallelism_config.is_ep_mode,
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
        # Inspect the trainer-wrapped model: attention LoRA is applied by the trainer's
        # get_peft_model (peft_config), so the adapters only exist after construction.
        wrapped = unwrap(trainer.model)
        ok, detail = assert_only_adapters_trainable(wrapped)
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
        ok, detail = assert_adapters_moved(adapters_before, adapters_after)
        checks["adapters_moved"] = ok
        log(f"  [moved] {detail}")

        ok, detail = assert_adapter_checkpoint(trainer, output_dir, rank, expert_lora=expert_lora)
        checks["adapter_checkpoint"] = ok
        log(f"  [checkpoint] {detail}")

        passed = all(checks.values())
        log(f"\n  OFFLINE-GRPO {mode.upper()} " + ("PASSED" if passed else "FAILED"))
        for k, v in checks.items():
            log(f"    - {'PASS' if v else 'FAIL'}: {k}")
        return passed
    except Exception as e:
        log(f"\n  OFFLINE-GRPO {mode} FAILED WITH ERROR: {e}")
        if rank == 0:
            traceback.print_exc()
        return False
    finally:
        barrier()
        cleanup_dirs(output_dir, cache_dir)
        cleanup_memory()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="lora", choices=["lora", "qlora", "lora_ep", "expert_lora", "lora_etp"])
    parser.add_argument("--ep", type=int, default=2)
    args = parser.parse_args()
    success = run(args.mode, args.ep)
    teardown_distributed()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
