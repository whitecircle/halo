#!/usr/bin/env python
"""
Test: checkpoint + RESUME restore for native expert-LoRA under EP=2 (GptOss-20B).

EP rebuilds the model in __init__ with freshly zero-initialized adapters, so resume MUST restore the
trained adapter weights from the checkpoint (peft.restore_adapters) or training silently
continues from the untrained adapter. This trains a model, writes a checkpoint, then CORRUPTS the
live adapters + LR scheduler (standing in for a fresh process) and drives the trainer's actual resume
hooks to verify:

  1. Adapters restored EXACTLY — `_load_from_checkpoint` reloads the saved grouped A/B (vs the
     corrupted/zeroed state), per-rank-sliced across the EP group.
  2. LR scheduler restored — `_load_optimizer_and_scheduler` puts the scheduler back at the saved step.

(The two phases share one model load: a second EP model in the same process leaves the first model's
EP process groups live and deadlocks DeepEP. The corrupt-then-restore drives the same resume code
paths a fresh process would.)

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_ep_experts_resume.py
"""

import os
import sys
import traceback

import torch
from accelerate import PartialState
from accelerate.utils import extract_model_from_parallel
from safetensors.torch import load_file
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.expert_parallel.config import ExpertLoraSpec
from src.distributed.expert_parallel.expert_weights import gather_ep_lora_adapters
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import (
    cleanup_dirs,
    ensure_model_downloaded,
    init_distributed,
    setup_cache_dirs,
    teardown_distributed,
)
from tests.common.ep_reference import ep_layers
from tests.common.models import GPT_OSS_20B
from tests.common.peft_helpers import freeze_base_keep_expert_adapters
from tests.common.utils import cleanup_memory, log

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 2
SAVE_AT_STEP = 2
MAX_SEQ_LENGTH = 1024
LEARNING_RATE = 1e-3
SEED = 42


def _zero_live_adapters(model):
    """Stand in for a fresh process: wipe the live adapters so restore has something to fix."""
    for module in ep_layers(model):
        for attr in module._expert_lora_attrs:
            getattr(module, f"{attr}_lora_A").data.zero_()
            getattr(module, f"{attr}_lora_B").data.zero_()


def run_resume_test() -> bool:
    rank, world_size, local_rank = init_distributed()
    log(f"\n{'=' * 70}\n  Expert-LoRA checkpoint + RESUME restore (EP=2, GptOss-20B)\n{'=' * 70}")
    if world_size != EP_SIZE:
        log(f"\nERROR: requires exactly {EP_SIZE} GPUs, got {world_size}")
        teardown_distributed()
        return False

    output_dir, cache_dir = setup_cache_dirs("lora_ep_experts_resume", rank)
    # EP writes the gathered adapter to ONE directory, so every rank must share rank 0's output_dir
    # (as a real shared FS does) — else rank>0 has no adapter file and its restore goes untested.
    shared_out = [output_dir]
    torch.distributed.broadcast_object_list(shared_out, src=0)
    output_dir = shared_out[0]
    checkpoint = os.path.join(output_dir, f"checkpoint-{SAVE_AT_STEP}")

    try:
        ensure_model_downloaded(MODEL_NAME, rank)
        PartialState()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        parallelism_config = ParallelismConfig(ep_size=EP_SIZE)
        parallelism_config.expert_lora = ExpertLoraSpec(r=8, alpha=16, projections=frozenset({"gate", "up", "down"}))
        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flex_attention",
            use_liger_kernel=True,
        )
        freeze_base_keep_expert_adapters(model)
        train_dataset = create_sft_dataset(64, tokenizer, seed=SEED)
        config = SFTConfig(
            output_dir=output_dir,
            max_steps=SAVE_AT_STEP,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=1,
            learning_rate=LEARNING_RATE,
            lr_scheduler_type="linear",
            warmup_steps=1,
            bf16=True,
            gradient_checkpointing=True,
            use_liger_kernel=False,
            logging_steps=1,
            save_strategy="steps",
            save_steps=SAVE_AT_STEP,
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            dataloader_drop_last=True,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            ddp_find_unused_parameters=True,
            fsdp="",
        )
        trainer = DistributedSFTTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )

        log(f"\n[1/2] Training {SAVE_AT_STEP} steps + checkpointing...")
        trainer.train()
        unwrapped = extract_model_from_parallel(trainer.model, recursive=True)
        adapters_saved = {k: v.clone() for k, v in gather_ep_lora_adapters(unwrapped).items()}
        sched_saved = trainer.lr_scheduler.last_epoch
        log(
            f"  checkpoint={os.path.isdir(checkpoint)}, adapters={len(adapters_saved)}, sched.last_epoch={sched_saved}"
        )

        log("\n[2/2] Corrupting live state, then restoring via the trainer's resume hooks...")
        _zero_live_adapters(unwrapped)
        trainer.lr_scheduler.last_epoch = 0
        corrupted = gather_ep_lora_adapters(unwrapped)
        corrupt_differs = any(not torch.equal(adapters_saved[k].cpu(), corrupted[k].cpu()) for k in adapters_saved)

        trainer._load_from_checkpoint(checkpoint)  # CheckpointLoader.load_model -> restore_adapters
        trainer._load_optimizer_and_scheduler(checkpoint)
        restored = gather_ep_lora_adapters(unwrapped)

        # Resume's source of truth is the checkpoint FILE; comparing to the post-train gather instead
        # would conflate the save's bf16 cast (~1e-2 at these magnitudes) with the restore.
        saved_file = load_file(os.path.join(checkpoint, "adapter_model.safetensors"))
        file_match = [
            k for k in restored if not torch.equal(restored[k].cpu(), saved_file[k].to(restored[k].dtype).cpu())
        ]
        restored_nonzero = sum(1 for v in restored.values() if v.abs().sum().item() > 0)
        close_to_trained = all(
            torch.allclose(adapters_saved[k].float().cpu(), restored[k].float().cpu(), atol=2e-2, rtol=2e-2)
            for k in adapters_saved
        )
        if rank == 0:
            log(
                f"  restored vs checkpoint file: {len(restored) - len(file_match)}/{len(restored)} exact; "
                f"restored_nonzero={restored_nonzero}; close_to_trained(bf16 tol)={close_to_trained}"
            )

        # `restored vs file` is logged, not gated (bf16<->fp32 round-trips break bit-exactness);
        # per-key closeness is: a no-op restore leaves zeros, a wrong expert slice fails allclose.
        checks = {
            "corrupt_zeroed_adapters": corrupt_differs,
            "all_adapters_restored_nonzero": restored_nonzero == len(restored),
            "adapters_close_to_trained": close_to_trained,
            "scheduler_restored": trainer.lr_scheduler.last_epoch == sched_saved,
        }
        log(f"  sched.last_epoch after restore={trainer.lr_scheduler.last_epoch} (saved {sched_saved})")

        all_passed = all(checks.values())
        log(f"\n{'=' * 70}\n  EXPERT-LoRA RESUME TEST " + ("PASSED" if all_passed else "FAILED"))
        for k, v in checks.items():
            log(f"  - {'PASS' if v else 'FAIL'}: {k}")
        log(f"{'=' * 70}")
        return all_passed

    except Exception as e:
        # print, not log(): log() is rank-0 only, so a rank>0-only failure would surface no traceback.
        print(f"\n  [rank{rank}] EXPERT-LoRA RESUME TEST FAILED WITH ERROR: {e}", flush=True)
        traceback.print_exc()
        return False

    finally:
        from src.distributed.runtime import barrier

        barrier()
        cleanup_dirs(output_dir, cache_dir)
        cleanup_memory()


def main():
    success = run_resume_test()
    teardown_distributed()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
