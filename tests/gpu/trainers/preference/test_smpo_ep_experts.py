#!/usr/bin/env python
"""
Test: native grouped-LoRA on EP experts with a NON-SFT trainer (SmoothMarginPOTrainer, EP=2).

Expert-LoRA mechanics (adapter build, grad-sync, save) live in the shared DistributedTrainerMixin /
EP layers, so this confirms they ride a preference trainer's path end to end — frozen base + only
expert adapters trainable, adapters move, and trainer.save_model writes a standalone adapter file.

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/preference/test_smpo_ep_experts.py
"""

import math
import os
import sys
import traceback

import torch
from accelerate import PartialState
from accelerate.utils import extract_model_from_parallel
from safetensors.torch import load_file
from transformers import AutoTokenizer

from src.configs.smpo_config import SmoothMarginPOConfig
from src.distributed.expert_parallel.config import ExpertLoraSpec
from src.distributed.expert_parallel.expert_weights import gather_ep_lora_adapters, has_ep_lora
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.preference.smpo import SmoothMarginPOTrainer
from tests.common.datasets import create_preference_dataset
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
from tests.common.utils import cleanup_memory, gpu_mem_gb, log

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 2
NUM_TRAIN_SAMPLES = 64
MAX_LENGTH = 2048
MAX_PROMPT_LENGTH = 1024
NUM_TRAIN_STEPS = 5
LEARNING_RATE = 1e-3  # large so adapters move clearly
SEED = 42


def run_smpo_ep_experts_test():
    rank, world_size, local_rank = init_distributed()
    log(f"\n{'=' * 70}\n  SMPO + native expert-LoRA (EP=2, GptOss-20B)\n{'=' * 70}")
    log(f"  World size: {world_size}, EP size: {EP_SIZE}")

    if world_size != EP_SIZE:
        log(f"\nERROR: requires exactly {EP_SIZE} GPUs, got {world_size}")
        teardown_distributed()
        return False

    output_dir, cache_dir = setup_cache_dirs("smpo_ep_experts", rank)

    try:
        ensure_model_downloaded(MODEL_NAME, rank)
        PartialState()
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        train_dataset = create_preference_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)

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
        log(f"  GPU memory after load: {gpu_mem_gb():.1f}GB")

        checks = {"has_expert_lora": has_ep_lora(model)}
        non_adapter_trainable = []
        for module in ep_layers(model):
            native = {f"{a}_lora_{w}" for a in module._expert_lora_attrs for w in ("A", "B")}
            non_adapter_trainable += [n for n, p in module.named_parameters() if p.requires_grad and n not in native]
        checks["only_adapters_trainable_in_ep"] = not non_adapter_trainable

        base_before = {k: v.clone() for k, v in ep_layers(model)[0].gather_expert_state_dict().items()}
        adapters_before = gather_ep_lora_adapters(model)

        config = SmoothMarginPOConfig(
            output_dir=output_dir,
            max_steps=NUM_TRAIN_STEPS,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=2,
            learning_rate=LEARNING_RATE,
            bf16=True,
            gradient_checkpointing=True,
            use_liger_kernel=False,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            max_length=MAX_LENGTH,
            max_prompt_length=MAX_PROMPT_LENGTH,
            dataloader_drop_last=True,
            remove_unused_columns=False,
            dataloader_num_workers=0,
            ddp_find_unused_parameters=True,
            fsdp="",
        )
        trainer = SmoothMarginPOTrainer(
            model=model,
            args=config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )
        barrier()
        train_result = trainer.train()
        log(f"  Training loss: {train_result.training_loss:.6f}, steps: {train_result.global_step}")
        checks["loss_finite"] = math.isfinite(train_result.training_loss)
        checks["steps_completed"] = train_result.global_step == NUM_TRAIN_STEPS

        unwrapped = extract_model_from_parallel(trainer.model, recursive=True)
        base_after = ep_layers(unwrapped)[0].gather_expert_state_dict()
        checks["base_frozen"] = all(torch.equal(base_before[k].cpu(), base_after[k].cpu()) for k in base_before)

        adapters_after = gather_ep_lora_adapters(unwrapped)
        b_moved = any(
            k.endswith(".lora_B") and not torch.equal(adapters_before[k].cpu(), adapters_after[k].cpu())
            for k in adapters_before
        )
        checks["adapters_moved"] = b_moved

        # trainer.save_model adapter-only write (save_ep_checkpoint Case B)
        save_dir = os.path.join(output_dir, "adapter")
        trainer.save_model(save_dir)
        barrier()
        if rank == 0:
            f = os.path.join(save_dir, "adapter_model.safetensors")
            ek = [k for k in load_file(f) if "experts." in k and ".lora_" in k] if os.path.exists(f) else []
            checks["save_model_adapter_file"] = len(ek) > 0
            log(f"  adapter_model.safetensors expert-lora keys: {len(ek)}")

        all_passed = all(checks.values())
        log(f"\n{'=' * 70}\n  SMPO EXPERT-LoRA TEST " + ("PASSED" if all_passed else "FAILED"))
        for k, v in checks.items():
            log(f"  - {'PASS' if v else 'FAIL'}: {k}")
        log(f"{'=' * 70}")
        return all_passed

    except Exception as e:
        log(f"\n  SMPO EXPERT-LoRA TEST FAILED WITH ERROR: {e}")
        if rank == 0:
            traceback.print_exc()
        return False

    finally:
        barrier()
        cleanup_dirs(output_dir, cache_dir)
        cleanup_memory()


def main():
    success = run_smpo_ep_experts_test()
    teardown_distributed()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
