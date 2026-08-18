#!/usr/bin/env python
"""
Test: expert-LoRA checkpoint + RESUME on the ep1 + FSDP-sharded-experts path (GptOss-20B, 2 GPUs).

``ep_size=1`` with ``fsdp_shard_ep1_experts`` (the default) leaves the EP module FSDP-managed, so the
grouped adapters are DTensors and ``load_expert_lora_state_dict`` re-shards each one with
``distribute_tensor`` — a COLLECTIVE. Every rank must therefore walk the adapters in the same order.
``_expert_lora_attrs`` is a frozenset, and frozenset iteration order over strings varies per process
(``PYTHONHASHSEED`` is unpinned under torchrun), so an unsorted loop lets ranks scatter different
tensors against each other: NCCL size mismatch, or a hang.

The ``ep_size=2`` sibling (test_lora_ep_experts_resume.py) cannot cover this — there the adapters are
plain FSDP-ignored tensors and the load does a local ``copy_`` with no collective at all.

To make the rank-consistency requirement deterministic rather than hash-seed dependent, rank 0 and
rank 1 are handed DIFFERENT container orders before the resume. A loop that iterates the container
visits them in opposite orders; only a name-sorted loop keeps the ranks in step.

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/lora/test_lora_ep1_fsdp_experts_resume.py
"""

import os
import shutil

import torch
from accelerate.utils import extract_model_from_parallel
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.expert_parallel.config import ExpertLoraSpec
from src.distributed.expert_parallel.expert_weights import gather_ep_lora_adapters, has_ep_lora
from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import ensure_model_downloaded, shared_scratch_dir
from tests.common.ep_reference import ep_layers
from tests.common.harness import gpu_test_main
from tests.common.models import GPT_OSS_20B
from tests.common.peft_helpers import freeze_base_keep_expert_adapters
from tests.common.utils import log

MODEL_NAME = GPT_OSS_20B
SAVE_AT_STEP = 2
MAX_SEQ_LENGTH = 512
LEARNING_RATE = 1e-3
LORA_R = 8
LORA_ALPHA = 16
SEED = 42


def _zero_live_adapters(model):
    for module in ep_layers(model):
        for attr in module._expert_lora_attrs:
            getattr(module, f"{attr}_lora_A").data.zero_()
            getattr(module, f"{attr}_lora_B").data.zero_()


def _skew_attr_order_per_rank(model, rank: int):
    """Give each rank a different container order, so only a sorted walk keeps them in step."""
    for module in ep_layers(model):
        attrs = sorted(module._expert_lora_attrs)
        module._expert_lora_attrs = attrs if rank % 2 == 0 else list(reversed(attrs))


def run(ctx) -> dict:
    log(f"\n{'=' * 70}\n  Expert-LoRA resume on ep1 + FSDP-sharded experts (DTensor adapters)\n{'=' * 70}")

    # The trainer writes the checkpoint on rank 0 and every rank resumes from it, so the output dir
    # must be one shared path rather than a per-rank temp dir.
    output_dir = shared_scratch_dir("lora_ep1_fsdp_resume")
    if ctx.rank == 0:
        ctx.on_teardown(lambda: shutil.rmtree(output_dir, ignore_errors=True))
    checkpoint = os.path.join(output_dir, f"checkpoint-{SAVE_AT_STEP}")

    ensure_model_downloaded(MODEL_NAME, ctx.rank)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log("\n[1/4] Loading GptOss-20B at ep_size=1 (experts FSDP-sharded) + expert-LoRA spec...")
    parallelism_config = ParallelismConfig(ep_size=1)
    parallelism_config.expert_lora = ExpertLoraSpec(
        r=LORA_R, alpha=LORA_ALPHA, projections=frozenset({"gate", "up", "down"})
    )
    model, _ = load_distributed_model(
        model_name_or_path=MODEL_NAME,
        parallelism_config=parallelism_config,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flex_attention",
        use_liger_kernel=True,
    )
    freeze_base_keep_expert_adapters(model)

    checks = {"experts_fsdp_managed": parallelism_config.create_ep_config().experts_fsdp_managed}
    checks["has_expert_lora"] = has_ep_lora(model)
    adapted = sorted({a for m in ep_layers(model) for a in m._expert_lora_attrs})
    checks["multiple_adapted_attrs"] = len(adapted) >= 2  # else ordering cannot differ
    log(f"  adapted attrs={adapted}, experts_fsdp_managed={checks['experts_fsdp_managed']}")

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
        fsdp="",
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=config,
        train_dataset=create_sft_dataset(64, tokenizer, seed=SEED),
        processing_class=tokenizer,
        parallelism_config=parallelism_config,
    )

    log(f"\n[2/4] Training {SAVE_AT_STEP} steps + checkpointing...")
    trainer.train()
    unwrapped = extract_model_from_parallel(trainer.model, recursive=True)
    # FSDP2 shards the EP module here, so the live adapters must be DTensors — that is what makes
    # the resume path collective. Without it this test would be a duplicate of the ep2 one.
    checks["adapters_are_dtensors"] = any(
        isinstance(getattr(m, f"{a}_lora_A").data, torch.distributed.tensor.DTensor)
        for m in ep_layers(unwrapped)
        for a in m._expert_lora_attrs
    )
    adapters_saved = {k: v.clone() for k, v in gather_ep_lora_adapters(unwrapped).items()}
    log(f"  checkpoint={os.path.isdir(checkpoint)}, adapters={len(adapters_saved)}")

    log("\n[3/4] Zeroing adapters and skewing the per-rank attr order...")
    _zero_live_adapters(unwrapped)
    corrupted = gather_ep_lora_adapters(unwrapped)
    checks["corrupt_zeroed_adapters"] = any(
        not torch.equal(adapters_saved[k].cpu(), corrupted[k].cpu()) for k in adapters_saved
    )
    _skew_attr_order_per_rank(unwrapped, ctx.rank)

    log("\n[4/4] Restoring through the trainer's resume hooks (collective per adapter)...")
    trainer._load_from_checkpoint(checkpoint)
    restored = gather_ep_lora_adapters(unwrapped)

    checks["all_adapters_restored_nonzero"] = all(v.abs().sum().item() > 0 for v in restored.values())
    checks["adapters_close_to_trained"] = all(
        torch.allclose(adapters_saved[k].float().cpu(), restored[k].float().cpu(), atol=2e-2, rtol=2e-2)
        for k in adapters_saved
    )

    return {"checks": checks}


main = gpu_test_main(min_world_size=2, prefix="lora_ep1_fsdp_resume")(run)

if __name__ == "__main__":
    main()
