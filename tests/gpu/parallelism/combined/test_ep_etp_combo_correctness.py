#!/usr/bin/env python
"""EP+ETP COMBO (ep_size>1 AND expert_tp_size>1) training correctness / deadlock regression.

Exercises the combination where the EP group splits into ``expert_tp_size`` DeepEP dispatch
groups coupled by the strided expert-TP all-reduce. That coupling deadlocks if the expert-TP
all-reduce sits BETWEEN DeepEP dispatch and combine: under FSDP2's multi-stream / multi-layer
execution it times out DeepEP's intranode combine barrier across the coupled dispatch groups
("CUDA error: unspecified launch failure", typically ~step 3). The reduction therefore runs in TOKEN
space, outside the dispatch->combine span (EPMoELayerBase._dispatch_compute_combine).

The deadlock needs several steps to surface, so this runs >3 steps through the REAL trainer
(FSDP2 + gradient checkpointing) — a single forward/backward does not trigger it.

Config: ep_size=2, expert_tp_size=2 (ep_group_size=4) on GptOss-20B → 4 GPUs.

Run:
    torchrun --nproc_per_node=4 \
        tests/gpu/parallelism/combined/test_ep_etp_combo_correctness.py
"""

import sys
import traceback

import torch
import torch.distributed as dist
from accelerate import PartialState
from transformers import AutoTokenizer
from trl import SFTConfig

from src.distributed.loading.model_loading import load_distributed_model
from src.distributed.parallelism_config import ParallelismConfig
from src.distributed.runtime import barrier
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import (
    cleanup_dirs,
    ensure_model_downloaded,
    init_distributed,
    setup_cache_dirs,
    teardown_distributed,
)
from tests.common.models import GPT_OSS_20B
from tests.common.utils import cleanup_memory, log

MODEL_NAME = GPT_OSS_20B
EP_SIZE = 2
EXPERT_TP_SIZE = 2
NUM_TRAIN_SAMPLES = 32
MAX_SEQ_LENGTH = 2048
NUM_TRAIN_STEPS = 6  # > 3 so the historical step-~3 deadlock would surface
BATCH_SIZE = 1
SEED = 42


def main():
    rank, world_size, local_rank = init_distributed()
    PartialState()

    log(f"\n{'#' * 70}")
    log(
        f"  EP+ETP COMBO correctness (ep_size={EP_SIZE}, expert_tp_size={EXPERT_TP_SIZE}, "
        f"ep_group_size={EP_SIZE * EXPERT_TP_SIZE})"
    )
    log(f"  World: {world_size}, Model: {MODEL_NAME}, Steps: {NUM_TRAIN_STEPS}")
    log(f"{'#' * 70}")

    if world_size < EP_SIZE * EXPERT_TP_SIZE:
        log(f"\nERROR: need >= {EP_SIZE * EXPERT_TP_SIZE} GPUs (ep_size*expert_tp_size), got {world_size}")
        teardown_distributed()
        return 1

    output_dir, cache_dir = setup_cache_dirs("ep_etp_combo_test", rank)
    try:
        ensure_model_downloaded(MODEL_NAME, rank)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        train_dataset = create_sft_dataset(NUM_TRAIN_SAMPLES, tokenizer, seed=SEED)

        parallelism_config = ParallelismConfig(ep_size=EP_SIZE, expert_tp_size=EXPERT_TP_SIZE)
        log(f"Config: {parallelism_config.summary()}")
        log(
            f"ep_group_size={parallelism_config.ep_group_size} dp_size={parallelism_config.data_parallel_size} "
            f"is_expert_tp_mode={parallelism_config.is_expert_tp_mode}"
        )
        assert parallelism_config.ep_size > 1 and parallelism_config.expert_tp_size > 1, (
            "this test must exercise the ep>1 AND expert_tp>1 combo"
        )

        model, _ = load_distributed_model(
            model_name_or_path=MODEL_NAME,
            parallelism_config=parallelism_config,
            dtype=torch.bfloat16,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            use_liger_kernel=True,
        )
        etp_layers = [
            m for _, m in model.named_modules() if hasattr(m, "ep_config") and getattr(m, "expert_tp_size", 1) > 1
        ]
        log(f"ETP layers (expert_tp_size>1): {len(etp_layers)}")
        assert etp_layers, "no ETP-wrapped layers found"

        sft_config = SFTConfig(
            output_dir=output_dir,
            max_steps=NUM_TRAIN_STEPS,
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=1,
            learning_rate=2e-5,
            bf16=True,
            gradient_checkpointing=True,  # required: GC + FSDP2 is the deadlock condition
            use_liger_kernel=False,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            max_length=MAX_SEQ_LENGTH,
            dataloader_drop_last=True,
            dataloader_num_workers=0,
            ddp_find_unused_parameters=True,
            fsdp="",
        )

        trainer = DistributedSFTTrainer(
            model=model,
            args=sft_config,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            parallelism_config=parallelism_config,
        )
        assert trainer.is_ep_mode, "trainer should be in EP mode"
        log(f"_fsdp_wrapped={getattr(trainer, '_fsdp_wrapped', '?')} (FSDP2 must be active for the deadlock path)")

        log(f"\n--- Training {NUM_TRAIN_STEPS} steps (deadlock would surface ~step 3) ---")
        train_result = trainer.train()  # a reduction inside dispatch->combine deadlocks here

        training_loss = train_result.training_loss
        step_losses = [e["loss"] for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e]
        grad_norms = [e["grad_norm"] for e in trainer.state.log_history if "grad_norm" in e]
        log(f"Final loss={training_loss:.6f}  step_losses={[f'{l:.4f}' for l in step_losses]}")
        if grad_norms:
            log(f"grad_norms={[f'{g:.2f}' for g in grad_norms]}")

        checks = {}
        checks["training_completed"] = len(step_losses) == NUM_TRAIN_STEPS
        checks["loss_finite"] = all(torch.isfinite(torch.tensor(l)) for l in step_losses + [training_loss])
        checks["loss_decreased"] = step_losses[-1] < step_losses[0]  # SFT on tiny data converges fast
        if grad_norms:
            checks["grad_finite"] = all(torch.isfinite(torch.tensor(g)) for g in grad_norms)

        # loss agreement across ranks (trainer reports the DP-averaged loss)
        lt = torch.tensor([training_loss], device=f"cuda:{local_rank}")
        gathered = [torch.zeros_like(lt) for _ in range(world_size)]
        dist.all_gather(gathered, lt)
        spread = max(v.item() for v in gathered) - min(v.item() for v in gathered)
        checks["loss_consistent"] = spread < 0.01

        all_passed = all(checks.values())
        if rank == 0:
            log("\n--- Checks ---")
            for k, v in checks.items():
                log(f"  {k}: {'PASS' if v else 'FAIL'}")
            log(f"  loss spread across ranks: {spread:.6f}")
            log(f"\n{'#' * 70}")
            log(f"  EP+ETP COMBO TEST {'PASSED' if all_passed else 'FAILED'}")
            if not all_passed:
                log(f"  Failed: {[k for k, v in checks.items() if not v]}")
            log(f"{'#' * 70}\n")

        trainer.cleanup_ep()
        del trainer, model
        cleanup_memory()
        cleanup_dirs(output_dir, cache_dir)
        barrier()
        teardown_distributed()
        return 0 if all_passed else 1
    except Exception as e:
        log(f"\nFATAL ERROR: {type(e).__name__}: {e}")
        if rank == 0:
            traceback.print_exc()
        cleanup_memory()
        cleanup_dirs(output_dir, cache_dir)
        teardown_distributed()
        return 1


if __name__ == "__main__":
    sys.exit(main())
