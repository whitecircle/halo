#!/usr/bin/env python
"""FSDP2: an ``evaluate()`` before ``train()`` must not freeze training.

A pre-train eval leaves transient unsharded params registered, and HF builds the optimizer after it. Without the
re-shard in ``create_optimizer`` the groups hold orphans: shards never move and the loss drifts with finite grad norms.
"""

import torch
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoTokenizer
from trl import SFTConfig

from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.distributed import world_all, world_min
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.pp_equivalence import build_dense_model
from tests.common.rank_log import log
from tests.common.training_checks import step_losses

STEPS = 8
LEARNING_RATE = 5e-3
MIN_LOSS_DROP = 0.05


def _shards(model) -> dict[str, torch.Tensor]:
    """This rank's sharded parameter data, what the optimizer must be updating."""
    return {
        name: (param.to_local() if isinstance(param, DTensor) else param).detach().clone()
        for name, param in model.named_parameters()
    }


def _arm(ctx, name: str, *, eval_first: bool) -> tuple[dict, dict]:
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    args = SFTConfig(
        output_dir=f"{ctx.output_dir}/{name}",
        max_steps=STEPS,
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant",
        bf16=True,
        logging_steps=1,
        eval_strategy="no",
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=64,
        dataloader_num_workers=0,
        seed=42,
    )
    trainer = DistributedSFTTrainer(
        model=build_dense_model(tokenizer, vocab_size=AutoConfig.from_pretrained(QWEN3_0_6B).vocab_size),
        args=args,
        train_dataset=create_sft_dataset(64, tokenizer, seed=0),
        eval_dataset=create_sft_dataset(16, tokenizer, seed=1),
        processing_class=tokenizer,
        parallelism_config=ParallelismConfig(),
    )
    before = _shards(trainer.model)
    gathered_after_eval = False
    if eval_first:
        trainer.evaluate()
        gathered_after_eval = not any(isinstance(param, DTensor) for param in trainer.model.parameters())
    trainer.train()
    losses = step_losses(trainer)
    after = _shards(trainer.model)
    delta = max(float((after[name] - init).abs().max()) for name, init in before.items())
    log(
        f"[rank {ctx.rank}] {name}: losses={[f'{value:.3f}' for value in losses]} max|delta|={delta:.3e} "
        f"gathered_after_eval={gathered_after_eval}"
    )
    checks = {
        f"{name}_shards_moved": world_min(delta) > 0.0,
        f"{name}_loss_decreased": world_all(len(losses) == STEPS and losses[-1] < losses[0] - MIN_LOSS_DROP),
    }
    metrics = {
        f"{name}_max_delta": world_min(delta),
        f"{name}_first_loss": losses[0],
        f"{name}_last_loss": losses[-1],
    }
    if eval_first:
        checks["pretrain_eval_leaves_model_gathered"] = world_all(gathered_after_eval)
    return checks, metrics


@gpu_test_main(exact_world_size=2, prefix="fsdp2_pretrain_eval")
def run(ctx):
    checks, metrics = {}, {}
    for name, eval_first in (("eval_then_train", True), ("control_train_only", False)):
        arm_checks, arm_metrics = _arm(ctx, name, eval_first=eval_first)
        checks.update(arm_checks)
        metrics.update(arm_metrics)
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
