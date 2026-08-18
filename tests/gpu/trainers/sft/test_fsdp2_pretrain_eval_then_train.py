#!/usr/bin/env python
"""FSDP2: an ``evaluate()`` before ``train()`` must not freeze training.

A gathered FSDP2 module registers its transient unsharded params, and the toolkit's
``reshard_after_forward=False`` leaves a pre-train eval in that state; HF builds the optimizer after
it. Without the re-shard in ``DistributedTrainerMixin.create_optimizer`` the param groups hold orphans
the next unshard discards: every step updates them, the shards never move, and the loss only drifts
with the batches while grad norms stay finite. It fails when:
  * the sharded params do not move after evaluate() -> train() (world MIN of the per-rank max |delta|)
  * the loss does not decrease over the run
  * the control arm (train() with no pre-train eval) fails either check — the measurement is broken
  * the pre-train eval no longer leaves the model gathered — the scenario stopped being exercised

Topology: 2 GPUs, plain FSDP2 (``ParallelismConfig()``), tiny Qwen3 over the Qwen3-0.6B vocabulary.
Run: torchrun --nproc_per_node=2 tests/gpu/trainers/sft/test_fsdp2_pretrain_eval_then_train.py
"""

import sys

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor
from transformers import AutoConfig, AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from trl import SFTConfig

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.harness import gpu_test_main, log
from tests.common.models import QWEN3_0_6B, TINY_QWEN3_CONFIG

STEPS = 8
LEARNING_RATE = 5e-3
MIN_LOSS_DROP = 0.05


def build_dense_model(tokenizer=None, *, vocab_size: int | None = None) -> Qwen3ForCausalLM:
    """Tiny dense Qwen3, identical on every rank (fixed seed).

    pad/eos ids must live inside the model's vocab: TRL copies the tokenizer's ids into the model
    config, and the saved config would otherwise fail ``from_pretrained``'s padding_idx assert. With
    no ``tokenizer`` the tiny config's own 1024-token vocab and ids 0/1 are used; a suite that
    tokenizes with a real tokenizer passes it together with that checkpoint's ``vocab_size``.
    """
    torch.manual_seed(1234)
    config = TINY_QWEN3_CONFIG if vocab_size is None else {**TINY_QWEN3_CONFIG, "vocab_size": vocab_size}
    return Qwen3ForCausalLM(
        Qwen3Config(
            **config,
            pad_token_id=0 if tokenizer is None else tokenizer.pad_token_id,
            eos_token_id=1 if tokenizer is None else tokenizer.eos_token_id,
        )
    )


def _shards(model) -> dict[str, torch.Tensor]:
    """This rank's sharded parameter data — what the optimizer must be updating."""
    return {
        name: (param.to_local() if isinstance(param, DTensor) else param).detach().clone()
        for name, param in model.named_parameters()
    }


def _world_min(value: float) -> float:
    reduced = torch.tensor([value], device=torch.cuda.current_device())
    dist.all_reduce(reduced, op=dist.ReduceOp.MIN)
    return float(reduced)


def _world_all(flag: bool) -> bool:
    return _world_min(float(flag)) > 0.0


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
    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    after = _shards(trainer.model)
    delta = max(float((after[name] - init).abs().max()) for name, init in before.items())
    log(
        f"[rank {ctx.rank}] {name}: losses={[f'{value:.3f}' for value in losses]} max|delta|={delta:.3e} "
        f"gathered_after_eval={gathered_after_eval}"
    )
    checks = {
        f"{name}_shards_moved": _world_min(delta) > 0.0,
        f"{name}_loss_decreased": _world_all(len(losses) == STEPS and losses[-1] < losses[0] - MIN_LOSS_DROP),
    }
    metrics = {
        f"{name}_max_delta": _world_min(delta),
        f"{name}_first_loss": losses[0],
        f"{name}_last_loss": losses[-1],
    }
    if eval_first:
        checks["pretrain_eval_leaves_model_gathered"] = _world_all(gathered_after_eval)
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
    sys.exit(run())
