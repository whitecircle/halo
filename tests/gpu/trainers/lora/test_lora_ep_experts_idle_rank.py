#!/usr/bin/env python
"""Expert-only LoRA at ep2 trains through the production trainer while one rank's experts idle.

Tiny Zaya has two experts and top-1 routing, so at ep2 each rank owns one expert and a layer whose
gate sends every token to one expert gives the other rank nothing to compute. Nothing upstream of the
experts trains, so that rank's backward has to take the same DeepEP collectives as its peer without
a gradient of its own to carry. Routing is pinned through the gate's native ``balancing_biases``
(selection only, saved in the checkpoint):

  --idle all    every layer routes to expert 0: rank 1 idles in every layer.
  --idle first  layer 0 routes to expert 0, later layers to expert 1: each rank idles somewhere.

Checks: every step completes with a finite loss; an adapter bank that received tokens moved, and one
that never did is bit-unchanged.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 tests/gpu/trainers/lora/test_lora_ep_experts_idle_rank.py --idle all
"""

import argparse
import os

import torch
from transformers import AutoTokenizer
from transformers.models.zaya.modeling_zaya import ZayaRouter
from trl import SFTConfig

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.ep_reference import ep_layers
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.peft_helpers import load_peft_model
from tests.common.tiny_models import TINY_MOE_FAMILIES, shared_tiny_family_checkpoint
from tests.common.utils import log, log_all, step_losses, training_run_checks

EP_SIZE = 2
STEPS = 4
MAX_LENGTH = 128
SEED = 42
# Selection bias that decides top-1 over any softmax probability: the chosen class gets +2, the other
# expert and the discard slot -2.
PIN_BIAS = 2.0


def expert_per_layer(idle: str, num_layers: int) -> list[int]:
    """The expert every token routes to, per MoE layer, under ``--idle``."""
    if idle == "all":
        return [0] * num_layers
    return [0] + [1] * (num_layers - 1)


def pin_routing(idle: str):
    """An ``edit`` for :func:`build_tiny_family_checkpoint`: every gate pins its tokens to one expert."""

    def edit(model) -> None:
        routers = [module for module in model.modules() if isinstance(module, ZayaRouter)]
        for router, expert in zip(routers, expert_per_layer(idle, len(routers)), strict=True):
            bias = torch.full_like(router.balancing_biases, -PIN_BIAS)
            bias[expert] = PIN_BIAS
            router.balancing_biases.copy_(bias)

    return edit


def adapter_snapshot(model) -> list[list[torch.Tensor]]:
    return [
        [
            getattr(layer, f"{attr}_lora_{side}").detach().clone()
            for attr in sorted(layer._expert_lora_attrs)
            for side in "AB"
        ]
        for layer in ep_layers(model)
    ]


@gpu_test_main(exact_world_size=EP_SIZE, prefix="lora_ep_experts_idle_rank")
def run(ctx):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--idle", choices=("all", "first"), required=True)
    idle = parser.parse_args().idle
    checks = {}
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    base = shared_tiny_family_checkpoint(
        ctx, TINY_MOE_FAMILIES["zaya"], f"idle_rank_zaya_{idle}", tokenizer, SEED, edit=pin_routing(idle)
    )

    pc = ParallelismConfig(ep_size=EP_SIZE)
    model, tokenizer, peft_config = load_peft_model(
        "expert_lora", pc, model_name=base, attn_implementation="eager", use_liger_kernel=False
    )
    pins = expert_per_layer(idle, len(ep_layers(model)))
    log(f"  --idle {idle}: expert per MoE layer {pins}")
    before = adapter_snapshot(model)
    args = SFTConfig(
        output_dir=os.path.join(ctx.output_dir, "out"),
        max_steps=STEPS,
        per_device_train_batch_size=2,
        learning_rate=2e-3,
        lr_scheduler_type="constant",
        bf16=True,
        gradient_checkpointing=False,
        use_liger_kernel=False,
        max_length=MAX_LENGTH,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
    )
    trainer = DistributedSFTTrainer(
        model=model,
        args=args,
        train_dataset=create_sft_dataset(16 * STEPS, tokenizer, seed=SEED),
        processing_class=tokenizer,
        parallelism_config=pc,
        peft_config=peft_config,
    )
    ctx.on_teardown(trainer.cleanup_ep)
    checks.update(training_run_checks(trainer.train(), trainer, STEPS))
    losses = step_losses(trainer)
    # Every step's loss must be logged for all_steps_finite to cover it.
    checks["every_step_logged"] = len(losses) == STEPS

    after = adapter_snapshot(model)
    layers = ep_layers(model)
    for index, (layer, pin) in enumerate(zip(layers, pins, strict=True)):
        received = layer.expert_start <= pin < layer.expert_end
        unchanged = all(torch.equal(a, b) for a, b in zip(before[index], after[index], strict=True))
        checks[f"l{index}_{'active_bank_moved' if received else 'idle_bank_unchanged'}"] = (
            not unchanged if received else unchanged
        )
    log_all(f"  rank {ctx.rank} owns experts {layers[0].expert_start}-{layers[0].expert_end - 1}")
    return {"checks": checks, "metrics": {"final_loss": losses[-1] if losses else float("nan")}}


if __name__ == "__main__":
    run()
