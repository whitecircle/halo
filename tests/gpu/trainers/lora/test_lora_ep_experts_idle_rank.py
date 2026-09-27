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

import math
import os
import sys

import torch
import torch.distributed as dist
from transformers import AutoTokenizer
from transformers.models.zaya.configuration_zaya import ZayaConfig
from transformers.models.zaya.modeling_zaya import ZayaForCausalLM, ZayaRouter
from trl import SFTConfig

from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.sft import DistributedSFTTrainer
from tests.common.datasets import create_sft_dataset
from tests.common.ep_reference import ep_layers
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B, TINY_ZAYA_CONFIG
from tests.common.peft_helpers import load_peft_model
from tests.common.utils import log, log_all, step_losses

IDLE = "all"
EP_SIZE = 2
STEPS = 4
MAX_LENGTH = 128
SEED = 42
# Selection bias that decides top-1 over any softmax probability: the chosen class gets +2, the other
# expert and the discard slot -2.
PIN_BIAS = 2.0


def expert_per_layer(num_layers: int) -> list[int]:
    """The expert every token routes to, per MoE layer, under ``IDLE``."""
    if IDLE == "all":
        return [0] * num_layers
    return [0] + [1] * (num_layers - 1)


def build_checkpoint(path: str, tokenizer) -> list[int]:
    """Save a seeded tiny Zaya whose gates pin every token to one expert per layer; returns the pins."""
    torch.manual_seed(SEED)
    config = ZayaConfig(
        **{
            **TINY_ZAYA_CONFIG,
            "max_position_embeddings": MAX_LENGTH,
            "vocab_size": len(tokenizer),
            "pad_token_id": tokenizer.pad_token_id,
        }
    )
    model = ZayaForCausalLM(config).to(torch.bfloat16)
    routers = [module for module in model.modules() if isinstance(module, ZayaRouter)]
    pins = expert_per_layer(len(routers))
    for router, expert in zip(routers, pins, strict=True):
        bias = torch.full_like(router.balancing_biases, -PIN_BIAS)
        bias[expert] = PIN_BIAS
        router.balancing_biases.copy_(bias)
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return pins


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
    checks = {}
    shared = [ctx.output_dir]
    dist.broadcast_object_list(shared, src=0)
    base = os.path.join(shared[0], "base")
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    pins = [None]
    if ctx.rank == 0:
        pins = [build_checkpoint(base, tokenizer)]
    dist.broadcast_object_list(pins, src=0)
    pins = pins[0]
    log(f"  --idle {IDLE}: expert per MoE layer {pins}")

    pc = ParallelismConfig(ep_size=EP_SIZE)
    model, tokenizer, peft_config = load_peft_model(
        "expert_lora", pc, model_name=base, attn_implementation="eager", use_liger_kernel=False
    )
    before = adapter_snapshot(model)
    args = SFTConfig(
        output_dir=os.path.join(shared[0], "out"),
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
    trainer.train()

    losses = step_losses(trainer)
    checks["all_steps_ran"] = trainer.state.global_step == STEPS and len(losses) == STEPS
    checks["losses_finite"] = all(math.isfinite(loss) for loss in losses)
    log(f"  losses {losses}")

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
    if "--idle" in sys.argv:
        i = sys.argv.index("--idle")
        IDLE = sys.argv[i + 1]
        del sys.argv[i : i + 2]
    if IDLE not in ("all", "first"):
        raise SystemExit(f"--idle must be 'all' or 'first', got {IDLE!r}")
    run()
