#!/usr/bin/env python
"""Offline GRPO on a family whose attention reads row indices, at batch 2 under FSDP2 and EP2: a left-padded row must
score as it does alone, through the per-row forwards that takes, K of them per backward, and the steps must complete,
past the trainer's guard against a group a backward left unreduced.

    torchrun --nproc_per_node=2 tests/gpu/trainers/grpo/test_offline_grpo_row_positions.py --family deepseek_v4 --gc off
    torchrun --nproc_per_node=2 tests/gpu/trainers/grpo/test_offline_grpo_row_positions.py --family mistral4 --gc reentrant
"""

import argparse
import math

import torch

from halo.distributed.parallelism_config import ParallelismConfig
from halo.runtime.collectives import barrier
from tests.common.distributed import shared_scratch_dir
from tests.common.harness import gpu_test_main
from tests.common.offline_grpo import (
    MISTRAL4_STEPPED_ROPE,
    build_offline_grpo_trainer,
    make_offline_tokenizer,
    mixed_length_groups,
    offline_grpo_config,
    token_logps,
)
from tests.common.rank_log import log
from tests.common.tiny_models import TINY_MOE_FAMILIES, build_tiny_family_checkpoint
from tests.common.training_checks import step_losses

# The released 0.1 moves a tiny row by less than bf16's kernel noise.
VISIBLE_SCALE_BETA = 50.0
FAMILIES = {
    "deepseek_v4": {},
    "mistral4": {"rope_parameters": {**MISTRAL4_STEPPED_ROPE, "llama_4_scaling_beta": VISIBLE_SCALE_BETA}},
}
STEPS = 4
BATCH = 2
SEQUENCE_LENGTH = 64
# Over a bf16 forward's kernel noise; a left-padded row moves by more than a tenth of a nat on these weights.
ALONE_TOL = 5e-2

parser = argparse.ArgumentParser()
parser.add_argument("--family", choices=sorted(FAMILIES), required=True)
parser.add_argument("--gc", choices=["off", "reentrant"], required=True)
ARGS = parser.parse_args()


def _first_left_padded_batch(trainer) -> dict[str, torch.Tensor]:
    """The sampler shuffles rows, so a batch can hold equal prompts, which no pad tells from the rows alone. Every rank
    takes one batch, so the forwards below stay paired."""
    for batch in trainer.get_train_dataloader():
        if not batch["prompt_attention_mask"][:, 0].all():
            return trainer._prepare_inputs(batch)
    raise RuntimeError("No batch of this seed left-pads a row, so the check below would pass on any scorer.")


@gpu_test_main(exact_world_size=2, prefix="offline_grpo_row_positions")
def run(ctx):
    torch.cuda.set_device(ctx.device)
    path = shared_scratch_dir(f"offline_row_positions_{ARGS.family}")
    tokenizer = make_offline_tokenizer()
    if ctx.rank == 0:
        build_tiny_family_checkpoint(TINY_MOE_FAMILIES[ARGS.family], path, tokenizer, overrides=FAMILIES[ARGS.family])
    barrier()

    args = offline_grpo_config(
        ctx.output_dir,
        steps=STEPS,
        save_steps=STEPS,
        seed=0,
        kl_beta=0.0,
        sequence_length=SEQUENCE_LENGTH,
        evaluate=False,
        save=False,
        per_device_train_batch_size=BATCH,
        gradient_checkpointing=ARGS.gc == "reentrant",
        gradient_checkpointing_kwargs={"use_reentrant": True} if ARGS.gc == "reentrant" else None,
        use_chunked_grpo_logprobs=False,
    )
    train = mixed_length_groups(BATCH * STEPS * ctx.world_size * 2)
    trainer = build_offline_grpo_trainer(ctx, path, ParallelismConfig(ep_size=2), args, train)
    checks = {"takes_the_chunked_scorer": trainer._use_chunked_grpo_logprobs}

    batch = _first_left_padded_batch(trainer)
    ids, mask, keep = trainer._joined_rows(batch)
    with torch.no_grad():
        together = trainer._get_per_token_logps(trainer.model, ids, mask, keep)
        worst = 0.0
        for row in range(ids.size(0)):
            completion = int(batch["completion_attention_mask"][row].sum())
            alone = token_logps(trainer.model, ids[row][mask[row].bool()].unsqueeze(0))[0, -completion:]
            worst = max(worst, float((together[row, :completion] - alone).abs().max()))
    log(f"  a row against itself alone: {worst:.2e} (tol {ALONE_TOL})")
    checks["a_row_scores_as_it_does_alone"] = worst < ALONE_TOL

    result = trainer.train()
    losses = step_losses(trainer)
    checks["steps_completed"] = result.global_step == STEPS
    checks["losses_finite"] = bool(losses) and all(math.isfinite(loss) for loss in losses)
    return {"checks": checks, "metrics": {"alone_gap": worst, "final_loss": result.training_loss}}


if __name__ == "__main__":
    run()
