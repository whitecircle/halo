#!/usr/bin/env python
"""PP+EP checkpoint resume through the real DistributedSFTTrainer (pp2 × ep2, gpt-oss tiny MoE).

The EP-specific resume seams on top of the dense PP resume test:

  * PP+EP weight resume loads at construction (the loader cannot ingest HF-format keys into the
    fused per-rank expert layout), so the resume trainer's model is built from_pretrained(ckpt) —
    exactly what prepare_distributed_resume arranges in the training scripts;
  * the construction guard: a model built from base weights must be rejected by the load path
    (its experts would silently resume from base weights);
  * router balancing biases restore through the stage's global_parameter_name remap — post-load
    biases bitwise equal the saved global-named entries for this stage's layers (without the
    remap, stage 1 would look up stage 0's names and keep zero biases);
  * EP expert optimizer moments (plain per-rank tensors) restore bitwise from this rank's shard;
  * end-to-end: train(resume) continues to completion with finite losses.

Topology: 4 GPUs as 2 simulated NVLink domains of 2, moe_balancing=bias_update.

Run: torchrun --nproc_per_node=4 tests/gpu/parallelism/pp/test_pp_ep_resume.py
"""

import json
import os
import sys

import torch
import torch.distributed as dist
from transformers import AutoTokenizer, GptOssForCausalLM
from trl import SFTConfig

from src.distributed.expert_parallel.moe_layer.layer import named_ep_layers
from src.distributed.parallelism_config import ParallelismConfig
from src.models.unwrapping import unwrap_toolkit_wrappers
from tests.common.distributed import shared_output_dir, world_all
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B, TINY_GPTOSS_CONFIG
from tests.common.pp_equivalence import build_gptoss_model
from tests.common.pp_trainers import (
    build_bias_balanced_sft_trainer,
    build_pp_args,
    build_sft_rows,
    losses_identical_across_ranks,
    restored_moments_match_shard,
)
from tests.common.utils import log

PP_SIZE = 2
EP_SIZE = 2
SEQ_LEN = 64
N_STEPS = 8
SAVE_STEP = 4


@gpu_test_main(exact_world_size=4, prefix="pp_ep_resume")
def run(ctx):
    checks, metrics = {}, {}
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    config = ParallelismConfig(
        world_size=ctx.world_size, gpus_per_node=2, nvlink_domain_size=2, pp_size=PP_SIZE, ep_size=EP_SIZE
    )

    root = shared_output_dir(ctx)

    def build_trainer(subdir: str, max_steps: int, save_steps: int | None = None, from_ckpt: str | None = None):
        model = (
            build_gptoss_model()
            if from_ckpt is None
            else GptOssForCausalLM.from_pretrained(from_ckpt, dtype=torch.bfloat16, attn_implementation="eager")
        )
        args = build_pp_args(
            SFTConfig,
            os.path.join(root, subdir),
            max_steps=max_steps,
            per_device_train_batch_size=4,
            max_length=SEQ_LEN,
            save_strategy="steps" if save_steps else "no",
            save_steps=save_steps or 500,
            bf16=True,
            gradient_checkpointing=False,  # resume compares optimizer state, not the checkpoint path
        )
        return build_bias_balanced_sft_trainer(
            model, args, config, tokenizer, build_sft_rows(TINY_GPTOSS_CONFIG["vocab_size"])
        )

    # ── train 4 steps, checkpoint at step 4 (EP gather + biases + optimizer shards) ─────────────
    trainer_a = build_trainer("runA", SAVE_STEP, save_steps=SAVE_STEP)
    ctx.on_teardown(trainer_a.cleanup_ep)
    trainer_a.train()
    ckpt = os.path.join(root, "runA", f"checkpoint-{SAVE_STEP}")
    dist.barrier()
    rank = dist.get_rank()
    # TRL copies the (full-size) tokenizer's special-token ids into the tiny config during
    # training, so the saved config would fail from_pretrained's padding_idx assert — restore ids
    # that exist in the tiny vocab so the resume path exercises PP mechanics, not the test's
    # tokenizer/model size mismatch.
    if rank == 0:
        for name in ("config.json", "generation_config.json"):
            path = os.path.join(ckpt, name)
            if os.path.isfile(path):
                with open(path) as f:
                    cfg = json.load(f)
                cfg["pad_token_id"], cfg["eos_token_id"] = 0, 1
                with open(path, "w") as f:
                    json.dump(cfg, f)
    dist.barrier()
    checks["checkpoint_written"] = (
        os.path.isfile(os.path.join(ckpt, "model.safetensors.index.json"))
        and os.path.isfile(os.path.join(ckpt, f"optimizer_shard_{rank:05d}.pt"))
        and os.path.isfile(os.path.join(ckpt, "router_balancing_biases.pt"))
    )

    # ── the construction guard: a base-built model must be rejected by the load path ────────────
    trainer_base = build_trainer("runBase", N_STEPS)
    ctx.on_teardown(trainer_base.cleanup_ep)
    guard_raised = False
    try:
        trainer_base._load_from_checkpoint(ckpt)
    except ValueError as exc:
        guard_raised = "constructed FROM the checkpoint" in str(exc)
        log(f"[guard] raised as expected: {str(exc)[:120]}")
    checks["base_built_model_rejected"] = world_all(guard_raised)

    # ── the production resume path: model constructed from the checkpoint ───────────────────────
    trainer_b = build_trainer("runB", N_STEPS, from_ckpt=ckpt)
    ctx.on_teardown(trainer_b.cleanup_ep)
    trainer_b._load_from_checkpoint(ckpt)
    stage = unwrap_toolkit_wrappers(trainer_b.model)

    # Balancing biases: restored through the global remap, bitwise equal the file's entries for
    # exactly this stage's layers.
    saved_biases = torch.load(os.path.join(ckpt, "router_balancing_biases.pt"), weights_only=False)
    live_biases = {
        name: module.balancing_biases
        for name, module in stage.named_modules()
        if getattr(module, "balancing_biases", None) is not None
    }
    biases_ok = len(live_biases) == 4  # 8 layers, 4 per stage
    nonzero = False
    for name, bias in live_biases.items():
        reference = saved_biases.get(stage.global_parameter_name(name))
        if reference is None or not torch.equal(bias.cpu().to(reference.dtype), reference):
            log(f"[biases] mismatch for {name} -> {stage.global_parameter_name(name)}")
            biases_ok = False
        elif bool((reference.abs() > 0).any()):
            nonzero = True
    checks["balancing_biases_restored_global_named"] = biases_ok
    checks["balancing_biases_nonzero"] = nonzero  # zero saved biases would make the check vacuous

    # Expert optimizer moments: plain per-rank tensors keyed by stage-local FQN, restored bitwise.
    trainer_b.create_optimizer_and_scheduler(num_training_steps=N_STEPS)
    trainer_b._load_optimizer_and_scheduler(ckpt)
    expert_prefixes = tuple(f"{name}." for name, _ in named_ep_layers(stage))
    checks["stage_has_ep_layers"] = bool(expert_prefixes)
    expert_ok, n_expert_fqns = restored_moments_match_shard(stage, trainer_b.optimizer, ckpt, rank, expert_prefixes)
    checks["expert_optimizer_moments_bitwise_equal_shard"] = expert_ok
    metrics["expert_state_fqns"] = n_expert_fqns

    # ── end-to-end: resume to completion with finite losses ─────────────────────────────────────
    trainer_b.train(resume_from_checkpoint=ckpt)
    losses = [e["loss"] for e in trainer_b.state.log_history if "loss" in e]
    checks["resume_trained_to_completion"] = trainer_b.state.global_step == N_STEPS and len(losses) == N_STEPS
    checks["resumed_losses_finite"] = bool(torch.isfinite(torch.tensor(losses)).all())
    checks["losses_identical_across_ranks"] = losses_identical_across_ranks(losses)
    metrics["losses"] = ", ".join(f"{v:.5f}" for v in losses)

    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
