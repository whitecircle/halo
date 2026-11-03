#!/usr/bin/env python
"""SMPO's first step under every parallel layout equals an undistributed recompute, then saves, resumes and exports.

SMPO's loss is the rank's own mean, so a layout that miscounts it or mis-combines an expert still trains finite. The
routed path is compared where it is computed, since at this random init its output barely reaches the loss (a halved
combine moves the step-1 loss 1e-3 relative).
"""

import argparse
import contextlib
import os

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM, Qwen3MoeConfig, Qwen3MoeForCausalLM

from halo.configs.preference import SMPOConfig
from halo.distributed.parallelism_config import ParallelismConfig
from halo.model_loading.parallel.loader import load_distributed_model
from halo.trainers.preference.smpo import SMPOTrainer
from tests.common.checkpoint_io import save_schedule
from tests.common.distributed import pin_deterministic_ep_dispatch, shared_output_dir
from tests.common.dp_layout import dp_layout_checks, hsdp_layout, probe_forward
from tests.common.first_step import (
    dense_gradient_checks,
    first_step_checks,
    score_first_step,
    train_recording_first_step,
)
from tests.common.harness import gpu_test_main
from tests.common.memory import finish_phase
from tests.common.models import TINY_QWEN3_CONFIG, TINY_QWEN3_MOE_CONFIG
from tests.common.parallel_shape import parallel_shape_checks
from tests.common.pp_trainers import PAD_ID, tiny_vocab_tokenizer
from tests.common.resume_export import ResumeExportRoute, load_route_model, run_resume_export, save_base_checkpoint
from tests.common.routed_first_step import pinned_reference_routing, record_routed_first_step, routed_first_step_checks
from tests.common.training_checks import training_run_checks

# ``None`` fields are HSDP's, sized by the world.
MODES = {
    "ep2": (2, "moe", {"ep_size": 2}),
    "ep1": (2, "moe", {"ep_size": 1}),
    "ep1_zero3": (2, "moe", {"ep_size": 1, "fsdp_reshard_after_forward": True}),
    "ep1_unsharded": (2, "moe", {"ep_size": 1, "fsdp_shard_ep1_experts": False}),
    "etp2": (2, "moe", {"expert_tp_size": 2}),
    "ep2tp2": (2, "moe", {"ep_size": 2, "tp_size": 2}),
    "ep2etp2": (4, "moe", {"ep_size": 2, "expert_tp_size": 2}),
    "ep2dp2": (4, "moe", {"ep_size": 2}),
    "ep2x2dom": (4, "moe", {"ep_size": 2, "gpus_per_node": 2}),
    "zero3": (2, "dense", {"fsdp_reshard_after_forward": True}),
    "hsdp": (4, "dense", None),
}
N_ROWS = 48
N_STEPS = 6
SAVE_STEP = N_STEPS // 2
PAIRS_PER_RANK = 2
GRAD_ACCUM = 2
SEQ_LEN = 64
LEARNING_RATE = 5e-3
# bf16 forwards over another reduction order, and under EP grouped-GEMM against the stock loop. A loss counted once per
# rank is 100% off.
LOSS_RTOL = 0.02
MISCOUNT_FACTOR = 2

parser = argparse.ArgumentParser()
parser.add_argument("--mode", choices=sorted(MODES), required=True)
ARGS = parser.parse_args()
WORLD_SIZE, FAMILY, FIELDS = MODES[ARGS.mode]


def build_model():
    torch.manual_seed(1234)
    if FAMILY == "moe":
        return Qwen3MoeForCausalLM(Qwen3MoeConfig(**TINY_QWEN3_MOE_CONFIG, pad_token_id=PAD_ID, eos_token_id=1))
    return Qwen3ForCausalLM(Qwen3Config(**TINY_QWEN3_CONFIG, pad_token_id=PAD_ID, eos_token_id=1))


def build_dataset(vocab_size: int, seed: int = 7) -> Dataset:
    g = torch.Generator().manual_seed(seed)

    def ids(lo: int, hi: int) -> list[int]:
        return torch.randint(4, vocab_size, (int(torch.randint(lo, hi, (1,), generator=g)),), generator=g).tolist()

    rows = {"prompt_input_ids": [], "chosen_input_ids": [], "rejected_input_ids": []}
    for _ in range(N_ROWS):
        rows["prompt_input_ids"].append(ids(4, 16))
        rows["chosen_input_ids"].append(ids(4, 24) + [1])
        rows["rejected_input_ids"].append(ids(4, 24) + [1])
    return Dataset.from_dict(rows)


def smpo_args(output_dir: str, *, max_steps: int = N_STEPS, save_every: int | None = None) -> SMPOConfig:
    return SMPOConfig(
        output_dir=output_dir,
        max_steps=max_steps,
        per_device_train_batch_size=PAIRS_PER_RANK,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="constant",
        # Off, so the recorded step-1 gradient is the backward's own; HF still logs its norm.
        max_grad_norm=0.0,
        bf16=True,
        gradient_checkpointing=False,
        logging_steps=1,
        **save_schedule(save_every),
        report_to=[],
        logging_nan_inf_filter=False,
        seed=42,
        max_length=SEQ_LEN,
        max_prompt_length=16,
        # Batch quantiles of the percentile clip would compare bf16 tails across the two runs.
        lower_clip_percentile=None,
        upper_clip_percentile=None,
        dataloader_num_workers=0,
    )


@gpu_test_main(exact_world_size=WORLD_SIZE, prefix=f"smpo_layouts_{ARGS.mode}")
def run(ctx):
    fields = hsdp_layout(ctx.world_size) if FIELDS is None else FIELDS
    config = ParallelismConfig(**fields)
    if config.ep_size > 1:
        # The resume phase replays a run exactly, which DeepEP's atomic receive order would not.
        pin_deterministic_ep_dispatch()
    tokenizer = tiny_vocab_tokenizer()
    root = shared_output_dir(ctx)
    base = save_base_checkpoint(ctx, build_model().to(torch.bfloat16), tokenizer, os.path.join(root, "base"))
    vocab = (TINY_QWEN3_MOE_CONFIG if FAMILY == "moe" else TINY_QWEN3_CONFIG)["vocab_size"]
    dataset = build_dataset(vocab)

    def route_trainer(model, output_dir, *, max_steps, save_every, resume_checkpoint):
        # The SMPO script tells its trainer nothing about a resume: train() restores everything.
        return SMPOTrainer(
            model=model,
            args=smpo_args(output_dir, max_steps=max_steps, save_every=save_every),
            train_dataset=dataset,
            processing_class=tokenizer,
            parallelism_config=config,
        )

    route = ResumeExportRoute(route_trainer, AutoModelForCausalLM, dtype=torch.bfloat16, attn_implementation="eager")
    trainer = route_trainer(
        load_route_model(route, base, config, resume=False),
        ctx.output_dir,
        max_steps=N_STEPS,
        save_every=None,
        resume_checkpoint=None,
    )
    ctx.on_teardown(trainer.cleanup_ep)
    checks = parallel_shape_checks(trainer.model, config)
    routed = record_routed_first_step(trainer) if FAMILY == "moe" else None
    result, record = train_recording_first_step(trainer)
    checks |= training_run_checks(result, trainer, N_STEPS, grad_norms=True)
    checks |= dp_layout_checks(
        trainer.model,
        fields,
        world_size=ctx.world_size,
        ep_config=trainer._ep_config,
        forward=lambda: probe_forward(trainer.model, ctx.device),
    )
    finish_phase(trainer)

    # Plain FSDP2 DP over the same initial weights, the experts HF's own loop.
    reference_config = ParallelismConfig(use_grouped_gemm=False)
    reference_model, _ = load_distributed_model(
        base, reference_config, torch.bfloat16, trust_remote_code=True, attn_implementation="eager"
    )
    reference_trainer = SMPOTrainer(
        model=reference_model,
        args=smpo_args(os.path.join(ctx.output_dir, "reference")),
        processing_class=tokenizer,
        parallelism_config=reference_config,
    )
    pinning = contextlib.nullcontext() if routed is None else pinned_reference_routing(reference_trainer.model, routed)
    with pinning as pinned:
        reference = score_first_step(reference_trainer, record)
    loss_checks, metrics = first_step_checks(record, reference, miscount_factor=MISCOUNT_FACTOR, loss_rtol=LOSS_RTOL)
    grad_checks, grad_metrics = dense_gradient_checks(record, reference)
    checks |= loss_checks | grad_checks
    metrics |= grad_metrics
    if routed is not None:
        routed_checks, routed_metrics = routed_first_step_checks(routed, pinned, reference.grads, device=ctx.device)
        checks |= routed_checks
        metrics |= routed_metrics
    finish_phase(reference_trainer)

    phase_checks, phase_metrics = run_resume_export(
        ctx, route, config, base=base, root=os.path.join(root, "resume"), save_step=SAVE_STEP, total_steps=N_STEPS
    )
    return {"checks": checks | phase_checks, "metrics": metrics | phase_metrics}


if __name__ == "__main__":
    run()
