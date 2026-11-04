#!/usr/bin/env python
"""Reward (Bradley-Terry) and Classification on a tiny Qwen3-MoE across the parallel layouts of ``MODES``.

The only trainers whose loss reads one pooled position per row, so the parallel MoE forward must reproduce the dense
output at that token. A mis-permuted combine changes the pooled score, not the shape, so the step-1 loss is pinned
against the exact objective on a dense copy of the seeded weights over the captured batch.
"""

import argparse
import os
import sys

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Qwen3MoeConfig,
    Qwen3MoeForSequenceClassification,
)
from trl import RewardConfig

from halo.configs.classification import ClassificationConfig
from halo.distributed.parallelism_config import ParallelismConfig
from halo.trainers.reward.bradley_terry import DistributedRewardTrainer
from halo.trainers.reward.classification import ClassificationTrainer
from tests.common.checkpoint_io import save_schedule
from tests.common.distributed import pin_deterministic_ep_dispatch, shared_output_dir, world_mean
from tests.common.dp_layout import dp_layout_checks, probe_forward
from tests.common.ep_reference import ep_layers
from tests.common.harness import gpu_test_main
from tests.common.memory import finish_phase
from tests.common.models import QWEN3_0_6B, TINY_QWEN3_MOE_CONFIG
from tests.common.parallel_shape import parallel_shape_checks
from tests.common.rank_log import log
from tests.common.resume_export import ResumeExportRoute, load_route_model, run_resume_export, save_base_checkpoint

MODES = {
    "ep2": (2, {"ep_size": 2}),
    "tp2": (2, {"tp_size": 2}),
    "etp2": (2, {"expert_tp_size": 2}),
    "ep1": (2, {"ep_size": 1}),
    "ep1_zero3": (2, {"ep_size": 1, "fsdp_reshard_after_forward": True}),
    "ep1_unsharded": (2, {"ep_size": 1, "fsdp_shard_ep1_experts": False}),
    "ep2tp2": (2, {"ep_size": 2, "tp_size": 2}),
    "ep2etp2": (4, {"ep_size": 2, "expert_tp_size": 2}),
    "tp2dp2": (4, {"tp_size": 2}),
    "ep2dp2": (4, {"ep_size": 2}),
    "ep2x2dom": (4, {"ep_size": 2, "gpus_per_node": 2}),
}
N_ROWS = 32
N_STEPS = 6
# Lands mid-epoch on every row (4 or more steps per epoch).
SAVE_STEP = N_STEPS // 2
PAIRS_PER_RANK = 2
SEQ_LEN = 64
PAD_ID = 0
NUM_LABELS = 3
CENTER_COEF = 0.05

# bf16 kernel and reduction-order noise against the reference's dense HF experts, plus a bf16-rounded logged loss.
# Measured under ep2 / tp2 / etp2: 4.4e-4 / 2.0e-3 / 3.2e-4 (reward), 1.0e-3 / 1.5e-3 / 7.7e-4 (classification).
LOSS_RTOL = 5e-3
# A control inside this band can't see a real bug. Measured: 0.42-0.43 (reward, halves swapped), 0.11-0.26
# (classification, labels rolled).
CONTROL_MIN_RTOL = 0.02


def build_model(num_labels: int) -> Qwen3MoeForSequenceClassification:
    torch.manual_seed(1234)
    config = Qwen3MoeConfig(**TINY_QWEN3_MOE_CONFIG, num_labels=num_labels, pad_token_id=PAD_ID, eos_token_id=1)
    return Qwen3MoeForSequenceClassification(config).to(torch.bfloat16)


def build_reward_dataset(seed: int) -> Dataset:
    """Ragged lengths move the pooled position per row, so right values at wrong positions change the loss."""
    g = torch.Generator().manual_seed(seed)
    vocab = TINY_QWEN3_MOE_CONFIG["vocab_size"]
    rows = []
    for _ in range(N_ROWS):
        chosen_len = int(torch.randint(8, SEQ_LEN, (1,), generator=g))
        rejected_len = int(torch.randint(8, SEQ_LEN, (1,), generator=g))
        rows.append(
            {
                "chosen_ids": torch.randint(4, vocab, (chosen_len,), generator=g).tolist(),
                "rejected_ids": torch.randint(4, vocab, (rejected_len,), generator=g).tolist(),
                "margin": round(float(torch.rand(1, generator=g)) * 0.5, 3),
            }
        )
    return Dataset.from_list(rows)


def build_classification_dataset(seed: int) -> Dataset:
    g = torch.Generator().manual_seed(seed)
    vocab = TINY_QWEN3_MOE_CONFIG["vocab_size"]
    rows = []
    for i in range(N_ROWS):
        length = int(torch.randint(8, SEQ_LEN, (1,), generator=g))
        ids = torch.randint(4, vocab, (length,), generator=g).tolist()
        rows.append({"input_ids": ids, "attention_mask": [1] * length, "labels": i % NUM_LABELS})
    return Dataset.from_list(rows)


def reward_loss(pooled: torch.Tensor, margin: torch.Tensor) -> float:
    chosen, rejected = torch.chunk(pooled.float(), chunks=2)
    loss = -F.logsigmoid(chosen - rejected - margin.float()).mean()
    return float(loss + CENTER_COEF * ((chosen + rejected) ** 2).mean())


def classification_loss(pooled: torch.Tensor, labels: torch.Tensor) -> float:
    return float(F.cross_entropy(pooled.float(), labels))


def pooled_logits(model, batch: dict, device) -> torch.Tensor:
    with torch.no_grad():
        outputs = model(input_ids=batch["input_ids"].to(device), attention_mask=batch["attention_mask"].to(device))
    return outputs.logits.squeeze(-1) if outputs.logits.shape[-1] == 1 else outputs.logits


def build_trainer(kind: str, model, dataset, tokenizer, config, output_dir, *, max_steps=N_STEPS, save_every=None):
    common = {
        "output_dir": output_dir,
        "max_steps": max_steps,
        "learning_rate": 5e-3,
        "lr_scheduler_type": "constant",
        "logging_steps": 1,
        **save_schedule(save_every),
        "report_to": [],
        "logging_nan_inf_filter": False,
        "seed": 42,
        "bf16": True,
        "gradient_checkpointing": False,
        "max_length": SEQ_LEN,
        "dataloader_num_workers": 0,
    }
    if kind == "reward":
        return DistributedRewardTrainer(
            model=model,
            args=RewardConfig(
                per_device_train_batch_size=PAIRS_PER_RANK,
                center_rewards_coefficient=CENTER_COEF,
                **common,
            ),
            train_dataset=dataset,
            processing_class=tokenizer,
            parallelism_config=config,
        )
    return ClassificationTrainer(
        model=model,
        args=ClassificationConfig(per_device_train_batch_size=2 * PAIRS_PER_RANK, **common),
        train_dataset=dataset,
        processing_class=tokenizer,
        parallelism_config=config,
        is_binary=False,
        is_multi_label=False,
    )


parser = argparse.ArgumentParser()
parser.add_argument("--trainer", choices=("reward", "classification"), default="reward")
parser.add_argument("--mode", choices=sorted(MODES), default="ep2")
ARGS = parser.parse_args()
WORLD_SIZE, LAYOUT = MODES[ARGS.mode]


@gpu_test_main(exact_world_size=WORLD_SIZE, prefix="ep_pooled_head")
def run(ctx):
    kind = ARGS.trainer

    checks, metrics = {}, {}
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    # The tiny vocab is 1024, far below Qwen's real pad id. Data rows are drawn from [4, vocab), so token 0 never
    # collides.
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(PAD_ID)

    num_labels = 1 if kind == "reward" else NUM_LABELS
    config = ParallelismConfig(**LAYOUT)
    if config.ep_size > 1:
        # The resume phase replays a run exactly, which DeepEP's atomic receive order would not.
        pin_deterministic_ep_dispatch()
    root = shared_output_dir(ctx)
    base = save_base_checkpoint(ctx, build_model(num_labels), tokenizer, os.path.join(root, "base"))
    dataset = build_reward_dataset(7) if kind == "reward" else build_classification_dataset(7)

    def route_trainer(model, output_dir, *, max_steps, save_every, resume_checkpoint):
        # Neither script tells its trainer about a resume: train() restores everything from the checkpoint.
        return build_trainer(
            kind, model, dataset, tokenizer, config, output_dir, max_steps=max_steps, save_every=save_every
        )

    # The dense reference's attention, so the pin compares the parallel forward to its own kernel.
    route = ResumeExportRoute(
        route_trainer,
        AutoModelForSequenceClassification,
        dtype=torch.bfloat16,
        attn_implementation=TINY_QWEN3_MOE_CONFIG["attn_implementation"],
    )
    trainer = build_trainer(
        kind, load_route_model(route, base, config, resume=False), dataset, tokenizer, config, ctx.output_dir
    )
    ctx.on_teardown(trainer.cleanup_ep)
    checks |= parallel_shape_checks(trainer.model, config)
    # Every decoder layer is sparse, so a load that wrapped only some of them passes the check above.
    checks["every_moe_layer_wrapped"] = len(ep_layers(trainer.model)) == TINY_QWEN3_MOE_CONFIG["num_hidden_layers"]

    # So the reference consumes exactly the rows and padding the parallel forward saw.
    captured: dict = {}
    real_compute_loss = trainer.compute_loss

    def spying_compute_loss(model_arg, inputs, *args, **kwargs):
        if not captured:
            captured.update({key: value.clone() for key, value in inputs.items() if torch.is_tensor(value)})
        return real_compute_loss(model_arg, inputs, *args, **kwargs)

    trainer.compute_loss = spying_compute_loss
    trainer.train()

    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    checks["ran_all_steps"] = trainer.state.global_step == N_STEPS and len(losses) == N_STEPS
    checks["losses_finite"] = bool(torch.isfinite(torch.tensor(losses)).all())
    # Random-token rows need no monotonic descent, but a dead EP backward shows as a zero grad norm and a frozen loss.
    grad_norms = [entry["grad_norm"] for entry in trainer.state.log_history if "grad_norm" in entry]
    checks["grad_norm_logged_finite"] = bool(grad_norms) and bool(torch.isfinite(torch.tensor(grad_norms)).all())
    checks["grad_norm_nonzero"] = bool(grad_norms) and min(grad_norms) > 1e-6
    checks["loss_responded_to_updates"] = abs(losses[-1] - losses[0]) > 1e-3
    metrics["first_loss"], metrics["last_loss"] = losses[0], losses[-1]
    checks |= dp_layout_checks(
        trainer.model,
        LAYOUT,
        world_size=ctx.world_size,
        ep_config=trainer._ep_config,
        forward=lambda: probe_forward(trainer.model, ctx.device),
    )

    # The logged loss is the world mean over ranks, each holding its DP shard, so the reference is averaged the same
    # way.
    checks["captured_step1_batch"] = bool(captured)
    reference_model = build_model(num_labels).to(ctx.device).eval()
    scores = pooled_logits(reference_model, captured, ctx.device)
    # A collapsed head would make every comparison below vacuous.
    checks["reference_scores_nondegenerate"] = bool(torch.isfinite(scores).all() and scores.float().std() > 1e-4)

    if kind == "reward":
        margin = captured["margin"]
        reference = reward_loss(scores, margin)
        # Control: every pair scored backwards.
        swapped = torch.cat(torch.chunk(scores, 2)[::-1])
        control = reward_loss(swapped, margin)
    else:
        labels = captured["labels"]
        reference = classification_loss(scores, labels)
        # Control: same logits, wrong targets.
        control = classification_loss(scores, labels.roll(1))

    expected = world_mean(reference, ctx.device)
    expected_control = world_mean(control, ctx.device)
    error = abs(losses[0] - expected) / abs(expected)
    control_error = abs(losses[0] - expected_control) / abs(expected_control)
    log(f"[rank {ctx.rank}] step1={losses[0]:.6f} reference={expected:.6f} control={expected_control:.6f}")
    metrics["step1_loss"], metrics["step1_reference"] = losses[0], expected
    metrics["step1_rel_err"], metrics["control_rel_err"] = error, control_error
    checks["step1_loss_matches_dense_reference"] = error < LOSS_RTOL
    checks["control_misses"] = control_error > CONTROL_MIN_RTOL
    finish_phase(trainer)

    phase_checks, phase_metrics = run_resume_export(
        ctx, route, config, base=base, root=os.path.join(root, "resume"), save_step=SAVE_STEP, total_steps=N_STEPS
    )
    checks |= phase_checks
    metrics |= phase_metrics
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    sys.exit(run())
