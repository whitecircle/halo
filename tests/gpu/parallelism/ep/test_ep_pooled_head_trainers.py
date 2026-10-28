#!/usr/bin/env python
"""Reward (Bradley-Terry) and Classification on a tiny Qwen3-MoE under EP (ep2), attention TP (tp2) and expert
TP (etp2), each on two ranks.

These two are the only supported trainers whose loss reads a single pooled position per row instead of
per-token logits: the parallel MoE forward has to reproduce the dense output at exactly the token the
sequence-classification pooling later selects. Under EP that is the dispatch→expert→combine round trip,
under TP the attention projections split across the ranks, under ETP each expert's FFN; the head itself
(``score``) is a non-expert parameter in every mode.

Every seam is silent when it breaks — a mis-permuted combine or a dropped expert changes the pooled
score, not the shape — so the step-1 loss is pinned against the trainer's exact objective recomputed
on a DENSE, unpatched copy of the same seeded weights over the CAPTURED batch:

  * reward: TRL's ``-logsigmoid(chosen - rejected - margin).mean() + coef * mean((c+r)**2)`` over the
    ``[chosen ⧺ rejected]`` chunk layout, margin included.
  * classification: the pooled cross-entropy mean the model's own head computes.

Each comparison carries a perturbation control that must MISS by orders of magnitude (reward: the
pair halves swapped; classification: the labels rolled), so a reference that had silently collapsed
to a constant could not pass. The axis is read off the model (``parallel_shape_checks``), or the mode
would be a dense run wearing a parallel config.

The model loads as the reward and classification scripts load it, from a saved base through
``load_distributed_model``. A second phase (:func:`tests.common.resume_export.run_resume_export`) trains with
a checkpoint midway and at the end, resumes a fresh trainer built from the midway checkpoint and holds it to an
exact replay, and loads the final checkpoint through ``AutoModelForSequenceClassification`` with the trained
score head.

Run: torchrun --nproc_per_node=2 tests/gpu/parallelism/ep/test_ep_pooled_head_trainers.py \
         --trainer reward --mode ep2
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
from tests.common.ep_reference import ep_layers
from tests.common.harness import gpu_test_main
from tests.common.memory import finish_phase
from tests.common.models import QWEN3_0_6B, TINY_QWEN3_MOE_CONFIG
from tests.common.parallel_shape import parallel_shape_checks
from tests.common.rank_log import log
from tests.common.resume_export import ResumeExportRoute, load_route_model, run_resume_export, save_base_checkpoint

# Each row's parallel shape on its two ranks; tp2 keeps the experts on the grouped-GEMM EP layers at ep_size 1.
MODES = {"ep2": {"ep_size": 2}, "tp2": {"tp_size": 2}, "etp2": {"expert_tp_size": 2}}
N_ROWS = 32
N_STEPS = 6
# Midway through N_STEPS, which lands mid-epoch on every row (4 or more steps per epoch).
SAVE_STEP = N_STEPS // 2
PAIRS_PER_RANK = 2
SEQ_LEN = 64
PAD_ID = 0
NUM_LABELS = 3
CENTER_COEF = 0.05

# The EP path computes the expert FFN through DeepEP dispatch/combine in bf16 against the
# reference's dense HF experts, so the two agree only to bf16 kernel and reduction-order noise (plus
# the logged loss itself being a bf16-rounded scalar). Measured relative error under ep2: 1.2e-4
# (reward), 4.9e-4 (classification); tp2 and etp2 sum bf16 partials across their ranks instead, noise
# of the same kind. Relative, because the two objectives differ in scale.
LOSS_RTOL = 5e-3
# A control that lands inside this band would mean the comparison cannot see a real bug. Measured
# control error: 0.42 (reward, halves swapped), 0.11 (classification, labels rolled).
CONTROL_MIN_RTOL = 0.02


def build_model(num_labels: int) -> Qwen3MoeForSequenceClassification:
    """The seeded tiny MoE seq-cls model. Identical on every rank and for the dense reference."""
    torch.manual_seed(1234)
    config = Qwen3MoeConfig(**TINY_QWEN3_MOE_CONFIG, num_labels=num_labels, pad_token_id=PAD_ID, eos_token_id=1)
    return Qwen3MoeForSequenceClassification(config).to(torch.bfloat16)


def build_reward_dataset(seed: int) -> Dataset:
    """Pre-tokenized preference rows with RAGGED lengths and a nonzero margin column.

    Ragged lengths move the pooled position per row, so a combine that returned the right values at
    the wrong positions changes the loss; the margin column keeps that leg of TRL's objective live.
    """
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
    """Pre-tokenized classification rows with ragged lengths and every class represented."""
    g = torch.Generator().manual_seed(seed)
    vocab = TINY_QWEN3_MOE_CONFIG["vocab_size"]
    rows = []
    for i in range(N_ROWS):
        length = int(torch.randint(8, SEQ_LEN, (1,), generator=g))
        ids = torch.randint(4, vocab, (length,), generator=g).tolist()
        rows.append({"input_ids": ids, "attention_mask": [1] * length, "labels": i % NUM_LABELS})
    return Dataset.from_list(rows)


def reward_loss(pooled: torch.Tensor, margin: torch.Tensor) -> float:
    """TRL's exact Bradley-Terry objective on the ``[chosen ⧺ rejected]`` chunk layout."""
    chosen, rejected = torch.chunk(pooled.float(), chunks=2)
    loss = -F.logsigmoid(chosen - rejected - margin.float()).mean()
    return float(loss + CENTER_COEF * ((chosen + rejected) ** 2).mean())


def classification_loss(pooled: torch.Tensor, labels: torch.Tensor) -> float:
    """The pooled single-label cross-entropy the model's own head computes."""
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


@gpu_test_main(exact_world_size=2, prefix="ep_pooled_head")
def run(ctx):
    parser = argparse.ArgumentParser()
    parser.add_argument("--trainer", choices=("reward", "classification"), default="reward")
    parser.add_argument("--mode", choices=sorted(MODES), default="ep2")
    cli = parser.parse_args()
    kind = cli.trainer

    checks, metrics = {}, {}
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    # The tiny model's vocab is 1024, far below Qwen's real pad id, so pad with token 0 — data rows
    # are drawn from [4, vocab), so the pad token never collides with a real one.
    tokenizer.pad_token = tokenizer.convert_ids_to_tokens(PAD_ID)

    num_labels = 1 if kind == "reward" else NUM_LABELS
    config = ParallelismConfig(**MODES[cli.mode])
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

    # The dense reference's attention, so the step-1 pin compares the parallel forward to its own kernel.
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

    # ── Capture step 1's batch straight off the trainer's own compute_loss, so the reference
    # consumes exactly the rows and padding the parallel forward saw.
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
    # The rows are random token noise, so the objective is not expected to descend monotonically;
    # what must hold is that gradients reach the pooled head at all. A dead EP backward (the combine
    # returning detached or zero-grad tensors) shows as a zero grad norm and a frozen loss.
    grad_norms = [entry["grad_norm"] for entry in trainer.state.log_history if "grad_norm" in entry]
    checks["grad_norm_logged_finite"] = bool(grad_norms) and bool(torch.isfinite(torch.tensor(grad_norms)).all())
    checks["grad_norm_nonzero"] = bool(grad_norms) and min(grad_norms) > 1e-6
    checks["loss_responded_to_updates"] = abs(losses[-1] - losses[0]) > 1e-3
    metrics["first_loss"], metrics["last_loss"] = losses[0], losses[-1]

    # ── Step-1 loss vs the exact objective on a DENSE copy of the init weights. The logged loss is
    # the world mean over ranks, each holding its own DP shard (or, under tp2 and etp2, the same batch),
    # so the reference is averaged the same way.
    checks["captured_step1_batch"] = bool(captured)
    reference_model = build_model(num_labels).to(ctx.device).eval()
    scores = pooled_logits(reference_model, captured, ctx.device)
    # A collapsed head would make every comparison below vacuous.
    checks["reference_scores_nondegenerate"] = bool(torch.isfinite(scores).all() and scores.float().std() > 1e-4)

    if kind == "reward":
        margin = captured["margin"]
        reference = reward_loss(scores, margin)
        # Control: swap the chunk halves, i.e. score every pair backwards.
        swapped = torch.cat(torch.chunk(scores, 2)[::-1])
        control = reward_loss(swapped, margin)
    else:
        labels = captured["labels"]
        reference = classification_loss(scores, labels)
        # Control: roll the labels by one — same logits, wrong targets.
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
