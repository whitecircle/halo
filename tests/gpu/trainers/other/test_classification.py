#!/usr/bin/env python
"""ClassificationTrainer end to end, with the logged loss pinned to its exact objective.

Both loss paths fail silently: wrong pooling scores padding, label misalignment trains on the wrong class. Focal's
``(1-pt)^gamma`` modulator comes from the unweighted cross-entropy while the class weight scales only the summand. That
distinction exists only with class weights, so the focal leg runs with them.
"""

import argparse
import math
import random

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from halo.configs.classification import ClassificationConfig
from halo.configs.script_arguments import CommonScriptArguments
from halo.distributed.parallelism_config import ParallelismConfig
from halo.distributed.tensor_parallel.hf_plan import is_input_embedding_tp_sharded
from halo.model_loading.tokenizer_setup import configure_run_tokenizer
from halo.trainers.reward.classification import ClassificationTrainer
from tests.common.distributed import ensure_model_downloaded, snapshot_full_weights, world_any, world_mean
from tests.common.dp_layout import dp_layout_checks, hsdp_layout, probe_forward
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log
from tests.common.training_checks import step_losses

MODEL_NAME = QWEN3_0_6B
NUM_TRAIN_SAMPLES = 64
MAX_LENGTH = 128
NUM_TRAIN_STEPS = 10
FOCAL_STEPS = 3
FOCAL_GAMMA = 2.0
# Asymmetric: with equal weights the two cross-entropies coincide and the modulator-source control cannot discriminate.
FOCAL_CLASS_WEIGHTS = [0.4, 1.8]
NUM_LABELS = 2
# >1 so the batch is ragged and the `pooling_at_pad` control has padding to differ on.
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
SEED = 42
PINNED_STEP = NUM_TRAIN_STEPS

# What separates the logged loss from a rescoring of the same weights is the two bf16 forwards' logits. A bound near
# that lets the pooling-at-pad control, which can move this loss by 1e-2, fail it.
LOSS_ABS_TOL = 5e-3
# A control shows the pin would fail, so its threshold is the pin's tolerance.
CONTROL_MIN_GAP = LOSS_ABS_TOL
# Relative: the modulator errors this leg catches are multiplicative. The modulator-source control misses by 0.31, 27x
# this bound.
FOCAL_REL_TOL = 2e-2


def create_classification_dataset(num_samples: int, tokenizer, max_length: int = MAX_LENGTH, seed: int = SEED):
    """Phrase lengths differ, so the collated batch is ragged and the pooling position is a real question."""
    random.seed(seed)
    positive_phrases = [
        "Great product overall",
        "Excellent quality and fast shipping",
        "Love this item so much",
        "Amazing value for the price",
        "Highly recommend to everyone",
        "Best purchase I have ever made",
        "Very satisfied with the result",
        "Perfect quality and design",
        "Outstanding customer service experience",
        "Wonderful and delightful product",
    ]
    negative_phrases = [
        "Terrible experience with this",
        "Waste of money completely",
        "Very poor build quality",
        "Do not recommend at all",
        "Worst product I have tried",
        "Extremely disappointed with this",
        "Awful quality and packaging",
        "Horrible customer support response",
        "Complete junk not worth it",
        "Broken on arrival very bad",
    ]

    data = {"input_ids": [], "attention_mask": [], "labels": []}
    for i in range(num_samples):
        positive = i % 2 == 0
        phrase = random.choice(positive_phrases if positive else negative_phrases)
        tokenized = tokenizer(f"{phrase}. Sample {i}.", truncation=True, max_length=max_length, padding=False)
        data["input_ids"].append(tokenized["input_ids"])
        data["attention_mask"].append(tokenized["attention_mask"])
        data["labels"].append(1 if positive else 0)
    return Dataset.from_dict(data)


def build_model(tokenizer):
    # The head is not in the checkpoint, so from_pretrained initializes it randomly. Seed first, or the objective
    # references compare against a different model each run.
    torch.manual_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    # Through the seam the training script runs: the pooling rule reads this id back, so recording it by hand would
    # keep the test green through a regression in the sync.
    configure_run_tokenizer(
        CommonScriptArguments(), model, tokenizer, MAX_LENGTH, embeddings_sharded=is_input_embedding_tp_sharded
    )
    return model


def pooled_logits(model, input_ids, attention_mask, pad_token_id, *, at_pad: bool = False) -> torch.Tensor:
    """Class logits in fp32, pooled at the last non-pad token. Backbone and head run separately so this test owns the
    pooling rule; ``at_pad`` pools at the final column, the control for that bug."""
    with torch.no_grad():
        hidden = model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        per_token = model.score(hidden).float()
    rows, seq_len = input_ids.shape
    if at_pad:
        positions = torch.full((rows,), seq_len - 1, device=input_ids.device, dtype=torch.long)
    else:
        non_pad = (input_ids != pad_token_id).to(torch.int32)
        token_indices = torch.arange(seq_len, device=input_ids.device, dtype=torch.int32)
        positions = (token_indices * non_pad).argmax(-1)
    return per_token[torch.arange(rows, device=input_ids.device), positions]


def focal_objective(
    logits: torch.Tensor, labels: torch.Tensor, gamma: float, weight: torch.Tensor, *, modulate_weighted: bool = False
) -> float:
    """``mean((1-pt)^gamma * weighted_CE)`` with ``pt = exp(-UNWEIGHTED_CE)``. ``exp(-ce)`` on the weighted value is
    ``p_y ** w_y``, which distorts the focusing; ``modulate_weighted=True`` is that mistake, this leg's control."""
    plain = F.cross_entropy(logits, labels, reduction="none")
    weighted = F.cross_entropy(logits, labels, weight=weight, reduction="none")
    modulator_source = weighted if modulate_weighted else plain
    return float((((1 - torch.exp(-modulator_source)) ** gamma) * weighted).mean())


def make_config(output_dir: str, **overrides) -> ClassificationConfig:
    base = {
        "output_dir": output_dir,
        "max_steps": NUM_TRAIN_STEPS,
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "gradient_accumulation_steps": 1,
        "learning_rate": LEARNING_RATE,
        "bf16": True,
        "gradient_checkpointing": True,
        "use_liger_kernel": True,
        "logging_steps": 1,
        "save_strategy": "no",
        "eval_strategy": "no",
        "report_to": "none",
        "logging_nan_inf_filter": False,
        "dataloader_drop_last": True,
        "remove_unused_columns": False,
    }
    base.update(overrides)
    return ClassificationConfig(**base)


def train_and_capture(ctx, tokenizer, dataset, config, pinned_step: int, layout: dict):
    trainer = ClassificationTrainer(
        model=build_model(tokenizer),
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        parallelism_config=ParallelismConfig(**layout),
    )
    ctx.on_teardown(trainer.cleanup_ep)

    captured: dict = {}
    weights: dict = {}
    real_compute_loss = trainer.compute_loss

    def spying_compute_loss(model, inputs, *args, **kwargs):
        # global_step is still N-1 while step N's loss is being computed.
        if trainer.state.global_step == pinned_step - 1 and not captured:
            captured.update({k: v.clone() for k, v in inputs.items() if torch.is_tensor(v)})
            weights.update(snapshot_full_weights(model))
        return real_compute_loss(model, inputs, *args, **kwargs)

    trainer.compute_loss = spying_compute_loss
    trainer.train()
    return trainer, captured, weights, step_losses(trainer)


parser = argparse.ArgumentParser()
parser.add_argument("--hsdp", action="store_true", help="HSDP over two simulated NVLink domains (4+ ranks)")
parser.add_argument("--world", type=int, default=None, help="the exact rank count (1: one process, no FSDP wrap)")
ARGS = parser.parse_args()


@gpu_test_main(min_world_size=4 if ARGS.hsdp else 1, exact_world_size=ARGS.world, prefix="classification")
def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}
    layout = hsdp_layout(ctx.world_size) if ARGS.hsdp else {}

    ensure_model_downloaded(MODEL_NAME, ctx.rank)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = create_classification_dataset(NUM_TRAIN_SAMPLES, tokenizer)

    trainer, captured, weights, losses = train_and_capture(
        ctx, tokenizer, dataset, make_config(ctx.output_dir), PINNED_STEP, layout
    )
    checks |= dp_layout_checks(
        trainer.model,
        layout,
        world_size=ctx.world_size,
        ep_config=None,
        forward=lambda: probe_forward(trainer.model, ctx.device),
    )
    loss = trainer._classification_loss
    checks["cross_entropy_builds_unweighted_ce"] = (
        not loss.multi_label and loss.focal_gamma is None and loss.weight is None and loss.label_smoothing == 0.0
    )
    # The trainer must reject CP (SUPPORTS_CP is False), where the world holds a CP group at all.
    if ctx.world_size > 1:
        try:
            ClassificationTrainer(
                model=build_model(tokenizer),
                args=make_config(ctx.output_dir + "/cp"),
                train_dataset=dataset,
                processing_class=tokenizer,
                parallelism_config=ParallelismConfig(cp_size=ctx.world_size),
            )
            checks["cp_rejected_at_construction"] = False
        except ValueError as exc:
            checks["cp_rejected_at_construction"] = "Context Parallelism" in str(exc)
    checks["ran_all_steps"] = trainer.state.global_step == NUM_TRAIN_STEPS and len(losses) == NUM_TRAIN_STEPS
    checks["losses_finite"] = bool(torch.isfinite(torch.tensor(losses)).all())
    # Early return: without the capture every line below raises KeyError, and the harness reports `error` over FAIL.
    checks["pinned_step_captured"] = bool(captured) and bool(weights)
    if not checks["pinned_step_captured"]:
        return {"checks": checks, "metrics": metrics}

    # Anti-vacuity, world-wide since the compared loss is a DP mean: requiring padding and both classes on every shard
    # would flake on a shard whose four rows tokenize to one length.
    lengths = captured["attention_mask"].sum(-1)
    labels = captured["labels"]
    checks["batch_is_ragged"] = world_any(int(lengths.min()) < int(lengths.max()), ctx.device)
    checks["batch_has_both_classes"] = world_any(int(labels.unique().numel()) == NUM_LABELS, ctx.device)

    scorer = build_model(tokenizer)
    scorer.load_state_dict(weights)
    scorer = scorer.to(ctx.device).eval()
    logits = pooled_logits(scorer, captured["input_ids"], captured["attention_mask"], tokenizer.pad_token_id)
    expected = world_mean(float(F.cross_entropy(logits, labels)), ctx.device)
    logged = losses[PINNED_STEP - 1]
    metrics["ce_logged"], metrics["ce_reference"] = logged, expected
    checks["ce_loss_matches_reference"] = abs(logged - expected) < LOSS_ABS_TOL
    # A classifier that learned nothing scores exactly ln(NUM_LABELS), a derived reference point that can actually
    # fail.
    uninformative = math.log(NUM_LABELS)
    metrics["uninformative_baseline"] = uninformative
    checks["ce_below_uninformative_baseline"] = expected < uninformative - CONTROL_MIN_GAP

    ce_controls = {
        "labels_misaligned": world_mean(float(F.cross_entropy(logits, labels.roll(1))), ctx.device),
        "pooling_at_pad": world_mean(
            float(
                F.cross_entropy(
                    pooled_logits(
                        scorer,
                        captured["input_ids"],
                        captured["attention_mask"],
                        tokenizer.pad_token_id,
                        at_pad=True,
                    ),
                    labels,
                )
            ),
            ctx.device,
        ),
    }
    for name, value in ce_controls.items():
        gap = abs(logged - value)
        metrics[f"control_{name}_gap"] = gap
        log(f"  control {name}: reference={value:.6f} gap-from-logged={gap:.3e}")
        checks[f"control_{name}_breaks_the_pin"] = gap > CONTROL_MIN_GAP

    focal_trainer, focal_captured, focal_weights, focal_losses = train_and_capture(
        ctx,
        tokenizer,
        dataset,
        make_config(
            ctx.output_dir + "/focal",
            max_steps=FOCAL_STEPS,
            loss_type="focal",
            focal_gamma=FOCAL_GAMMA,
            class_weights=FOCAL_CLASS_WEIGHTS,
        ),
        FOCAL_STEPS,
        layout,
    )
    checks["focal_selects_focal_loss_fn"] = focal_trainer._classification_loss.focal_gamma == FOCAL_GAMMA

    focal_scorer = build_model(tokenizer)
    focal_scorer.load_state_dict(focal_weights)
    focal_scorer = focal_scorer.to(ctx.device).eval()
    focal_logits = pooled_logits(
        focal_scorer, focal_captured["input_ids"], focal_captured["attention_mask"], tokenizer.pad_token_id
    )
    focal_labels = focal_captured["labels"]
    weight = torch.tensor(FOCAL_CLASS_WEIGHTS, device=ctx.device, dtype=torch.float32)
    focal_expected = world_mean(focal_objective(focal_logits, focal_labels, FOCAL_GAMMA, weight), ctx.device)
    focal_logged = focal_losses[FOCAL_STEPS - 1]
    plain_ce = world_mean(float(F.cross_entropy(focal_logits, focal_labels)), ctx.device)
    modulated_by_weighted = world_mean(
        focal_objective(focal_logits, focal_labels, FOCAL_GAMMA, weight, modulate_weighted=True), ctx.device
    )
    metrics["focal_logged"], metrics["focal_reference"], metrics["focal_plain_ce"] = (
        focal_logged,
        focal_expected,
        plain_ce,
    )
    metrics["focal_modulated_by_weighted_ce"] = modulated_by_weighted
    # The configured weights must be inside the loss the trainer built. Compared at the trainer's stored dtype: fp32.
    configured = focal_trainer._classification_loss.weight
    checks["focal_class_weights_reached_the_loss"] = configured is not None and torch.equal(
        configured.float().cpu(),
        torch.tensor(FOCAL_CLASS_WEIGHTS).to(configured.dtype).float(),
    )
    checks["focal_loss_matches_reference"] = abs(focal_logged - focal_expected) < FOCAL_REL_TOL * abs(focal_expected)
    # Controls: the modulator must be live and come from the unweighted cross-entropy.
    checks["focal_modulator_is_live"] = abs(focal_logged - plain_ce) > FOCAL_REL_TOL * abs(focal_expected)
    checks["focal_modulator_uses_unweighted_ce"] = abs(focal_logged - modulated_by_weighted) > FOCAL_REL_TOL * abs(
        focal_expected
    )

    log(f"  CE    step {PINNED_STEP}: logged={logged:.6f} reference={expected:.6f}")
    log(f"  focal step {FOCAL_STEPS}: logged={focal_logged:.6f} reference={focal_expected:.6f} CE={plain_ce:.6f}")
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
