#!/usr/bin/env python
"""ClassificationTrainer end-to-end, with the logged loss pinned to its exact objective.

The trainer has two loss paths and both fail silently. The default routes to the model's built-in
head loss (cross-entropy over logits pooled at each row's last non-pad token), where a wrong pooling
position scores padding and a label misalignment trains on the wrong class — both finite, both
step-count-clean. The ``focal`` path is halo's own code (``ClassificationTrainer._focal_loss``) and
carries a detail that is easy to get wrong and impossible to see from finiteness: the ``(1-pt)^gamma``
modulator is built from the UNWEIGHTED cross-entropy while the class weight scales only the summand.
That distinction exists ONLY when class weights are configured — with ``weight=None`` the weighted and
unweighted cross-entropies are the same object — so the focal leg runs WITH ``class_weights``, and its
control is the same objective with the modulator taken from the weighted CE instead.

So both paths are pinned here against references this test computes itself, from an independently
constructed copy of the model holding the weight snapshot the trainer had at the pinned step:

  * cross-entropy leg — the logged loss equals ``F.cross_entropy(pooled_logits, labels)``, and two
    NEGATIVE CONTROLS (labels rolled by one row, logits pooled at the final padded column) must each
    miss it by more than the pin's tolerance;
  * focal leg (with class weights) — the logged loss equals the focal objective, must NOT equal plain
    cross-entropy on the same logits (or the modulator is inert), and must NOT equal the variant whose
    modulator is built from the WEIGHTED cross-entropy (the mistake the implementation documents).

Usage:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/other/test_classification.py
"""

import math
import random

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from src.args.common_script_args import CommonScriptArguments
from src.configs.classification_config import ClassificationConfig
from src.distributed.parallelism_config import ParallelismConfig
from src.models.loading.tokenizer_setup import setup_model_and_tokenizer
from src.trainers.reward.classification import ClassificationTrainer
from tests.common.distributed import ensure_model_downloaded, snapshot_full_weights, world_any, world_mean
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.tolerances import TOL
from tests.common.utils import log

MODEL_NAME = QWEN3_0_6B
NUM_TRAIN_SAMPLES = 64
NUM_EVAL_SAMPLES = 16
MAX_LENGTH = 128
NUM_TRAIN_STEPS = 10
FOCAL_STEPS = 3
FOCAL_GAMMA = 2.0
# Asymmetric on purpose: with equal weights the weighted and unweighted cross-entropies coincide and
# the modulator-source control below cannot discriminate. Far enough from 1.0 that p^w differs from p.
FOCAL_CLASS_WEIGHTS = [0.4, 1.8]
NUM_LABELS = 2
# >1 so the batch is ragged and the last-non-pad pooling rule is observable at all (the
# `pooling_at_pad` control below only differs when some row actually carries padding).
BATCH_SIZE = 4
LEARNING_RATE = 2e-5
SEED = 42
PINNED_STEP = NUM_TRAIN_STEPS

LOSS_REL_TOL = TOL.exact_objective_rel
# A control's job is to show the pin would FAIL, so its threshold IS the pin's tolerance.
CONTROL_MIN_GAP = LOSS_REL_TOL
# The focal pin is RELATIVE: the shared absolute bound is ~1% of the focal value here but would be a
# far weaker statement at a smaller one, and the modulator errors this leg exists to catch are
# multiplicative. The floor is the bf16 quantization of the logged scalar (half a ULP, ~0.3% at this
# magnitude, and the measured residual of 1.8e-3 on a value of 0.58 is exactly that). The
# modulator-source control misses by 0.31 — 27x this bound — so the leg still discriminates hard.
FOCAL_REL_TOL = 2e-2


def create_classification_dataset(num_samples: int, tokenizer, max_length: int = MAX_LENGTH, seed: int = SEED):
    """Synthetic binary sentiment data, pre-tokenized to ``input_ids``/``attention_mask``/``labels``.

    Phrase lengths differ, so the collated batch is ragged — which is what makes the pooling position
    a real question rather than a formality.
    """
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
    # The classification head is not in the checkpoint, so from_pretrained initialises it randomly.
    # Seed before construction (which precedes the Trainer's own seeding) or the head differs run to
    # run and the objective references below compare against a different model each time.
    torch.manual_seed(SEED)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_LABELS,
        dtype=torch.bfloat16,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    # Through the seam the training script runs (apply_max_length → setup_model_and_tokenizer),
    # never a hand-written config.pad_token_id: the pooling rule this test pins reads that id back,
    # so recording it here by hand would keep the test green through a regression in the sync.
    setup_model_and_tokenizer(CommonScriptArguments(), model, tokenizer, MAX_LENGTH)
    return model


def pooled_logits(model, input_ids, attention_mask, pad_token_id, *, at_pad: bool = False) -> torch.Tensor:
    """Class logits ``[B, num_labels]`` in fp32, pooled at the LAST NON-PAD token.

    The backbone and the head are run separately so this test owns the pooling rule (``model(...)``
    would return already-pooled logits, hiding a wrong position). ``at_pad`` pools at the final
    column instead — the control for exactly that bug.
    """
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
    """Halo's focal loss: ``mean((1-pt)^gamma * weighted_CE)`` with ``pt = exp(-UNWEIGHTED_CE)``.

    The modulator source is the whole point. ``F.cross_entropy(weight=w)`` returns ``w_y * -log p_y``,
    so ``exp(-ce)`` on the weighted value is ``p_y ** w_y`` rather than ``p_y`` — building the
    modulator from it distorts the focusing instead of scaling the loss. ``modulate_weighted=True``
    is that mistake, and is this leg's control.
    """
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
        "dataloader_drop_last": True,
        "remove_unused_columns": False,
        "fsdp": "",  # the mixin owns FSDP wrapping
    }
    base.update(overrides)
    return ClassificationConfig(**base)


def train_and_capture(ctx, tokenizer, dataset, config, pinned_step: int):
    """Train, capturing the batch and the FULL weights held at ``pinned_step``.

    Returns ``(trainer, captured_batch, weights, logged_losses)``.
    """
    trainer = ClassificationTrainer(
        model=build_model(tokenizer),
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        parallelism_config=ParallelismConfig(),
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
    return trainer, captured, weights, [e["loss"] for e in trainer.state.log_history if "loss" in e]


@gpu_test_main(min_world_size=2, prefix="classification")
def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}

    ensure_model_downloaded(MODEL_NAME, ctx.rank)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    dataset = create_classification_dataset(NUM_TRAIN_SAMPLES, tokenizer)

    # ── Cross-entropy leg (the default path: the model's built-in head loss) ─────────────────
    trainer, captured, weights, losses = train_and_capture(
        ctx, tokenizer, dataset, make_config(ctx.output_dir), PINNED_STEP
    )
    checks["cross_entropy_selects_builtin_head_loss"] = trainer._loss_fn is None
    # Constructing with cp_size=1 and then asserting is_cp_mode is False only restates the test's own
    # input. What matters is that the trainer REJECTS CP (_supports_cp is False), so ask it to.
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
    # Early return, not a bare check: without the capture every line below raises KeyError and the
    # harness reports `error`, so this check could never be the thing that says FAIL.
    checks["pinned_step_captured"] = bool(captured) and bool(weights)
    if not checks["pinned_step_captured"]:
        return {"checks": checks, "metrics": metrics}

    # ANTI-VACUITY, stated world-wide because the compared loss is a DP mean: padding has to exist
    # SOMEWHERE for the pooling control to differ, and both classes have to appear somewhere for the
    # label-roll control to be more than a no-op. Requiring it of every shard would flake on a shard
    # whose four rows happen to tokenize to the same length.
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
    checks["ce_loss_matches_reference"] = abs(logged - expected) < LOSS_REL_TOL * max(1.0, abs(expected))
    # A classifier that has learned nothing predicts uniformly and scores exactly ln(NUM_LABELS).
    # Requiring the pinned loss to sit clear of that is a derived reference point, not a magic floor
    # (and unlike "> 0" it can actually fail).
    uninformative = math.log(NUM_LABELS)
    metrics["uninformative_baseline"] = uninformative
    checks["ce_below_uninformative_baseline"] = expected < uninformative - CONTROL_MIN_GAP

    ce_controls = {
        # Labels rolled by one row: the trainer would be scoring each row against another row's class.
        "labels_misaligned": world_mean(float(F.cross_entropy(logits, labels.roll(1))), ctx.device),
        # Pooled at the final column, so every padded row is scored on padding.
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

    # ── Focal leg (halo's own loss function, with its unweighted-CE modulator) ───────────────
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
    )
    checks["focal_selects_custom_loss_fn"] = focal_trainer._loss_fn is not None

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
    # The configured weights must actually be inside the partial the trainer built, or the
    # weighted-vs-unweighted distinction this leg exists to pin is not being exercised at all.
    # Compared at the dtype the trainer stores them in: the weights follow the logits to bf16, so an
    # fp32 expectation would differ by bf16 rounding alone (0.4 -> 0.400390625) and fail spuriously.
    configured = focal_trainer._loss_fn.keywords.get("weight")
    checks["focal_class_weights_reached_the_loss"] = configured is not None and torch.equal(
        configured.float().cpu(),
        torch.tensor(FOCAL_CLASS_WEIGHTS).to(configured.dtype).float(),
    )
    checks["focal_loss_matches_reference"] = abs(focal_logged - focal_expected) < FOCAL_REL_TOL * abs(focal_expected)
    # CONTROLS: the modulator must be live at all, and must come from the UNWEIGHTED cross-entropy.
    checks["focal_modulator_is_live"] = abs(focal_logged - plain_ce) > FOCAL_REL_TOL * abs(focal_expected)
    checks["focal_modulator_uses_unweighted_ce"] = abs(focal_logged - modulated_by_weighted) > FOCAL_REL_TOL * abs(
        focal_expected
    )

    log(f"  CE    step {PINNED_STEP}: logged={logged:.6f} reference={expected:.6f}")
    log(f"  focal step {FOCAL_STEPS}: logged={focal_logged:.6f} reference={focal_expected:.6f} CE={plain_ce:.6f}")
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
