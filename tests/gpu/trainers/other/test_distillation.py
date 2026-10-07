#!/usr/bin/env python
"""DistributedDistillationTrainer end-to-end, with the logged loss pinned to its exact objective.

The objective is ``alpha * KL(teacher_T || student_T) * T^2 + (1 - alpha) * CE(student, hard_labels)``,
each term ONE mean over the batch's supervised positions. Everything that can go wrong there is
silent: a dropped ``T^2`` rescales the whole distillation gradient, a reversed KL argument order
optimises the wrong direction (KL is asymmetric), an ignored ``alpha`` trains a different objective
entirely, a per-row mean reweights short rows, and a teacher forward whose logits never reach the
loss degenerates it to self-distillation. Each of those keeps the loss finite and the step count intact.

So the objective is reimplemented here and pinned, on the batch the trainer consumed and an
independently constructed student holding the weight snapshot it held at the pinned step:

  * the logged loss, the logged ``distillation_loss`` and the logged ``sft_loss`` each equal their
    independent reference;
  * five NEGATIVE CONTROLS — temperature-squared dropped, KL arguments reversed, alpha ignored, a
    per-row KL mean, and the teacher logits replaced by the student's own — must each miss the logged
    loss by more than the pin's tolerance.

Teacher and student start from the same checkpoint, so the pinned step is not the first: at step 1
the two are identical, the KL term is exactly 0 and every control agrees with the real objective.
Every batch pairs a few supervised tokens of low KL with hundreds of higher KL, so the per-row control
is separated by construction rather than by a lucky draw of row lengths.

Run with 2 GPUs:
    torchrun --nproc_per_node=2 \
        tests/gpu/trainers/other/test_distillation.py
"""

import random

import torch
import torch.nn.functional as F
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.configs.distillation_config import DistillationConfig
from src.data.collators.packing import DataCollatorForCausalLMWithPadding
from src.data.spans import LABEL_IGNORE_INDEX
from src.distributed.parallelism_config import ParallelismConfig
from src.trainers.distillation.teacher_distillation import DistributedDistillationTrainer
from tests.common.distributed import ensure_model_downloaded, snapshot_full_weights, world_all, world_mean
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.tolerances import TOL
from tests.common.utils import log, step_losses

MODEL_NAME = QWEN3_0_6B
NUM_TRAIN_SAMPLES = 64
MAX_SEQ_LENGTH = 512
NUM_TRAIN_STEPS = 8
BATCH_SIZE = 2
LEARNING_RATE = 5e-5
# T != 1 so a dropped T^2 rescaling is observable, and alpha strictly between 0 and 1 so BOTH terms
# are live (the 1.0 default drops the CLM term, leaving the alpha weighting untested).
TEMPERATURE = 2.0
ALPHA = 0.5
SEED = 42
# Teacher and student are the same checkpoint, so at step 1 the KL term is exactly 0 and every
# control below coincides with the real objective. One update leaves forward and reverse KL nearly
# equal, and later steps pull the student back toward the teacher, narrowing every control's gap.
PINNED_STEP = 3
# Copy rows: a sentence of COPY_WORDS random words, repeated COPY_REPEATS times; the repeats before
# the supervised last copy dictate it.
COPY_WORDS = 6
COPY_REPEATS = 4
# Dialogue rows: this many math exchanges, so a dialogue supervises dozens of times a copy's tokens.
DIALOGUE_EXCHANGES = 6
# Two rows with per-token KL means k_a, k_b and supervised counts in ratio r separate the per-row
# mean from the token mean by |k_a - k_b| / 2 * (r - 1) / (r + 1); this floor on r keeps most of it.
MIN_COUNT_RATIO = 10

LOSS_REL_TOL = TOL.exact_objective_rel
# A control's job is to show the pin would FAIL, so its threshold IS the pin's tolerance.
CONTROL_MIN_GAP = LOSS_REL_TOL
# The KL term must be non-trivial at the pinned step, or the alpha weighting and the KL-direction
# control are both vacuous (a zero KL is symmetric and alpha-invariant).
MIN_KL = 1e-3


def create_distillation_dataset(num_samples: int, tokenizer, seed: int = SEED) -> Dataset:
    """Tokenized rows alternating two kinds; with the sequential sampler every batch holds one of each.

    Even rows are copy rows: only the last repeat of a random sentence is supervised, a handful of
    tokens the context dictates, where a student that still copies stays close to the teacher. Odd
    rows are chat-templated math dialogues, every token supervised; their turn boundaries and
    openings carry most of the student's drift from the teacher. A per-row mean weighs the copy's
    few tokens as much as the whole dialogue; the trainer's token mean weighs every token once.
    """
    random.seed(seed)
    words = (
        "amber basket candle desert engine forest garden harbor island jacket kettle ladder marble needle "
        "orchid pepper quartz ribbon saddle tunnel velvet walnut yellow zipper"
    ).split()
    question_templates = [
        ("What is {a} + {b}?", lambda a, b: a + b),
        ("Calculate {a} * {b}.", lambda a, b: a * b),
        ("What is {a} - {b}?", lambda a, b: a - b),
        ("How much is {a} plus {b}?", lambda a, b: a + b),
        ("Compute the product of {a} and {b}.", lambda a, b: a * b),
    ]
    answer_templates = [
        "The answer is {result}. I calculated this by performing the operation on {a} and {b}.",
        "That equals {result}. Here is how: {a} combined with {b} gives {result}.",
        "The result is {result}. This is a straightforward arithmetic computation.",
        "{result} is the answer. Working through the math: {a} and {b} yield {result}.",
    ]

    rows = []
    for index in range(num_samples):
        if index % 2 == 0:
            sentence = " " + " ".join(random.sample(words, COPY_WORDS)) + "."
            copy = tokenizer(sentence, add_special_tokens=False)["input_ids"]
            input_ids = copy * COPY_REPEATS
            labels = [LABEL_IGNORE_INDEX] * (len(input_ids) - len(copy)) + copy
        else:
            messages = []
            for _ in range(DIALOGUE_EXCHANGES):
                template_q, op_fn = random.choice(question_templates)
                a, b = random.randint(1, 100), random.randint(1, 100)
                answer = random.choice(answer_templates).format(a=a, b=b, result=op_fn(a, b))
                messages += [
                    {"role": "user", "content": template_q.format(a=a, b=b)},
                    {"role": "assistant", "content": answer},
                ]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
            input_ids = tokenizer(text, truncation=True, max_length=MAX_SEQ_LENGTH)["input_ids"]
            labels = list(input_ids)
        rows.append({"input_ids": input_ids, "attention_mask": [1] * len(input_ids), "labels": labels})
    return Dataset.from_list(rows)


def build_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="flash_attention_2"
    )


def shifted_logits(model, input_ids, attention_mask) -> torch.Tensor:
    """Next-token logits ``[B, S-1, V]`` in fp32 — the span the objective scores."""
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    return logits[..., :-1, :].float()


def global_token_mean(per_token: torch.Tensor, mask: torch.Tensor, *, per_row: bool = False) -> torch.Tensor:
    """One mean over every kept position of the batch — the trainer's reduction for both terms.

    ``per_row=True`` is a control: a per-row mean, then a mean over rows, which weighs a short row's
    tokens more than a long row's.
    """
    if per_row:
        counts = mask.sum(-1, dtype=torch.float32).clamp(min=1e-9)
        return ((per_token * mask.float()).sum(-1) / counts).mean()
    return (per_token * mask.float()).sum() / mask.sum().clamp(min=1)


def kl_term(
    student_logits,
    teacher_logits,
    mask,
    *,
    temperature: float,
    scale_by_t2: bool = True,
    reverse: bool = False,
    per_row: bool = False,
) -> float:
    """``KL(teacher_T || student_T) * T^2``, token-mean reduced over ``mask``.

    The keyword flags are three of this test's controls for the term: ``scale_by_t2=False`` drops the
    temperature-squared rescaling, ``reverse=True`` swaps the two distributions and ``per_row=True``
    reduces per row.
    """
    student_logprobs = F.log_softmax(student_logits / temperature, dim=-1)
    teacher_logprobs = F.log_softmax(teacher_logits / temperature, dim=-1)
    if reverse:
        student_logprobs, teacher_logprobs = teacher_logprobs, student_logprobs
    per_token = F.kl_div(student_logprobs, teacher_logprobs.exp(), reduction="none").sum(-1)
    if scale_by_t2:
        per_token = per_token * temperature**2
    return float(global_token_mean(per_token, mask, per_row=per_row))


def clm_term(student_logits, hard_labels) -> float:
    """The trainer's CLM term: fp32 cross-entropy, one mean over the valid tokens."""
    valid = (hard_labels != LABEL_IGNORE_INDEX).sum().clamp(min=1)
    total = F.cross_entropy(
        student_logits.reshape(-1, student_logits.size(-1)),
        hard_labels.reshape(-1),
        ignore_index=LABEL_IGNORE_INDEX,
        reduction="sum",
    )
    return float(total / valid)


@gpu_test_main(min_world_size=2, prefix="test_distill")
def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}

    ensure_model_downloaded(MODEL_NAME, ctx.rank)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = create_distillation_dataset(NUM_TRAIN_SAMPLES, tokenizer)

    config = DistillationConfig(
        output_dir=ctx.output_dir,
        max_steps=NUM_TRAIN_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=True,
        logging_steps=1,
        save_strategy="no",
        eval_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=MAX_SEQ_LENGTH,
        dataloader_drop_last=True,
        # In dataset order, so each two-row batch is one copy row and one dialogue.
        train_sampling_strategy="sequential",
        dataset_num_proc=1,
        dataloader_num_workers=0,
        fsdp="",  # the mixin owns FSDP wrapping
        distill_loss="kl_divergence",
        distill_temperature=TEMPERATURE,
        distill_alpha=ALPHA,
    )

    trainer = DistributedDistillationTrainer(
        student_model=build_model(),
        teacher_model=build_model(),
        teacher_tokenizer=tokenizer,
        args=config,
        train_dataset=dataset,
        data_collator=DataCollatorForCausalLMWithPadding(tokenizer=tokenizer, mlm=False),
        processing_class=tokenizer,
        parallelism_config=ParallelismConfig(),
    )
    ctx.on_teardown(trainer.cleanup_ep)

    checks["teacher_frozen"] = not any(p.requires_grad for p in trainer.teacher_model.parameters())

    captured: dict = {}
    weights: dict = {}
    real_compute_loss = trainer.compute_loss

    def spying_compute_loss(model, inputs, *args, **kwargs):
        # global_step is still N-1 while step N's loss is being computed.
        if trainer.state.global_step == PINNED_STEP - 1 and not captured:
            captured.update({k: v.clone() for k, v in inputs.items() if torch.is_tensor(v)})
            weights.update(snapshot_full_weights(model))
        return real_compute_loss(model, inputs, *args, **kwargs)

    trainer.compute_loss = spying_compute_loss
    trainer.train()

    losses = step_losses(trainer)
    checks["ran_all_steps"] = trainer.state.global_step == NUM_TRAIN_STEPS and len(losses) == NUM_TRAIN_STEPS
    checks["losses_finite"] = bool(torch.isfinite(torch.tensor(losses)).all())
    # Early return, not a bare check: without the capture every line below raises KeyError and the
    # harness reports `error`, so this check could never be the thing that says FAIL.
    checks["pinned_step_captured"] = bool(captured) and bool(weights)
    if not checks["pinned_step_captured"]:
        return {"checks": checks, "metrics": metrics}
    checks["teacher_still_frozen_after_training"] = not any(
        p.requires_grad for p in trainer.teacher_model.parameters()
    )

    # ── Independent reference ────────────────────────────────────────────────────────────────
    student = build_model()
    student.load_state_dict(weights)
    student = student.to(ctx.device).eval()
    # The teacher is frozen throughout: an input to the objective, not part of the math under test.
    teacher = trainer.teacher_model

    input_ids, attention_mask = captured["input_ids"], captured["attention_mask"]
    hard_labels = captured["labels"][..., 1:].contiguous()
    supervised = hard_labels != LABEL_IGNORE_INDEX
    student_logits = shifted_logits(student, input_ids, attention_mask)
    teacher_logits = shifted_logits(teacher, input_ids, attention_mask)

    # Without a few-token row beside a many-token one, the trainer's token mean is indistinguishable
    # from a per-row one. Required on EVERY rank: the compared loss is a DP mean, so a rank whose rows
    # are alike only dilutes the per-row control's gap.
    counts = supervised.sum(-1)
    checks["every_rank_pairs_a_copy_row_with_a_dialogue"] = world_all(
        int(counts.max()) >= MIN_COUNT_RATIO * int(counts.min()), ctx.device
    )

    kl = kl_term(student_logits, teacher_logits, supervised, temperature=TEMPERATURE)
    clm = clm_term(student_logits, hard_labels)
    expected = world_mean(ALPHA * kl + (1 - ALPHA) * clm, ctx.device)
    logged = losses[PINNED_STEP - 1]
    metrics["pinned_loss"], metrics["pinned_reference"] = logged, expected
    metrics["kl_reference"], metrics["clm_reference"] = kl, clm
    checks["loss_matches_reference"] = abs(logged - expected) < LOSS_REL_TOL * max(1.0, abs(expected))
    # ANTI-VACUITY: a zero KL term makes the direction and alpha controls meaningless.
    checks["kl_term_nontrivial"] = kl > MIN_KL

    # The two components are logged separately; pin each so a compensating error in one is visible.
    # A stored metric is logged as its world mean, every rank weighing one, so its reference is the
    # world mean of the per-rank references.
    entry = [e for e in trainer.state.log_history if "loss" in e][PINNED_STEP - 1]
    for key, local in (("distillation_loss", kl), ("sft_loss", clm)):
        want = world_mean(local, ctx.device)
        got = entry.get(key)
        metrics[f"logged_{key}"], metrics[f"expected_{key}"] = (got if got is not None else float("nan")), want
        checks[f"metric_matches_reference[{key}]"] = got is not None and abs(got - want) < LOSS_REL_TOL * max(
            1.0, abs(want)
        )

    # ── NEGATIVE CONTROLS ────────────────────────────────────────────────────────────────────
    controls = {
        # T^2 dropped: the distillation gradient is silently rescaled by 1/T^2.
        "temperature_squared_dropped": ALPHA
        * kl_term(student_logits, teacher_logits, supervised, temperature=TEMPERATURE, scale_by_t2=False)
        + (1 - ALPHA) * clm,
        # KL arguments reversed: KL is asymmetric, so this optimises the wrong direction.
        "kl_direction_reversed": ALPHA
        * kl_term(student_logits, teacher_logits, supervised, temperature=TEMPERATURE, reverse=True)
        + (1 - ALPHA) * clm,
        # alpha ignored: the CLM term silently disappears from the objective.
        "alpha_ignored": kl,
        # A per-row KL mean: short rows' tokens outweigh long rows'.
        "kl_reduced_per_row": ALPHA
        * kl_term(student_logits, teacher_logits, supervised, temperature=TEMPERATURE, per_row=True)
        + (1 - ALPHA) * clm,
        # The teacher forward's logits never reach the loss (a dropped wiring), degenerating the
        # distillation term to self-distillation: KL(student || student) == 0.
        "teacher_logits_unused": ALPHA * kl_term(student_logits, student_logits, supervised, temperature=TEMPERATURE)
        + (1 - ALPHA) * clm,
    }
    for name, value in controls.items():
        gap = abs(logged - world_mean(value, ctx.device))
        metrics[f"control_{name}_gap"] = gap
        log(f"  control {name}: gap-from-logged={gap:.3e}")
        checks[f"control_{name}_breaks_the_pin"] = gap > CONTROL_MIN_GAP

    log(
        f"  pinned step {PINNED_STEP}: loss={logged:.6f} reference={expected:.6f} kl={kl:.6f} clm={clm:.6f} "
        f"supervised counts={counts.tolist()}"
    )
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
