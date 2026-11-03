#!/usr/bin/env python
"""DistributedKTOTrainer end to end, with the logged loss pinned to TRL 1.14's exact KTO objective.

A constant loss, swapped desirable and undesirable branches (training then rewards wrong answers) and ignored reference
log-probs all stay finite with an intact step count. The trainer's frozen reference model is an input to the objective,
so only its transport is under test here. The KL's world-global gather is ``test_kto_fsdp_multi_gpu.py``'s.
"""

import random

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import KTOConfig

from halo.distributed.parallelism_config import ParallelismConfig
from halo.env import env_int
from halo.trainers.preference.kto import DistributedKTOTrainer
from tests.common.distributed import snapshot_full_weights
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.rank_log import log
from tests.common.training_checks import step_losses

MODEL_NAME = QWEN3_0_6B
NUM_TRAIN_SAMPLES = 64
NUM_TRAIN_STEPS = 6
# TRL's KL estimate cycles completions within each block of per_device_train_batch_size rows, so the batch needs >1
# row; 4 makes both labels likely in the pinned batch.
BATCH_SIZE = 4
LEARNING_RATE = 2e-6
# Above the 0.1 production default: there the sigmoid is linear, and swapping branches moves the loss by ~0.01.
# LEARNING_RATE lands the log-ratios inside RESPONSIVE_LOGRATIO: below it a sign flip barely moves the loss, above it
# the sigmoid saturates.
BETA = 1.0
RESPONSIVE_LOGRATIO = (0.3, 20.0)
SEED = 42
# At step 1 the policy equals the implicit reference, every log-ratio is ~0, the KL clamps to 0, and a sign-flipped or
# KL-free implementation lands on the same number.
PINNED_STEP = NUM_TRAIN_STEPS

# Both sides score in fp32, so the pin agrees to fp32 rounding (measured below 1e-6); the controls miss by >0.5.
LOSS_REL_TOL = 1e-3
# The groups sit ~13 nats apart, so a swapped split misses by three orders of magnitude.
LOGPS_ABS_TOL = 2e-2
# Clear of TRL's clamp, or the `kl` pin compares 0.0 with 0.0 and holds for a trainer with no KL term.
MIN_LIVE_KL = 0.1
# A control shows the pin would fail, so its threshold is the pin's tolerance.
CONTROL_MIN_GAP = LOSS_REL_TOL


def _parallelism_from_env() -> ParallelismConfig:
    ep = env_int("HALO_TEST_EP", 1)
    tp = env_int("HALO_TEST_TP", 1)
    if ep > 1:
        return ParallelismConfig(ep_size=ep)
    if tp > 1:
        return ParallelismConfig(tp_size=tp)
    return ParallelismConfig()


def create_kto_dataset(num_samples: int, seed: int = SEED) -> Dataset:
    random.seed(seed)
    rows = []
    for _ in range(num_samples):
        a, b = random.randint(1, 50), random.randint(1, 50)
        correct = random.random() < 0.5
        answer = a + b if correct else a + b + random.randint(1, 9)
        rows.append(
            {
                "prompt": [{"role": "user", "content": f"What is {a} + {b}?"}],
                "completion": [{"role": "assistant", "content": f"The answer is {answer}."}],
                "label": correct,
            }
        )
    return Dataset.from_list(rows)


def build_model():
    return AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, dtype=torch.bfloat16, trust_remote_code=True, attn_implementation="flash_attention_2"
    )


def sequence_logps(model, input_ids, attention_mask, completion_mask) -> torch.Tensor:
    """Mirrors TRL's masking contract: ``completion_mask[:, 1:]``, so position 0 is never scored."""
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    token_logps = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    picked = token_logps.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (picked * completion_mask[:, 1:].float()).sum(-1)


def kto_objective(logratios: torch.Tensor, desirable: torch.Tensor, kl: float) -> float:
    """Per-row weighted ``1 - sigmoid(beta * ±(logratio -/+ KL))``, meaned over the unweighted row count."""
    per_row = torch.where(
        desirable,
        1 - torch.sigmoid(BETA * (logratios - kl)),
        1 - torch.sigmoid(BETA * (kl - logratios)),
    )
    return float(per_row.nanmean())


@gpu_test_main(exact_world_size=1, prefix="test_kto")
def run(ctx):
    checks: dict[str, bool] = {}
    metrics: dict[str, float] = {}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_dataset = create_kto_dataset(NUM_TRAIN_SAMPLES)
    config = KTOConfig(
        output_dir=ctx.output_dir,
        max_steps=NUM_TRAIN_STEPS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=LEARNING_RATE,
        beta=BETA,
        bf16=True,
        gradient_checkpointing=True,
        use_liger_kernel=True,  # The trainer must force this off (TRL's chunked KTO path refuses the aux loss).
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        logging_nan_inf_filter=False,
        max_length=1024,
        dataloader_drop_last=True,
        dataset_num_proc=1,
        dataloader_num_workers=0,
    )

    trainer = DistributedKTOTrainer(
        model=build_model(),
        args=config,
        train_dataset=train_dataset,
        processing_class=tokenizer,
        parallelism_config=_parallelism_from_env(),
    )
    ctx.on_teardown(trainer.cleanup_ep)

    checks["liger_kto_loss_disabled"] = trainer.args.use_liger_kernel is False
    checks["implicit_reference_model_built"] = trainer.ref_model is not None

    captured: dict = {}
    weights: dict = {}
    real_compute_loss = trainer.compute_loss

    def spying_compute_loss(model, inputs, *args, **kwargs):
        # global_step is still N-1 while step N's loss is being computed.
        if trainer.state.global_step == PINNED_STEP - 1 and not captured:
            captured.update({k: v.clone() if torch.is_tensor(v) else v for k, v in inputs.items()})
            weights.update(snapshot_full_weights(model))
        return real_compute_loss(model, inputs, *args, **kwargs)

    trainer.compute_loss = spying_compute_loss
    trainer.train()

    losses = step_losses(trainer)
    checks["ran_all_steps"] = trainer.state.global_step == NUM_TRAIN_STEPS and len(losses) == NUM_TRAIN_STEPS
    checks["losses_finite"] = bool(torch.isfinite(torch.tensor(losses)).all())
    # Early return: without the capture every line below raises KeyError, and the harness reports `error` over FAIL.
    checks["pinned_step_captured"] = bool(captured) and bool(weights)
    if not checks["pinned_step_captured"]:
        return {"checks": checks, "metrics": metrics}

    desirable = torch.as_tensor(captured["label"], dtype=torch.bool, device=ctx.device)
    # Anti-vacuity: with one-sided labels only half the objective runs and the split cannot be wrong.
    checks["pinned_batch_has_both_labels"] = bool(desirable.any()) and bool((~desirable).any())
    # A KL batch that copied the scored one would make the term identically 0.
    checks["kl_batch_is_mismatched"] = not torch.equal(captured["KL_input_ids"], captured["input_ids"])

    policy = build_model()
    policy.load_state_dict(weights)
    policy = policy.to(ctx.device).eval()
    reference_model = trainer.ref_model  # Frozen: an input to the objective.
    forwards = {}
    for tag, model in (("policy", policy), ("ref", reference_model)):
        for name, prefix in (("completion", ""), ("KL_completion", "KL_")):
            forwards[f"{tag}_{name}"] = sequence_logps(
                model,
                captured[f"{prefix}input_ids"],
                captured[f"{prefix}attention_mask"],
                captured[f"{prefix}completion_mask"],
            )

    raw_kl = float((forwards["policy_KL_completion"] - forwards["ref_KL_completion"]).mean())
    kl = max(0.0, raw_kl)
    logratios = forwards["policy_completion"] - forwards["ref_completion"]
    expected = kto_objective(logratios, desirable, kl)
    logged = losses[PINNED_STEP - 1]
    metrics["pinned_loss"], metrics["pinned_reference"] = logged, expected
    metrics["kl_reference"], metrics["kl_raw_estimate"] = kl, raw_kl
    # At exactly 0 a trainer with no KL branch logs the same value, so a change pushing the shift negative must fail
    # here.
    checks["kl_baseline_survives_clamp"] = kl > MIN_LIVE_KL
    checks["loss_matches_reference"] = abs(logged - expected) < LOSS_REL_TOL * max(1.0, abs(expected))

    # Every KTO variant lands on 0.5 at zero log-ratio, so the logged loss must sit away from it.
    metrics["distance_from_degenerate_half"] = abs(logged - 0.5)
    checks["logged_loss_away_from_degenerate_half"] = abs(logged - 0.5) > 10 * LOSS_REL_TOL
    # Or every control below is vacuous.
    drift = float(logratios.abs().mean())
    metrics["mean_abs_logratio"] = drift
    checks["logratios_in_responsive_region"] = RESPONSIVE_LOGRATIO[0] < drift < RESPONSIVE_LOGRATIO[1]

    # Pinned independently of the loss, which separates the desirability split.
    entry = [e for e in trainer.state.log_history if "loss" in e][PINNED_STEP - 1]
    logps = forwards["policy_completion"]
    metric_pins = {
        "kl": (kl, LOSS_REL_TOL * max(1.0, abs(kl))),
        "logps/chosen": (float(logps[desirable].mean()), LOGPS_ABS_TOL),
        "logps/rejected": (float(logps[~desirable].mean()), LOGPS_ABS_TOL),
    }
    for key, (want, tol) in metric_pins.items():
        got = entry.get(key)
        metrics[f"logged_{key}"] = got if got is not None else float("nan")
        metrics[f"expected_{key}"] = want
        checks[f"metric_matches_reference[{key}]"] = got is not None and abs(got - want) < tol
    # Anti-vacuity for the logps pins: a swapped split must miss by far more than LOGPS_ABS_TOL.
    group_gap = abs(float(logps[desirable].mean()) - float(logps[~desirable].mean()))
    metrics["logps_group_gap"] = group_gap
    checks["logps_groups_separable"] = group_gap > 10 * LOGPS_ABS_TOL

    controls = {
        # Rewards wrong answers with a finite loss.
        "sign_flip": kto_objective(logratios, ~desirable, kl),
        # The log-ratio degenerates to the raw policy log-prob.
        "ref_logps_ignored": kto_objective(logps, desirable, kl),
    }
    for name, value in controls.items():
        gap = abs(logged - value)
        metrics[f"control_{name}_gap"] = gap
        log(f"  control {name}: reference={value:.6f} gap-from-logged={gap:.3e}")
        checks[f"control_{name}_breaks_the_pin"] = gap > CONTROL_MIN_GAP

    log(f"  pinned step {PINNED_STEP}: loss={logged:.6f} reference={expected:.6f} kl={kl:.6f}")
    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()
