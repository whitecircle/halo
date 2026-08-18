#!/usr/bin/env python
"""DistributedKTOTrainer end-to-end, with the logged loss pinned to TRL 1.6's exact KTO objective.

KTO is unpaired: each row carries a boolean desirability label, and the loss pushes desirable rows'
log-ratio above — and undesirable rows' below — a KL estimate taken on deliberately MISMATCHED
prompt/completion pairs. Every part of that is a silent-failure surface. A constant loss, a loss with
the desirable and undesirable branches swapped (the sign flip: training would then reward wrong
answers), or ignored reference log-probs all stay finite and all keep the step count intact, so
finiteness cannot see any of them. This test reimplements the objective independently and pins it:

  * the logged loss at ``PINNED_STEP`` equals ``mean_rows (1 - sigmoid(beta * ±(logratio -/+ KL)))``
    computed from log-probs this test gathers itself, on the batch the trainer actually consumed and
    the weights it actually held;
  * the logged ``kl`` and ``logps/{chosen,rejected}`` metrics equal the same independent quantities,
    which pins the desirability split separately from the loss (the ``kl`` pin is guarded by
    ``kl_baseline_survives_clamp``, since a clamped-to-zero baseline is indistinguishable from a
    trainer with no KL branch; its WORLD-GLOBAL gather is covered by ``test_kto_fsdp_multi_gpu.py``);
  * two NEGATIVE CONTROLS prove the pin is not satisfiable by a degenerate objective — swapping the
    desirable mask (the sign flip) and ignoring the reference log-probs must each move the reference
    loss far outside the tolerance the pin is asserted at.

The reference consumes the trainer's own frozen reference model, exactly as
``pp/test_pp_dpo_precompute.py`` consumes captured ref columns: the reference model is an *input* to
the objective, so only its transport is under test here, not its provenance. The policy side is an
independently constructed copy of the checkpoint loaded with the weight snapshot taken at the pinned
step, so the two sides see byte-identical policy weights.

Run:
    torchrun --nproc_per_node=1 \
        tests/gpu/trainers/preference/test_kto.py
"""

import random

import torch
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import KTOConfig

from src.distributed.parallelism_config import ParallelismConfig
from src.env import env_int
from src.trainers.preference.kto import DistributedKTOTrainer
from tests.common.distributed import snapshot_full_weights
from tests.common.harness import gpu_test_main
from tests.common.models import QWEN3_0_6B
from tests.common.tolerances import TOL
from tests.common.utils import log

MODEL_NAME = QWEN3_0_6B
NUM_TRAIN_SAMPLES = 64
NUM_TRAIN_STEPS = 6
# TRL's KL estimate cycles completions within each block of per_device_train_batch_size rows, so the
# batch must hold >1 row; 4 also makes both desirability labels likely in the pinned batch.
BATCH_SIZE = 4
LEARNING_RATE = 2e-6
# 1.0, not the 0.1 production default: at 0.1 the sigmoid sits in its linear region, where swapping
# the desirable and undesirable branches moves the loss by ~0.01 and the pin cannot separate that
# bug from its own tolerance. LEARNING_RATE then lands the pinned log-ratios inside
# RESPONSIVE_LOGRATIO — below the band a sign flip barely moves the loss, above it the sigmoid
# saturates and every variant collapses to the same number.
BETA = 1.0
RESPONSIVE_LOGRATIO = (0.3, 20.0)
SEED = 42
# The LAST step: at step 1 the policy is byte-identical to the implicit reference, every log-ratio is
# ~0, the KL clamps to 0, and a sign-flipped or KL-free implementation lands on the same number.
PINNED_STEP = NUM_TRAIN_STEPS

# TRL's `selective_log_softmax` keeps bf16 logits in bf16 while this reference is fp32; that dtype
# gap is the whole residual (measured 4.0e-3 on the loss, 7.4e-3 on KL, against controls off by >0.53).
LOSS_REL_TOL = TOL.exact_objective_rel
# Absolute, because each sequence log-prob sums ~40 bf16 per-token terms whose error does not cancel.
# The two groups sit ~10 nats apart, so a swapped split still misses by two orders of magnitude.
LOGPS_ABS_TOL = 0.2
# Clear of TRL's clamp, or the `kl` pin compares 0.0 with 0.0 and holds for a trainer with no KL term.
MIN_LIVE_KL = 0.1
# A control must show the pin would FAIL, so its threshold IS the pin's tolerance.
CONTROL_MIN_GAP = LOSS_REL_TOL


def _parallelism_from_env() -> ParallelismConfig:
    """Standard by default; HALO_TEST_TP / HALO_TEST_EP select TP / EP (for parallelism sweeps)."""
    ep = env_int("HALO_TEST_EP", 1)
    tp = env_int("HALO_TEST_TP", 1)
    if ep > 1:
        return ParallelismConfig(ep_size=ep)
    if tp > 1:
        return ParallelismConfig(tp_size=tp)
    return ParallelismConfig()


def create_kto_dataset(num_samples: int, seed: int = SEED) -> Dataset:
    """Synthetic unpaired preference data: correct answers labelled desirable, wrong ones not."""
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
    """Σ log p(token_t | <t) over the positions ``completion_mask`` scores, in fp32.

    Mirrors TRL's masking contract and nothing else: the mask is applied as ``completion_mask[:, 1:]``,
    so position 0 is never scored and prompt/padding positions contribute exactly 0.
    """
    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).logits
    token_logps = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
    picked = token_logps.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (picked * completion_mask[:, 1:].float()).sum(-1)


def kto_objective(logratios: torch.Tensor, desirable: torch.Tensor, kl: float) -> float:
    """TRL's exact KTO loss: per-row weighted ``1 - sigmoid(beta * ±(logratio -/+ KL))``, then the mean
    over the UNWEIGHTED row count (the weights scale the summands, never the denominator)."""
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
        use_liger_kernel=True,  # the trainer must force this OFF (TRL 1.6's Liger KTO loss is broken)
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        max_length=1024,
        dataloader_drop_last=True,
        dataset_num_proc=1,
        dataloader_num_workers=0,
        fsdp="",
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

    # The reference must run on exactly the pinned step's batch and weights, or it compares
    # unrelated numbers.
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

    losses = [e["loss"] for e in trainer.state.log_history if "loss" in e]
    checks["ran_all_steps"] = trainer.state.global_step == NUM_TRAIN_STEPS and len(losses) == NUM_TRAIN_STEPS
    checks["losses_finite"] = bool(torch.isfinite(torch.tensor(losses)).all())
    # Early return, not a bare check: without the capture every line below raises KeyError and the
    # harness reports `error` instead of this check failing.
    checks["pinned_step_captured"] = bool(captured) and bool(weights)
    if not checks["pinned_step_captured"]:
        return {"checks": checks, "metrics": metrics}

    desirable = torch.as_tensor(captured["label"], dtype=torch.bool, device=ctx.device)
    # ANTI-VACUITY: with one-sided labels only half the objective runs and the split cannot be wrong.
    checks["pinned_batch_has_both_labels"] = bool(desirable.any()) and bool((~desirable).any())
    # A KL batch that copied the scored one would make the term identically 0, leaving it untested.
    checks["kl_batch_is_mismatched"] = not torch.equal(
        captured["KL_completion_input_ids"], captured["completion_input_ids"]
    )

    policy = build_model()
    policy.load_state_dict(weights)
    policy = policy.to(ctx.device).eval()
    reference_model = trainer.ref_model  # frozen: an input to the objective, not part of the math
    forwards = {}
    for tag, model in (("policy", policy), ("ref", reference_model)):
        for prefix in ("completion", "KL_completion"):
            forwards[f"{tag}_{prefix}"] = sequence_logps(
                model,
                captured[f"{prefix}_input_ids"],
                captured[f"{prefix}_attention_mask"],
                captured[f"{prefix}_mask"],
            )

    raw_kl = float((forwards["policy_KL_completion"] - forwards["ref_KL_completion"]).mean())
    kl = max(0.0, raw_kl)
    logratios = forwards["policy_completion"] - forwards["ref_completion"]
    expected = kto_objective(logratios, desirable, kl)
    logged = losses[PINNED_STEP - 1]
    metrics["pinned_loss"], metrics["pinned_reference"] = logged, expected
    metrics["kl_reference"], metrics["kl_raw_estimate"] = kl, raw_kl
    # The `kl` pin only says anything while the clamp is NOT firing: at exactly 0 a trainer with no
    # KL branch logs the same value, so a change pushing the shift negative must fail here.
    checks["kl_baseline_survives_clamp"] = kl > MIN_LIVE_KL
    checks["loss_matches_reference"] = abs(logged - expected) < LOSS_REL_TOL * max(1.0, abs(expected))

    # 0.5 is where every KTO variant lands at zero log-ratio — sign flip, dropped reference and the
    # real objective all coincide there, so the logged loss must sit away from it.
    metrics["distance_from_degenerate_half"] = abs(logged - 0.5)
    checks["logged_loss_away_from_degenerate_half"] = abs(logged - 0.5) > 10 * LOSS_REL_TOL
    # The policy must actually have drifted from the reference, or every control below is vacuous.
    drift = float(logratios.abs().mean())
    metrics["mean_abs_logratio"] = drift
    checks["logratios_in_responsive_region"] = RESPONSIVE_LOGRATIO[0] < drift < RESPONSIVE_LOGRATIO[1]

    # Pinned independently of the loss, which is what separates the desirability split.
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
    # ANTI-VACUITY for the two logps pins: a swapped split must miss by far more than LOGPS_ABS_TOL.
    group_gap = abs(float(logps[desirable].mean()) - float(logps[~desirable].mean()))
    metrics["logps_group_gap"] = group_gap
    checks["logps_groups_separable"] = group_gap > 10 * LOGPS_ABS_TOL

    # NEGATIVE CONTROLS: each degenerate objective must MISS the logged loss.
    controls = {
        # Sign flip: a trainer with this bug rewards wrong answers and its loss stays finite.
        "sign_flip": kto_objective(logratios, ~desirable, kl),
        # Reference log-probs ignored — the log-ratio degenerates to the raw policy log-prob.
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
