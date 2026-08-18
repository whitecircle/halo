#!/usr/bin/env python
"""KTO under multi-GPU FSDP2 data parallelism (2 GPUs, live reference model).

The registered KTO GPU test runs at ``nproc=1``; this one covers KTO's two distributed seams, both
silent when they break:

  * **the world-global KL baseline.** KTO's ``kto`` loss subtracts a KL term that is *not* local:
    TRL computes each rank's mean ``logpi(y'|x) - logpi_ref(y'|x)`` over the batch's rotated
    completions, ``gather_for_metrics`` it across the world, means the per-rank means and clamps at
    0. Every rank must therefore see the SAME baseline. Drop the gather and each rank silently
    optimizes a different objective — the loss stays finite and still decreases, so only a value
    comparison catches it. Pinned here against the exact objective recomputed on unsharded replicas
    of the policy and the reference, with the baseline rebuilt from every rank's own captured KL
    batch. The control is the same reference computed with the LOCAL baseline instead of the
    gathered one: it must miss, which is exactly what a broken gather would produce.
  * **FSDP2 gradient sync on an unpaired objective.** KTO's rows are independent and the desirable /
    undesirable terms are concatenated before a single ``nanmean``, so a rank whose batch is
    all-desirable contributes a differently-shaped vector. If the reduce-scatter did not cover that,
    each rank would optimize its own half. Comparing gathered PARAMETERS across ranks cannot see
    this: under 1D full-shard FSDP2 every parameter is a ``Shard(0)`` DTensor, so ``full_tensor()``
    is an all-gather that returns byte-identical data on every rank by construction — there are no
    replicas to disagree, and a dropped reduce-scatter would have every rank corrupt its own unique
    shard and still read back the same concatenation. What is pinned instead is the GRADIENT: the
    tensor FSDP2 produces must equal the DP average of the per-rank gradients, recomputed on an
    unsharded replica of the same weights. The negative control turns the sync off for real with
    ``set_requires_gradient_sync(False)`` and requires the same comparison to fail.

The reference and its controls only mean something if the KL baseline is non-degenerate, so the
policy and the reference model are built from DIFFERENT seeds (a reference equal to the policy makes
every log-ratio 0, the baseline clamp to 0 and the loss the constant 0.5) and the test asserts, as
preconditions, that the gathered baseline survived the ``clamp(min=0)`` and that the ranks' local
baselines actually disagree. Both preconditions are load-bearing: a negative gathered mean clamps to
0, at which point the local-only baseline and no baseline at all are indistinguishable.

Run: torchrun --nproc_per_node=2 tests/gpu/trainers/preference/test_kto_fsdp_multi_gpu.py
"""

import torch
import torch.distributed as dist
from datasets import Dataset
from torch.distributed.tensor import DTensor
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from trl import KTOConfig

from src.distributed.parallelism_config import ParallelismConfig
from tests.common.distributed import snapshot_full_weights, world_mean
from tests.common.harness import gpu_test_main, log
from tests.common.models import QWEN3_0_6B, TINY_QWEN3_CONFIG

N_ROWS = 32
N_STEPS = 6
ROWS_PER_RANK = 4  # KTO requires > 1 per device so the rotated KL completions are mismatched
MAX_LENGTH = 96
# beta scales how hard the KL baseline pushes on the loss, so it also scales the gap between the
# real (gathered) baseline and the local-only one the controls stand in for. 0.3 lifts that gap
# well clear of the numerical floor without saturating the sigmoids.
BETA = 0.3
DESIRABLE_WEIGHT = 1.5
UNDESIRABLE_WEIGHT = 0.75

# The REAL Qwen tokenizer tokenizes the text rows, so the tiny model must span its full id range.
VOCAB_SIZE = 151936

# The reference must not equal the policy: with identical weights every log-ratio is 0, the KL
# baseline clamps to 0 and the loss is the constant 0.5 — reference and controls all go vacuous.
# The assignment is not arbitrary either: swapped, the mean log-ratio is negative, the baseline
# clamps to 0 and the local-vs-global control cannot discriminate (asserted below).
POLICY_SEED = 4321
REFERENCE_SEED = 1234

# Both sides run fp32, so the residual is reduction-order noise only: measured 1.3e-6.
LOSS_RTOL = 1e-3
# The gradient the optimizer consumes is compared to the same fp32 math on an unsharded replica, so
# this bound covers reduction-order noise only. The controls are held to the same number on purpose:
# a control's job is to show the pin would FAIL, so the threshold it must clear IS the pin's bound.
GRAD_RTOL = 1e-3
# Each control must miss by far more than that residual. Measured: 7.3e-2 for the local-baseline
# control (the shape a dropped gather actually takes) and 1.3e-2 for switching the KL term off
# entirely — the smaller of the two is still four orders of magnitude above the residual.
CONTROL_MIN_RTOL = 5e-3


def build_model(seed: int, tokenizer) -> Qwen3ForCausalLM:
    torch.manual_seed(seed)
    config = Qwen3Config(
        **{**TINY_QWEN3_CONFIG, "vocab_size": VOCAB_SIZE},
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return Qwen3ForCausalLM(config)


def build_dataset() -> Dataset:
    """Unpaired KTO rows with ragged completions and blocked desirability labels.

    Rows are grouped in blocks of ``ROWS_PER_RANK`` — one block per device per step — with
    alternating labels, so each rank's batch is all-desirable or all-undesirable. That is the ragged
    concat shape the FSDP grad sync has to survive (TRL concatenates the desirable and undesirable
    terms, so one of the two is empty on each rank) and it makes the ranks' KL log-ratio means
    disagree, which is what gives the local-vs-gathered baseline control its resolution.
    """
    rows = {"prompt": [], "completion": [], "label": []}
    for i in range(N_ROWS):
        rows["prompt"].append(f"Question {i}: what is {i} plus {i}?")
        rows["completion"].append(" The answer is " + " ".join(str(2 * i + j) for j in range(1 + i % 5)) + ".")
        rows["label"].append((i // ROWS_PER_RANK) % 2 == 0)
    return Dataset.from_dict(rows)


def sequence_logprobs(model, input_ids: torch.Tensor, attention_mask: torch.Tensor, mask: torch.Tensor):
    """TRL's per-sequence completion log-prob: shift by one, score where the mask is 1, sum.

    Independent of ``src/`` and of TRL's own helper — a plain fp32 log-softmax. Differentiable: the
    gradient reference below backprops through it; callers that only want a value wrap it in
    ``torch.no_grad()``.
    """
    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    token_logps = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    token_logps = token_logps.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)
    return (token_logps * mask[:, 1:]).sum(-1)


def local_kl_mean(policy, reference, batch, device) -> float:
    """This rank's contribution to the KL baseline: the mean policy-minus-reference log-ratio over
    the batch's ROTATED (mismatched) completions."""
    keys = ("KL_completion_input_ids", "KL_completion_attention_mask", "KL_completion_mask")
    tensors = [batch[key].to(device) for key in keys]
    with torch.no_grad():
        policy_logps = sequence_logprobs(policy, *tensors)
        reference_logps = sequence_logprobs(reference, *tensors)
    return float((policy_logps - reference_logps).mean())


def kto_loss(policy, reference, batch, kl: float, device) -> torch.Tensor:
    """TRL's exact ``kto`` objective on one rank's batch, given the KL baseline.

    Returns a differentiable scalar: the value pins are ``float()`` of it, and the gradient pin
    backprops through it. The reference model is frozen, so only the policy side carries grad — the
    KL baseline is a detached constant in TRL too.
    """
    keys = ("completion_input_ids", "completion_attention_mask", "completion_mask")
    tensors = [batch[key].to(device) for key in keys]
    with torch.no_grad():
        reference_logps = sequence_logprobs(reference, *tensors)
    logratios = sequence_logprobs(policy, *tensors) - reference_logps
    desirable = torch.as_tensor(batch["label"], dtype=torch.bool, device=device)
    chosen_losses = 1 - torch.sigmoid(BETA * (logratios[desirable] - kl))
    rejected_losses = 1 - torch.sigmoid(BETA * (kl - logratios[~desirable]))
    losses = torch.cat((DESIRABLE_WEIGHT * chosen_losses, UNDESIRABLE_WEIGHT * rejected_losses))
    return losses.nanmean()


def flat_gradient(model, order: list[str]) -> torch.Tensor:
    """Every named gradient in ``order``, unsharded and concatenated into one fp32 vector.

    FSDP2 leaves gradients as DTensors sharded like their parameter, so ``full_tensor()`` rebuilds
    the gradient the optimizer will actually consume.
    """
    grads = dict(model.named_parameters())
    flat = []
    for name in order:
        grad = grads[name].grad
        if grad is None:
            raise AssertionError(f"{name} has no gradient — backward did not reach it")
        flat.append((grad.full_tensor() if isinstance(grad, DTensor) else grad).detach().float().flatten())
    return torch.cat(flat)


def relative_error(got: torch.Tensor, want: torch.Tensor) -> float:
    return float((got - want).norm() / want.norm().clamp(min=1e-12))


@gpu_test_main(exact_world_size=2, prefix="kto_fsdp_multi_gpu")
def run(ctx):
    from src.trainers.preference.kto import DistributedKTOTrainer

    checks, metrics = {}, {}
    tokenizer = AutoTokenizer.from_pretrained(QWEN3_0_6B)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    config = ParallelismConfig()  # plain FSDP2 data parallelism over both ranks
    args = KTOConfig(
        output_dir=ctx.output_dir,
        max_steps=N_STEPS,
        per_device_train_batch_size=ROWS_PER_RANK,
        learning_rate=2e-3,
        lr_scheduler_type="constant",
        logging_steps=1,
        save_strategy="no",
        report_to=[],
        seed=42,
        # fp32: the pinned quantity is the KL gather, and in bf16 the per-sequence log-prob sums
        # (~-1e2) carry ~0.4 of rounding error, which beta scales straight into the sigmoid and
        # swamps the objective. fp32 is affordable on a tiny model and makes the residual ~1e-4.
        bf16=False,
        gradient_checkpointing=False,
        max_length=MAX_LENGTH,
        beta=BETA,
        desirable_weight=DESIRABLE_WEIGHT,
        undesirable_weight=UNDESIRABLE_WEIGHT,
        dataloader_num_workers=0,
        fsdp="",  # the mixin owns FSDP wrapping
    )

    trainer = DistributedKTOTrainer(
        model=build_model(POLICY_SEED, tokenizer),
        ref_model=build_model(REFERENCE_SEED, tokenizer),
        args=args,
        train_dataset=build_dataset(),
        processing_class=tokenizer,
        parallelism_config=config,
    )
    checks["kl_term_is_live"] = trainer.calculate_KL is True
    checks["reference_model_is_live"] = trainer.ref_model is not None

    # ── Capture step 1's batch off the trainer's own compute_loss, so the reference consumes exactly
    # the rows, rotation and padding the trained forward saw.
    captured: dict = {}
    real_compute_loss = trainer.compute_loss

    def spying_compute_loss(model_arg, inputs, *args_, **kwargs):
        if not captured:
            captured.update({key: value.clone() if torch.is_tensor(value) else value for key, value in inputs.items()})
        return real_compute_loss(model_arg, inputs, *args_, **kwargs)

    trainer.compute_loss = spying_compute_loss
    ctx.on_teardown(trainer.cleanup_ep)
    trainer.train()

    losses = [entry["loss"] for entry in trainer.state.log_history if "loss" in entry]
    checks["ran_all_steps"] = trainer.state.global_step == N_STEPS and len(losses) == N_STEPS
    checks["losses_finite"] = bool(torch.isfinite(torch.tensor(losses)).all())
    # The rows are synthetic and the two ranks hold opposite-desirability batches, so the objective
    # is not expected to descend monotonically; what must hold is that gradients reach the policy.
    checks["loss_responded_to_updates"] = abs(losses[-1] - losses[0]) > 1e-4
    metrics["first_loss"], metrics["last_loss"] = losses[0], losses[-1]

    grad_norms = [entry["grad_norm"] for entry in trainer.state.log_history if "grad_norm" in entry]
    checks["grad_norm_logged_finite"] = bool(grad_norms) and bool(torch.isfinite(torch.tensor(grad_norms)).all())
    checks["grad_norm_nonzero"] = bool(grad_norms) and min(grad_norms) > 1e-6

    # ── The world-global KL baseline, rebuilt from every rank's own captured KL batch.
    # Early return, not a bare check: without the capture every line below raises KeyError and the
    # harness reports `error`, so the check itself could never be the thing that says FAIL.
    checks["captured_step1_batch"] = "KL_completion_input_ids" in captured
    if not checks["captured_step1_batch"]:
        return {"checks": checks, "metrics": metrics}
    policy_replica = build_model(POLICY_SEED, tokenizer).to(ctx.device).eval()
    reference_replica = build_model(REFERENCE_SEED, tokenizer).to(ctx.device).eval()
    local_kl = local_kl_mean(policy_replica, reference_replica, captured, ctx.device)
    gathered_kl: list = [None] * ctx.world_size
    dist.all_gather_object(gathered_kl, local_kl)
    global_kl = max(sum(gathered_kl) / len(gathered_kl), 0.0)
    metrics["local_kl"], metrics["global_kl"] = local_kl, global_kl
    kl_spread = max(gathered_kl) - min(gathered_kl)
    metrics["kl_spread"] = kl_spread
    # Preconditions for the local-vs-global control to mean anything: the baseline must survive the
    # clamp, and the ranks must genuinely disagree about it.
    checks["kl_baseline_survives_clamp"] = global_kl > 0.0
    checks["ranks_disagree_on_local_kl"] = kl_spread > 1e-3

    reference = world_mean(
        float(kto_loss(policy_replica, reference_replica, captured, global_kl, ctx.device)), ctx.device
    )
    error = abs(losses[0] - reference) / abs(reference)
    metrics["step1_loss"], metrics["step1_reference"], metrics["step1_rel_err"] = losses[0], reference, error
    log(f"[rank {ctx.rank}] step1={losses[0]:.6f} reference={reference:.6f} rel_err={error:.3e}")
    checks["step1_loss_matches_reference"] = error < LOSS_RTOL

    # Control: the SAME objective with each rank's LOCAL baseline — what a dropped gather produces.
    local_baseline = world_mean(
        float(kto_loss(policy_replica, reference_replica, captured, max(local_kl, 0.0), ctx.device)), ctx.device
    )
    # Control: the KL term switched off entirely.
    no_baseline = world_mean(float(kto_loss(policy_replica, reference_replica, captured, 0.0, ctx.device)), ctx.device)
    for name, value in (("local_kl_baseline", local_baseline), ("no_kl_baseline", no_baseline)):
        control_error = abs(losses[0] - value) / abs(value)
        metrics[f"{name}_rel_err"] = control_error
        checks[f"{name}_control_misses"] = control_error > CONTROL_MIN_RTOL

    # ── FSDP2 gradient sync ──────────────────────────────────────────────────────────────────
    # The gradient FSDP2 hands the optimizer must be the DP AVERAGE of the two ranks' local
    # gradients. Comparing gathered PARAMETERS instead cannot see this at all: 1D full-shard makes
    # every parameter a Shard(0) DTensor, so full_tensor() is an all-gather that returns the same
    # bytes on every rank whatever each shard contains.
    trained_weights = snapshot_full_weights(trainer.model)  # collective
    order = sorted(name for name, param in trainer.model.named_parameters() if param.requires_grad)

    trainer.model.train()
    trainer.model.zero_grad(set_to_none=True)
    trainer.compute_loss(trainer.model, captured).backward()  # FSDP2 reduce-scatters in here
    fsdp_gradient = flat_gradient(trainer.model, order)

    # Same weights, no sharding: this rank's own gradient, then its DP average.
    replica = build_model(POLICY_SEED, tokenizer)
    replica.load_state_dict(trained_weights)
    replica = replica.to(ctx.device).train()
    replica_kl = max(world_mean(local_kl_mean(replica, reference_replica, captured, ctx.device), ctx.device), 0.0)
    replica.zero_grad(set_to_none=True)
    kto_loss(replica, reference_replica, captured, replica_kl, ctx.device).backward()
    local_gradient = flat_gradient(replica, order)
    reference_gradient = local_gradient.clone()
    dist.all_reduce(reference_gradient, op=dist.ReduceOp.AVG)

    sync_error = relative_error(fsdp_gradient, reference_gradient)
    metrics["grad_sync_rel_err"] = sync_error
    checks["fsdp_gradient_is_the_dp_average"] = sync_error < GRAD_RTOL

    # NEGATIVE CONTROL: ``local_gradient`` is exactly what this rank would hold with the DP reduce
    # dropped — its own batch's gradient and nothing else. It must MISS the pin, or "FSDP2 produced
    # the average" carries no information because the average and the local value coincide.
    unreduced_error = relative_error(local_gradient, reference_gradient)
    metrics["unreduced_grad_rel_err"] = unreduced_error
    checks["unreduced_gradient_would_miss_the_pin"] = unreduced_error > GRAD_RTOL
    # And FSDP2's gradient must not BE the local one, which is the same statement from the other side.
    fsdp_vs_local = relative_error(fsdp_gradient, local_gradient)
    metrics["fsdp_vs_local_rel_err"] = fsdp_vs_local
    checks["fsdp_gradient_is_not_the_local_gradient"] = fsdp_vs_local > GRAD_RTOL
    log(
        f"[rank {ctx.rank}] grad vs DP average={sync_error:.3e}  "
        f"un-reduced vs average={unreduced_error:.3e}  fsdp vs local={fsdp_vs_local:.3e}"
    )

    return {"checks": checks, "metrics": metrics}


if __name__ == "__main__":
    run()  # the harness owns sys.exit
