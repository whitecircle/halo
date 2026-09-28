"""A run's first optimizer step, recorded off the trainer and scored by a reference trainer.

A parallel mode that miscounts its objective (a CP loss every rank's all-reduce hands in full, then
rescaled by ``cp_size``) still trains to finite losses and passes every smoke check. The value has to
be compared. :func:`train_recording_first_step` runs ``trainer.train()`` and records what the first
optimizer step consumed, off the trainer's own seams: each microbatch ``compute_loss`` saw with the
loss it returned, the logged step loss, the gradients handed to the optimizer, and every backward that
reached PyTorch's c10d autograd fallback. :func:`score_first_step` runs those microbatches through a
reference trainer on the initial weights as the training step would; :func:`first_step_checks` and
:func:`first_step_gradient_checks` compare the two.

Gradients are compared on the FSDP2-sharded (DTensor) parameters, which FSDP2 and the trainer's clip
seam finalize the same way on both sides. EP experts and routers are FSDP-ignored and live in a
layout-specific form, so they are left out.
"""

import contextlib
import math
import warnings
from dataclasses import dataclass, field

import torch
from torch.distributed.tensor import DTensor
from transformers import TrainerCallback

from tests.common.distributed import world_mean
from tests.common.tolerances import TOL
from tests.common.utils import log, step_losses

# What PyTorch warns when a backward passes through a collective with no autograd kernel (an in-place
# ``dist.all_reduce`` on a grad-carrying tensor): the gradient goes through as the identity.
AUTOGRAD_FALLBACK_WARNING = "autograd kernel was not registered"


@dataclass
class FirstStep:
    """What a run's first optimizer step consumed, on the host, so the trainer can be freed."""

    batches: list[dict] = field(default_factory=list)
    losses: list[float] = field(default_factory=list)
    grads: dict[str, torch.Tensor] = field(default_factory=dict)
    logged_loss: float = math.nan
    autograd_fallbacks: list[str] = field(default_factory=list)


def _sharded_grads(model) -> dict[str, torch.Tensor]:
    """Every FSDP2-sharded parameter's gradient, unsharded to fp32 on the host. Collective.

    Selected by the parameter rather than its gradient, so every rank reaches the same ``full_tensor``
    calls in the same order.
    """
    grads = {}
    for name, param in model.named_parameters():
        if not (param.requires_grad and isinstance(param, DTensor)):
            continue
        if param.grad is None:
            raise AssertionError(f"{name} carries no gradient at the optimizer step")
        grads[name] = param.grad.full_tensor().float().cpu()
    return grads


def _host_copy(value):
    return value.detach().cpu().clone() if torch.is_tensor(value) else value


class _FirstStepRecorder(TrainerCallback):
    """Records step 1's microbatches and losses through ``compute_loss``, and on request its gradients."""

    def __init__(self, trainer, record: FirstStep, *, gradients: bool):
        self._model = trainer.model
        self._record = record
        self._gradients = gradients
        microbatches = trainer.args.gradient_accumulation_steps
        compute_loss = trainer.compute_loss

        def recording_compute_loss(model, inputs, *args, **kwargs):
            # Training microbatches only: an evaluation pass reaches compute_loss with the model in eval mode.
            recording = model.training and len(record.batches) < microbatches
            if recording:
                record.batches.append({key: _host_copy(value) for key, value in inputs.items()})
            loss = compute_loss(model, inputs, *args, **kwargs)
            if recording:
                record.losses.append(float(loss))
            return loss

        trainer.compute_loss = recording_compute_loss
        trainer.add_callback(self)

    def on_pre_optimizer_step(self, args, state, control, **kwargs):
        if self._gradients and state.global_step == 0:
            self._record.grads = _sharded_grads(self._model)


@contextlib.contextmanager
def recording_autograd_fallbacks(sink: list[str]):
    """Append every c10d autograd-fallback warning raised inside to ``sink``; other warnings show as usual."""
    with warnings.catch_warnings():
        show = warnings.showwarning

        def record_fallback(message, category, filename, lineno, file=None, line=None):
            if AUTOGRAD_FALLBACK_WARNING in str(message):
                sink.append(str(message))
            else:
                show(message, category, filename, lineno, file, line)

        warnings.showwarning = record_fallback
        warnings.filterwarnings("always", message=f".*{AUTOGRAD_FALLBACK_WARNING}")
        yield


def train_recording_first_step(trainer, *, gradients: bool = True):
    """``trainer.train()``, returning its result and the :class:`FirstStep` it took.

    ``gradients`` records the step's sharded gradients too, an all-gather of each onto the host.
    """
    record = FirstStep()
    _FirstStepRecorder(trainer, record, gradients=gradients)
    with recording_autograd_fallbacks(record.autograd_fallbacks):
        result = trainer.train()
    record.logged_loss = step_losses(trainer)[0]
    return result, record


def score_first_step(trainer, record: FirstStep) -> FirstStep:
    """``trainer``'s own first step on ``record``'s microbatches. Collective.

    Mirrors the training step outside the loop: ``compute_loss`` under the trainer's loss context and,
    when ``record`` carries gradients, each loss divided by the microbatch count, backpropagated through
    the accelerator and finalized by the clip seam HF runs before the optimizer step. The result carries
    the losses, the step loss HF would log and those gradients. Only a trainer whose loss is its own
    mean is mirrored: HF skips the division when it normalizes by the batch's item count instead.
    """
    if trainer.model_accepts_loss_kwargs:
        raise ValueError(
            f"{type(trainer).__name__} normalizes its loss by the batch's item count, a step score_first_step "
            f"does not mirror"
        )
    trainer.model.train()
    scored = FirstStep()
    for batch in record.batches:
        inputs = trainer._prepare_inputs(batch)
        with torch.set_grad_enabled(bool(record.grads)), trainer.compute_loss_context_manager():
            loss = trainer.compute_loss(trainer.model, inputs)
        if record.grads:
            trainer.accelerator.backward(loss / len(record.batches))
        scored.losses.append(float(loss))
    if record.grads:
        # HF's clip, or its norm-only pass when clipping is off: where the EP, TP and QLoRA syncs run.
        grad_norm = trainer._clip_grad_norm(trainer.model) if trainer.args.max_grad_norm > 0 else None
        trainer._get_grad_norm(trainer.model, grad_norm=grad_norm)
        scored.grads = _sharded_grads(trainer.model)
    # HF logs the world mean of the per-rank step losses.
    scored.logged_loss = world_mean(sum(scored.losses) / len(scored.losses), trainer.accelerator.device)
    return scored


def _relative_error(got: float, want: float) -> float:
    """``|got - want|`` relative to ``max(1, |want|)``, so a loss near zero is judged absolutely."""
    return abs(got - want) / max(1.0, abs(want))


def first_step_checks(
    record: FirstStep, reference: FirstStep, *, miscount_factor: int, loss_rtol: float
) -> tuple[dict[str, bool], dict[str, float]]:
    """The loss checks on ``record``'s step against ``reference``'s, and the metrics behind them.

    Losses are compared relative to ``max(1, |reference|)``. ``miscount_factor`` is the factor a
    miscounted objective carries (the parallel axis size): the bound has to resolve a loss scaled by
    it, or a match says nothing. Also fails a backward that reached the c10d autograd fallback.
    """
    loss_error = max(_relative_error(got, want) for got, want in zip(record.losses, reference.losses, strict=True))
    logged_error = _relative_error(record.logged_loss, reference.logged_loss)
    miscount_error = _relative_error(miscount_factor * reference.logged_loss, reference.logged_loss)
    checks = {
        "step1_losses_match_reference": loss_error <= loss_rtol,
        "logged_step1_loss_is_the_reference_loss": logged_error <= loss_rtol,
        "loss_bound_resolves_a_miscounted_loss": miscount_error > TOL.control_min_loss_shift(loss_rtol),
        "backward_never_took_the_c10d_autograd_fallback": not record.autograd_fallbacks,
    }
    metrics = {
        "step1_loss_max_rel_err": loss_error,
        "logged_step1_loss": record.logged_loss,
        "reference_step1_loss": reference.logged_loss,
        "autograd_fallback_warnings": len(record.autograd_fallbacks),
    }
    log(
        f"  step 1 vs reference: losses {[f'{x:.5f}' for x in record.losses]} vs "
        f"{[f'{x:.5f}' for x in reference.losses]} (max rel err {loss_error:.2e}, tol {loss_rtol}); logged "
        f"{record.logged_loss:.5f} vs {reference.logged_loss:.5f} (rel err {logged_error:.2e}); "
        f"{len(record.autograd_fallbacks)} autograd-fallback warnings"
    )
    return checks, metrics


def _gradient_agreement(got: dict[str, torch.Tensor], want: dict[str, torch.Tensor]) -> tuple[float, float]:
    """(cosine, norm ratio) of two gradients taken as one vector each over ``want``'s parameters, in fp64.

    Accumulated per parameter rather than over a concatenation, which would hold a second full copy of
    the model's gradient on the host. NaN when either side is non-finite, zero or missing a parameter.
    """
    if not want or want.keys() - got.keys():
        return math.nan, math.nan
    dot = got_sq = want_sq = 0.0
    for name, reference in want.items():
        a, b = got[name].double().flatten(), reference.double().flatten()
        dot += float(a @ b)
        got_sq += float(a @ a)
        want_sq += float(b @ b)
    if not (got_sq > 0 and want_sq > 0):
        return math.nan, math.nan
    return dot / math.sqrt(got_sq * want_sq), math.sqrt(got_sq / want_sq)


def first_step_gradient_checks(
    record: FirstStep,
    reference: FirstStep,
    *,
    cosine_min: float = TOL.cp_grad_cosine_min,
    norm_rtol: float = TOL.cp_grad_norm_rtol,
) -> tuple[dict[str, bool], dict[str, float]]:
    """The gradient checks on ``record``'s step against ``reference``'s, and the metrics behind them.

    The gradient is judged as one vector, by direction and by norm: a mis-scaled reduction keeps the
    direction and a mis-routed one keeps the norm. Per parameter, bf16 noise on the small norm-weight
    gradients is too large to gate on.
    """
    cosine, norm_ratio = _gradient_agreement(record.grads, reference.grads)
    checks = {
        "step1_grads_cover_the_reference_params": bool(reference.grads)
        and record.grads.keys() == reference.grads.keys(),
        "step1_grad_direction_matches_reference": cosine >= cosine_min,
        "step1_grad_norm_matches_reference": abs(norm_ratio - 1.0) <= norm_rtol,
    }
    log(
        f"  step 1 gradient over {len(reference.grads)} params: cosine {cosine:.6f} (min {cosine_min}), "
        f"norm ratio {norm_ratio:.5f} (tol {norm_rtol})"
    )
    return checks, {"step1_grad_cosine": cosine, "step1_grad_norm_ratio": norm_ratio}
