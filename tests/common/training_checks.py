"""Checks of a finished training run: its per-step losses, the loss band and the finite-run verdicts the SFT
and LoRA suites share, the loss deltas of a resumed run, and the trainable parameters a run moved."""

import math

import torch
from accelerate.utils import extract_model_from_parallel
from torch.distributed.tensor import DTensor

from tests.common.rank_log import log

# Where a trained causal-LM loss lands: the floor catches a collapsed or masked-away objective, the ceiling
# an untrained or garbage forward. Finiteness alone passes both.
LM_TRAINING_LOSS_BAND = (0.05, 20.0)


def step_losses(trainer) -> list[float]:
    """Per-step training losses from a trainer's log history, eval entries and the run summary excluded."""
    return [e["loss"] for e in trainer.state.log_history if "loss" in e and "eval_loss" not in e]


def training_run_checks(
    train_result,
    trainer,
    max_steps: int,
    *,
    loss_band: tuple[float, float] | None = None,
    grad_norms: bool = False,
    loss_decreased: bool = False,
) -> dict[str, bool]:
    """The finished-run checks the SFT and LoRA suites share, each logged with its verdict.

    ``loss_finite`` (the reported training loss), ``all_steps_finite`` (at least one step loss logged,
    every one finite), ``steps_completed`` (``max_steps`` optimizer steps ran), given ``loss_band``
    ``loss_reasonable`` (the reported training loss strictly inside it), with ``grad_norms``
    ``grad_norms_finite`` (at least one gradient norm logged, every one finite), and with
    ``loss_decreased`` ``loss_decreased`` (the last logged step loss below the first, which fails on
    fewer than two logged steps).
    """
    training_loss = train_result.training_loss
    losses = step_losses(trainer)
    checks = {
        "loss_finite": math.isfinite(training_loss),
        "all_steps_finite": bool(losses) and all(math.isfinite(loss) for loss in losses),
        "steps_completed": train_result.global_step == max_steps,
    }
    log(f"  Loss is finite: {'PASS' if checks['loss_finite'] else 'FAIL'} ({training_loss:.6f})")
    log(f"  All step losses finite: {'PASS' if checks['all_steps_finite'] else 'FAIL'}")
    log(f"  Per-step losses: {[f'{loss:.4f}' for loss in losses]}")
    log(
        f"  Steps completed: {'PASS' if checks['steps_completed'] else 'FAIL'} "
        f"({train_result.global_step}/{max_steps})"
    )
    if loss_band is not None:
        low, high = loss_band
        checks["loss_reasonable"] = low < training_loss < high
        log(f"  Loss in band ({low}, {high}): {'PASS' if checks['loss_reasonable'] else 'FAIL'} ({training_loss:.4f})")
    if grad_norms:
        norms = [e["grad_norm"] for e in trainer.state.log_history if "grad_norm" in e]
        checks["grad_norms_finite"] = bool(norms) and all(math.isfinite(norm) for norm in norms)
        log(
            f"  All grad norms finite: {'PASS' if checks['grad_norms_finite'] else 'FAIL'} "
            f"({[f'{norm:.2f}' for norm in norms]})"
        )
    if loss_decreased:
        checks["loss_decreased"] = len(losses) >= 2 and losses[-1] < losses[0]
        trend = f"{losses[0]:.4f} -> {losses[-1]:.4f}" if losses else "no step logged"
        log(f"  Loss decreased: {'PASS' if checks['loss_decreased'] else 'FAIL'} ({trend})")
    return checks


def resumed_loss_deltas(
    uninterrupted: list[float], resumed: list[float], *, save_step: int, total_steps: int
) -> list[float] | None:
    """``|resumed - uninterrupted|`` at each step after ``save_step`` (``inf`` where either loss is not
    finite), or ``None`` when either run is missing one of those steps.

    Both lists are :func:`step_losses`; a resumed run's log history opens with the steps its
    checkpoint carried, so its last ``total_steps - save_step`` entries are the resumed ones.
    """
    remaining = total_steps - save_step
    tail, reference = resumed[-remaining:], uninterrupted[save_step:]
    if not len(tail) == len(reference) == remaining:
        log(
            f"  resumed run logged {len(tail)} of the {remaining} steps after {save_step} (uninterrupted {len(reference)})"
        )
        return None
    deltas = [
        abs(a - b) if math.isfinite(a) and math.isfinite(b) else math.inf for a, b in zip(tail, reference, strict=True)
    ]
    log(
        f"  uninterrupted {[f'{x:.6f}' for x in reference]}  resumed {[f'{x:.6f}' for x in tail]}  "
        f"deltas {[f'{d:.2e}' for d in deltas]}"
    )
    return deltas


def snapshot_trainable(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Every trainable parameter of ``model`` (framework wrappers stripped), whole and on the host:
    FSDP2 / TP DTensors are gathered. Collective when any is a DTensor."""
    snapshot = {}
    for name, param in extract_model_from_parallel(model, recursive=True).named_parameters():
        if param.requires_grad:
            data = param.data.full_tensor() if isinstance(param.data, DTensor) else param.data
            snapshot[name] = data.detach().cpu().clone()
    return snapshot
