"""Checks of a finished training run: losses, the finite-run verdicts, resumed loss deltas, and parameters stepped and
moved."""

import math
from collections.abc import Collection

import torch
from accelerate.utils import extract_model_from_parallel
from torch.distributed.tensor import DTensor

from halo.distributed.fsdp import reshard_fsdp2_modules
from halo.runtime.dtensor import to_local
from tests.common.rank_log import log, log_all

# The floor catches a collapsed or masked-away objective, the ceiling an untrained forward. Finiteness alone passes
# both.
LM_TRAINING_LOSS_BAND = (0.05, 20.0)
# Elements one update must be able to move before a bit-identical tensor fails the run; all of them staying put, under
# nearest or stochastic rounding, is out of reach of chance.
MIN_MOVABLE_ELEMENTS = 64


def step_losses(trainer) -> list[float]:
    """Per-step training losses, eval entries and the run summary excluded."""
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
    """The finished-run checks the SFT and LoRA suites share, each logged with its verdict; ``loss_decreased`` fails on
    fewer than two steps."""
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
    """``|resumed - uninterrupted|`` at each step after ``save_step`` (``inf`` if not finite), or None when a step is
    missing. A resumed log opens with the checkpoint's steps, so its last ``total_steps - save_step`` entries are the
    resumed ones."""
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
    """Every trainable parameter, whole and on the host; collective when any is a DTensor."""
    snapshot = {}
    for name, param in extract_model_from_parallel(model, recursive=True).named_parameters():
        if param.requires_grad:
            data = param.data.full_tensor() if isinstance(param.data, DTensor) else param.data
            snapshot[name] = data.detach().cpu().clone()
    return snapshot


def movable_elements(tensor: torch.Tensor, step: float) -> int:
    """How many elements an update of ``step`` moves under nearest rounding: spacing to the next larger magnitude under
    ``2 * step``."""
    finfo = torch.finfo(tensor.dtype)
    _, exponent = torch.frexp(tensor.detach().float().abs().clamp_min(finfo.tiny))
    spacing = torch.ldexp(torch.full(exponent.shape, finfo.eps, device=exponent.device), exponent - 1)
    return int((spacing < 2 * step).sum())


def movable_tensors(tensors: dict[str, torch.Tensor], step: float) -> set[str]:
    """Names holding at least :data:`MIN_MOVABLE_ELEMENTS` elements one update moves. Adam's first update is the LR; at
    ``1e-5`` a bf16 element moves only below ``2**-8``, so norm gains near 1 stay bit-identical."""
    return {name for name, tensor in tensors.items() if movable_elements(tensor, step) >= MIN_MOVABLE_ELEMENTS}


def stepped_parameters(model: torch.nn.Module, optimizer, names: Collection[str]) -> set[str]:
    """``names`` this rank's AdamW-family ``optimizer`` stepped: a non-zero local ``exp_avg``. Resharded first, so the
    params are its shards. Rank-local."""
    reshard_fsdp2_modules(model)
    wanted = set(names)
    stepped = set()
    for name, param in model.named_parameters():
        exp_avg = optimizer.state.get(param, {}).get("exp_avg") if name in wanted else None
        if exp_avg is not None and bool(to_local(exp_avg).ne(0).any()):
            stepped.add(name)
    return stepped


def weight_update_checks(
    model: torch.nn.Module,
    optimizer,
    before: dict[str, torch.Tensor],
    moved: Collection[str],
    learning_rate: float,
) -> dict[str, bool]:
    """A full fine-tune's verdict on ``before``, given the names in ``moved``. Rank-local. ``weights_stepped`` is the
    only signal on a tensor a few updates cannot change; ``movable_weights_moved`` fails a zero rate or a missed
    update."""
    names = set(before)
    stepped = stepped_parameters(model, optimizer, names)
    movable = movable_tensors(before, learning_rate)
    unmoved = movable - set(moved)
    log_all(
        f"  stepped {len(stepped)}/{len(names)} (not stepped: {sorted(names - stepped)}), moved "
        f"{len(movable) - len(unmoved)}/{len(movable)} within one update's reach (unmoved: {sorted(unmoved)}; beyond "
        f"it: {sorted(names - movable)})"
    )
    return {
        "weights_stepped": bool(names) and stepped == names,
        "movable_weights_moved": bool(movable) and not unmoved,
    }
