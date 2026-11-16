"""Float64 references for the bf16 optimizers, written from the update equations rather than from the kernels: AdamW as
``torch.optim.AdamW`` documents it (decoupled weight decay scaled by lr) and Muon's momentum, Nesterov look-ahead and
decayed weight update. A bf16 output is held to the two bf16 values around its float64 value, widened by the fp32
roundoff of the arithmetic that feeds it; a stochastically rounded one also to its mean over seeds. Run with
``TRITON_INTERPRET=1`` set before Triton is imported. The interpreter rounds a kernel's bf16 cast toward zero, so a
nearest-rounded kernel output is held to the two neighbours; the eager step's ``Tensor.bfloat16()`` to the nearest."""

import math

import torch
import triton
from torch import Tensor

from halo.optimizers.adamw_bf16 import AdamWBF16, _fused_update, _launch_fused, _launch_tables
from halo.optimizers.muon import _momentum_nesterov_kernel, _weight_decay_update_kernel
from halo.optimizers.stochastic_rounding import SR_BLOCK_SIZE

# fp32 unit roundoff; a slack of k * U32 * (sum of |terms|) covers k roundings of the terms' fp32 arithmetic.
U32 = 2.0**-24
BETAS = (0.9, 0.999)
# Seeds a rounding-bias check averages over; with 1024 elements each, a pooled |z| past 4 is a 6e-5 chance unbiased.
# A z that is not finite (no target between two bf16 values, so no variance) is a finding too: the check saw nothing.
SEEDS = 256
Z_LIMIT = 4.0


def bf16_neighbours(value: Tensor) -> tuple[Tensor, Tensor]:
    """The largest bf16 at or below each float64 ``value`` and the smallest at or above it."""
    nearest = value.to(torch.bfloat16)
    up = torch.nextafter(nearest, torch.full_like(nearest, math.inf))
    down = torch.nextafter(nearest, torch.full_like(nearest, -math.inf))
    return torch.where(nearest.double() > value, down, nearest), torch.where(nearest.double() < value, up, nearest)


def _round_to_nearest_even(value: Tensor) -> Tensor:
    below, above = bf16_neighbours(value)
    to_below, to_above = value - below.double(), above.double() - value
    even = (below.view(torch.int16) & 1) == 0
    return torch.where((to_below < to_above) | ((to_below == to_above) & even), below, above)


def _between_neighbours(out: Tensor, value: Tensor, slack: Tensor) -> Tensor:
    """``out`` is a bf16 neighbour of some value within ``slack`` of ``value``."""
    return (out.double() >= bf16_neighbours(value - slack)[0].double()) & (
        out.double() <= bf16_neighbours(value + slack)[1].double()
    )


def _nearest_rounded(out: Tensor, value: Tensor, slack: Tensor) -> Tensor:
    """``out`` is the round-to-nearest-even of some value within ``slack`` of ``value``."""
    low, high = _round_to_nearest_even(value - slack).double(), _round_to_nearest_even(value + slack).double()
    return (out.double() >= low) & (out.double() <= high)


def _pooled_z(draws: Tensor, value: Tensor) -> float:
    """Summed error of ``draws`` ``[seeds, n]`` over its standard deviation under unbiased rounding to the neighbours."""
    below, above = (t.double() for t in bf16_neighbours(value))
    width = above - below
    share = torch.where(width > 0, (value - below) / width.clamp(min=1e-300), 0.0)
    variance = draws.shape[0] * (width**2 * share * (1 - share)).sum()
    return float((draws.double() - value).sum() / variance.sqrt())


def adamw_float64(state: dict, *, step: int, lr: float, eps: float, weight_decay: float) -> dict:
    """``m = b1 m + (1 - b1) g``, ``v = b2 v + (1 - b2) g^2``, ``p = p (1 - lr wd) - lr m_hat / (sqrt(v_hat) + eps)``
    from bf16 inputs, in float64, with each output's fp32 slack."""
    beta1, beta2 = BETAS
    p, g, m, v = (state[k].double() for k in ("p", "grad", "exp_avg", "exp_avg_sq"))
    m_new = beta1 * m + (1 - beta1) * g
    v_new = beta2 * v + (1 - beta2) * g * g
    update = lr * (m_new / (1 - beta1**step)) / ((v_new / (1 - beta2**step)).sqrt() + eps)
    decayed = p * (1 - lr * weight_decay)
    return {
        "exp_avg": (m_new, 16 * U32 * (beta1 * m.abs() + (1 - beta1) * g.abs())),
        "exp_avg_sq": (v_new, 16 * U32 * (beta2 * v + (1 - beta2) * g * g)),
        "p": (decayed - update, 32 * U32 * (decayed.abs() + update.abs())),
    }


def _adamw_state(n: int, step: int, seed: int) -> dict:
    """bf16 inputs; zero moments at step 1, as a fresh optimizer holds them."""
    generator = torch.Generator().manual_seed(seed)
    state = {
        "p": (torch.randn(n, generator=generator) * 0.05).bfloat16(),
        "grad": (torch.randn(n, generator=generator) * 1e-3).bfloat16(),
        "exp_avg": (torch.randn(n, generator=generator) * 1e-3).bfloat16(),
        "exp_avg_sq": (torch.rand(n, generator=generator) * 1e-6).bfloat16(),
    }
    if step == 1:
        state["exp_avg"].zero_()
        state["exp_avg_sq"].zero_()
    return state


def _adamw_step(state: dict, *, fused: bool, step: int, lr: float, eps: float, weight_decay: float, name: str) -> dict:
    """One step from ``state`` through the fused kernel or the eager bf16 step; returns the stepped bf16 tensors."""
    p = torch.nn.Parameter(state["p"].clone())
    p.grad = state["grad"].clone()
    moments = {"exp_avg": state["exp_avg"].clone(), "exp_avg_sq": state["exp_avg_sq"].clone()}
    group = {"betas": BETAS, "lr": lr, "eps": eps, "weight_decay": weight_decay}
    if fused:
        update = _fused_update(p, name, {"step": step, **moments}, group)
        _launch_fused([update], *_launch_tables([update]), None, None, 1)
    else:
        optimizer = AdamWBF16(
            [{"params": [p], "param_names": [name]}], lr=lr, betas=BETAS, eps=eps, weight_decay=weight_decay
        )
        optimizer.state[p] = {"step": step - 1, **moments}
        optimizer.step()
    return {"p": p.detach(), **moments}


def adamw_float64_mismatches() -> list[str]:
    """Each case and output where the fused or eager AdamWBF16 step leaves its float64 bracket, and each ``lr = 0``
    step that moved a weight."""
    findings = []
    for n, step, lr, weight_decay, eps in (
        (n, step, lr, wd, eps)
        for n in (1, 1025)
        for step in (1, 3)
        for lr in (0.0, 1e-3)
        for wd in (0.0, 0.1)
        for eps in (1e-8, 1.0)
    ):
        state = _adamw_state(n, step, seed=n * 7 + step)
        reference = adamw_float64(state, step=step, lr=lr, eps=eps, weight_decay=weight_decay)
        for fused in (True, False):
            out = _adamw_step(state, fused=fused, step=step, lr=lr, eps=eps, weight_decay=weight_decay, name="w")
            case = f"{'fused' if fused else 'eager'} n={n} step={step} lr={lr} wd={weight_decay} eps={eps}"
            for key, (value, slack) in reference.items():
                nearest = key == "exp_avg"
                held = (_between_neighbours if fused or not nearest else _nearest_rounded)(out[key], value, slack)
                if not held.all():
                    findings.append(f"{case}: {key} leaves its float64 bracket at {int((~held).sum())} of {n}")
            if lr == 0.0 and not torch.equal(out["p"], state["p"]):
                findings.append(f"{case}: lr = 0 moved a weight")
    return findings


def adamw_rounding_bias_mismatches() -> list[str]:
    """The stochastically rounded weight and second moment of one sub-ulp step, averaged over :data:`SEEDS` seeds
    (each parameter name seeds its own noise), against float64."""
    state = _adamw_state(SR_BLOCK_SIZE, 2, seed=11)
    hyper = {"step": 2, "lr": 1e-3, "eps": 1e-8, "weight_decay": 0.1}
    reference = adamw_float64(state, **hyper)
    findings = []
    for fused in (True, False):
        draws = [_adamw_step(state, fused=fused, name=f"w{k}", **hyper) for k in range(SEEDS)]
        for key in ("p", "exp_avg_sq"):
            z = _pooled_z(torch.stack([d[key] for d in draws]), reference[key][0])
            if not math.isfinite(z) or abs(z) > Z_LIMIT:
                findings.append(f"{'fused' if fused else 'eager'} {key}: pooled z {z:.2f} over {SEEDS} seeds")
    return findings


def _momentum(momentum: Tensor, grad: Tensor, mu: float, nesterov: bool) -> tuple[Tensor, Tensor]:
    momentum, update = momentum.clone(), torch.empty_like(momentum)
    n = momentum.numel()
    _momentum_nesterov_kernel[(triton.cdiv(n, SR_BLOCK_SIZE),)](
        momentum, grad, update, mu, n, momentum, HAS_SKIP=False, NESTEROV=nesterov, BLOCK_SIZE=SR_BLOCK_SIZE
    )
    return momentum, update


def _decayed_update(p: Tensor, update: Tensor, decay: float, seed: int) -> Tensor:
    p = p.clone()
    n = p.numel()
    _weight_decay_update_kernel[(triton.cdiv(n, SR_BLOCK_SIZE),)](
        p, update, decay, n, seed, p, HAS_SKIP=False, BLOCK_SIZE=SR_BLOCK_SIZE
    )
    return p


def muon_float64_mismatches() -> list[str]:
    """Muon's two kernels against ``m = mu m + g``, ``u = mu m + g`` (Nesterov) or ``u = m``, and
    ``p = p * decay - u`` stochastically rounded, in float64."""
    findings = []
    for n, mu, nesterov, decay in (
        (n, mu, nesterov, decay)
        for n in (1, 1025)
        for mu in (0.95, 0.0)
        for nesterov in (True, False)
        for decay in (1.0, 1.0 - 1e-4)
    ):
        generator = torch.Generator().manual_seed(n)
        momentum = (torch.randn(n, generator=generator) * 1e-2).bfloat16()
        grad = (torch.randn(n, generator=generator) * 1e-2).bfloat16()
        p = (torch.randn(n, generator=generator) * 0.05).bfloat16()
        new_momentum, update = _momentum(momentum, grad, mu, nesterov)
        m, g = momentum.double(), grad.double()
        m_new = mu * m + g
        u = mu * m_new + g if nesterov else m_new
        m_slack = 16 * U32 * (mu * m.abs() + g.abs())
        u_slack = 16 * U32 * (mu * m_new.abs() + g.abs()) + m_slack
        stepped = _decayed_update(p, update, decay, seed=12345)
        p_new = p.double() * decay - update.double()
        p_slack = 16 * U32 * ((p.double() * decay).abs() + update.double().abs())
        case = f"n={n} mu={mu} nesterov={nesterov} decay={decay}"
        for key, out, value, slack in (
            ("momentum", new_momentum, m_new, m_slack),
            ("update", update, u, u_slack),
            ("weight", stepped, p_new, p_slack),
        ):
            held = _between_neighbours(out, value, slack)
            if not held.all():
                findings.append(f"{case}: {key} leaves its float64 bracket at {int((~held).sum())} of {n}")
    generator = torch.Generator().manual_seed(3)
    p = (torch.randn(SR_BLOCK_SIZE, generator=generator) * 0.05).bfloat16()
    update = (torch.randn(SR_BLOCK_SIZE, generator=generator) * 1e-5).bfloat16()
    decay = 1.0 - 1e-4
    draws = torch.stack([_decayed_update(p, update, decay, seed=s) for s in range(SEEDS)])
    z = _pooled_z(draws, p.double() * decay - update.double())
    if not math.isfinite(z) or abs(z) > Z_LIMIT:
        findings.append(f"weight update: pooled z {z:.2f} over {SEEDS} seeds")
    return findings
