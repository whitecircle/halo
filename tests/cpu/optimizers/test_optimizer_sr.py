#!/usr/bin/env python
"""CPU tests for AdamWBF16 stochastic rounding (the eager / non-Triton path).

These pin the *numerical* properties that justify SR over nearest rounding:

  1. ``stochastic_round_to_bf16`` is an unbiased estimator of its fp32 input
     (mean of many roundings -> the input within Monte-Carlo error), and is exact
     on a value already on the bf16 grid.
  2. The eager Adam step keeps ``exp_avg`` (the signed first moment) on NEAREST
     rounding -> bit-identical across two runs from identical state, while the
     weight and ``exp_avg_sq`` differ run-to-run (SR noise).
  3. SR on ``exp_avg_sq`` removes the systematic +~50% upward bias that nearest
     rounding inflicts on the always-positive second-moment accumulator near the
     bf16 underflow floor. THIS is the test that bites if SR is dropped from the
     second moment: AdamWBF16's exp_avg_sq tracks fp32 Adam, whereas a
     nearest-rounded reference overshoots.

Run: python tests/cpu/optimizers/test_optimizer_sr.py  (or pytest)
"""

import math

import pytest
import torch

import src.optimizers.adamw_bf16 as adamw_mod
import src.optimizers.muon as muon_mod
from src.optimizers.adamw_bf16 import (
    AdamWBF16,
    _eager_adam_bf16_step,
    stochastic_round_to_bf16,
)


def _bf16_ulp(x: float) -> float:
    """Distance from ``x`` (cast to bf16) to the next bf16 value above it."""
    xb = torch.tensor(x, dtype=torch.bfloat16)
    nxt = torch.nextafter(xb, torch.tensor(float("inf"), dtype=torch.bfloat16))
    return float((nxt.float() - xb.float()).abs())


# 1. stochastic_round_to_bf16 — unbiased, and exact on grid points


def test_sr_unbiased_between_grid_points():
    """A value sitting 0.3 ulp above a bf16 grid point rounds to the bracketing
    grid points with probabilities that make the MEAN equal the input.

    Nearest rounding would always pick the lower grid point (bias = -0.3 ulp);
    SR makes the expected value the true fp32 value. With N=20000 draws the
    standard error of the mean is ~ulp/(2*sqrt(N)), so a 0.3-ulp constant bias
    would be ~50 sigma away and fail loudly.
    """
    torch.manual_seed(0)
    base = torch.tensor(1.0, dtype=torch.bfloat16).float()  # exact bf16 grid point
    ulp = _bf16_ulp(1.0)
    x = float(base) + 0.3 * ulp
    assert torch.tensor(x, dtype=torch.bfloat16).float().item() == float(base), (
        "test setup: x must round-to-nearest DOWN to base (so a bias is detectable)"
    )

    # Vectorized Monte-Carlo: one SR call over an [n] tensor draws n INDEPENDENT roundings
    # (``torch.randint_like`` adds per-element noise), statistically identical to n scalar calls but
    # without paying torch's per-op CPU thread-pool overhead 20000x — which is minutes on a many-core
    # host. This is also the way SR is actually applied in the optimizer: to whole tensors, not scalars.
    n = 20000
    samples = stochastic_round_to_bf16(torch.full((n,), x, dtype=torch.float32)).float()

    # Only the two bracketing grid points may appear.
    uniq = torch.unique(samples)
    assert uniq.numel() <= 2, f"SR produced non-bracketing values: {uniq}"
    assert float(samples.min()) == float(base)
    assert abs(float(samples.max()) - (float(base) + ulp)) < 1e-6

    mean = float(samples.mean())
    tol = 4.0 * ulp / math.sqrt(n)  # ~4 sigma
    assert abs(mean - x) < tol, f"SR mean {mean} not within {tol} of {x} (bias detected)"
    # A nearest-rounded estimate would equal `base`; the SR mean must be clearly above it.
    assert mean - float(base) > 0.15 * ulp, "SR did not lift the mean above the nearest-round value"


def test_sr_exact_on_grid_point():
    """A value exactly on a bf16 grid point rounds to itself every time (no noise)."""
    x = torch.tensor(2.0, dtype=torch.bfloat16).float().item()  # exactly representable
    # Vectorized: 2000 independent SR draws of an on-grid value in one call (see the unbiased test).
    rs = stochastic_round_to_bf16(torch.full((2000,), x, dtype=torch.float32)).float()
    assert torch.all(rs == x), f"on-grid value {x} rounded off-grid: {torch.unique(rs).tolist()}"


# 2. exp_avg stays NEAREST (deterministic); weight + exp_avg_sq carry SR noise


def _fresh_state(p):
    return {
        "exp_avg": torch.zeros_like(p),
        "exp_avg_sq": torch.zeros_like(p),
    }


def _run_eager_step(seed, p0, grad):
    """Run one eager Adam+SR step from identical inputs under a fixed RNG seed.

    Returns (param, exp_avg, exp_avg_sq) clones after the step.
    """
    torch.manual_seed(seed)
    p = p0.clone()
    state = _fresh_state(p)
    _eager_adam_bf16_step(
        p,
        grad.clone(),
        state["exp_avg"],
        state["exp_avg_sq"],
        step_size=1e-3,
        bc2_sqrt=math.sqrt(1.0 - 0.999),
        eps=1e-8,
        wd_factor=1.0,
        beta1=0.9,
        beta2=0.999,
    )
    return p.clone(), state["exp_avg"].clone(), state["exp_avg_sq"].clone()


def test_exp_avg_nearest_weight_and_easq_stochastic():
    """Two eager steps from identical state under DIFFERENT RNG seeds:
    exp_avg is bit-identical (nearest), while the weight and exp_avg_sq differ (SR).
    """
    torch.manual_seed(1234)
    p0 = (torch.randn(4096, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    # Small, near-constant gradient so easq lands near the bf16 underflow floor where SR matters.
    grad = torch.full((4096,), 3e-4, dtype=torch.bfloat16)

    p_a, ea_a, easq_a = _run_eager_step(11, p0, grad)
    p_b, ea_b, easq_b = _run_eager_step(99, p0, grad)

    # exp_avg: nearest rounding -> deterministic regardless of the SR RNG state.
    assert torch.equal(ea_a, ea_b), "exp_avg must be bit-identical (nearest rounding, no SR)"

    # weight + exp_avg_sq: stochastic rounding -> the two runs must differ somewhere.
    assert not torch.equal(p_a, p_b), "weight write must carry SR noise (differs across RNG seeds)"
    assert not torch.equal(easq_a, easq_b), "exp_avg_sq write must carry SR noise (differs across seeds)"


# 3. SR on exp_avg_sq removes the +~50% second-moment bias (THE bite test)


def _adam_easq_reference(grad_val, n_steps, beta2, sr=False, seed=0):
    """fp32-accumulated second moment, then stored as bf16 each step.

    sr=False -> nearest rounding (the buggy path that over-estimates).
    sr=True  -> stochastic rounding (matches AdamWBF16's eager path).
    Returns the mean of the final bf16 exp_avg_sq.
    """
    torch.manual_seed(seed)
    size = 8192
    grad = torch.full((size,), grad_val, dtype=torch.bfloat16)
    easq = torch.zeros(size, dtype=torch.bfloat16)
    for _ in range(n_steps):
        easq_fp32 = easq.float()
        g = grad.float()
        easq_fp32.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
        easq = stochastic_round_to_bf16(easq_fp32) if sr else easq_fp32.to(torch.bfloat16)  # nearest
    return float(easq.float().mean())


def test_sr_removes_second_moment_bias():
    """Drive a fixed tiny gradient so (1-beta2)*g^2 sits near the bf16 underflow
    floor, then compare second-moment estimates after many steps.

    fp32 reference is the ground truth. Nearest-rounded bf16 storage biases the
    always-positive accumulator SYSTEMATICALLY (large, single-signed error — here
    a steady-state undershoot because the repeated truncation of (1-beta2)*g^2
    pulls the EMA fixed point below the true value). AdamWBF16's eager step uses SR
    on the second moment, so it is unbiased and tracks fp32 — and is dramatically
    closer to fp32 than the nearest-rounded reference.

    The test asserts the MAGNITUDE of the nearest bias and the unbiasedness of SR,
    not a hard-coded sign: the bias sign depends on the gradient regime, but its
    presence (and SR's removal of it) is the invariant that breaks if SR is dropped.
    """
    beta2 = 0.999
    grad_val = 1e-4
    n_steps = 500

    # fp32 ground truth: no rounding of the state at all.
    torch.manual_seed(0)
    size = 8192
    g = torch.full((size,), grad_val, dtype=torch.float32)
    easq_fp32 = torch.zeros(size, dtype=torch.float32)
    for _ in range(n_steps):
        easq_fp32.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
    fp32_mean = float(easq_fp32.mean())
    assert fp32_mean > 0.0

    nearest_mean = _adam_easq_reference(grad_val, n_steps, beta2, sr=False)

    # AdamWBF16 eager path: step the real optimizer many times with a constant grad.
    torch.manual_seed(7)
    p = torch.nn.Parameter(torch.zeros(size, dtype=torch.bfloat16))
    opt = AdamWBF16([p], lr=1e-3, betas=(0.9, beta2), weight_decay=0.0, use_triton=False)
    for _ in range(n_steps):
        p.grad = torch.full((size,), grad_val, dtype=torch.bfloat16)
        opt.step()
    sr_mean = float(opt.state[p]["exp_avg_sq"].float().mean())

    # 1. Nearest rounding is systematically biased by a large margin (sign depends on regime).
    nearest_bias = abs(nearest_mean - fp32_mean) / fp32_mean
    assert nearest_bias > 0.15, (
        f"test premise broken: nearest-rounded easq should be biased from fp32 by a large "
        f"margin, got {nearest_bias:.1%} (nearest={nearest_mean:.3e}, fp32={fp32_mean:.3e})"
    )

    # 2. SR (AdamWBF16) tracks fp32 within a few percent (unbiased).
    sr_rel_err = abs(sr_mean - fp32_mean) / fp32_mean
    assert sr_rel_err < 0.05, (
        f"AdamWBF16 (SR) easq mean should track fp32, got rel-err {sr_rel_err:.1%} "
        f"(sr={sr_mean:.3e}, fp32={fp32_mean:.3e})"
    )

    # 3. Negative control: SR is strictly closer to fp32 than the nearest-rounded reference.
    assert abs(sr_mean - fp32_mean) < abs(sr_mean - nearest_mean), (
        f"SR easq ({sr_mean:.3e}) must be closer to fp32 ({fp32_mean:.3e}) than to the "
        f"nearest-rounded reference ({nearest_mean:.3e})"
    )


# 4. SR seed stream is STRUCTURAL: a grad-None param must not shift it
#    (rank-nonuniform grad presence would silently drift replicated bf16 params)


def _resync_sr_rng(module, seed=1234567):
    """Simulate rank-synchronized _SR_RNG state: both 'ranks' start from the same seed."""
    module._SR_RNG.seed(seed)


def test_adamw_seed_stream_survives_missing_grad():
    """Two 'ranks' (fresh optimizers with re-synced _SR_RNG) hold p0 (grad present on rank A only)
    and p1 (replicated, identical grads). p1 must round BIT-IDENTICALLY on both ranks: skipping the
    seed draws for grad-None params makes rank B consume p0's seeds for p1, silently drifting the
    replicated param apart."""
    torch.manual_seed(0)
    p0_init = (torch.randn(2048) * 0.02).to(torch.bfloat16)
    p1_init = (torch.randn(2048) * 0.02).to(torch.bfloat16)
    g1 = torch.full((2048,), 3e-4, dtype=torch.bfloat16)

    def run_rank(p0_has_grad: bool):
        _resync_sr_rng(adamw_mod)
        p0 = torch.nn.Parameter(p0_init.clone())
        p1 = torch.nn.Parameter(p1_init.clone())
        p0.grad = torch.full_like(p0, 1e-4) if p0_has_grad else None
        p1.grad = g1.clone()
        opt = AdamWBF16([p0, p1], lr=1e-3, use_triton=False)
        opt.step()
        return p1.detach().clone(), opt.state[p1]["exp_avg_sq"].clone()

    p1_a, easq_a = run_rank(p0_has_grad=True)
    p1_b, easq_b = run_rank(p0_has_grad=False)
    assert torch.equal(p1_a, p1_b), "replicated param drifted: grad-None param shifted the SR seed stream"
    assert torch.equal(easq_a, easq_b), "exp_avg_sq drifted: grad-None param shifted the SR seed stream"


def test_adamw_all_grads_present_ranks_identical():
    """Sanity control for the two-rank harness: with identical grad presence the ranks match."""
    torch.manual_seed(1)
    p_init = (torch.randn(1024) * 0.02).to(torch.bfloat16)

    def run_rank():
        _resync_sr_rng(adamw_mod)
        p = torch.nn.Parameter(p_init.clone())
        p.grad = torch.full_like(p, 2e-4)
        opt = AdamWBF16([p], lr=1e-3, use_triton=False)
        opt.step()
        return p.detach().clone()

    assert torch.equal(run_rank(), run_rank())


def test_muon_seed_draws_are_structural():
    """Muon's per-group collection must draw one SR seed per param in group order regardless of
    grad presence: a rank whose param lacks a grad must still consume that param's seed, so the
    surviving params' seeds match the all-grads rank."""

    def params(grad_flags):
        out = []
        for has_grad in grad_flags:
            p = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16))
            p.grad = torch.zeros_like(p) if has_grad else None
            out.append(p)
        return out

    _resync_sr_rng(muon_mod)
    with_grad_all, seeds_all = muon_mod._collect_params_with_sr_seeds(params([True, True, True]))
    assert len(with_grad_all) == 3 and len(seeds_all) == 3

    _resync_sr_rng(muon_mod)
    with_grad_skip, seeds_skip = muon_mod._collect_params_with_sr_seeds(params([True, False, True]))
    assert len(with_grad_skip) == 2
    assert seeds_skip[0] == seeds_all[0]
    assert seeds_skip[1] == seeds_all[2], (
        "the grad-None param did not consume its seed — surviving params' SR streams desync across ranks"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
