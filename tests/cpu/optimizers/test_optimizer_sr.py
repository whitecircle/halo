#!/usr/bin/env python
"""CPU tests for AdamWBF16 stochastic rounding (the eager / non-Triton path).

These pin the *numerical* properties that justify SR over nearest rounding:

  1. ``stochastic_round_to_bf16`` is an unbiased estimator of its fp32 input
     (mean of many roundings -> the input within Monte-Carlo error), and is exact
     on a value already on the bf16 grid.
  2. The eager Adam step keeps ``exp_avg`` (the signed first moment) on NEAREST
     rounding -> bit-identical across two runs from identical state, while the
     weight and ``exp_avg_sq`` differ run-to-run (SR noise).
  3. SR on ``exp_avg_sq`` removes the systematic bias, tens of percent with a
     regime-dependent sign, that nearest rounding inflicts on the always-positive
     second-moment accumulator near the bf16 underflow floor. THIS is the test that
     bites if SR is dropped from the second moment: AdamWBF16's exp_avg_sq tracks
     fp32 Adam, whereas a nearest-rounded reference misses it by more than 15%.
  4. The rounding noise is keyed by the parameter's step and name, not drawn: a
     replica missing another param's grad rounds alike, and an optimizer rebuilt and
     restored from a state dict rounds exactly as the uninterrupted one.

Run: python tests/cpu/optimizers/test_optimizer_sr.py  (or pytest)
"""

import copy
import math

import pytest
import torch
from torch.distributed.checkpoint.state_dict import _init_optim_state

import src.optimizers.adamw_bf16 as adamw_mod
import src.optimizers.muon as muon_mod
from src.optimizers.adamw_bf16 import (
    AdamWBF16,
    _eager_adam_bf16_step,
    sr_seed_pair,
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
    samples = stochastic_round_to_bf16(torch.full((n,), x, dtype=torch.float32), seed=0).float()

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
    rs = stochastic_round_to_bf16(torch.full((2000,), x, dtype=torch.float32), seed=0).float()
    assert torch.all(rs == x), f"on-grid value {x} rounded off-grid: {torch.unique(rs).tolist()}"


# 2. exp_avg stays NEAREST (deterministic); weight + exp_avg_sq carry SR noise


def _fresh_state(p):
    return {
        "exp_avg": torch.zeros_like(p),
        "exp_avg_sq": torch.zeros_like(p),
    }


def _run_eager_step(seed, p0, grad):
    """Run one eager Adam+SR step from identical inputs under the SR seed pair ``(seed, seed + 1)``.

    Returns (param, exp_avg, exp_avg_sq) clones after the step.
    """
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
        sr_seeds=(seed, seed + 1),
    )
    return p.clone(), state["exp_avg"].clone(), state["exp_avg_sq"].clone()


def test_exp_avg_nearest_weight_and_easq_stochastic():
    """Two eager steps from identical state under DIFFERENT SR seeds:
    exp_avg is bit-identical (nearest), while the weight and exp_avg_sq differ (SR) — and a rerun
    under the SAME seed reproduces them, so the difference is the seed's noise.
    """
    torch.manual_seed(1234)
    p0 = (torch.randn(4096, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    # Small, near-constant gradient so easq lands near the bf16 underflow floor where SR matters.
    grad = torch.full((4096,), 3e-4, dtype=torch.bfloat16)

    p_a, ea_a, easq_a = _run_eager_step(11, p0, grad)
    p_b, ea_b, easq_b = _run_eager_step(99, p0, grad)
    p_a2, _, easq_a2 = _run_eager_step(11, p0, grad)

    # exp_avg: nearest rounding -> deterministic regardless of the SR RNG state.
    assert torch.equal(ea_a, ea_b), "exp_avg must be bit-identical (nearest rounding, no SR)"

    # weight + exp_avg_sq: stochastic rounding -> the two runs must differ somewhere.
    assert not torch.equal(p_a, p_b), "weight write must carry SR noise (differs across RNG seeds)"
    assert not torch.equal(easq_a, easq_b), "exp_avg_sq write must carry SR noise (differs across seeds)"
    assert torch.equal(p_a, p_a2) and torch.equal(easq_a, easq_a2), "the SR seed must fully determine the noise"


# 3. SR on exp_avg_sq removes the nearest-rounding second-moment bias (THE bite test)


def _adam_easq_reference(grad_val, n_steps, beta2):
    """fp32-accumulated second moment, then stored as nearest-rounded bf16 each step (the biased
    path). Returns the mean of the final bf16 exp_avg_sq.
    """
    size = 8192
    grad = torch.full((size,), grad_val, dtype=torch.bfloat16)
    easq = torch.zeros(size, dtype=torch.bfloat16)
    for _ in range(n_steps):
        easq_fp32 = easq.float()
        g = grad.float()
        easq_fp32.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
        easq = easq_fp32.to(torch.bfloat16)
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
    size = 8192
    g = torch.full((size,), grad_val, dtype=torch.float32)
    easq_fp32 = torch.zeros(size, dtype=torch.float32)
    for _ in range(n_steps):
        easq_fp32.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
    fp32_mean = float(easq_fp32.mean())
    assert fp32_mean > 0.0

    nearest_mean = _adam_easq_reference(grad_val, n_steps, beta2)

    # AdamWBF16 eager path: step the real optimizer many times with a constant grad, so each step's
    # noise must be fresh for the average to come out unbiased.
    p = torch.nn.Parameter(torch.zeros(size, dtype=torch.bfloat16))
    opt = AdamWBF16([("p", p)], lr=1e-3, betas=(0.9, beta2), weight_decay=0.0, use_triton=False)
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


# 4. SR seeds are keyed by (step, param name): replicas and resumes round alike


def test_seed_pairs_differ_across_steps_and_params():
    """Each (step, param key) draws its own seeds, inside the kernel's int32-safe range, and the two
    optimizers' keys give separate streams: a key that ignored the step would replay one step's noise
    on the next and bias the rounding, one that ignored the param would share it across params."""
    key = adamw_mod._SR_KEY
    seeds = [sr_seed_pair(key, step, index) for step in range(1, 21) for index in range(10)]
    assert len({pair[0] for pair in seeds}) == len(seeds), "two (step, param) pairs share a kernel seed"
    assert all(pair[0] != pair[1] for pair in seeds), "the eager step's two SR writes share a seed"
    assert all(0 <= seed < 2**30 for pair in seeds for seed in pair)
    assert sr_seed_pair(key, 7, 3) == sr_seed_pair(key, 7, 3)
    assert sr_seed_pair(key, 7, 3) != sr_seed_pair(muon_mod._SR_KEY, 7, 3), "the keys do not separate the streams"


def test_adamw_missing_grad_leaves_the_replicated_params_rounding_alone():
    """Two 'ranks' hold p0 (grad present on rank A only) and p1 (replicated, identical grads). p1 must
    round BIT-IDENTICALLY on both: its seed is keyed by its own name and step, which a grad-None p0
    ahead of it does not shift."""
    torch.manual_seed(0)
    p0_init = (torch.randn(2048) * 0.02).to(torch.bfloat16)
    p1_init = (torch.randn(2048) * 0.02).to(torch.bfloat16)
    g1 = torch.full((2048,), 3e-4, dtype=torch.bfloat16)

    def run_rank(p0_has_grad: bool):
        p0 = torch.nn.Parameter(p0_init.clone())
        p1 = torch.nn.Parameter(p1_init.clone())
        opt = AdamWBF16([("p0", p0), ("p1", p1)], lr=1e-3, use_triton=False)
        for _ in range(3):
            p0.grad = torch.full_like(p0, 1e-4) if p0_has_grad else None
            p1.grad = g1.clone()
            opt.step()
        return p1.detach().clone(), opt.state[p1]["exp_avg_sq"].clone()

    p1_a, easq_a = run_rank(p0_has_grad=True)
    p1_b, easq_b = run_rank(p0_has_grad=False)
    assert torch.equal(p1_a, p1_b), "replicated param drifted: a grad-None param shifted its rounding"
    assert torch.equal(easq_a, easq_b), "exp_avg_sq drifted: a grad-None param shifted its rounding"


def test_a_param_rounds_by_its_name_whatever_else_the_rank_holds():
    """A replicated param must round as its replicas do even where one rank's optimizer holds a param
    the others lack (an expert-TP rank's bias its partners do not own), so a param ahead of it does
    not shift its noise; the same param and grad under another name draws other noise, so the name
    really keys the seed (else every param would share one noise pattern)."""
    torch.manual_seed(1)
    p_init = (torch.randn(1024) * 0.02).to(torch.bfloat16)

    def run(name: str, leading: int):
        pads = [(f"pad{i}", torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))) for i in range(leading)]
        p = torch.nn.Parameter(p_init.clone())
        p.grad = torch.full_like(p, 2e-4)
        AdamWBF16([*pads, (name, p)], lr=1e-3, use_triton=False).step()
        return p.detach().clone()

    assert torch.equal(run("router.bias", 0), run("router.bias", 1)), "a param on one rank only shifted the rounding"
    assert not torch.equal(run("router.bias", 0), run("router.weight", 0))


def test_an_optimizer_built_without_names_is_refused():
    with pytest.raises(ValueError, match="keys its stochastic rounding by parameter name"):
        AdamWBF16([torch.nn.Parameter(torch.zeros(4, dtype=torch.bfloat16))], lr=1e-3, use_triton=False)


def _optimizer(params: list[torch.nn.Parameter]) -> AdamWBF16:
    """Two groups, so the restored positions span a group boundary."""
    named = [(f"p{i}", p) for i, p in enumerate(params)]
    groups = [{"params": named[:2]}, {"params": named[2:], "weight_decay": 0.0}]
    return AdamWBF16(groups, lr=1e-3, use_triton=False)


def _train(opt: AdamWBF16, params: list[torch.nn.Parameter], grads: list[list[torch.Tensor]]) -> None:
    for step_grads in grads:
        for p, g in zip(params, step_grads, strict=True):
            p.grad = g.clone()
        opt.step()


def test_restored_adamw_rounds_like_the_uninterrupted_one():
    """Resume in one process, the way the trainer restores: a rebuilt optimizer materializes its state
    with torch's zero-LR ``_init_optim_state`` step, then loads the saved state. The steps after the
    restore must be bit-identical to the uninterrupted run's with nothing reset in between, and an
    unrelated optimizer stepping in the same process meanwhile changes nothing."""
    torch.manual_seed(3)
    inits = [(torch.randn(512) * 0.02).to(torch.bfloat16) for _ in range(3)]
    grads = [[torch.randn(512, dtype=torch.bfloat16) * 1e-3 for _ in inits] for _ in range(6)]

    def fresh_params(values):
        return [torch.nn.Parameter(t.clone()) for t in values]

    uninterrupted = fresh_params(inits)
    uninterrupted_opt = _optimizer(uninterrupted)
    _train(uninterrupted_opt, uninterrupted, grads)

    first = fresh_params(inits)
    first_opt = _optimizer(first)
    _train(first_opt, first, grads[:3])
    saved_state = copy.deepcopy(first_opt.state_dict())

    bystander = fresh_params(inits)
    _train(_optimizer(bystander), bystander, grads[:2])

    resumed = fresh_params(first)
    resumed_opt = _optimizer(resumed)
    _init_optim_state(resumed_opt)
    assert all(torch.equal(p, s) for p, s in zip(resumed, first, strict=True)), "the zero-LR step moved a weight"
    resumed_opt.load_state_dict(saved_state)
    _train(resumed_opt, resumed, grads[3:])

    for index, (a, b) in enumerate(zip(uninterrupted, resumed, strict=True)):
        assert torch.equal(a, b), f"param {index} rounded differently after the restore"
        assert torch.equal(uninterrupted_opt.state[a]["exp_avg_sq"], resumed_opt.state[b]["exp_avg_sq"])


def test_muon_seeds_are_keyed_by_step_and_name():
    """Muon's matrix step keys each seed by the param's step count and its name: a rank whose param
    lacks a grad still counts that param's step, so the surviving params' seeds match the all-grads
    rank's, a later step draws new ones, and a param's seed follows its name, not its place."""

    def params(grad_flags):
        out = []
        for has_grad in grad_flags:
            p = torch.nn.Parameter(torch.zeros(4, 4, dtype=torch.bfloat16))
            p.grad = torch.zeros_like(p) if has_grad else None
            out.append(p)
        return out

    all_grads = params([True, True, True])
    state_all: dict = {p: {} for p in all_grads}
    names = ["w0", "w1", "w2"]
    with_grad_all, seeds_all = muon_mod._collect_params_with_sr_seeds(state_all, all_grads, names)
    assert len(with_grad_all) == 3 and len(seeds_all) == 3

    skipped = params([True, False, True])
    state_skip: dict = {p: {} for p in skipped}
    with_grad_skip, seeds_skip = muon_mod._collect_params_with_sr_seeds(state_skip, skipped, names)
    assert len(with_grad_skip) == 2
    assert seeds_skip == [seeds_all[0], seeds_all[2]], "a grad-None param shifted its neighbours' seeds"
    assert [state_skip[p]["step"] for p in skipped] == [1, 1, 1], "the grad-None param did not count the step"
    assert "momentum" not in state_skip[skipped[1]], "a param with no grad was given a momentum buffer"

    _, seeds_next = muon_mod._collect_params_with_sr_seeds(state_all, all_grads, names)
    assert not set(seeds_next) & set(seeds_all), "the next step replayed a seed"
    # The name keys the seed: the same names in another order draw the same seeds, reordered.
    _, seeds_reordered = muon_mod._collect_params_with_sr_seeds({p: {} for p in all_grads}, all_grads, names[::-1])
    assert seeds_reordered == seeds_all[::-1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
