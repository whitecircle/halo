#!/usr/bin/env python
"""AdamWBF16 stochastic rounding, eager path: the numerical properties that justify SR over nearest rounding.
SR on ``exp_avg_sq`` removes the systematic bias nearest rounding leaves near the bf16 underflow floor."""

import copy
import math

import pytest
import torch
from torch.distributed.checkpoint.state_dict import _init_optim_state

import halo.optimizers.adamw_bf16 as adamw_mod
import halo.optimizers.muon as muon_mod
from halo.optimizers.adamw_bf16 import AdamWBF16, _eager_adam_bf16_step
from halo.optimizers.stochastic_rounding import sr_seed_pair, stochastic_round_to_bf16


def _bf16_ulp(x: float) -> float:
    xb = torch.tensor(x, dtype=torch.bfloat16)
    nxt = torch.nextafter(xb, torch.tensor(float("inf"), dtype=torch.bfloat16))
    return float((nxt.float() - xb.float()).abs())


def test_sr_unbiased_between_grid_points():
    """At N=20000 the standard error is ~ulp/(2*sqrt(N)), so a 0.3-ulp constant bias sits ~50 sigma away."""
    base = torch.tensor(1.0, dtype=torch.bfloat16).float()
    ulp = _bf16_ulp(1.0)
    x = float(base) + 0.3 * ulp
    assert torch.tensor(x, dtype=torch.bfloat16).float().item() == float(base), (
        "test setup: x must round-to-nearest DOWN to base (so a bias is detectable)"
    )

    # One SR call over an [n] tensor draws n independent roundings, without torch's per-op thread-pool overhead 20000x
    # (minutes on a many-core host).
    n = 20000
    samples = stochastic_round_to_bf16(torch.full((n,), x, dtype=torch.float32), seed=0).float()

    uniq = torch.unique(samples)
    assert uniq.numel() <= 2, f"SR produced non-bracketing values: {uniq}"
    assert float(samples.min()) == float(base)
    assert abs(float(samples.max()) - (float(base) + ulp)) < 1e-6

    mean = float(samples.mean())
    tol = 4.0 * ulp / math.sqrt(n)  # ~4 sigma.
    assert abs(mean - x) < tol, f"SR mean {mean} not within {tol} of {x} (bias detected)"
    # A nearest-rounded estimate would equal `base`.
    assert mean - float(base) > 0.15 * ulp, "SR did not lift the mean above the nearest-round value"


def test_sr_exact_on_grid_point():
    x = torch.tensor(2.0, dtype=torch.bfloat16).float().item()
    rs = stochastic_round_to_bf16(torch.full((2000,), x, dtype=torch.float32), seed=0).float()
    assert torch.all(rs == x), f"on-grid value {x} rounded off-grid: {torch.unique(rs).tolist()}"


def _fresh_state(p):
    return {
        "exp_avg": torch.zeros_like(p),
        "exp_avg_sq": torch.zeros_like(p),
    }


def _run_eager_step(seed, p0, grad):
    """Returns (param, exp_avg, exp_avg_sq) clones after the step."""
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
    """A rerun under the same seed reproduces them, so the difference is the seed's noise."""
    torch.manual_seed(1234)
    p0 = (torch.randn(4096, dtype=torch.float32) * 0.02).to(torch.bfloat16)
    # Small, near-constant gradient so easq lands near the bf16 underflow floor where SR matters.
    grad = torch.full((4096,), 3e-4, dtype=torch.bfloat16)

    p_a, ea_a, easq_a = _run_eager_step(11, p0, grad)
    p_b, ea_b, easq_b = _run_eager_step(99, p0, grad)
    p_a2, _, easq_a2 = _run_eager_step(11, p0, grad)

    assert torch.equal(ea_a, ea_b), "exp_avg must be bit-identical (nearest rounding, no SR)"

    assert not torch.equal(p_a, p_b), "weight write must carry SR noise (differs across RNG seeds)"
    assert not torch.equal(easq_a, easq_b), "exp_avg_sq write must carry SR noise (differs across seeds)"
    assert torch.equal(p_a, p_a2) and torch.equal(easq_a, easq_a2), "the SR seed must fully determine the noise"


def _adam_easq_reference(grad_val, n_steps, beta2):
    """fp32-accumulated, then stored as nearest-rounded bf16 each step (the biased path)."""
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
    """Asserts the magnitude of the nearest bias and SR's removal of it; the sign depends on the gradient regime."""
    beta2 = 0.999
    grad_val = 1e-4
    n_steps = 500

    size = 8192
    g = torch.full((size,), grad_val, dtype=torch.float32)
    easq_fp32 = torch.zeros(size, dtype=torch.float32)
    for _ in range(n_steps):
        easq_fp32.mul_(beta2).addcmul_(g, g, value=1.0 - beta2)
    fp32_mean = float(easq_fp32.mean())
    assert fp32_mean > 0.0

    nearest_mean = _adam_easq_reference(grad_val, n_steps, beta2)

    # Each step's noise must be fresh for the average to come out unbiased.
    p = torch.nn.Parameter(torch.zeros(size, dtype=torch.bfloat16))
    opt = AdamWBF16([("p", p)], lr=1e-3, betas=(0.9, beta2), weight_decay=0.0, use_triton=False)
    for _ in range(n_steps):
        p.grad = torch.full((size,), grad_val, dtype=torch.bfloat16)
        opt.step()
    sr_mean = float(opt.state[p]["exp_avg_sq"].float().mean())

    # Sign depends on regime.
    nearest_bias = abs(nearest_mean - fp32_mean) / fp32_mean
    assert nearest_bias > 0.15, (
        f"test premise broken: nearest-rounded easq should be biased from fp32 by a large "
        f"margin, got {nearest_bias:.1%} (nearest={nearest_mean:.3e}, fp32={fp32_mean:.3e})"
    )

    sr_rel_err = abs(sr_mean - fp32_mean) / fp32_mean
    assert sr_rel_err < 0.05, (
        f"AdamWBF16 (SR) easq mean should track fp32, got rel-err {sr_rel_err:.1%} "
        f"(sr={sr_mean:.3e}, fp32={fp32_mean:.3e})"
    )

    # Negative control: SR is strictly closer to fp32 than the nearest-rounded reference.
    assert abs(sr_mean - fp32_mean) < abs(sr_mean - nearest_mean), (
        f"SR easq ({sr_mean:.3e}) must be closer to fp32 ({fp32_mean:.3e}) than to the "
        f"nearest-rounded reference ({nearest_mean:.3e})"
    )


def test_seed_pairs_differ_across_steps_and_params():
    """Ignoring the step replays one step's noise on the next and biases the rounding; ignoring the param shares it
    across params."""
    key = adamw_mod._SR_KEY
    seeds = [sr_seed_pair(key, step, index) for step in range(1, 21) for index in range(10)]
    assert len({pair[0] for pair in seeds}) == len(seeds), "two (step, param) pairs share a kernel seed"
    assert all(pair[0] != pair[1] for pair in seeds), "the eager step's two SR writes share a seed"
    assert all(0 <= seed < 2**30 for pair in seeds for seed in pair)
    assert sr_seed_pair(key, 7, 3) == sr_seed_pair(key, 7, 3)
    assert sr_seed_pair(key, 7, 3) != sr_seed_pair(muon_mod._SR_KEY, 7, 3), "the keys do not separate the streams"


def test_adamw_missing_grad_leaves_the_replicated_params_rounding_alone():
    """p1's seed is keyed by its own name and step, which a grad-None p0 ahead of it does not shift."""
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
    """An expert-TP rank can hold a bias its partners lack, so a param ahead must not shift its noise."""
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
    """A rebuilt optimizer materializes state with torch's zero-LR ``_init_optim_state`` step, then loads, and must
    still match bit for bit."""
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
    """A rank whose param lacks a grad still counts that param's step, so surviving seeds match the all-grads
    rank's."""

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
    with_grad_all, seeds_all = muon_mod._advance_steps_and_collect_grads(state_all, all_grads, names)
    assert len(with_grad_all) == 3 and len(seeds_all) == 3

    skipped = params([True, False, True])
    state_skip: dict = {p: {} for p in skipped}
    with_grad_skip, seeds_skip = muon_mod._advance_steps_and_collect_grads(state_skip, skipped, names)
    assert len(with_grad_skip) == 2
    assert seeds_skip == [seeds_all[0], seeds_all[2]], "a grad-None param shifted its neighbours' seeds"
    assert [state_skip[p]["step"] for p in skipped] == [1, 1, 1], "the grad-None param did not count the step"
    assert "momentum" not in state_skip[skipped[1]], "a param with no grad was given a momentum buffer"

    _, seeds_next = muon_mod._advance_steps_and_collect_grads(state_all, all_grads, names)
    assert not set(seeds_next) & set(seeds_all), "the next step replayed a seed"
    _, seeds_reordered = muon_mod._advance_steps_and_collect_grads({p: {} for p in all_grads}, all_grads, names[::-1])
    assert seeds_reordered == seeds_all[::-1]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
