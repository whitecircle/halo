#!/usr/bin/env python
"""``tests.common.first_step`` grades a recorded first step the way the SMPO CP GPU rows rely on.

Those rows run only on the GPU tiers, so a check that stopped failing there would go unnoticed. These
pins drive :func:`first_step_checks` and :func:`first_step_gradient_checks` with each miscount they exist
to catch and assert that exactly its own verdict flips, drive :func:`recording_autograd_fallbacks`
with a real backward through an in-place gloo all-reduce, and drive :func:`parameter_gradient_checks` (the
CP roster's gradient check) with a near-tie routing flip it must absorb and the corruptions it must catch.

Run: python tests/cpu/conventions/test_first_step_checks.py
"""

import warnings

import pytest
import torch
import torch.distributed as dist

from halo.models.moe.config_fields import has_routed_experts
from tests.common.equivalence.cp_equivalence import FAMILIES as CP_FAMILIES
from tests.common.equivalence.cp_equivalence import tiny_model as cp_tiny_model
from tests.common.first_step import (
    AUTOGRAD_FALLBACK_WARNING,
    ROUTING_DEPENDENT_PARAM,
    FirstStep,
    first_step_checks,
    first_step_gradient_checks,
    parameter_gradient_checks,
    recording_autograd_fallbacks,
)
from tests.common.gloo import run_gloo_ranks
from tests.common.numerics import cos_sim
from tests.common.tolerances import TOL

CP_SIZE = 2
LOSS_RTOL = 0.1
LOSSES = [1.7, 2.3]


def _grads() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(0)
    return {name: torch.randn(4, 3, generator=generator) for name in ("embed.weight", "q_proj.weight", "norm.weight")}


def _step(losses=LOSSES, grads=None, logged_loss=None, autograd_fallbacks=()) -> FirstStep:
    return FirstStep(
        losses=list(losses),
        grads=_grads() if grads is None else grads,
        logged_loss=sum(losses) / len(losses) if logged_loss is None else logged_loss,
        autograd_fallbacks=list(autograd_fallbacks),
    )


def _failed(record: FirstStep, reference: FirstStep | None = None) -> list[str]:
    reference = reference or _step()
    checks, _ = first_step_checks(record, reference, miscount_factor=CP_SIZE, loss_rtol=LOSS_RTOL)
    grad_checks, _ = first_step_gradient_checks(record, reference)
    return [name for name, ok in (checks | grad_checks).items() if not ok]


def test_a_matching_step_passes_every_check():
    assert _failed(_step()) == []


def test_losses_within_the_relative_bound_pass():
    assert _failed(_step(losses=[(1 + LOSS_RTOL / 2) * loss for loss in LOSSES])) == []


def test_a_loss_counted_once_per_cp_rank_fails_both_loss_checks():
    assert _failed(_step(losses=[CP_SIZE * loss for loss in LOSSES])) == [
        "step1_losses_match_reference",
        "logged_step1_loss_is_the_reference_loss",
    ]


def test_a_logged_loss_off_the_trained_one_fails_only_the_logged_check():
    true_loss = sum(LOSSES) / len(LOSSES)
    assert _failed(_step(logged_loss=CP_SIZE * true_loss)) == ["logged_step1_loss_is_the_reference_loss"]


@pytest.mark.parametrize("scale", [CP_SIZE, 1 / CP_SIZE])
def test_a_gradient_scaled_by_the_cp_size_fails_only_the_norm_check(scale):
    grads = {name: scale * grad for name, grad in _grads().items()}
    assert _failed(_step(grads=grads)) == ["step1_grad_norm_matches_reference"]


def test_a_reoriented_gradient_fails_only_the_direction_check():
    grads = _grads()
    grads["q_proj.weight"] = grads["q_proj.weight"].flip(0)
    assert _failed(_step(grads=grads)) == ["step1_grad_direction_matches_reference"]


def test_a_parameter_left_out_of_the_step_fails_every_gradient_check():
    grads = _grads()
    del grads["norm.weight"]
    assert _failed(_step(grads=grads)) == [
        "step1_grads_cover_the_reference_params",
        "step1_grad_direction_matches_reference",
        "step1_grad_norm_matches_reference",
    ]


def test_a_backward_through_the_c10d_fallback_fails_its_check():
    assert _failed(_step(autograd_fallbacks=[AUTOGRAD_FALLBACK_WARNING])) == [
        "backward_never_took_the_c10d_autograd_fallback"
    ]


def test_a_loss_too_small_to_show_a_miscount_fails_the_resolution_precondition():
    tiny = [1e-3, 2e-3]
    assert _failed(_step(losses=tiny), _step(losses=tiny)) == ["loss_bound_resolves_a_miscounted_loss"]


def _moe_grads() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    shapes = {
        "model.embed_tokens.weight": (32, 8),
        "model.layers.0.self_attn.q_proj.weight": (8, 8),
        "model.layers.0.mlp.router.weight": (4, 8),
        "model.layers.0.mlp.router.bias": (4,),
        "model.layers.0.mlp.experts.gate_up_proj": (4, 8, 16),
        "model.layers.0.mlp.experts.gate_up_proj_bias": (4, 16),
        "model.layers.1.mlp.gate.weight": (4, 8),
        "model.layers.1.mlp.experts.down_proj": (4, 8, 8),
        "model.layers.1.mlp.shared_expert_gate.weight": (1, 8),
    }
    grads = {name: torch.randn(*shape, generator=generator) for name, shape in shapes.items()}
    # As in the roster's tiny MoE models, the routers carry about 1% of the routing group's squared norm.
    for router in (
        "model.layers.0.mlp.router.weight",
        "model.layers.0.mlp.router.bias",
        "model.layers.1.mlp.gate.weight",
    ):
        grads[router] *= 0.1
    grads["model.layers.0.mlp.experts.gate_up_proj_bias"] *= 0.01
    return grads


def _flip_one_token(grad: torch.Tensor, weight: float) -> torch.Tensor:
    """``grad`` plus a rank-one term ``weight`` times its norm, the shape a re-routed token adds to a gradient."""
    generator = torch.Generator().manual_seed(2)
    token = torch.outer(
        torch.randn(grad.shape[0], generator=generator), torch.randn(grad.shape[1], generator=generator)
    )
    return grad + weight * grad.norm() / token.norm() * token


def _with(name: str, grad: torch.Tensor, grads: dict[str, torch.Tensor] | None = None) -> dict[str, torch.Tensor]:
    return {**(_moe_grads() if grads is None else grads), name: grad}


def _cosine(got: dict[str, torch.Tensor], want: dict[str, torch.Tensor], name: str) -> float:
    return cos_sim(got[name], want[name], label=name)


@pytest.mark.parametrize("family", sorted(CP_FAMILIES))
def test_routing_dependent_params_are_each_roster_familys_routers_and_routed_experts(family):
    model = cp_tiny_model(family, "cpu", torch.device("cpu"))
    names = [name for name, _ in model.named_parameters()]
    routed = [name for name in names if ROUTING_DEPENDENT_PARAM.search(name)]
    assert all(".mlp.router." in n or ".mlp.gate." in n or ".mlp.experts." in n for n in routed), routed
    assert {n for n in names if ".mlp.router." in n or ".mlp.gate." in n} <= set(routed)
    assert not [n for n in routed if "shared_expert" in n]
    if has_routed_experts(model.config):
        assert any(".mlp.experts." in n for n in routed) and any(".experts." not in n for n in routed), routed
    else:
        assert routed == []


def test_a_matching_backward_passes_the_parameter_check():
    assert parameter_gradient_checks(_moe_grads(), _moe_grads())[0]


def test_one_re_routed_token_on_a_router_passes_as_one_routing_vector():
    want = _moe_grads()
    router = "model.layers.0.mlp.router.weight"
    got = _with(router, _flip_one_token(want[router], 0.2))
    assert TOL.grad_direction_cosine_min < _cosine(got, want, router) < TOL.cp_grad_cosine_min
    matches, metrics = parameter_gradient_checks(got, want)
    assert matches, metrics
    assert metrics["grad_routed_cosine"] >= TOL.cp_grad_cosine_min


@pytest.mark.parametrize("router", ["model.layers.0.mlp.router.weight", "model.layers.1.mlp.gate.weight"])
@pytest.mark.parametrize("broken", ["zero", "negated"])
def test_a_router_without_its_gradient_fails_the_parameter_check(router, broken):
    want = _moe_grads()
    got = _with(router, torch.zeros_like(want[router]) if broken == "zero" else -want[router])
    matches, metrics = parameter_gradient_checks(got, want)
    assert not matches
    assert metrics["grad_routed_cosine"] >= TOL.cp_grad_cosine_min, "the vector alone must not be what fails it"


def test_an_expert_no_token_reached_on_one_side_is_skipped():
    bias = "model.layers.0.mlp.experts.gate_up_proj_bias"
    want = _moe_grads()
    assert parameter_gradient_checks(_with(bias, torch.zeros_like(want[bias])), want)[0]


def _dense(grads: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: grad for name, grad in grads.items() if not ROUTING_DEPENDENT_PARAM.search(name)}


@pytest.mark.parametrize("routed", [True, False])
def test_downstream_noise_on_a_dense_parameter_passes_only_in_a_routed_model(routed):
    q_proj = "model.layers.0.self_attn.q_proj.weight"
    want = _moe_grads() if routed else _dense(_moe_grads())
    got = _with(q_proj, _flip_one_token(want[q_proj], 0.16), want)
    assert TOL.cp_routed_grad_cosine_min < _cosine(got, want, q_proj) < TOL.cp_grad_cosine_min
    assert parameter_gradient_checks(got, want)[0] is routed


@pytest.mark.parametrize(("weight", "whole_gradient_holds"), [(0.3, True), (1.0, False)])
def test_a_reoriented_dense_parameter_fails_in_a_routed_model(weight, whole_gradient_holds):
    q_proj = "model.layers.0.self_attn.q_proj.weight"
    want = _moe_grads()
    got = _with(q_proj, _flip_one_token(want[q_proj], weight))
    assert _cosine(got, want, q_proj) < TOL.cp_routed_grad_cosine_min
    matches, metrics = parameter_gradient_checks(got, want)
    assert not matches
    assert (metrics["grad_whole_cosine"] >= TOL.cp_grad_cosine_min) is whole_gradient_holds


def test_a_reoriented_whole_gradient_fails_the_parameter_check():
    want = _moe_grads()
    got = {name: _flip_one_token(grad, 0.3) if grad.dim() == 2 else grad for name, grad in want.items()}
    matches, metrics = parameter_gradient_checks(got, want)
    assert not matches
    assert metrics["grad_whole_cosine"] < TOL.cp_grad_cosine_min


def test_a_reoriented_expert_bank_fails_the_parameter_check():
    bank = "model.layers.1.mlp.experts.down_proj"
    want = _moe_grads()
    matches, metrics = parameter_gradient_checks(_with(bank, want[bank].flip(0)), want)
    assert not matches
    assert metrics["grad_routed_cosine"] < TOL.cp_grad_cosine_min


@pytest.mark.parametrize("scale", [CP_SIZE, 1 / CP_SIZE])
def test_a_gradient_scaled_by_the_cp_size_fails_the_parameter_check(scale):
    want = _moe_grads()
    matches, metrics = parameter_gradient_checks({name: scale * grad for name, grad in want.items()}, want)
    assert not matches
    assert metrics["grad_min_cosine"] == pytest.approx(1.0)


def test_a_non_finite_gradient_fails_the_parameter_check():
    bias = "model.layers.0.mlp.router.bias"
    want = _moe_grads()
    assert not parameter_gradient_checks(_with(bias, torch.full_like(want[bias], float("nan"))), want)[0]


def _backward_through_an_in_place_all_reduce(rank: int) -> None:
    del rank
    fallbacks: list[str] = []
    # The recorder inside, so a warning it passes on reaches the outer log.
    with warnings.catch_warnings(record=True) as shown, recording_autograd_fallbacks(fallbacks):
        warnings.simplefilter("always")
        partial = torch.ones(3, requires_grad=True) * 2.0
        dist.all_reduce(partial)
        partial.sum().backward()
        warnings.warn("an unrelated warning", UserWarning, stacklevel=1)
    assert fallbacks, "the in-place all-reduce's backward raised no autograd-fallback warning"
    assert all(AUTOGRAD_FALLBACK_WARNING in message for message in fallbacks)
    assert [str(w.message) for w in shown] == ["an unrelated warning"], "other warnings must still show"


def test_the_recorder_catches_a_backward_through_an_in_place_collective():
    run_gloo_ranks(_backward_through_an_in_place_all_reduce, 1)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
