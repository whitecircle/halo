#!/usr/bin/env python
"""``score_ep_grad_pairs``, ``pair_step_grads`` and ``routing_pinned`` fail each way an EP gradient or a replayed
route can disagree with its reference. Their GPU suites run only on the GPU tiers, so each failure mode is pinned
here."""

import contextlib
import math

import pytest
import torch
import torch.nn.functional as F

from halo.distributed.expert_parallel.parallelize import apply_ep_layers
from tests.common.ep_reference import (
    KIND_SCORED_CHECK,
    MISSING_GRAD_SCORE,
    best_resolved_scores,
    captured_picks,
    decoder_layer_index,
    ep_layers,
    fault_passes_bf16_noise_scoring,
    groups_never_scored,
    pair_step_grads,
    parameter_kind,
    parameters_outside_the_bars,
    per_layer_grad_floors,
    random_token_batch,
    router_parameter_names,
    routing_can_be_pinned,
    routing_pinned,
    score_against_bf16_noise,
    score_ep_grad_pairs,
    scores_any,
)
from tests.common.parallelism import single_process_ep_config
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model
from tests.common.tolerances import TOL

LOW, HIGH = TOL.ep_grad_norm_ratio_band
REFERENCE = torch.randn(4, 16, 8, generator=torch.Generator().manual_seed(0)).to(torch.bfloat16)
A_LOG = torch.randn(4, generator=torch.Generator().manual_seed(1))
# Two decoder layers shaped like the tiny Qwen3.5's: a projection gradient and a per-head GatedDeltaNet A_log one,
# resolvable on layer 0 (~1e-4 of the layer's largest) and negligible on layer 6, whose heads all decay hard (~1e-9).
L0_A_LOG, L6_A_LOG = "model.layers.0.linear_attn.A_log", "model.layers.6.linear_attn.A_log"
L0_PROJ, L6_PROJ = "model.layers.0.mlp.down_proj.weight", "model.layers.6.mlp.down_proj.weight"
LAYERED = {L0_PROJ: REFERENCE.float(), L0_A_LOG: A_LOG * 1e-3, L6_PROJ: REFERENCE.float() / 2, L6_A_LOG: A_LOG * 1e-8}
# DeepSeek-V4-shaped gradients over three steps at the tiny model's scales: per layer a routed expert bank (its layer's
# largest), a router at 4e-2 of it, a 3-element hc scale and a 4-element dt_bias; outside the layers an lm_head and the
# head's 2- and 1-element hc vectors. The fp32 reference routes some tokens differently from the bf16 one, so routers
# and banks sit 25% of their norm from it, while the policy stays 2% from bf16. Layer 1's hc scale cancels to 2.6e-5 on
# step 1 and the head base to 5e-6 on step 0, under bf16 rounding (2e-5 at most); the one-element head scale rounds by
# 12-44% of itself, the dt_bias by a quarter. Every other gradient stands 10x above its rounding.
L0_EXPERTS, L1_EXPERTS = "model.layers.0.mlp.experts.down_proj", "model.layers.1.mlp.experts.down_proj"
L0_GATE, L1_GATE = "model.layers.0.mlp.gate.weight", "model.layers.1.mlp.gate.weight"
HC0, HC1 = "model.layers.0.attn_hc.scale", "model.layers.1.attn_hc.scale"
DT0, DT1 = "model.layers.0.linear_attn.dt_bias", "model.layers.1.linear_attn.dt_bias"
HEAD, HC_BASE, HC_SCALE = "lm_head.weight", "model.hc_head.hc_base", "model.hc_head.hc_scale"
HC_STEPS = 3
CANCELLED = {(1, HC1), (0, HC_BASE)}
# The largest per layer differ, so a per-layer fraction differs from one of the step's largest.
RESOLVED = {
    L0_EXPERTS: ((4, 8, 6), 1.0),
    L1_EXPERTS: ((4, 8, 6), 0.5),
    L0_GATE: ((8, 16), 4e-2),
    L1_GATE: ((8, 16), 2e-2),
    HEAD: ((16, 8), 2.0),
    DT0: ((4,), 2e-4),
    DT1: ((4,), 2e-4),
}
# Each resolved gradient's bf16 distance from fp32 and the policy's from the bf16 reference, as shares of its norm.
BF16_SHARE = {L0_EXPERTS: 0.25, L1_EXPERTS: 0.25, L0_GATE: 0.25, L1_GATE: 0.25, HEAD: 1e-2, DT0: 0.25, DT1: 0.25}
POLICY_SHARE = {L0_EXPERTS: 2e-2, L1_EXPERTS: 2e-2, L0_GATE: 2e-2, L1_GATE: 2e-2, HEAD: 5e-3, DT0: 5e-2, DT1: 5e-2}


def _vector(shape: tuple[int, ...], norm: float, seed: int) -> torch.Tensor:
    vector = torch.randn(shape, generator=torch.Generator().manual_seed(seed))
    return vector * (norm / vector.norm())


def _resolved(share: dict[str, float], seed: int) -> dict[tuple[int, str], torch.Tensor]:
    return {
        (step, name): _vector(shape, share.get(name, 1.0) * norm, seed + 10 * index + step)
        for index, (name, (shape, norm)) in enumerate(RESOLVED.items())
        for step in range(HC_STEPS)
    }


# Per (step, name): the fp32 reference's gradient, and the bf16 reference's and the policy's rounding errors from it.
HC_FP32 = {
    **_resolved({}, 200),
    **{(step, HC0): _vector((3,), 6e-4, 10 + step) for step in range(HC_STEPS)},
    **{(step, HC1): _vector((3,), 3e-4, 20 + step) for step in (0, 2)},
    (1, HC1): torch.tensor([2.0e-5, 1.5e-5, 0.8e-5]),
    (0, HC_BASE): torch.tensor([4e-6, -3e-6]),
    **{(step, HC_BASE): _vector((2,), 4e-4, 30 + step) for step in (1, 2)},
    **{(step, HC_SCALE): torch.tensor([value]) for step, value in enumerate((3.4e-5, 4.1e-5, 9.1e-5))},
}
BF16_ERROR = {
    **_resolved(BF16_SHARE, 300),
    **{(step, HC0): _vector((3,), 1e-5, 110 + step) for step in range(HC_STEPS)},
    (0, HC1): _vector((3,), 2e-5, 120),
    (1, HC1): torch.tensor([0.2e-5, -0.2e-5, 0.1e-5]),
    (2, HC1): _vector((3,), 1e-5, 122),
    (0, HC_BASE): torch.tensor([1.2e-5, 1.1e-5]),
    **{(step, HC_BASE): _vector((2,), 2e-5, 130 + step) for step in (1, 2)},
    **{(step, HC_SCALE): torch.tensor([value]) for step, value in enumerate((0.98e-5, 1.8e-5, 1.05e-5))},
}
POLICY_ERROR = {
    **{
        key: _vector(tuple(error.shape), error.norm().item(), 1000 + index)
        for index, (key, error) in enumerate(BF16_ERROR.items())
    },
    **{key: BF16_ERROR[key] + shift for key, shift in _resolved(POLICY_SHARE, 400).items()},
    (1, HC1): torch.tensor([-2.0e-5, 1.0e-5, 2.2e-5]),
    (0, HC_BASE): torch.tensor([-2e-6, 5e-6]),
    **{(step, HC_SCALE): BF16_ERROR[step, HC_SCALE] for step in range(HC_STEPS)},
}


def _score(got):
    checks, metrics = {}, {}
    score_ep_grad_pairs({"l0_gate_up_grad": (got, REFERENCE)}, checks, metrics, cos_min=TOL.ep_grad_cosine_min)
    return checks["l0_gate_up_grad_matches"], metrics


def test_a_matching_gradient_passes():
    matches, metrics = _score(REFERENCE.clone())
    assert matches
    assert metrics["l0_gate_up_grad_cos"] == pytest.approx(1.0) and metrics["l0_gate_up_grad_norm_ratio"] == 1.0


@pytest.mark.parametrize("got", [None, REFERENCE[:2]], ids=["missing", "misshapen"])
def test_a_missing_or_misshapen_gradient_fails_by_name(got):
    matches, metrics = _score(got)
    assert not matches
    assert metrics["l0_gate_up_grad_cos"] == metrics["l0_gate_up_grad_norm_ratio"] == MISSING_GRAD_SCORE


@pytest.mark.parametrize("scale", [2.0, 0.5], ids=["doubled", "halved"])
def test_a_rescaled_gradient_fails_on_the_norm_ratio(scale):
    matches, metrics = _score(REFERENCE * scale)
    assert not matches
    assert metrics["l0_gate_up_grad_cos"] == pytest.approx(1.0)


def test_a_reoriented_gradient_fails_on_the_cosine():
    matches, metrics = _score(REFERENCE.roll(1, dims=0))
    assert not matches
    assert metrics["l0_gate_up_grad_norm_ratio"] == pytest.approx(1.0)


def _score_layered(got: dict[str, torch.Tensor | None], references: dict[str, torch.Tensor] = LAYERED):
    """Score ``references`` under their per-layer floors, each gradient the reference's own unless ``got`` names it."""
    pairs = {name: (got[name] if name in got else ref.clone(), ref) for name, ref in references.items()}
    checks, metrics = {}, {}
    floors = per_layer_grad_floors(pairs, TOL.grad_negligible_layer_fraction)
    score_ep_grad_pairs(pairs, checks, metrics, cos_min=TOL.ep_grad_cosine_min, negligible_floors=floors)
    return checks, metrics, floors


def test_the_floor_is_the_fraction_of_the_largest_gradient_in_the_layer():
    _, _, floors = _score_layered({})
    fraction = TOL.grad_negligible_layer_fraction
    assert floors[L0_A_LOG] == floors[L0_PROJ] == pytest.approx(fraction * LAYERED[L0_PROJ].norm().item())
    assert floors[L6_A_LOG] == floors[L6_PROJ] == pytest.approx(fraction * LAYERED[L6_PROJ].norm().item())
    shrunk = {**LAYERED, L6_PROJ: LAYERED[L6_PROJ] * 1e-6, L6_A_LOG: LAYERED[L6_A_LOG] * 1e-6}
    checks, metrics, _ = _score_layered({}, shrunk)
    assert f"{L6_PROJ}_cos" in metrics and f"{L6_A_LOG}_negligible_abs_error" in metrics
    assert all(checks.values()), checks


def test_a_negligible_reference_is_held_to_absolute_error_not_its_direction():
    """Rounding noise as large as the reference itself can reverse it (cosine -1 here) and sit far under the floor."""
    checks, metrics, floors = _score_layered({L6_A_LOG: -LAYERED[L6_A_LOG]})
    assert all(checks.values()), checks
    assert f"{L6_A_LOG}_cos" not in metrics and metrics[f"{L6_A_LOG}_negligible_abs_error"] < floors[L6_A_LOG]
    assert f"{L0_A_LOG}_cos" in metrics, "premise: the resolvable layer's A_log is scored by direction and norm"


def test_a_negligible_reference_fails_an_error_past_its_floor():
    _, _, floors = _score_layered({})
    checks, metrics, _ = _score_layered({L6_A_LOG: LAYERED[L6_A_LOG] + 2 * floors[L6_A_LOG]})
    assert not checks[f"{L6_A_LOG}_matches"], metrics


@pytest.mark.parametrize("scale", [0.5, 2.0], ids=["dropped_reduce", "doubled_reduce"])
def test_a_mis_reduced_per_head_gradient_fails_on_its_resolvable_layer(scale):
    """A TP sync that drops one rank's A_log or sums it twice: the negligible layer cannot tell, the resolvable one
    fails on its norm ratio."""
    checks, metrics, _ = _score_layered({L0_A_LOG: LAYERED[L0_A_LOG] * scale, L6_A_LOG: LAYERED[L6_A_LOG] * scale})
    assert checks[f"{L6_A_LOG}_matches"] and not checks[f"{L0_A_LOG}_matches"], checks
    assert metrics[f"{L0_A_LOG}_norm_ratio"] == pytest.approx(scale)


def test_a_kind_wholly_under_the_floor_fails():
    """A parameter kind negligible on every layer is scored nowhere, so a per-kind fault would hide behind the floor."""
    checks, _, _ = _score_layered({}, {**LAYERED, L0_A_LOG: A_LOG * 1e-9})
    assert all(checks[f"{name}_matches"] for name in LAYERED), checks
    assert not checks[KIND_SCORED_CHECK]
    checks, _, _ = _score_layered({})
    assert checks[KIND_SCORED_CHECK]


def test_a_missing_gradient_under_the_floor_still_fails_by_name():
    checks, metrics, _ = _score_layered({L6_A_LOG: None})
    assert not checks[f"{L6_A_LOG}_matches"]
    assert metrics[f"{L6_A_LOG}_cos"] == metrics[f"{L6_A_LOG}_norm_ratio"] == MISSING_GRAD_SCORE


def _hc_steps(got: dict | None = None, fp32: dict = HC_FP32) -> list[dict]:
    """The three steps' ``(gradient, bf16 reference, fp32 reference)`` pairs, each gradient the fp32 one plus the
    policy's rounding unless ``got`` names it."""
    got = got or {}
    return [
        {
            name: (
                got[step, name] if (step, name) in got else fp32[step, name] + POLICY_ERROR[step, name],
                fp32[step, name] + BF16_ERROR[step, name],
                fp32[step, name],
            )
            for (at, name) in fp32
            if at == step
        }
        for step in range(HC_STEPS)
    ]


def _scored(steps: list[dict]) -> list[dict]:
    return score_against_bf16_noise(steps, cos_min=TOL.ep_grad_cosine_min)


def _scaled(name: str, scale: float, steps: tuple[int, ...] = tuple(range(HC_STEPS))) -> list[dict]:
    clean = _hc_steps()
    return _hc_steps({(step, name): scale * clean[step][name][0] for step in steps})


def test_bf16_noise_excuses_only_the_cancelled_pairs():
    """The cancelled pairs land outside the bars and within their noise of the fp32 gradient; every other pair is held
    to the bars, and each parameter lands inside them at some step."""
    scored = _scored(_hc_steps())
    excused = {(step, name) for step, scores in enumerate(scored) for name, score in scores.items() if score.excused}
    assert excused == CANCELLED
    assert all(
        score.in_bars
        for step, scores in enumerate(scored)
        for name, score in scores.items()
        if (step, name) not in CANCELLED
    )
    assert not parameters_outside_the_bars(scored)
    for step, name in CANCELLED:
        score = scored[step][name]
        assert not score.in_bars and score.fp32_error <= TOL.bf16_noise_excuse_factor * score.noise


def test_the_excusable_fraction_is_the_bf16_reference_over_its_layers_largest():
    """Per decoder layer, the parameters outside the stack one more layer: layer 1's bank and the lm_head set the
    fractions of its hc scale and of the head's hc base."""
    steps, scored = _hc_steps(), _scored(_hc_steps())
    for name, largest in ((HC1, L1_EXPERTS), (HC_BASE, HEAD)):
        _, bf16, _ = steps[1][name]
        assert scored[1][name].layer_fraction == pytest.approx(bf16.norm().item() / steps[1][largest][1].norm().item())


def test_the_noise_is_the_largest_over_the_steps():
    """Layer 1's scale rounds by 3e-6 on its cancelled step and 2e-5 on step 0: that step's draw alone would not cover
    the policy's 3.1e-5."""
    score = _scored(_hc_steps())[1][HC1]
    assert score.noise == pytest.approx(BF16_ERROR[0, HC1].norm().item())
    assert score.fp32_error > TOL.bf16_noise_excuse_factor * BF16_ERROR[1, HC1].norm().item()


def test_a_gradient_past_its_noise_factor_fails():
    noise = _scored(_hc_steps())[1][HC1].noise
    error = POLICY_ERROR[1, HC1] / POLICY_ERROR[1, HC1].norm() * 1.1 * TOL.bf16_noise_excuse_factor * noise
    score = _scored(_hc_steps({(1, HC1): HC_FP32[1, HC1] + error}))[1][HC1]
    assert not score.matches and score.noise == noise


@pytest.mark.parametrize("got", [None, torch.zeros(1)], ids=["missing", "misshapen"])
def test_a_missing_or_misshapen_gradient_is_never_excused(got):
    score = _scored(_hc_steps({(0, HC_BASE): got}))[0][HC_BASE]
    assert not score.matches
    assert score.ratio == score.cosine == MISSING_GRAD_SCORE


@pytest.mark.parametrize("name", [L1_GATE, L1_EXPERTS], ids=["router", "expert_bank"])
@pytest.mark.parametrize("steps", [(1,), tuple(range(HC_STEPS))], ids=["one_step", "every_step"])
def test_a_halved_router_or_expert_bank_on_one_layer_fails(name, steps):
    """The fp32 reference's other routing puts both a quarter of their norm from it, so a halved gradient sits within
    the noise factor; they stand far above the excusable layer fraction, on one step as on every step."""
    scored = _scored(_scaled(name, 0.5, steps))
    for step in steps:
        score = scored[step][name]
        assert score.fp32_error <= TOL.bf16_noise_excuse_factor * score.noise, "premise: inside the noise factor"
        assert score.layer_fraction > TOL.bf16_noise_excusable_layer_fraction
        assert not score.matches


def test_a_small_vector_halved_on_one_layer_fails_on_its_parameter_check():
    """Halving layer 1's dt_bias moves each step's gradient by less than the noise factor times its rounding, and layer
    0's keeps the kind inside the bars: only the check that each parameter lands inside them at some step catches it."""
    scored = _scored(_scaled(DT1, 0.5))
    assert all(scored[step][DT1].excused for step in range(HC_STEPS))
    entries = [(name, score.in_bars) for scores in scored for name, score in scores.items()]
    assert not groups_never_scored(entries, parameter_kind), "premise: the kind lands inside the bars"
    assert parameters_outside_the_bars(scored) == [DT1]


def _replica(step: int, name: str, grad: torch.Tensor) -> torch.Tensor:
    """One DP replica's own gradient: as large as the mean or larger, in another direction."""
    true = HC_FP32[step, name]
    return true + _vector(tuple(true.shape), 2 * true.norm().item(), 500 + step)


@pytest.mark.parametrize("kind", [parameter_kind(HC1), HC_BASE])
@pytest.mark.parametrize(
    "fault",
    [lambda step, name, grad: 2 * grad, lambda step, name, grad: grad / 2, _replica],
    ids=["doubled", "halved", "dropped"],
)
def test_a_mis_synced_kind_fails_despite_its_excused_pair(kind, fault):
    """The kind's well-resolved pairs fail past their noise, or one of its parameters lands outside the bars on every
    step; the same control passes the clean gradients, its excusable fractions read off every layer's gradients."""
    steps = _hc_steps()
    assert fault_passes_bf16_noise_scoring(steps, kind, lambda step, name, grad: grad, cos_min=TOL.ep_grad_cosine_min)
    assert not fault_passes_bf16_noise_scoring(steps, kind, fault, cos_min=TOL.ep_grad_cosine_min)


def test_a_halved_parameter_inside_its_noise_on_every_step_fails_on_the_parameter_check():
    """Halving the head scale moves each step's gradient by less than the noise factor times its rounding, so every
    pair is excused, and only the check, which finds none of its steps inside the bars any more, catches it."""
    scored = _scored(_scaled(HC_SCALE, 0.5))
    assert all(scored[step][HC_SCALE].excused for step in range(HC_STEPS))
    assert parameters_outside_the_bars(scored) == [HC_SCALE]
    assert not fault_passes_bf16_noise_scoring(
        _hc_steps(), HC_SCALE, lambda step, name, grad: grad / 2, cos_min=TOL.ep_grad_cosine_min
    )


@pytest.mark.parametrize("name", [L1_GATE, L1_EXPERTS], ids=["router", "expert_bank"])
def test_a_routed_parameter_is_never_excused(name):
    """Every pair counted near-cancelled, the size rule off: a router or bank halved on step 1 sits within the factor of
    the noise an fp32 reference routing other tokens gives it, so only its being routed fails it."""
    steps = _scaled(name, 0.5, (1,))
    loose = score_against_bf16_noise(steps, cos_min=TOL.ep_grad_cosine_min, excusable_fraction=math.inf)
    assert loose[1][name].excused, "premise: the halved gradient sits inside the noise factor"
    routed = score_against_bf16_noise(
        steps, cos_min=TOL.ep_grad_cosine_min, excusable_fraction=math.inf, routed=frozenset({L1_GATE, L1_EXPERTS})
    )
    assert not routed[1][name].matches
    assert [
        (step, other) for step, scores in enumerate(routed) for other, score in scores.items() if not score.matches
    ] == [(1, name)]


def test_a_parameter_the_noise_excuses_on_every_step_fails_the_parameter_check():
    """The head base cancelled on every step: each pair is excused, so no step scores it."""
    cancelled = {**HC_FP32, **{(step, HC_BASE): HC_FP32[0, HC_BASE] for step in range(HC_STEPS)}}
    scored = _scored(_hc_steps(fp32=cancelled))
    assert all(scored[step][HC_BASE].excused for step in range(HC_STEPS))
    assert parameters_outside_the_bars(scored) == [HC_BASE]


# DeepSeek-V4's head hc_base (2 elements) on the GPU ep_multi row, per step: its fp32 reference gradient's norm, and the
# bf16 reference's and the policy's (norm ratio, angle in degrees) against it. Cancelled on step 0 and best resolved on
# step 1; on step 2 the bf16 reference misses fp32 by 29% of its norm, so the policy sits at 0.72 of it.
GPU_HC_BASE = (
    (4.871e-6, (4.1972, 27.3), (1.3256, 61.7)),
    (1.930e-4, (0.8960, 3.34), (0.8076, -1.15)),
    (6.653e-5, (1.2902, 7.2), (0.9328, 5.9)),
)


def _planar(norm: float, ratio: float, degrees: float) -> torch.Tensor:
    angle = torch.tensor(degrees).deg2rad()
    return ratio * norm * torch.stack([angle.cos(), angle.sin()])


def _gpu_hc_base_steps(scale: float = 1.0) -> list[dict]:
    """The hc_base steps beside a head gradient that sets its excusable fraction, its policy gradient times ``scale``."""
    head = _vector((16, 8), 2.0, 700)
    return [
        {
            HC_BASE: (scale * _planar(norm, *policy), _planar(norm, *bf16), _planar(norm, 1.0, 0.0)),
            HEAD: (head, head, head),
        }
        for norm, bf16, policy in GPU_HC_BASE
    ]


def test_a_doubled_hc_base_fails_at_its_best_resolved_step():
    """The policy's own: excused at the cancelled step 0, inside the bars at steps 1 and 2. Doubled, it leaves the bars at
    step 1 within the factor of its noise, so every pair still matches, and at step 2 it lands inside them (2 x 0.72):
    only the bars at step 1, where its fp32 reference is largest, fail it."""
    clean = _scored(_gpu_hc_base_steps())
    assert [scores[HC_BASE].excused for scores in clean] == [True, False, False]
    assert [scores[HC_BASE].in_bars for scores in clean] == [False, True, True]
    assert not parameters_outside_the_bars(clean)
    doubled = _scored(_gpu_hc_base_steps(scale=2.0))
    assert all(scores[HC_BASE].matches for scores in doubled), "premise: the factor excuses every doubled pair"
    assert doubled[2][HC_BASE].in_bars and not doubled[1][HC_BASE].in_bars
    assert parameters_outside_the_bars(doubled) == [HC_BASE]
    assert not fault_passes_bf16_noise_scoring(
        _gpu_hc_base_steps(), HC_BASE, lambda step, name, grad: 2 * grad, cos_min=TOL.ep_grad_cosine_min
    )


# A GLM-5 Next ffn_hc.base on the CPU calibration's two-replica rows, per step: its fp32 reference gradient's norm, and
# the bf16 reference's and the policy's (norm ratio, angle in degrees) against it. The bf16 reference misses fp32 by
# 0.7-1.0 of its norm on every step, inflating it past the fp32 peak at step 2; the fp32 reference peaks at step 1, where
# the policy lands inside the bars.
INFLATED_HC = "model.layers.1.ffn_hc.base"
INFLATED_HC_STEPS = (
    (2.4e-4, (0.5, 70.0), (1.0, -20.0)),
    (6.94e-4, (0.41, 40.0), (0.418, 40.0)),
    (4.24e-4, (1.7, 30.0), (1.0, 0.0)),
)


def test_the_best_resolved_step_is_the_fp32_references_largest():
    """Scored at the bf16 reference's largest step, step 2, the parameter would land outside the bars; at the fp32
    reference's, step 1, it lands inside them, and the noise excuses it elsewhere."""
    bank = _vector((4, 8, 6), 1.0, 701)
    steps = [
        {
            INFLATED_HC: (_planar(norm, *policy), _planar(norm, *bf16), _planar(norm, 1.0, 0.0)),
            L1_EXPERTS: (bank, bank, bank),
        }
        for norm, bf16, policy in INFLATED_HC_STEPS
    ]
    bf16_norms = [pairs[INFLATED_HC][1].norm().item() for pairs in steps]
    assert bf16_norms.index(max(bf16_norms)) == 2, "premise: the bf16 reference peaks at step 2"
    scored = _scored(steps)
    assert [scores[INFLATED_HC].in_bars for scores in scored] == [False, True, False]
    assert all(scores[INFLATED_HC].matches for scores in scored)
    assert not parameters_outside_the_bars(scored)


# GLM-5 Next's attn_hc.scale (3 elements) on its four layers on the GPU ep_multi row, per step: its fp32 reference
# gradient's norm, and the bf16 reference's and the policy's (norm ratio, angle in degrees) against it. Each layer's
# largest gradient sits at 0.25. Layer 3 is best resolved on step 2, where the policy sits at 0.72 of the bf16
# reference: of the row's hc vectors, the closest to a bar at its best resolved step.
GLM5N_LAYER = "model.language_model.layers.{}.{}"
GLM5N_HC_SCALE = (
    (
        (4.959e-4, (0.7920, 2.1), (0.5767, 7.2)),
        (1.119e-3, (0.9752, 0.0), (1.0093, 0.0)),
        (1.617e-3, (1.1232, 0.0), (1.0713, 0.0)),
    ),
    (
        (1.517e-4, (0.8162, 18.6), (0.8547, 13.9)),
        (1.708e-4, (0.7672, 68.5), (0.7319, 66.4)),
        (4.938e-4, (0.9933, 0.8), (1.0232, 0.0)),
    ),
    (
        (2.231e-4, (1.1344, 1.4), (1.0864, -0.8)),
        (6.005e-5, (1.1232, 13.0), (1.2001, 16.4)),
        (2.994e-4, (0.9893, 0.0), (1.0653, 1.4)),
    ),
    (
        (1.105e-4, (0.5465, 20.0), (0.6316, 11.9)),
        (3.323e-5, (1.4343, 12.8), (1.4274, 12.2)),
        (2.713e-4, (1.1263, 0.8), (0.8061, 3.9)),
    ),
)
GLM5N_SCALES = [GLM5N_LAYER.format(layer, "attn_hc.scale") for layer in range(len(GLM5N_HC_SCALE))]


def _glm5n_hc_scale_steps(doubled: tuple[int, ...] = ()) -> list[dict]:
    """The attn_hc.scale steps, the policy's gradient doubled on the ``doubled`` layers, beside each layer's largest."""
    steps = []
    for step in range(HC_STEPS):
        pairs = {}
        for layer, per_step in enumerate(GLM5N_HC_SCALE):
            norm, bf16, policy = per_step[step]
            scale = 2.0 if layer in doubled else 1.0
            pairs[GLM5N_SCALES[layer]] = (
                scale * _planar(norm, *policy),
                _planar(norm, *bf16),
                _planar(norm, 1.0, 0.0),
            )
            largest = _vector((16, 8), 0.25, 800 + layer)
            pairs[GLM5N_LAYER.format(layer, "self_attn.o_proj.weight")] = (largest, largest, largest)
        steps.append(pairs)
    return steps


def test_the_best_step_bar_margin_is_the_closest_bar_at_the_fp32_largest_step():
    resolved = best_resolved_scores(_scored(_glm5n_hc_scale_steps()))
    closest = resolved[GLM5N_SCALES[3]]
    assert closest.fp32_norm == pytest.approx(GLM5N_HC_SCALE[3][2][0])
    assert closest.bar_margin == pytest.approx(0.8061 / 1.1263 / LOW, rel=1e-4)
    assert min(resolved.values(), key=lambda score: score.bar_margin) is closest


@pytest.mark.parametrize(
    ("ratio", "degrees", "margin"),
    [
        (0.8, 0.0, 0.8 / LOW),
        (1.25, 0.0, HIGH / 1.25),
        (1.0, 25.0, (1 - TOL.ep_grad_cosine_min) / (1 - math.cos(math.radians(25)))),
        (0.5, 0.0, 0.5 / LOW),
    ],
    ids=["low_end", "high_end", "direction", "outside"],
)
def test_the_bar_margin_is_the_closest_bar_as_a_factor(ratio, degrees, margin):
    reference = _vector((2,), 1e-2, 801)
    angle = math.radians(degrees)
    turned = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
    score = _scored([{HEAD: (ratio * turned @ reference, reference, reference)}])[0][HEAD]
    assert score.bar_margin == pytest.approx(margin, rel=1e-4)
    assert score.in_bars == (score.bar_margin > 1)


def test_a_doubling_inside_the_bars_at_its_best_step_is_held_by_its_kind():
    """Layer 3's attn_hc.scale doubled alone leaves the bars only at the steps cancelled enough for the noise to excuse it
    and lands inside them at its best resolved step (2 x 0.72), so one vector's doubled sync passes. A doubled sync on
    the kind fails at layers 0-2's best resolved steps."""
    clean = _glm5n_hc_scale_steps()
    assert not parameters_outside_the_bars(_scored(clean))
    alone = _scored(_glm5n_hc_scale_steps(doubled=(3,)))
    assert all(score.matches for scores in alone for score in scores.values()), "premise: the noise excuses the rest"
    assert alone[2][GLM5N_SCALES[3]].in_bars and not parameters_outside_the_bars(alone)
    every_layer = tuple(range(len(GLM5N_HC_SCALE)))
    assert parameters_outside_the_bars(_scored(_glm5n_hc_scale_steps(doubled=every_layer))) == GLM5N_SCALES[:3]
    assert not fault_passes_bf16_noise_scoring(
        clean, parameter_kind(GLM5N_SCALES[3]), lambda step, name, grad: 2 * grad, cos_min=TOL.ep_grad_cosine_min
    )


# An EP rank holding experts 2-3 of a 4-expert bank, and a replicated norm.
BANK, NORM = L0_EXPERTS, "model.layers.0.input_layernorm.weight"
RANK_SLICES = {BANK: slice(2, 4)}
FULL_BANK = _vector((4, 8, 6), 1.0, 600)
NORM_GRAD = _vector((8,), 0.1, 601)


def _paired(policy_bank, reference_bank):
    """``pair_step_grads`` over the rank's bank slice and the norm, the fp32 reference a copy of the bf16 one."""
    references = {BANK: reference_bank, NORM: NORM_GRAD}
    fp32 = {name: None if grad is None else grad.clone() for name, grad in references.items()}
    return pair_step_grads({BANK: policy_bank, NORM: NORM_GRAD.clone()}, references, fp32, RANK_SLICES)


def _without_rank_experts(bank: torch.Tensor) -> torch.Tensor:
    """``bank`` with the rank's experts at zero: no token reached them."""
    bank = bank.clone()
    bank[RANK_SLICES[BANK]] = 0
    return bank


@pytest.mark.parametrize(
    ("policy", "reference"),
    [(None, _without_rank_experts(FULL_BANK)), (torch.zeros(2, 8, 6), None), (None, None)],
    ids=["none_vs_zero_slice", "zeros_vs_none", "none_vs_none"],
)
def test_none_and_zeros_are_one_gradient_less_state(policy, reference):
    """A rank whose experts no token reached keeps None where the unsharded reference sums zeros; the bank is left
    unscored and the norm still pairs."""
    paired = _paired(policy, reference)
    assert paired.gradless == paired.reference_gradless == [BANK]
    assert list(paired.pairs) == [NORM] and not paired.missing


@pytest.mark.parametrize("policy", [None, torch.zeros(2, 8, 6)], ids=["none", "zeros"])
def test_a_dropped_gradient_fails_the_match_and_scores_as_missing(policy):
    """A policy without the reference's gradient fails the match and pairs as None, which scores as missing and does
    not raise on a zero norm."""
    paired = _paired(policy, FULL_BANK)
    assert paired.gradless == [BANK] and paired.reference_gradless == []
    assert paired.pairs[BANK][0] is None and torch.equal(paired.pairs[BANK][1], FULL_BANK[RANK_SLICES[BANK]])
    score = score_against_bf16_noise([paired.pairs], cos_min=TOL.ep_grad_cosine_min)[0][BANK]
    assert not score.matches and score.ratio == score.cosine == MISSING_GRAD_SCORE


def test_a_gradient_the_reference_lacks_fails_the_match():
    paired = _paired(FULL_BANK[RANK_SLICES[BANK]].clone(), _without_rank_experts(FULL_BANK))
    assert paired.gradless == [] and paired.reference_gradless == [BANK]
    assert BANK not in paired.pairs


def test_a_matching_rank_slice_pairs_with_the_reference_cut():
    paired = _paired(FULL_BANK[RANK_SLICES[BANK]].clone(), FULL_BANK)
    assert paired.gradless == paired.reference_gradless == []
    got, bf16, fp32 = paired.pairs[BANK]
    assert torch.equal(got, bf16) and torch.equal(bf16, fp32) and bf16.shape == (2, 8, 6)


def test_a_parameter_without_a_reference_is_missing():
    paired = pair_step_grads({NORM: NORM_GRAD, "extra.weight": NORM_GRAD}, {NORM: NORM_GRAD}, {NORM: NORM_GRAD}, {})
    assert paired.missing == ["extra.weight"] and list(paired.pairs) == [NORM]


def test_a_bank_without_a_gradient_on_both_sides_at_every_step_scores_nothing():
    """Each step passes the gradient-less match, so only ``scores_any`` tells that the rank's expert sync was
    compared on nothing."""
    unreached = [_paired(None, _without_rank_experts(FULL_BANK)), _paired(torch.zeros(2, 8, 6), None)]
    assert all(paired.gradless == paired.reference_gradless for paired in unreached)
    assert not scores_any([paired.pairs for paired in unreached], [BANK])
    assert scores_any([paired.pairs for paired in unreached], [NORM])
    reached = [*unreached, _paired(FULL_BANK[RANK_SLICES[BANK]].clone(), FULL_BANK)]
    assert scores_any([paired.pairs for paired in reached], [BANK])


# Two families whose EP layers hand routing replay their HF router's logits.
PINNED_FAMILIES = ("qwen3_moe", "gpt_oss")
ROWS, ROW_TOKENS = 4, 48


def _rows_loss(model, rows: list[torch.Tensor]) -> torch.Tensor:
    loss = 0.0
    for ids in rows:
        logits = model(input_ids=ids).logits.float()
        loss = loss + F.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="sum")
    return loss


def _picks(model, rows) -> list[torch.Tensor]:
    with captured_picks(model) as captured, torch.no_grad():
        _rows_loss(model, rows)
    return [torch.cat(chunks) for chunks in captured]


def _backward(model, rows, picks: list[torch.Tensor] | None = None) -> list:
    """The rows' loss backward, routed on ``picks`` when given; the pinned routes."""
    model.zero_grad(set_to_none=True)
    with routing_pinned(model, picks) if picks is not None else contextlib.nullcontext([]) as routes:
        loss = _rows_loss(model, rows)
    loss.backward()
    return routes


def _expert_grads(model) -> list[torch.Tensor]:
    return [param.grad.float() for layer in ep_layers(model) for _, param in layer.expert_named_params()]


def _ep_model(family: str, dtype: torch.dtype = torch.bfloat16) -> torch.nn.Module:
    """``family``'s tiny model at bf16 weights under EP layers, computing at ``dtype``."""
    torch.manual_seed(0)
    base = tiny_family_model(TINY_MOE_FAMILIES[family]).to(torch.bfloat16).to(dtype)
    return apply_ep_layers(base, single_process_ep_config()).train()


@pytest.mark.parametrize(
    ("family", "router"),
    [("qwen3_moe", ".mlp.gate."), ("gpt_oss", ".mlp.router."), ("zaya", ".mlp.gate."), ("gemma4_text", None)],
)
def test_router_parameter_names_are_each_ep_layers_own_router(family, router):
    """Every trainable parameter of each EP layer's own router module, by name (Zaya's an MLP of eight or nine); Gemma
    4's router is a sibling of its EP layers."""
    model = _ep_model(family)
    names = router_parameter_names(model)
    assert names == {
        name
        for name, param in model.named_parameters()
        if router is not None and router in name and param.requires_grad
    }
    assert router is None or len({decoder_layer_index(name) for name in names}) == len(ep_layers(model))


@pytest.mark.parametrize(
    ("family", "pinnable"),
    [
        ("qwen3_moe", True),
        ("gpt_oss", True),
        ("qwen3_5_moe_text", True),
        # It hands replay its gate's logits, but its gate returns (indices, weights, logits).
        ("bailing_moe", False),
        # Its router returns (logits, scores, indices), but it routes without the replay seam.
        ("mistral4", False),
        ("zaya", False),
    ],
)
def test_routing_can_be_pinned_reads_the_router_contract(family, pinnable):
    model = _ep_model(family)
    ids = random_token_batch(model.get_input_embeddings().num_embeddings, 1, ROW_TOKENS, "cpu")[0]
    assert routing_can_be_pinned(model, ids) is pinnable
    assert not any("_replay_router_choice" in vars(layer) for layer in ep_layers(model))


def test_one_layer_routing_past_replay_leaves_the_model_unpinnable():
    """Pinning would leave layer 1 on its own picks, so the model is refused."""
    model = _ep_model("qwen3_moe")
    ids = random_token_batch(model.get_input_embeddings().num_embeddings, 1, ROW_TOKENS, "cpu")[0]
    assert routing_can_be_pinned(model, ids), "premise: every layer hands replay its router's logits"
    layer = ep_layers(model)[1]

    def unreplayed(flat, **kwargs):
        _, weights, experts = layer.gate(layer._router_input(flat))
        return experts, weights

    layer._route_hidden_states = unreplayed
    assert not routing_can_be_pinned(model, ids)


@pytest.fixture(scope="module", params=PINNED_FAMILIES)
def ep_models(request):
    """The family's tiny model under EP layers at bf16 weights, as a bf16 and an fp32 copy, and the rows."""
    bf16, fp32 = _ep_model(request.param), _ep_model(request.param, torch.float32)
    vocab = bf16.get_input_embeddings().num_embeddings
    rows = [random_token_batch(vocab, 1, ROW_TOKENS, "cpu", seed=row)[0] for row in range(ROWS)]
    assert routing_can_be_pinned(bf16, rows[0]), "premise: the family's routing can be pinned"
    return bf16, fp32, rows


def test_captured_picks_are_each_forwards_top_k(ep_models):
    bf16, _, rows = ep_models
    with captured_picks(bf16) as captured, torch.no_grad():
        _rows_loss(bf16, rows)
    layers = ep_layers(bf16)
    assert len(captured) == len(layers) and all(len(chunks) == ROWS for chunks in captured)
    assert all(chunk.shape == (ROW_TOKENS, layers[0].top_k) for chunks in captured for chunk in chunks)
    assert not any(layer._capture_routing or layer._captured_routing_chunks for layer in layers)


def test_a_capture_of_layers_another_reader_armed_is_refused(ep_models):
    bf16, _, rows = ep_models
    with captured_picks(bf16) as captured, torch.no_grad():
        with pytest.raises(RuntimeError, match="already armed by another reader"), captured_picks(bf16):
            pass
        _rows_loss(bf16, rows[:1])
    assert all(len(chunks) == 1 for chunks in captured)
    assert not any(layer._capture_routing or layer._captured_routing_chunks for layer in ep_layers(bf16))


def test_routing_pinned_on_its_own_picks_changes_nothing(ep_models):
    bf16, _, rows = ep_models
    _backward(bf16, rows)
    natural = {name: param.grad.float().clone() for name, param in bf16.named_parameters()}
    routes = _backward(bf16, rows, _picks(bf16, rows))
    assert len(routes) == len(ep_layers(bf16))
    assert all(route.regret == route.own_regret == route.flipped == route.duplicated == 0 for route in routes)
    assert all(route.tokens == ROWS * ROW_TOKENS for route in routes)
    for name, param in bf16.named_parameters():
        assert F.cosine_similarity(param.grad.float().flatten(), natural[name].flatten(), dim=0) > 0.9999, name
    assert all(layer._forced_topk_indices is None and layer._forced_cursor == 0 for layer in ep_layers(bf16))


def test_a_bf16_copy_on_fp32_picks_routes_near_ties_alike(ep_models):
    """The fp32 copy's picks differ from the bf16 router's own on near-ties, all within the bar, and routed on them
    every expert's gradient of the bf16 copy lands on the fp32 one."""
    bf16, fp32, rows = ep_models
    picks = _picks(fp32, rows)
    fp32_routes = _backward(fp32, rows, picks)
    routes = _backward(bf16, rows, picks)
    assert all(route.own_regret <= TOL.router_own_pick_logit_abs for route in routes + fp32_routes)
    assert sum(route.flipped for route in routes) > 0, "premise: the fp32 copy flips some of the bf16 router's picks"
    assert max(route.regret for route in routes) <= TOL.router_near_tie_logit_abs
    assert min(route.next_regret for route in routes) > TOL.router_near_tie_logit_abs
    for got, want in zip(_expert_grads(bf16), _expert_grads(fp32), strict=True):
        for expert in range(want.shape[0]):
            if want[expert].any():
                assert F.cosine_similarity(got[expert].flatten(), want[expert].flatten(), dim=0) > 0.999


def _layer_logits(model, rows, index: int) -> torch.Tensor:
    logits = []
    router = ep_layers(model)[index]._live_router_module()
    handle = router.register_forward_hook(lambda module, args, output: logits.append(output[0].detach().float()))
    try:
        _picks(model, rows)
    finally:
        handle.remove()
    return torch.cat(logits)


def test_each_tokens_next_best_pick_falls_outside_the_bar(ep_models):
    """Every token of layer 1 routed to its (k+1)-th best expert in place of its k-th, the nearest wrong selection:
    its regret is the top-k gap ``next_regret`` predicts, outside the near-tie bar."""
    bf16, _, rows = ep_models
    picks = _picks(bf16, rows)
    top_k = picks[1].shape[-1]
    ranked = _layer_logits(bf16, rows, 1).topk(top_k + 1, dim=-1).indices
    routes = _backward(bf16, rows, [picks[0], torch.cat([ranked[:, : top_k - 1], ranked[:, top_k:]], -1), *picks[2:]])
    assert routes[0].regret == 0 and routes[1].flipped == routes[1].tokens and routes[1].duplicated == 0
    assert routes[1].regret > TOL.router_near_tie_logit_abs
    assert routes[1].regret == pytest.approx(routes[1].next_regret, abs=TOL.router_near_tie_logit_abs)


def test_a_token_picking_its_top_expert_twice_is_counted(ep_models):
    """Each token of layer 1 routed to its own top-1 expert ``top_k`` times: no pick falls below the k-th logit, so
    only the duplicate count tells it from the router's own selection."""
    bf16, _, rows = ep_models
    picks = _picks(bf16, rows)
    top = _layer_logits(bf16, rows, 1).argmax(dim=-1, keepdim=True)
    routes = _backward(bf16, rows, [picks[0], top.expand_as(picks[1]).contiguous(), *picks[2:]])
    assert routes[0].duplicated == 0 and routes[1].duplicated == routes[1].tokens
    assert routes[1].regret <= 0


@pytest.mark.parametrize(("extra", "error"), [(-1, RuntimeError), (1, AssertionError)], ids=["short", "long"])
def test_picks_that_do_not_tile_the_forwards_raise(ep_models, extra, error):
    """A mask a token short exhausts in the last forward; one a token long is left unconsumed. Either way every layer
    leaves routing replay disarmed."""
    bf16, _, rows = ep_models
    picks = _picks(bf16, rows)
    mask = picks[0][:extra] if extra < 0 else torch.cat([picks[0], picks[0][:extra]])
    with pytest.raises(error), routing_pinned(bf16, [mask, *picks[1:]]), torch.no_grad():
        _rows_loss(bf16, rows)
    assert all(layer._forced_topk_indices is None and layer._forced_cursor == 0 for layer in ep_layers(bf16))


# fp32 hyper-connection gradients as the PP row scores them: base and scale 5e-4 and 3e-5 of their layer's largest, over
# the floor, so the direction and norm bars take them.
HC_LAYER_BASE, HC_LAYER_SCALE = "model.layers.2.attn_hc.base", "model.layers.2.attn_hc.scale"
HC_LAYER_PROJ = "model.layers.2.mlp.down_proj.weight"
HC_LAYERED = {
    HC_LAYER_PROJ: _vector((8, 16), 1.0, 10),
    HC_LAYER_BASE: _vector((24,), 5e-4, 11),
    HC_LAYER_SCALE: _vector((3,), 3e-5, 12),
}


def test_an_fp32_hyper_connection_gradient_inside_rounding_passes():
    got = {name: ref * (1 + 1e-4) for name, ref in HC_LAYERED.items()}
    checks, _, floors = _score_layered(got, HC_LAYERED)
    assert all(checks[f"{name}_matches"] for name in HC_LAYERED), checks
    assert floors[HC_LAYER_SCALE] < HC_LAYERED[HC_LAYER_SCALE].norm()


@pytest.mark.parametrize("name", [HC_LAYER_BASE, HC_LAYER_SCALE])
@pytest.mark.parametrize("scale", [2.0, 0.5], ids=["doubled", "dropped_dp_average"])
def test_a_mis_reduced_hyper_connection_gradient_fails(name, scale):
    checks, _, _ = _score_layered({name: HC_LAYERED[name] * scale}, HC_LAYERED)
    assert not checks[f"{name}_matches"]
    assert all(checks[f"{other}_matches"] for other in HC_LAYERED if other != name)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
