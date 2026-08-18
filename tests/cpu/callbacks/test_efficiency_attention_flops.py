"""``EfficiencyCallback``'s attention term must describe what THIS RANK computes, on both axes.

The ``6*N`` half of the FLOPs estimate is already per-rank — ``local_numel`` reads each shard off its
DTensor under TP, and each stage's own parameters under PP. The ``12*L*S*H`` half is derived from the
config instead, so both corrections have to be explicit:

* **TP** — ``H`` stays global on every rank, so the term is divided by ``tp_size``. Without it a TP
  rank is credited with the whole model's attention scores and MFU over-reports by up to
  ``tp_size``x on the attention share of the estimate. The parameter term must NOT be divided again.
* **PP** — ``L`` is the rank's own decoder-layer count, never ``num_hidden_layers / pp_size``. The
  default partition is head-weighted, not even (the last stage gives up layer budget for the
  ``lm_head`` matmul), so a pp_size average makes the two halves of the estimate describe different
  models, and the error is large and one-sided.

The expected per-stage layer counts come from the PP splitter itself, not from constants restated
here, so these tests keep policing the callback if the partitioning heuristic changes.

    python tests/cpu/callbacks/test_efficiency_attention_flops.py
"""

import pytest
import torch
from torch import nn

from src.callbacks.efficiency import EfficiencyCallback
from src.callbacks.model_flops import estimate_attention_flops, estimate_model_flops_per_token
from src.distributed.pipeline_parallel.split import (
    compute_layer_partition,
    head_cost_layer_equivalents,
    layer_types_period,
)
from tests.common.parallelism import make_parallelism_config

# gpt-oss-20b geometry: a large vocab head against a narrow MoE FFN, which is what makes the default
# partition uneven. 32k sequence is where the attention term is a material share of total FLOPs.
NUM_LAYERS = 24
HIDDEN = 2880
SEQ_LEN = 32768
TP_SIZES = (1, 2, 4, 8)
PP_SIZES = (1, 2, 4, 8)


class _Config:
    model_type = "gpt_oss"
    hidden_size = HIDDEN
    num_hidden_layers = NUM_LAYERS
    vocab_size = 201088
    num_attention_heads = 64
    num_key_value_heads = 8
    head_dim = 64
    intermediate_size = HIDDEN
    moe_intermediate_size = HIDDEN
    num_experts_per_tok = 4
    layer_types = ["sliding_attention", "full_attention"] * (NUM_LAYERS // 2)

    def get_text_config(self):
        return self


class _Model(nn.Module):
    """A causal-LM-shaped model whose backbone holds this rank's slice of the decoder-layer list.

    Layers are allocated on ``meta``: the estimators read parameter SHAPES and never values, so a
    full-width stack costs nothing here.
    """

    def __init__(self, n_layers: int):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList(
            nn.Linear(HIDDEN, HIDDEN, bias=False, device="meta") for _ in range(n_layers)
        )
        self.config = _Config()


def _default_partition(pp_size: int) -> list[tuple[int, int]]:
    """The partition the toolkit would actually use, from the splitter."""
    config = _Config()
    return compute_layer_partition(
        NUM_LAYERS,
        pp_size,
        head_layer_equivalents=head_cost_layer_equivalents(config),
        boundary_period=layer_types_period(config),
    )


def _attention_flops(n_layers: float) -> float:
    """12 * L * S * H — QK^T and Attn*V, forward plus backward, for an unsharded ``n_layers`` stack."""
    return 12.0 * n_layers * SEQ_LEN * HIDDEN


def _param_term(n_layers: int) -> float:
    """The ``6*N`` term for :class:`_Model`, which TP shards through ``local_numel``, not through here."""
    return 6.0 * n_layers * HIDDEN * HIDDEN


def _tp_parallelism(tp_size: int):
    """A TP-only config on a world of exactly ``tp_size`` — one TP group, no data parallelism."""
    return make_parallelism_config(tp_size=tp_size, world_size=tp_size, gpus_per_node=tp_size)


def test_premise_attention_is_a_material_share_of_the_estimate():
    """Guard the premise: at this geometry the attention term is large enough for the bug to matter."""
    assert _attention_flops(NUM_LAYERS) > 0.25 * _param_term(NUM_LAYERS), (
        "the attention term is negligible at this geometry, so halving it would be unmeasurable and "
        "these tests would prove nothing — re-derive the geometry"
    )


def test_premise_default_partition_is_uneven():
    """Guard the premise: if the default split ever became even, dividing by pp_size would be correct."""
    sizes = [hi - lo for lo, hi in _default_partition(4)]
    assert sum(sizes) == NUM_LAYERS
    assert len(set(sizes)) > 1, (
        f"the default pp4 partition {sizes} is even, so dividing by pp_size would be correct — "
        "this test's premise no longer holds and it must be re-derived"
    )


@pytest.mark.parametrize("tp_size", TP_SIZES)
def test_the_attention_term_scales_inversely_with_tp_size(tp_size):
    """A TP rank computes scores for 1/tp_size of the heads — against the formula AND the tp1 value."""
    model = _Model(NUM_LAYERS)
    got = estimate_attention_flops(model, SEQ_LEN, 1, tp_size)
    expected = _attention_flops(NUM_LAYERS) / tp_size

    assert got == pytest.approx(expected, rel=1e-9), (
        f"attention FLOPs/token at tp_size={tp_size} is {got:.6e}, expected {expected:.6e} "
        f"(the tp1 value {_attention_flops(NUM_LAYERS):.6e} divided by {tp_size})"
    )
    assert got == pytest.approx(estimate_attention_flops(model, SEQ_LEN, 1, 1) / tp_size, rel=1e-9)


@pytest.mark.parametrize("tp_size", TP_SIZES)
def test_only_the_attention_term_moves_with_tp_size(tp_size):
    """The parameter term is TP-sharded via ``local_numel``; dividing it here too would double-count."""
    total = estimate_model_flops_per_token(_Model(NUM_LAYERS), SEQ_LEN, 1, tp_size)
    residual = total - _attention_flops(NUM_LAYERS) / tp_size

    assert residual == pytest.approx(_param_term(NUM_LAYERS), rel=1e-9), (
        f"at tp_size={tp_size} the non-attention part of the estimate is {residual:.6e}, expected the "
        f"unchanged 6*N term {_param_term(NUM_LAYERS):.6e}"
    )


@pytest.mark.parametrize("tp_size", TP_SIZES)
def test_the_callback_applies_tp_size_to_both_estimates(tp_size):
    """The dense and active FLOPs/token the callback stores must both carry the division.

    They are computed at two separate call sites; a fix applied to only one leaves MFU or S-MFU
    over-reporting under TP.
    """
    baseline = EfficiencyCallback(_tp_parallelism(1))
    baseline._initialize_model_flops(_Model(NUM_LAYERS), SEQ_LEN)

    callback = EfficiencyCallback(_tp_parallelism(tp_size))
    callback._initialize_model_flops(_Model(NUM_LAYERS), SEQ_LEN)

    shard = _attention_flops(NUM_LAYERS) * (1.0 - 1.0 / tp_size)
    for name in ("model_flops_per_token", "active_model_flops_per_token"):
        got = getattr(callback.state, name)
        expected = getattr(baseline.state, name) - shard
        assert got == pytest.approx(expected, rel=1e-9), (
            f"{name} at tp_size={tp_size} is {got:.6e}, expected {expected:.6e} — the tp1 value minus "
            f"the {tp_size - 1}/{tp_size} of the attention term this rank does not compute"
        )


@pytest.mark.parametrize("pp_size", PP_SIZES)
def test_attention_flops_track_the_stages_own_layer_count(pp_size):
    """Every stage's attention term must match its real layer count, not the pp_size average."""
    for stage_index, (lo, hi) in enumerate(_default_partition(pp_size)):
        stage_layers = hi - lo
        got = estimate_attention_flops(_Model(stage_layers), SEQ_LEN, pp_size)
        expected = _attention_flops(stage_layers)
        assert got == pytest.approx(expected, rel=1e-9), (
            f"pp{pp_size} stage {stage_index} holds {stage_layers} layers but its attention term "
            f"corresponds to {got / _attention_flops(1):.2f} layers "
            f"(pp_size average would be {NUM_LAYERS / pp_size:.2f})"
        )


@pytest.mark.parametrize("pp_size", [size for size in PP_SIZES if size > 1])
def test_stage_terms_sum_to_the_whole_model(pp_size):
    """No layer's attention may be double-counted or dropped across the pipeline."""
    total = sum(estimate_attention_flops(_Model(hi - lo), SEQ_LEN, pp_size) for lo, hi in _default_partition(pp_size))
    assert total == pytest.approx(estimate_attention_flops(_Model(NUM_LAYERS), SEQ_LEN, 1), rel=1e-9)


def test_both_axes_default_to_one():
    """The common path — neither axis enabled — must keep the full-depth, full-width attention term."""
    model = _Model(NUM_LAYERS)

    assert estimate_attention_flops(model, SEQ_LEN) == pytest.approx(_attention_flops(NUM_LAYERS))
    assert estimate_model_flops_per_token(model, SEQ_LEN) == pytest.approx(
        _attention_flops(NUM_LAYERS) + _param_term(NUM_LAYERS)
    )


@pytest.mark.parametrize(("pp_size", "tp_size"), [(1, 1), (4, 1), (1, 4), (2, 4)])
def test_the_config_depth_fallback_carries_both_divisors(pp_size, tp_size):
    """A module tree exposing no layer list falls back to config depth — which is the FULL model's,
    so it needs the stage share (``/pp_size``) and the head share (``/tp_size``) applied to it."""

    class _Bare:
        config = _Config()

    got = estimate_attention_flops(_Bare(), SEQ_LEN, pp_size, tp_size)
    assert got == pytest.approx(_attention_flops(NUM_LAYERS / pp_size) / tp_size), (
        f"the fallback at pp{pp_size}/tp{tp_size} is {got:.6e}, expected the full-depth term divided by both axes"
    )


def test_no_config_and_no_layers_omits_the_term():
    """Neither source available: omit the term rather than guess (documented under-estimate)."""
    assert estimate_attention_flops(torch.nn.Identity(), SEQ_LEN, 1) == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
