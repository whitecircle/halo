#!/usr/bin/env python
"""GptOss under expert TP runs its experts through grouped GEMM, on the storage the per-expert loop reads.

ETP stores this rank's slice of every local expert de-interleaved under the plain ``gate_proj``/``up_proj``
names (and ``gate_proj_bias``/``up_proj_bias``, ``down_proj`` sharded on its input axis, ``down_proj_bias`` on
expert-TP rank 0 only). The grouped path reads that pair where it reads ``*_gmm`` outside ETP, so a checkpoint an
ETP run wrote on the per-expert loop resumes unchanged.

Every layer here is a real ``EPGptOssMoELayer`` built on a real ``EPConfig`` for one rank of a pure-ETP (world 2)
or an ep4 x etp2 (world 8) job, around the real ``transformers`` ``GptOssMLP``; only DeepEP is stubbed, since the
expert compute is called on a dispatched view directly. Asserted in fp32 on CPU, where ``F.grouped_mm`` runs:

* the grouped path is the one that runs under ETP, on this rank's ETP-sharded weights;
* it matches the per-expert loop on the same rank, forward and backward (tokens, routing weights, every expert
  weight and bias), DeepEP's ``-1`` padding included;
* summed over the expert-TP partners, as the token-space all-reduce sums them, it reproduces the upstream
  ``GptOssExperts`` output and input gradients, and each rank's weight gradients are its slice of upstream's;
* the stored parameters do not depend on which compute path the run asked for;
* outside ETP the same grouped path reads the ``*_gmm`` pair and reproduces upstream.

Run: ``python tests/cpu/parallelism/test_gptoss_etp_grouped_mm.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import copy
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn
from transformers.models.gpt_oss.configuration_gpt_oss import GptOssConfig
from transformers.models.gpt_oss.modeling_gpt_oss import GptOssMLP

from halo.distributed.expert_parallel import config as ep_config_module
from halo.distributed.expert_parallel.families.gpt_oss import EPGptOssMoELayer
from halo.distributed.expert_parallel.moe_layer import compute as compute_module
from halo.distributed.expert_parallel.moe_layer import layer as layer_module
from halo.models.moe.expert_layout import split_interleaved_gate_up
from tests.common.ep_reference import dispatched_receive_counts
from tests.common.parallelism import stubbed_rank_ep_config

E, H, M, K = 16, 16, 32, 2  # experts, hidden, expert intermediate, top_k
TOKENS = 24
# Inside the fixture's activation scale, so the clamped SwiGLU's clamp and its zero-gradient region are exercised.
SWIGLU_LIMIT = 1.0
# (world size, ep_size); expert_tp_size is the quotient.
LAYOUTS = {"etp2": (2, 1), "ep4_etp2": (8, 4), "ep2": (2, 2)}
ETP_LAYOUTS = ("etp2", "ep4_etp2")
# Fraction of the reference's max magnitude two fp32 paths summing the same terms in another order may differ by:
# ~12x the measured worst case (1.7e-7, grouped vs loop), far below a bf16 rounding of one operand (~4e-3).
RELATIVE_BAND = 2e-6
# The storage every ETP rank holds on either compute path, which its checkpoints carry.
ETP_STORAGE = ("router.weight", "router.bias", "gate_proj", "up_proj", "gate_proj_bias", "up_proj_bias", "down_proj")
ETP_RANK0_ONLY = ("down_proj_bias",)


class _NoDispatcher:
    """DeepEP's stand-in: the tests call the expert compute on a dispatched view, never the dispatch."""

    def __init__(self, *args, **kwargs):
        pass


def _block() -> GptOssMLP:
    """The upstream block, every parameter (biases included, which upstream leaves uninitialized) drawn."""
    torch.manual_seed(0)
    block = GptOssMLP(
        GptOssConfig(
            hidden_size=H,
            intermediate_size=M,
            num_local_experts=E,
            num_experts_per_tok=K,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            vocab_size=32,
            swiglu_limit=SWIGLU_LIMIT,
        )
    )
    for parameter in block.parameters():
        nn.init.normal_(parameter, std=0.2)
    return block


def _rank_layer(block: GptOssMLP, layout: str, rank: int, *, grouped_at_build: bool = False) -> EPGptOssMoELayer:
    """Rank ``rank``'s layer around a copy of ``block``. Built outside the stubbed groups, so no grad-sync hook
    registers. ``grouped_at_build`` makes ``EPConfig.grouped_gemm_enabled`` hold at construction, as on SM90+."""
    config = stubbed_rank_ep_config(*LAYOUTS[layout], rank, E)
    with (
        patch.object(layer_module, "DeepEPDispatcher", _NoDispatcher),
        patch.object(ep_config_module, "supports_grouped_mm", return_value=grouped_at_build),
    ):
        return EPGptOssMoELayer(copy.deepcopy(block), config)


def _all_rank_layers(block: GptOssMLP, layout: str) -> list[EPGptOssMoELayer]:
    return [_rank_layer(block, layout, rank) for rank in range(LAYOUTS[layout][0])]


def _dispatched_view(layer: EPGptOssMoELayer, *, padded: bool, seed: int):
    """``(tokens, local expert ids, routing weights)`` as this rank receives them: ``K`` distinct experts per
    token, ids outside this rank's range padded with DeepEP's ``-1`` when ``padded``, else drawn from it only."""
    generator = torch.Generator().manual_seed(seed)
    pool = E if padded else layer.experts_per_rank
    picks = torch.rand(TOKENS, pool, generator=generator).argsort(dim=-1)[:, :K]
    if padded:
        local = picks - layer.expert_start
        local[(local < 0) | (local >= layer.experts_per_rank)] = -1
        keep = (local >= 0).any(dim=-1)  # DeepEP delivers only tokens with a local expert
        local = local[keep]
    else:
        local = picks
    count = local.shape[0]
    tokens = torch.randn(count, H, generator=generator)
    weights = torch.rand(count, K, generator=generator)
    return tokens, local, weights


def _expert_params(layer: EPGptOssMoELayer) -> dict[str, nn.Parameter]:
    return dict(layer.expert_named_params())


def _forward_backward(layer, tokens, experts, weights, *, grouped: bool, cotangent):
    """The rank's expert output and its gradients on tokens, routing weights and every expert parameter."""
    layer._use_grouped_mm = grouped
    assert layer._grouped_mm_enabled() is grouped, "the arm would not run the compute path it is named for"
    layer.zero_grad(set_to_none=True)
    tokens = tokens.clone().requires_grad_(True)
    weights = weights.clone().requires_grad_(True)
    out = layer._compute_experts(
        tokens, experts, weights, torch.float32, recv_counts=dispatched_receive_counts(layer, experts)
    )
    out.backward(cotangent)
    grads = {name: param.grad.clone() for name, param in _expert_params(layer).items()}
    return out.detach(), tokens.grad, weights.grad, grads


def _within_band(actual: torch.Tensor, expected: torch.Tensor) -> tuple[bool, float, float]:
    difference = (actual - expected).abs().max().item() if expected.numel() else 0.0
    band = RELATIVE_BAND * expected.abs().max().item() if expected.numel() else 0.0
    return difference <= band, difference, band


def _assert_close(actual: torch.Tensor, expected: torch.Tensor, label: str) -> None:
    assert actual.shape == expected.shape, f"{label}: shape {tuple(actual.shape)} vs {tuple(expected.shape)}"
    assert expected.abs().max().item() > 1e-3, f"{label}: the reference is ~zero, nothing was compared"
    ok, difference, band = _within_band(actual, expected)
    assert ok, f"{label}: diverges by {difference:.3e} (band {band:.3e})"


@pytest.mark.parametrize("layout", ETP_LAYOUTS)
def test_etp_runs_the_grouped_gemm_on_its_sharded_weights(monkeypatch, layout):
    """Asked for grouped GEMM, an ETP layer reports and runs it: three grouped projections over this rank's
    ``[E_local, H, M/etp]`` gate and up halves and its ``[E_local, M/etp, H]`` down slice."""
    layer = _rank_layer(_block(), layout, rank=1, grouped_at_build=True)
    assert layer.expert_tp_size == 2 and layer._grouped_mm_enabled()

    seen = []
    real = compute_module.expert_gemm
    monkeypatch.setattr(
        compute_module, "expert_gemm", lambda a, b, **kwargs: seen.append(tuple(b.shape)) or real(a, b, **kwargs)
    )
    tokens, experts, weights = _dispatched_view(layer, padded=layout != "etp2", seed=3)
    layer._compute_experts(
        tokens, experts, weights, torch.float32, recv_counts=dispatched_receive_counts(layer, experts)
    )

    local, shard = layer.experts_per_rank, M // layer.expert_tp_size
    assert seen == [(local, H, shard), (local, H, shard), (local, shard, H)]


@pytest.mark.parametrize("layout", ETP_LAYOUTS)
def test_etp_grouped_gemm_matches_the_per_expert_loop(layout):
    """Every rank of the layout, the same parameters through both paths: output, token and routing-weight
    gradients, and every expert weight and bias gradient agree to fp32 reordering."""
    block = _block()
    for rank, layer in enumerate(_all_rank_layers(block, layout)):
        tokens, experts, weights = _dispatched_view(layer, padded=layout != "etp2", seed=10 + rank)
        cotangent = torch.randn(tokens.shape, generator=torch.Generator().manual_seed(rank))
        grouped = _forward_backward(layer, tokens, experts, weights, grouped=True, cotangent=cotangent)
        loop = _forward_backward(layer, tokens, experts, weights, grouped=False, cotangent=cotangent)

        label = f"{layout} rank {rank}"
        _assert_close(grouped[0], loop[0], f"{label} output")
        _assert_close(grouped[1], loop[1], f"{label} token grad")
        _assert_close(grouped[2], loop[2], f"{label} routing-weight grad")
        assert grouped[3].keys() == loop[3].keys()
        for name in loop[3]:
            _assert_close(grouped[3][name], loop[3][name], f"{label} {name} grad")


def _upstream_slices(block: GptOssMLP, layer: EPGptOssMoELayer, source: str) -> dict[str, torch.Tensor]:
    """``source`` (``"data"`` or ``"grad"``) of upstream's expert tensors, cut to what ``layer`` stores."""
    experts = block.experts
    start, end = layer.expert_start, layer.expert_end
    shard = M // layer.expert_tp_size
    offset = layer.expert_tp_rank * shard

    def read(param):
        return getattr(param, source)[start:end]

    gate, up = split_interleaved_gate_up(read(experts.gate_up_proj))
    gate_bias, up_bias = split_interleaved_gate_up(read(experts.gate_up_proj_bias))
    slices = {
        "gate_proj": gate[..., offset : offset + shard],
        "up_proj": up[..., offset : offset + shard],
        "gate_proj_bias": gate_bias[..., offset : offset + shard],
        "up_proj_bias": up_bias[..., offset : offset + shard],
        "down_proj": read(experts.down_proj)[:, offset : offset + shard],
    }
    if layer.expert_tp_rank == 0:
        slices["down_proj_bias"] = read(experts.down_proj_bias)
    return slices


@pytest.mark.parametrize("layout", ETP_LAYOUTS)
def test_etp_partners_sum_to_the_upstream_experts(layout):
    """Each dispatch-EP rank's expert-TP partners, on tokens routed to that rank's experts: their grouped outputs
    and input gradients summed (the token-space all-reduce and the ETP backward sum) equal upstream's
    ``GptOssExperts``, the down bias counted once, and each rank's weight gradients are its own slice of
    upstream's."""
    block = _block()
    layers = _all_rank_layers(block, layout)
    partners: dict[int, list[EPGptOssMoELayer]] = {}
    for layer in layers:
        partners.setdefault(layer.expert_start, []).append(layer)
    assert all(len(group) == 2 for group in partners.values())

    for start, group in partners.items():
        tokens, experts, weights = _dispatched_view(group[0], padded=False, seed=start)
        cotangent = torch.randn(tokens.shape, generator=torch.Generator().manual_seed(start))
        reference_block = copy.deepcopy(block)
        reference_tokens = tokens.clone().requires_grad_(True)
        reference_weights = weights.clone().requires_grad_(True)
        expected = reference_block.experts(reference_tokens, experts + start, reference_weights)
        expected.backward(cotangent)

        results = [
            _forward_backward(layer, tokens, experts, weights, grouped=True, cotangent=cotangent) for layer in group
        ]
        label = f"{layout} experts [{start}, {start + group[0].experts_per_rank})"
        _assert_close(sum(r[0] for r in results), expected.detach(), f"{label} summed output")
        _assert_close(sum(r[1] for r in results), reference_tokens.grad, f"{label} summed token grad")
        _assert_close(sum(r[2] for r in results), reference_weights.grad, f"{label} summed routing-weight grad")
        for layer, result in zip(group, results, strict=True):
            upstream = _upstream_slices(reference_block, layer, "grad")
            assert result[3].keys() == upstream.keys(), f"{label}: stored {sorted(result[3])}"
            for name, grad in upstream.items():
                _assert_close(result[3][name], grad, f"{label} etp rank {layer.expert_tp_rank} {name} grad")


@pytest.mark.parametrize("layout", ETP_LAYOUTS)
def test_etp_storage_does_not_depend_on_the_compute_path(layout):
    """Every rank stores the same tensors under the same names whether its run asked for grouped GEMM or not, and
    they are the ETP storage checkpoints carry: an ETP checkpoint written on the per-expert loop resumes on
    grouped GEMM with its weights and its optimizer moments' parameter names unchanged."""
    block = _block()
    for rank in range(LAYOUTS[layout][0]):
        grouped = _rank_layer(block, layout, rank, grouped_at_build=True)
        loop = _rank_layer(block, layout, rank, grouped_at_build=False)
        assert grouped._grouped_mm_enabled() and not loop._grouped_mm_enabled()

        expected = {*ETP_STORAGE, *(ETP_RANK0_ONLY if grouped.expert_tp_rank == 0 else ())}
        grouped_state, loop_state = grouped.state_dict(), loop.state_dict()
        assert set(grouped_state) == set(loop_state) == expected, f"rank {rank}: {sorted(grouped_state)}"
        for name, tensor in loop_state.items():
            assert grouped_state[name].dtype == tensor.dtype and torch.equal(grouped_state[name], tensor), name
        assert [name for name, _ in grouped.expert_named_params()] == [name for name, _ in loop.expert_named_params()]
        for name, upstream in _upstream_slices(block, grouped, "data").items():
            assert torch.equal(grouped_state[name], upstream), f"rank {rank}: {name} is not upstream's ETP slice"


def test_grouped_gemm_outside_etp_reads_the_gmm_pair():
    """Outside ETP the grouped path reads the de-interleaved ``*_gmm`` pair: each ep2 rank's output and input
    gradients equal upstream's ``GptOssExperts`` on the tokens routed to its experts, and its gate and up
    gradients are upstream's de-interleaved halves."""
    block = _block()
    for rank in range(LAYOUTS["ep2"][0]):
        layer = _rank_layer(block, "ep2", rank, grouped_at_build=True)
        assert layer.expert_tp_size == 1 and hasattr(layer, "gate_proj_gmm")
        tokens, experts, weights = _dispatched_view(layer, padded=False, seed=20 + rank)
        cotangent = torch.randn(tokens.shape, generator=torch.Generator().manual_seed(20 + rank))
        reference_block = copy.deepcopy(block)
        reference_tokens = tokens.clone().requires_grad_(True)
        reference_weights = weights.clone().requires_grad_(True)
        expected = reference_block.experts(reference_tokens, experts + layer.expert_start, reference_weights)
        expected.backward(cotangent)

        out, token_grad, weight_grad, grads = _forward_backward(
            layer, tokens, experts, weights, grouped=True, cotangent=cotangent
        )
        label = f"ep2 rank {rank}"
        _assert_close(out, expected.detach(), f"{label} output")
        _assert_close(token_grad, reference_tokens.grad, f"{label} token grad")
        _assert_close(weight_grad, reference_weights.grad, f"{label} routing-weight grad")
        upstream = reference_block.experts.gate_up_proj.grad[layer.expert_start : layer.expert_end]
        gate, up = split_interleaved_gate_up(upstream)
        _assert_close(grads["gate_proj_gmm"], gate, f"{label} gate_proj_gmm grad")
        _assert_close(grads["up_proj_gmm"], up, f"{label} up_proj_gmm grad")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
