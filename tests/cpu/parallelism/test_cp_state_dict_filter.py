#!/usr/bin/env python
"""CPU tests for the CP wrapper's stale-dense-MLP state_dict filter.

``UlyssesCPModelWrapper.state_dict`` drops stale duplicate dense-MLP keys on sparse (MoE) layers.
A filter that substring-matches a hardcoded projection list drops the REAL expert weights of an
EP-wrapped GptOss layer under EP+CP (``mlp.gate_up_proj`` / ``mlp.down_proj`` /
``mlp.down_proj_bias`` — ".mlp.down_proj" is a substring of ".mlp.down_proj_bias"). The filter
derives everything from the model itself instead: EP layers are exempt by ``isinstance`` and
dense-projection names are harvested from the model's own genuinely-dense ``.mlp`` layers with
exact-key matching.

Run: ``python tests/cpu/parallelism/test_cp_state_dict_filter.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.distributed.context_parallel.wrapper import UlyssesCPModelWrapper, stale_dense_mlp_keys
from src.distributed.expert_parallel.config import EPConfig
from src.distributed.expert_parallel.layers.gpt_oss import EPGptOssMoELayer

E, H, M, K = 4, 8, 16, 2  # experts, hidden, intermediate, top_k


_EP_GPTOSS_LAYER_KEYS = {
    "model.layers.0.mlp.router.weight",
    "model.layers.0.mlp.router.bias",
    "model.layers.0.mlp.gate_up_proj",
    "model.layers.0.mlp.gate_up_proj_bias",
    "model.layers.0.mlp.down_proj",
    "model.layers.0.mlp.down_proj_bias",
    "model.layers.0.self_attn.q_proj.weight",
}


def test_ep_gptoss_expert_keys_are_never_stale():
    """An EP-wrapped GptOss layer's grouped expert params (dense-NAMED but real weights) survive —
    both via the EP exemption and because an all-sparse model has no dense layer to harvest names
    from. A substring filter drops all four expert keys."""
    assert stale_dense_mlp_keys(_EP_GPTOSS_LAYER_KEYS, ep_mlp_paths={"model.layers.0.mlp"}) == set()
    assert stale_dense_mlp_keys(_EP_GPTOSS_LAYER_KEYS) == set()


def test_hybrid_model_stale_dense_duplicates_dropped():
    """Hybrid (GLM-4-MoE-Lite-shaped) model: the sparse layer's stale dense duplicates — exact
    matches of the genuinely-dense layer's param names — are dropped; everything else survives."""
    keys = {
        # layer 0: genuinely dense — harvest source, must survive.
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        # layer 1: sparse — experts + router-as-gate + shared experts survive.
        "model.layers.1.mlp.experts.gate_up_proj",
        "model.layers.1.mlp.experts.down_proj",
        "model.layers.1.mlp.gate.weight",
        "model.layers.1.mlp.shared_experts.gate_proj.weight",
        "model.layers.1.mlp.shared_experts.down_proj.weight",
        # layer 1: stale dense duplicates — must drop.
        "model.layers.1.mlp.gate_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
    }
    assert stale_dense_mlp_keys(keys) == {
        "model.layers.1.mlp.gate_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
    }


def test_no_substring_collisions_on_sparse_layers():
    """EP-style grouped params on a sparse layer never match dense names by substring: only EXACT
    relative-key matches are stale (".mlp.gate_proj" substring-hits ".mlp.gate_proj_gmm")."""
    keys = {
        "model.layers.0.mlp.gate_proj.weight",  # dense harvest source
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.1.mlp.router.weight",
        "model.layers.1.mlp.gate_proj_gmm",
        "model.layers.1.mlp.down_proj_bias",
    }
    assert stale_dense_mlp_keys(keys) == set()


def test_ep_exemption_wins_over_exact_match():
    """Even a hypothetical EP-layer key that exactly matches a dense param name survives — the live
    module IS the expert owner, so nothing under it is a stale copy."""
    keys = {
        "model.layers.0.mlp.down_proj.weight",  # dense harvest source
        "model.layers.1.mlp.router.weight",
        "model.layers.1.mlp.down_proj.weight",  # would exact-match without the exemption
    }
    assert stale_dense_mlp_keys(keys, ep_mlp_paths={"model.layers.1.mlp"}) == set()
    assert stale_dense_mlp_keys(keys) == {"model.layers.1.mlp.down_proj.weight"}


# Integration: wrapper.state_dict() on a toy model with a REAL EP GptOss layer


class _GptOssRouter(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(E, H))
        self.bias = nn.Parameter(torch.randn(E))
        self.top_k = K
        self.num_experts = E


class _GptOssExperts(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_up_proj = nn.Parameter(torch.randn(E, H, 2 * M))
        self.gate_up_proj_bias = nn.Parameter(torch.randn(E, 2 * M))
        self.down_proj = nn.Parameter(torch.randn(E, M, H))
        self.down_proj_bias = nn.Parameter(torch.randn(E, H))
        self.alpha = 1.702
        self.limit = 7.0
        self.num_experts = E


class _GptOssBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.router = _GptOssRouter()
        self.experts = _GptOssExperts()


class _DenseMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate_proj = nn.Linear(H, M, bias=False)
        self.up_proj = nn.Linear(H, M, bias=False)
        self.down_proj = nn.Linear(M, H, bias=False)


class _NativeSparseMLP(nn.Module):
    """HF-native-shaped sparse block carrying stale dense duplicates (the historical corruption)."""

    def __init__(self):
        super().__init__()
        self.router = _GptOssRouter()
        self.experts = _GptOssExperts()
        self.gate_proj = nn.Linear(H, M, bias=False)  # stale duplicate
        self.down_proj = nn.Linear(M, H, bias=False)  # stale duplicate


class _Layer(nn.Module):
    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.mlp = mlp


def _toy_model(ep_layer: nn.Module) -> nn.Module:
    root = nn.Module()
    root.model = nn.Module()
    root.model.layers = nn.ModuleList([_Layer(_DenseMLP()), _Layer(ep_layer), _Layer(_NativeSparseMLP())])
    return root


def _bare_wrapper(model: nn.Module) -> UlyssesCPModelWrapper:
    """Wrapper instance without CP group/patching side effects — state_dict only needs .model."""
    wrapper = UlyssesCPModelWrapper.__new__(UlyssesCPModelWrapper)
    nn.Module.__init__(wrapper)
    wrapper.model = model
    return wrapper


def test_wrapper_state_dict_keeps_ep_experts_and_drops_stale_duplicates():
    torch.manual_seed(0)
    cfg = EPConfig(ep_size=1, world_size=1, gpus_per_node=1, use_grouped_gemm=False)
    cfg.finalize_expert_assignment(E)
    ep_layer = EPGptOssMoELayer(_GptOssBlock(), cfg).cpu()

    wrapper = _bare_wrapper(_toy_model(ep_layer))
    keys = set(wrapper.state_dict().keys())

    expected_ep = {f"model.layers.1.mlp.{s}" for s in ep_layer.state_dict()}
    missing = expected_ep - keys
    assert not missing, f"EP expert/router keys dropped from wrapper.state_dict(): {sorted(missing)}"
    assert "model.layers.1.mlp.down_proj_bias" in keys
    assert "model.layers.1.mlp.gate_up_proj" in keys

    assert "model.layers.0.mlp.gate_proj.weight" in keys
    assert "model.layers.0.mlp.down_proj.weight" in keys

    assert "model.layers.2.mlp.gate_proj.weight" not in keys
    assert "model.layers.2.mlp.down_proj.weight" not in keys
    assert "model.layers.2.mlp.router.weight" in keys
    assert "model.layers.2.mlp.experts.gate_up_proj" in keys


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
