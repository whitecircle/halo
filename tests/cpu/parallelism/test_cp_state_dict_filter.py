#!/usr/bin/env python
"""The CP wrapper's stale-dense-MLP state_dict filter derives from the model, since substring matching drops real EP
GptOss experts. ".mlp.down_proj" is a substring of ".mlp.down_proj_bias"."""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from halo.distributed.context_parallel.ulysses_model import UlyssesModel, stale_dense_mlp_keys
from halo.distributed.expert_parallel.families.gpt_oss import EPGptOssMoELayer
from tests.common.parallelism import single_process_ep_config

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
    """Dense-named but real weights; a substring filter drops all four."""
    assert stale_dense_mlp_keys(_EP_GPTOSS_LAYER_KEYS, ep_mlp_paths={"model.layers.0.mlp"}) == set()
    assert stale_dense_mlp_keys(_EP_GPTOSS_LAYER_KEYS) == set()


def test_hybrid_model_stale_dense_duplicates_dropped():
    keys = {
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.1.mlp.experts.gate_up_proj",
        "model.layers.1.mlp.experts.down_proj",
        "model.layers.1.mlp.gate.weight",
        "model.layers.1.mlp.shared_experts.gate_proj.weight",
        "model.layers.1.mlp.shared_experts.down_proj.weight",
        "model.layers.1.mlp.gate_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
    }
    assert stale_dense_mlp_keys(keys) == {
        "model.layers.1.mlp.gate_proj.weight",
        "model.layers.1.mlp.down_proj.weight",
    }


def test_no_substring_collisions_on_sparse_layers():
    """Only exact relative-key matches are stale: ``.mlp.gate_proj`` substring-hits ``.mlp.gate_proj_gmm``."""
    keys = {
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.1.mlp.router.weight",
        "model.layers.1.mlp.gate_proj_gmm",
        "model.layers.1.mlp.down_proj_bias",
    }
    assert stale_dense_mlp_keys(keys) == set()


def test_ep_exemption_wins_over_exact_match():
    """The live module is the expert owner, so nothing under it is a stale copy."""
    keys = {
        "model.layers.0.mlp.down_proj.weight",
        "model.layers.1.mlp.router.weight",
        "model.layers.1.mlp.down_proj.weight",  # Would exact-match without the exemption.
    }
    assert stale_dense_mlp_keys(keys, ep_mlp_paths={"model.layers.1.mlp"}) == set()
    assert stale_dense_mlp_keys(keys) == {"model.layers.1.mlp.down_proj.weight"}


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
    def __init__(self):
        super().__init__()
        self.router = _GptOssRouter()
        self.experts = _GptOssExperts()
        self.gate_proj = nn.Linear(H, M, bias=False)
        self.down_proj = nn.Linear(M, H, bias=False)


class _Layer(nn.Module):
    def __init__(self, mlp: nn.Module):
        super().__init__()
        self.mlp = mlp


def _toy_model(ep_layer: nn.Module) -> nn.Module:
    root = nn.Module()
    root.model = nn.Module()
    root.model.layers = nn.ModuleList([_Layer(_DenseMLP()), _Layer(ep_layer), _Layer(_NativeSparseMLP())])
    return root


def _bare_wrapper(model: nn.Module) -> UlyssesModel:
    wrapper = UlyssesModel.__new__(UlyssesModel)
    nn.Module.__init__(wrapper)
    wrapper.model = model
    return wrapper


def test_wrapper_state_dict_keeps_ep_experts_and_drops_stale_duplicates():
    torch.manual_seed(0)
    ep_layer = EPGptOssMoELayer(_GptOssBlock(), single_process_ep_config(E)).cpu()

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
