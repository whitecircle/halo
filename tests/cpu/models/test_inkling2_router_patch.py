"""Inkling 2's router patch: bitwise the upstream router where the weight is fp32, as transformers pins it, and still
routing in fp32 where the run dtype cast the weight to bf16, which the upstream router refuses."""

import importlib.util

import pytest
import torch
import torch.nn as nn
from transformers.models.inkling2.configuration_inkling2 import Inkling2TextConfig
from transformers.models.inkling2.modeling_inkling2 import Inkling2TopKRouter

from halo.models.families.inkling2 import _router_forward, apply_inkling2_patches


def _upstream_router_forward():
    """The router forward transformers ships, off a second execution of its module: the patch replaces the imported
    class's forward process-wide, so an earlier test in the session may have installed it."""
    spec = importlib.util.find_spec(Inkling2TopKRouter.__module__)
    pristine = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pristine)
    return pristine.Inkling2TopKRouter.forward


UPSTREAM_ROUTER_FORWARD = _upstream_router_forward()


def _router(dtype: torch.dtype) -> Inkling2TopKRouter:
    torch.manual_seed(0)
    router = Inkling2TopKRouter(Inkling2TextConfig(hidden_size=64, n_routed_experts=8, num_experts_per_tok=2))
    nn.init.normal_(router.weight, std=0.2)
    nn.init.normal_(router.e_score_correction_bias, std=0.3)
    return router.to(dtype)


def _hidden() -> torch.Tensor:
    return torch.randn((3, 5, 64), generator=torch.Generator().manual_seed(1)).to(torch.bfloat16)


def test_the_patch_is_the_upstream_router_on_an_fp32_weight():
    router = _router(torch.float32)
    for expected, actual in zip(
        UPSTREAM_ROUTER_FORWARD(router, _hidden()), _router_forward(router, _hidden()), strict=True
    ):
        assert torch.equal(actual, expected)


def test_the_patch_routes_a_run_dtype_weight_in_fp32():
    cast = _router(torch.bfloat16)
    with pytest.raises(RuntimeError, match="same dtype"):
        UPSTREAM_ROUTER_FORWARD(cast, _hidden())
    reference = _router(torch.float32)
    reference.weight.data = cast.weight.data.float()
    for expected, actual in zip(
        UPSTREAM_ROUTER_FORWARD(reference, _hidden()), _router_forward(cast, _hidden()), strict=True
    ):
        assert torch.equal(actual, expected)


def test_apply_installs_the_patch_once(monkeypatch):
    # Restores the class's forward on teardown, so the patch does not outlive the test.
    monkeypatch.setattr(Inkling2TopKRouter, "forward", UPSTREAM_ROUTER_FORWARD)
    apply_inkling2_patches()
    assert Inkling2TopKRouter.forward is _router_forward
    apply_inkling2_patches()
    assert Inkling2TopKRouter.forward is _router_forward


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
