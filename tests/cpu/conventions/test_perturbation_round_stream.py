#!/usr/bin/env python
"""``tests.common.on_policy_e2e.perturbation_round`` counts the stream a policy's sync must carry: expert tensors
for a MoE policy, the moved tensors for a dense one, read off the checkpoint's config."""

from types import SimpleNamespace

import pytest
import torch

from tests.common.on_policy_e2e import perturbation_round
from tests.common.tiny_models import TINY_DENSE_FAMILY, TINY_MOE_FAMILIES, tiny_family_model

CTX = SimpleNamespace(device=torch.device("cpu"), rank=0, barrier=lambda: None)


def _round(model) -> tuple[dict[str, bool], list[str]]:
    pushed: list[str] = []
    checks: dict[str, bool] = {}
    perturbation_round(CTX, model, adapter=None, push=lambda: pushed.append("push"), checks=checks)
    return checks, pushed


def test_a_dense_policy_counts_the_tensors_it_moves():
    torch.manual_seed(0)
    model = tiny_family_model(TINY_DENSE_FAMILY)
    before = {name: p.detach().clone() for name, p in model.named_parameters() if "layers.0" in name}
    checks, pushed = _round(model)
    assert checks == {"expert_perturbation_engaged": True}
    assert pushed == ["push"]
    assert any(not torch.equal(before[name], p) for name, p in model.named_parameters() if name in before)


def test_a_moe_policy_without_ep_layers_fails_the_expert_guard():
    torch.manual_seed(0)
    checks, _ = _round(tiny_family_model(TINY_MOE_FAMILIES["qwen3_moe"]))
    assert checks == {"expert_perturbation_engaged": False}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
