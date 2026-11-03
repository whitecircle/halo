#!/usr/bin/env python
"""The FLOPs estimator reads the family specs' declared expert spellings, since a copy would drift and count a new
family's experts as dense, over-reporting S-MFU. The EP import chain must stay free of ``deep_ep``."""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

from halo.callbacks.model_flops import _is_expert_param
from halo.models.moe import family_specs
from halo.models.moe.family_specs import MOE_FAMILY_SPECS, MoEFamilySpec, expert_weight_roots, experts_container_attrs
from tests.common.repo import REPO_ROOT


def test_every_declared_root_classifies_as_expert_param():
    """The roots are consumed on the last name segment, so a root only helps if the classifier matches it there."""
    declared = expert_weight_roots()
    assert declared, "no EP family registered any expert-weight roots — the subclass walk is broken"
    for root in sorted(declared):
        for container in ("mlp", "feed_forward"):
            name = f"model.layers.0.{container}.{root}"
            assert _is_expert_param(name), f"expert root {root!r} not classified as an expert param"


def test_every_registered_container_attr_classifies_as_expert_param():
    """A literal ``.experts.`` misses GLM-4/Laguna's ``routed_experts``."""
    attrs = experts_container_attrs()
    assert {"experts", "routed_experts"} <= set(attrs)
    for attr in attrs:
        name = f"model.layers.0.mlp.{attr}.0.up_proj.weight"
        assert _is_expert_param(name), f"container attr {attr!r} not classified as an expert param"


def test_container_exclusions_still_hold():
    """The container match must not swallow shared experts and router/gate params."""
    assert not _is_expert_param("model.layers.0.mlp.shared_experts.0.up_proj.weight")
    assert not _is_expert_param("model.layers.0.mlp.gate.weight")
    assert not _is_expert_param("model.layers.0.mlp.dense_proj.weight")


@pytest.mark.parametrize(
    ("declared", "param"),
    [
        ({"experts_container_attrs": ("probe_experts",)}, "model.layers.0.mlp.probe_experts.0.up_proj.weight"),
        ({"expert_weight_roots": ("probe_gate_proj",)}, "model.layers.0.mlp.probe_gate_proj"),
    ],
    ids=["container", "root"],
)
def test_a_new_family_spelling_needs_no_callback_edit(monkeypatch, declared, param):
    probe = MoEFamilySpec(model_types=("probe_moe",), **declared)
    monkeypatch.setattr(family_specs, "MOE_FAMILY_SPECS", (*MOE_FAMILY_SPECS, probe))
    expert_weight_roots.cache_clear()
    experts_container_attrs.cache_clear()
    try:
        assert _is_expert_param(param), f"the FLOPs estimator ignores a newly declared {next(iter(declared))}"
    finally:
        monkeypatch.undo()
        expert_weight_roots.cache_clear()
        experts_container_attrs.cache_clear()


def test_importing_the_callback_does_not_import_deep_ep():
    """Dense/DDP/CPU jobs build this callback; ``import deep_ep`` runs ``cuInit`` and latches
    ``CUDA_DEVICE_MAX_CONNECTIONS`` for the whole process."""
    if importlib.util.find_spec("deep_ep") is None:
        pytest.skip("deep_ep is not installed — the guard cannot distinguish deferred from absent")

    probe = "import sys\nimport halo.callbacks.efficiency\nprint('deep_ep' in sys.modules)\n"
    result = subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip().splitlines()[-1] == "False", (
        "importing EfficiencyCallback pulled in deep_ep — the EP import chain must stay free of the "
        f"native extension. Subprocess output:\n{result.stdout}\n{result.stderr}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
