#!/usr/bin/env python
"""CPU guards on how the FLOPs estimator learns which params are sparse (routed) MoE experts.

The estimator must read the family specs' declared ``expert_weight_roots`` and ``experts_container_attrs``
(:class:`~src.models.moe.family_specs.MoEFamilySpec`) — not a copy of them. A copy drifts silently: a
family adding an expert-weight root or container spelling the callback does not know is classified as a
DENSE parameter, so its params count as always-active and S-MFU (sparsity = top_k/num_experts applies
only to expert params) is over-reported for that model.

Reading the declaration only stays affordable while the EP import chain is free of ``deep_ep``, whose
import loads a CUDA extension and runs ``cuInit`` (latching ``CUDA_DEVICE_MAX_CONNECTIONS``) — the
callback also runs on dense/DDP/CPU jobs. Both halves are asserted here.

Run: ``python tests/cpu/callbacks/test_efficiency_expert_roots.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys

import pytest

from src.callbacks.model_flops import _is_expert_param
from src.models.moe import family_specs
from src.models.moe.family_specs import MOE_FAMILY_SPECS, MoEFamilySpec, expert_weight_roots, experts_container_attrs
from tests.common.repo import REPO_ROOT


def test_every_declared_root_classifies_as_expert_param():
    """Every root any EP layer class declares must reach ``_is_expert_param`` (the roots are consumed
    on the LAST name segment, so a root only helps if the classifier matches it there)."""
    declared = expert_weight_roots()
    assert declared, "no EP family registered any expert-weight roots — the subclass walk is broken"
    for root in sorted(declared):
        for container in ("mlp", "feed_forward"):
            name = f"model.layers.0.{container}.{root}"
            assert _is_expert_param(name), f"expert root {root!r} not classified as an expert param"


def test_every_registered_container_attr_classifies_as_expert_param():
    """A param INSIDE any registered expert container (per-expert module-list layouts) is an expert
    param under every spelling the families declare — a literal ``.experts.`` misses GLM-4/Laguna's
    ``routed_experts``, and a family added with a new spelling would need a callback edit."""
    attrs = experts_container_attrs()
    assert {"experts", "routed_experts"} <= set(attrs)
    for attr in attrs:
        name = f"model.layers.0.mlp.{attr}.0.up_proj.weight"
        assert _is_expert_param(name), f"container attr {attr!r} not classified as an expert param"


def test_container_exclusions_still_hold():
    """Shared experts and router/gate params stay always-active — the container match must not
    swallow them."""
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
    """A family spec declaring a novel expert spelling makes the callback classify it: a hand-maintained
    copy inside the callback could not know it and would count those params as dense."""
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
    """``import src.callbacks.efficiency`` must leave ``deep_ep`` unimported.

    Dense/DDP/CPU jobs build this callback; ``import deep_ep`` runs ``cuInit`` and latches
    ``CUDA_DEVICE_MAX_CONNECTIONS`` for the whole process.
    """
    if importlib.util.find_spec("deep_ep") is None:
        pytest.skip("deep_ep is not installed — the guard cannot distinguish deferred from absent")

    probe = "import sys\nimport src.callbacks.efficiency\nprint('deep_ep' in sys.modules)\n"
    result = subprocess.run([sys.executable, "-c", probe], cwd=REPO_ROOT, capture_output=True, text=True, check=True)
    assert result.stdout.strip().splitlines()[-1] == "False", (
        "importing EfficiencyCallback pulled in deep_ep — the EP import chain must stay free of the "
        f"native extension. Subprocess output:\n{result.stdout}\n{result.stderr}"
    )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
