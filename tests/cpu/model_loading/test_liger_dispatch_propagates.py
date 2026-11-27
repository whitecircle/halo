#!/usr/bin/env python
"""``_run_resolved_applier`` must not swallow the appliers' deliberate failures, or a run asking for
Liger trains unpatched."""

import inspect

import pytest

from halo.model_loading.liger import dispatch
from halo.model_loading.liger.dispatch import _run_resolved_applier
from tests.common.repo import probe_findings

# Each must reach the caller.
_APPLIER_EXCEPTIONS = (ImportError, ValueError, NotImplementedError, RuntimeError)


@pytest.fixture
def registry():
    """Restores ``_TOOLKIT_LIGER_APPLIERS``."""
    original = dict(dispatch._TOOLKIT_LIGER_APPLIERS)
    yield dispatch._TOOLKIT_LIGER_APPLIERS
    dispatch._TOOLKIT_LIGER_APPLIERS.clear()
    dispatch._TOOLKIT_LIGER_APPLIERS.update(original)


@pytest.mark.parametrize("exc_type", _APPLIER_EXCEPTIONS)
def test_applier_failure_reaches_the_caller(registry, exc_type):
    def _raising_applier(rope=True, cross_entropy=True, rms_norm=True, swiglu=True, geglu=True):
        raise exc_type("applier refused")

    registry["probe_family"] = _raising_applier
    with pytest.raises(exc_type, match="applier refused"):
        _run_resolved_applier("probe_family", {})


def test_unknown_model_type_still_returns_none(registry):
    assert _run_resolved_applier("no_such_model_type_at_all", {}) is None


def test_successful_applier_returns_the_filtered_config(registry):
    """Reports the signature-filtered config."""
    seen = {}

    def _applier(rope=True, cross_entropy=True, rms_norm=True):
        seen.update(rope=rope, cross_entropy=cross_entropy, rms_norm=rms_norm)
        return ["RMSNorm"]

    registry["probe_family"] = _applier
    applied = _run_resolved_applier("probe_family", {"rope": False})
    assert applied == {"rope": False, "cross_entropy": True, "rms_norm": True}
    assert seen == applied, "the applier was not called with the reported config"


def test_every_registered_model_type_applies_without_raising():
    """Resolved as a checkpoint of that type does, a wrapper with its tower as fallback.
    Subprocess: applying an applier rebinds upstream classes for the rest of the process."""
    script = """
import sys
from liger_kernel.transformers.auto_model import MODEL_TYPE_TO_APPLY_LIGER_FN
from transformers.models.auto.configuration_auto import CONFIG_MAPPING
from halo.model_loading.liger.dispatch import _TOOLKIT_LIGER_APPLIERS, _run_resolved_applier, _liger_model_types

failures = []
for model_type in sorted({**MODEL_TYPE_TO_APPLY_LIGER_FN, **_TOOLKIT_LIGER_APPLIERS}):
    fallback = _liger_model_types(CONFIG_MAPPING[model_type]())[1] if model_type in CONFIG_MAPPING else None
    try:
        if _run_resolved_applier(model_type, {}, fallback) is None:
            failures.append(f"{model_type}: resolved an applier but reported nothing applied")
    except Exception as exc:
        failures.append(f"{model_type}: {type(exc).__name__}: {exc}")
print("FAILURES:" + "|".join(failures))
"""
    failures = probe_findings(script, "FAILURES:")
    assert not failures, "model types whose Liger applier cannot run under the toolkit defaults:\n" + "\n".join(
        failures
    )


def test_real_toolkit_applier_rejections_reach_the_caller(subtests):
    """``cross_entropy`` and ``fused_linear_cross_entropy`` both patch the loss, so accepting both must raise."""
    both_losses = [
        model_type
        for model_type, applier in sorted(dispatch._TOOLKIT_LIGER_APPLIERS.items())
        if {"cross_entropy", "fused_linear_cross_entropy"} <= set(inspect.signature(applier).parameters)
    ]
    assert both_losses, "no toolkit Liger applier accepts both loss options"
    for model_type in both_losses:
        with subtests.test(model_type):
            with pytest.raises((ValueError, NotImplementedError)):
                _run_resolved_applier(model_type, {"cross_entropy": True, "fused_linear_cross_entropy": True})


def test_rope_false_appliers_refuse_an_explicit_rope_override(subtests):
    """``rope=False`` means Liger's rotary cannot serve this family; dispatch only forces it off when not asked, so
    a YAML request reaches the applier."""
    rope_off = [
        model_type
        for model_type, applier in sorted(dispatch._TOOLKIT_LIGER_APPLIERS.items())
        if (rope := inspect.signature(applier).parameters.get("rope")) is not None and rope.default is False
    ]
    assert rope_off, "no toolkit Liger applier declares rope=False"
    for model_type in rope_off:
        with subtests.test(model_type):
            with pytest.raises(NotImplementedError, match="rope"):
                _run_resolved_applier(model_type, {"rope": True})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
