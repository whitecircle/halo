#!/usr/bin/env python
"""``_apply_liger_for_standard_models`` must not swallow the appliers' deliberate failures.

The toolkit appliers raise on purpose: ``ImportError`` when Liger is absent (mistral4, zaya),
``ValueError`` for a contradictory option pair, ``NotImplementedError`` for an unimplemented one. A
blanket ``except ImportError: return None`` around the resolve-and-apply block turns any of them into
a log line about an unrelated cause, and the run that asked for Liger trains UNPATCHED.

Run: pytest tests/cpu/kernels/test_liger_orchestrator_propagates.py
"""

import inspect

import pytest
from accelerate import PartialState

import src.kernels.liger.orchestrator as orch
from src.kernels.liger.orchestrator import _apply_liger_for_standard_models
from tests.common.utils import probe_findings

# The orchestrator logs through accelerate's rank-aware logger, which requires an initialized state.
PartialState()

# The exception types the registered toolkit appliers actually raise. Each must reach the caller.
_APPLIER_EXCEPTIONS = (ImportError, ValueError, NotImplementedError, RuntimeError)


@pytest.fixture
def registry():
    """Restore ``_TOOLKIT_LIGER_APPLIERS`` after a test registers a probe applier into it."""
    original = dict(orch._TOOLKIT_LIGER_APPLIERS)
    yield orch._TOOLKIT_LIGER_APPLIERS
    orch._TOOLKIT_LIGER_APPLIERS.clear()
    orch._TOOLKIT_LIGER_APPLIERS.update(original)


@pytest.mark.parametrize("exc_type", _APPLIER_EXCEPTIONS)
def test_applier_failure_reaches_the_caller(registry, exc_type):
    """Whatever an applier raises must propagate, never become a ``None`` return."""

    def _raising_applier(rope=True, cross_entropy=True, rms_norm=True, swiglu=True, geglu=True):
        raise exc_type("applier refused")

    registry["probe_family"] = _raising_applier
    with pytest.raises(exc_type, match="applier refused"):
        _apply_liger_for_standard_models("probe_family", {})


def test_unknown_model_type_still_returns_none(registry):
    """The legitimate no-applier case stays a soft None — only raised failures propagate."""
    assert _apply_liger_for_standard_models("no_such_model_type_at_all", {}) is None


def test_successful_applier_returns_the_filtered_config(registry):
    """Baseline: a working applier is called and its signature-filtered config is reported."""
    seen = {}

    def _applier(rope=True, cross_entropy=True, rms_norm=True):
        seen.update(rope=rope, cross_entropy=cross_entropy, rms_norm=rms_norm)
        return ["RMSNorm"]

    registry["probe_family"] = _applier
    applied = _apply_liger_for_standard_models("probe_family", {"rope": False})
    assert applied == {"rope": False, "cross_entropy": True, "rms_norm": True}
    assert seen == applied, "the applier was not called with the reported config"


def test_every_registered_model_type_applies_without_raising():
    """Every model_type the orchestrator can resolve must apply cleanly under the toolkit defaults.

    Once an applier's failure propagates (above), any upstream applier that cannot honour those defaults
    takes the run down at model load. liger-kernel 0.8.0's Qwen3.5 applier is one (its ``cross_entropy``
    branch imports ``liger_cross_entropy`` from the wrong module while the toolkit defaults that flag ON);
    four more raise ``NotImplementedError`` on ``rope``. Sweeping the whole registry is what turns "we know
    about Qwen3.5" into "we find the next one".

    Run in a subprocess: applying an applier rebinds upstream HF module classes for the rest of the
    process, which would poison every later test in the session.
    """
    script = """
import sys
from accelerate import PartialState

PartialState()
from liger_kernel.transformers.auto_model import MODEL_TYPE_TO_APPLY_LIGER_FN
from src.kernels.liger.orchestrator import _TOOLKIT_LIGER_APPLIERS, _apply_liger_for_standard_models

failures = []
for model_type in sorted({**MODEL_TYPE_TO_APPLY_LIGER_FN, **_TOOLKIT_LIGER_APPLIERS}):
    try:
        if _apply_liger_for_standard_models(model_type, {}) is None:
            failures.append(f"{model_type}: resolved an applier but reported nothing applied")
    except Exception as exc:
        failures.append(f"{model_type}: {type(exc).__name__}: {exc}")
print("FAILURES:" + "|".join(failures))
"""
    failures = probe_findings(script, "FAILURES:")
    assert not failures, "model types whose Liger applier cannot run under the toolkit defaults:\n" + "\n".join(
        failures
    )


@pytest.mark.parametrize("model_type", sorted(orch._TOOLKIT_LIGER_APPLIERS))
def test_real_toolkit_applier_rejections_reach_the_caller(model_type):
    """Drive the REAL registered appliers through the orchestrator with a contradictory option pair.

    ``cross_entropy`` and ``fused_linear_cross_entropy`` are mutually exclusive (both patch the loss), so
    an applier accepting both must raise — and the raise must arrive here rather than being converted
    into a silently unpatched model. Enumerated over the registry and skipped by signature, so a family
    added later is covered without this file naming it.
    """
    applier = orch._TOOLKIT_LIGER_APPLIERS[model_type]
    params = set(inspect.signature(applier).parameters)
    if not {"cross_entropy", "fused_linear_cross_entropy"} <= params:
        pytest.skip(f"{model_type} applier does not accept both loss options")

    with pytest.raises((ValueError, NotImplementedError)):
        _apply_liger_for_standard_models(model_type, {"cross_entropy": True, "fused_linear_cross_entropy": True})


@pytest.mark.parametrize("model_type", sorted(orch._TOOLKIT_LIGER_APPLIERS))
def test_rope_false_appliers_refuse_an_explicit_rope_override(model_type):
    """An applier declaring ``rope=False`` states Liger's generic rotary cannot serve this family.

    The orchestrator only forces ``rope`` off when the user did NOT ask for it, so a YAML
    ``liger_kernel_config: {rope: true}`` survives the filter and reaches the applier. Ignoring it
    there trains a model whose rotary the user believes is fused and is not. Enumerated over the
    registry so a family added later is covered without this file naming it.
    """
    applier = orch._TOOLKIT_LIGER_APPLIERS[model_type]
    rope_param = inspect.signature(applier).parameters.get("rope")
    if rope_param is None or rope_param.default is not False:
        pytest.skip(f"{model_type} applier does not declare rope=False")

    with pytest.raises(NotImplementedError, match="rope"):
        _apply_liger_for_standard_models(model_type, {"rope": True})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
