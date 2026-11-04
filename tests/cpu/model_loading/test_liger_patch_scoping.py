#!/usr/bin/env python
"""The Liger CE patch reaches ``transformers.loss.loss_utils`` and nothing else, and no application rebinds
``torch.nn.LayerNorm``. A process-wide rebind breaks any 3-D or soft-target caller, CP's own fp32 loss included."""

import inspect
import re

import pytest
import torch
import torch.nn.functional as F
from liger_kernel.transformers.auto_model import MODEL_TYPE_TO_APPLY_LIGER_FN
from transformers.loss import loss_utils

import halo.model_loading.liger.cross_entropy as ce_module
from halo.model_loading.liger.cross_entropy import (
    _TORCH_CROSS_ENTROPY,
    liger_cross_entropy,
    patch_loss_utils_cross_entropy,
)
from halo.model_loading.liger.dispatch import _run_resolved_applier
from tests.common.repo import probe_findings


@pytest.fixture
def patched():
    original = loss_utils.nn
    patch_loss_utils_cross_entropy()
    yield
    loss_utils.nn = original


def test_process_wide_cross_entropy_is_untouched(patched):
    assert F.cross_entropy is _TORCH_CROSS_ENTROPY, (
        "the Liger CE patch rebound torch.nn.functional.cross_entropy process-wide"
    )
    assert torch.nn.functional.cross_entropy is _TORCH_CROSS_ENTROPY

    # Calls liger_cross_entropy cannot serve must keep working elsewhere.
    F.cross_entropy(torch.randn(2, 5, 7), torch.randint(0, 5, (2, 7)))
    F.cross_entropy(torch.randn(4, 7), torch.softmax(torch.randn(4, 7), dim=-1))


def test_loss_utils_is_actually_routed_through_liger(patched):
    assert loss_utils.nn.functional.cross_entropy is liger_cross_entropy


def test_hf_loss_really_reaches_the_override():
    """If HF stopped reaching CE through its module-level ``nn``, the patch would silently no-op with a proxy-only
    assert still green."""
    calls = []

    def _sentinel(*args, **kwargs):
        calls.append(True)
        return _TORCH_CROSS_ENTROPY(*args, **kwargs)

    original = loss_utils.nn
    loss_utils.nn = ce_module._AttributeOverride(
        torch.nn, functional=ce_module._AttributeOverride(torch.nn.functional, cross_entropy=_sentinel)
    )
    try:
        loss_utils.ForCausalLMLoss(torch.randn(1, 4, 9), torch.randint(0, 9, (1, 4)), vocab_size=9)
    finally:
        loss_utils.nn = original
    assert calls, "transformers' loss path no longer reaches cross-entropy through loss_utils.nn"


def test_a_later_writer_cannot_shadow_the_override(patched):
    """Their ``nn.functional.cross_entropy = ...`` must land on ``torch.nn.functional``, or it permanently shadows
    our override in a module-level singleton."""
    marker = object()
    try:
        loss_utils.nn.functional.cross_entropy = marker
        assert torch.nn.functional.cross_entropy is marker, "the write did not reach torch.nn.functional"
        assert loss_utils.nn.functional.cross_entropy is liger_cross_entropy, (
            "a later writer shadowed the scoped override inside the proxy"
        )
    finally:
        torch.nn.functional.cross_entropy = _TORCH_CROSS_ENTROPY


def test_patch_is_idempotent(patched):
    """The appliers may both run in one process."""
    before = loss_utils.nn
    patch_loss_utils_cross_entropy()
    assert loss_utils.nn is before


@pytest.mark.parametrize(
    "attribute",
    sorted(set(re.findall(r"\bnn\.([A-Za-z_][A-Za-z0-9_.]*)", inspect.getsource(loss_utils)))),
)
def test_every_nn_attribute_loss_utils_uses_still_resolves(patched, attribute):
    """Enumerated from the source, so a transformers upgrade reaching for a new ``nn`` attribute fails here."""
    resolved = loss_utils.nn
    for part in attribute.split("."):
        resolved = getattr(resolved, part)
    expected = torch.nn
    for part in attribute.split("."):
        expected = getattr(expected, part)
    if attribute != "functional.cross_entropy":
        assert resolved is expected, f"loss_utils.nn.{attribute} no longer delegates to torch.nn"


def test_an_upstream_family_leaves_the_process_wide_function_alone():
    """Otherwise a reward head calling ``F.cross_entropy`` twice on one logits tensor reads gradients Liger wrote in
    place. Subprocess: applying a real upstream applier rebinds the family's classes for the process."""
    script = """
import torch.nn.functional as F
from transformers.loss import loss_utils
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from halo.model_loading.liger.cross_entropy import _TORCH_CROSS_ENTROPY, liger_cross_entropy
from halo.model_loading.liger.dispatch import apply_liger_kernel

applied = apply_liger_kernel(CONFIG_MAPPING["llama"]())
failures = []
if not applied or applied.get("cross_entropy") is not True:
    failures.append(f"CE not applied for llama: {applied}")
if F.cross_entropy is not _TORCH_CROSS_ENTROPY:
    failures.append("torch.nn.functional.cross_entropy was rebound process-wide")
if loss_utils.nn.functional.cross_entropy is not liger_cross_entropy:
    failures.append("transformers' loss path is not routed through the scoped patch")
print("FAILURES:" + "|".join(failures))
"""
    failures = probe_findings(script, "FAILURES:")
    assert not failures, "\n".join(failures)


def test_no_upstream_family_rebinds_the_process_wide_layer_norm():
    """Anti-vacuity: called raw, before the model exists, at least one of those appliers does rebind it.
    Subprocess: the appliers rebind module classes."""
    script = """
import inspect

import torch
from liger_kernel.transformers.auto_model import MODEL_TYPE_TO_APPLY_LIGER_FN
from transformers.models.auto.configuration_auto import CONFIG_MAPPING

from halo.model_loading.liger.dispatch import (
    _TOOLKIT_LIGER_APPLIERS,
    _liger_model_types,
    _resolve_applier,
    apply_liger_kernel,
)

TORCH_LAYER_NORM = torch.nn.modules.normalization.LayerNorm
swept = {}
for model_type in sorted(MODEL_TYPE_TO_APPLY_LIGER_FN):
    if model_type not in CONFIG_MAPPING:
        continue
    config = CONFIG_MAPPING[model_type]()
    resolved, applier = _resolve_applier(*_liger_model_types(config))
    if resolved not in _TOOLKIT_LIGER_APPLIERS and "layer_norm" in inspect.signature(applier).parameters:
        swept[model_type] = (config, applier)

failures = [] if swept else ["no upstream family declares layer_norm: the sweep tests nothing"]
for model_type, (config, _) in swept.items():
    try:
        applied = apply_liger_kernel(config)
    except Exception as exc:
        failures.append(f"{model_type}: the dispatch raised {type(exc).__name__}: {exc}")
        applied = None
    if torch.nn.LayerNorm is not TORCH_LAYER_NORM:
        failures.append(f"{model_type}: the dispatch rebound torch.nn.LayerNorm")
    elif applied is not None and applied.get("layer_norm") is not False:
        failures.append(f"{model_type}: the applied config does not record layer_norm off: {applied}")
    torch.nn.LayerNorm = TORCH_LAYER_NORM

rebinding = []
for model_type, (_, applier) in swept.items():
    applier(model=None)
    if torch.nn.LayerNorm is not TORCH_LAYER_NORM:
        rebinding.append(model_type)
    torch.nn.LayerNorm = TORCH_LAYER_NORM
if not rebinding:
    failures.append("premise: no swept applier rebinds torch.nn.LayerNorm when called raw")
print("FAILURES:" + "|".join(failures))
"""
    failures = probe_findings(script, "FAILURES:")
    assert not failures, "\n".join(failures)


def test_an_applier_that_rebinds_the_layer_norm_is_refused(monkeypatch):
    """It fails the load instead of reaching every later build."""

    class _SwappedLayerNorm(torch.nn.LayerNorm):
        pass

    def rebinding_applier(rope: bool = True, model=None):
        torch.nn.LayerNorm = _SwappedLayerNorm

    monkeypatch.setattr(torch.nn, "LayerNorm", torch.nn.LayerNorm)  # Restored after the test.
    monkeypatch.setitem(MODEL_TYPE_TO_APPLY_LIGER_FN, "halo_test_layer_norm_swap", rebinding_applier)
    with pytest.raises(RuntimeError, match="rebound torch.nn.LayerNorm to _SwappedLayerNorm"):
        _run_resolved_applier("halo_test_layer_norm_swap", {})


def test_hf_loss_path_numerics_unchanged_on_cpu(patched):
    """Liger is Triton/CUDA-only."""
    torch.manual_seed(0)
    logits = torch.randn(1, 6, 11)
    labels = torch.randint(0, 11, (1, 6))
    patched_loss = loss_utils.ForCausalLMLoss(logits, labels, vocab_size=11)

    loss_utils.nn = torch.nn  # Unpatched reference.
    reference = loss_utils.ForCausalLMLoss(logits, labels, vocab_size=11)
    assert torch.equal(patched_loss, reference)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
