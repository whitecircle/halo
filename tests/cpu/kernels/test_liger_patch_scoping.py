#!/usr/bin/env python
"""The Liger cross-entropy patch must reach ``transformers.loss.loss_utils`` and NOTHING else.

Rebinding ``torch.nn.functional.cross_entropy`` (what upstream Liger's own appliers do) swaps the
kernel under every caller in the process. ``liger_cross_entropy`` serves only 2-D hard-label calls,
so a 3-D or soft-target caller elsewhere — CP's own fp32 loss, a reward head, third-party code —
would start raising, while the applier's docstring still promises the narrow scope.

Run: pytest tests/cpu/kernels/test_liger_patch_scoping.py
"""

import inspect
import re

import pytest
import torch
import torch.nn.functional as F
from transformers.loss import loss_utils

from src.kernels.liger.cross_entropy import (
    _TORCH_CROSS_ENTROPY,
    liger_cross_entropy,
    patch_loss_utils_cross_entropy,
)


@pytest.fixture
def patched():
    """Apply the patch and restore ``loss_utils``' original ``nn`` afterwards."""
    original = loss_utils.nn
    patch_loss_utils_cross_entropy()
    yield
    loss_utils.nn = original


def test_process_wide_cross_entropy_is_untouched(patched):
    """The blast radius: ``F.cross_entropy`` must still be torch's own function."""
    assert F.cross_entropy is _TORCH_CROSS_ENTROPY, (
        "the Liger CE patch rebound torch.nn.functional.cross_entropy process-wide"
    )
    assert torch.nn.functional.cross_entropy is _TORCH_CROSS_ENTROPY

    # A 3-D / soft-target call that liger_cross_entropy cannot serve must keep working elsewhere.
    F.cross_entropy(torch.randn(2, 5, 7), torch.randint(0, 5, (2, 7)))  # (N, C, d) hard labels
    F.cross_entropy(torch.randn(4, 7), torch.softmax(torch.randn(4, 7), dim=-1))  # soft targets


def test_loss_utils_is_actually_routed_through_liger(patched):
    """The patch is not a no-op: HF's single CE call site resolves to the Liger replacement."""
    assert loss_utils.nn.functional.cross_entropy is liger_cross_entropy


def test_hf_loss_really_reaches_the_override():
    """Drive HF's real loss and prove the substituted function is CALLED.

    Asserting only that ``loss_utils.nn.functional.cross_entropy`` is the replacement is a property of
    the proxy, not of transformers: if HF stopped reaching cross-entropy through its module-level ``nn``
    (a rename, or a switch to a direct ``F.cross_entropy`` import) the patch would become a silent no-op
    with that assertion still green. A sentinel installed the same way the patch installs its override
    catches that.
    """
    import src.kernels.liger.cross_entropy as ce_module

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
    """Upstream Liger's appliers reach torch.nn via ``from transformers.loss.loss_utils import nn``.

    After our patch that name IS the proxy, so their ``nn.functional.cross_entropy = ...`` assignment must
    land on ``torch.nn.functional`` rather than inside the proxy — otherwise it would permanently shadow
    our override (killing its CPU fallback) in a module-level singleton no re-patch could repair.
    """
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
    """Applying twice must not stack proxies (the appliers may both run in one process)."""
    before = loss_utils.nn
    patch_loss_utils_cross_entropy()
    assert loss_utils.nn is before


@pytest.mark.parametrize(
    "attribute",
    sorted(set(re.findall(r"\bnn\.([A-Za-z_][A-Za-z0-9_.]*)", inspect.getsource(loss_utils)))),
)
def test_every_nn_attribute_loss_utils_uses_still_resolves(patched, attribute):
    """The override is read-through: every ``nn.<attr>`` in ``loss_utils`` must still resolve.

    Enumerated from the installed ``loss_utils`` source rather than a hand-listed set, so a transformers
    upgrade that reaches for a new ``nn`` attribute fails here instead of at the first loss computation.
    """
    resolved = loss_utils.nn
    for part in attribute.split("."):
        resolved = getattr(resolved, part)
    expected = torch.nn
    for part in attribute.split("."):
        expected = getattr(expected, part)
    if attribute != "functional.cross_entropy":
        assert resolved is expected, f"loss_utils.nn.{attribute} no longer delegates to torch.nn"


def test_hf_loss_path_numerics_unchanged_on_cpu(patched):
    """Liger is Triton/CUDA-only; on CPU the replacement must fall back to torch, bit for bit."""
    torch.manual_seed(0)
    logits = torch.randn(1, 6, 11)
    labels = torch.randint(0, 11, (1, 6))
    patched_loss = loss_utils.ForCausalLMLoss(logits, labels, vocab_size=11)

    loss_utils.nn = torch.nn  # unpatched reference
    reference = loss_utils.ForCausalLMLoss(logits, labels, vocab_size=11)
    assert torch.equal(patched_loss, reference)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
