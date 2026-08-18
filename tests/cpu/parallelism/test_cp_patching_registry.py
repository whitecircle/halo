#!/usr/bin/env python
"""CPU tests for the Ulysses CP patching registry and the zero-patch guard.

* ``CP_SUPPORTED_ATTENTION_CLASSES`` (what the validator accepts) must BE the wrapper registry
  derived from the ``UlyssesAttentionBase`` subclass tree — a listed class without a wrapper would
  pass validation and then silently skip patching (local-chunk-only attention).
* ``patch_attention_for_ulysses`` must RAISE when zero attention layers were patched — a warning
  alone leaves the CP run training on truncated context.
* Every registered wrapper must be constructible the one way the patcher constructs them
  (``wrapper_cls(module, cp_group, cp_size)``); a family growing a fourth parameter breaks patching
  for that family alone.

Run: ``python tests/cpu/parallelism/test_cp_patching_registry.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import inspect

import pytest
import torch.nn as nn

from src.distributed.context_parallel.layers.registry import (
    CP_SUPPORTED_ATTENTION_CLASSES,
    WRAPPER_CLASS_MAP,
    build_wrapper_class_map,
)
from src.distributed.context_parallel.patching import patch_attention_for_ulysses
from src.distributed.context_parallel.validation import UlyssesConfigError


def test_supported_classes_match_wrapper_registry_exactly():
    assert set(WRAPPER_CLASS_MAP) == set(CP_SUPPORTED_ATTENTION_CLASSES)
    # The module-level map is the subclass-tree derivation (nothing hand-patched it since import).
    assert build_wrapper_class_map() == WRAPPER_CLASS_MAP


def test_validator_reads_the_derived_registry():
    """The validator must not carry its own copy of the list — importing it from anywhere else
    is what let the two drift."""
    from src.distributed.context_parallel import validation

    assert validation.CP_SUPPORTED_ATTENTION_CLASSES is CP_SUPPORTED_ATTENTION_CLASSES


def test_every_wrapper_takes_the_one_constructor_signature_the_patcher_calls():
    """``patching.py:67`` builds every wrapper as ``wrapper_cls(module, cp_group, cp_size)``, so a
    family declaring a fourth parameter (or renaming one) is a TypeError at patch time for that
    family only — invisible until someone runs its GPU suite."""
    expected = ("original_attention", "cp_group", "cp_size")
    for name, wrapper_cls in sorted(WRAPPER_CLASS_MAP.items()):
        params = tuple(inspect.signature(wrapper_cls.__init__).parameters)[1:]
        assert params == expected, f"{name} -> {wrapper_cls.__name__} takes {params}"


def test_zero_patched_layers_raises():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())  # no supported attention class anywhere
    with pytest.raises(UlyssesConfigError, match="No attention layers"):
        patch_attention_for_ulysses(model, cp_group=None, cp_size=2, validate=False)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
