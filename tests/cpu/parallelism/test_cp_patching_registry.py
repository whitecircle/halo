#!/usr/bin/env python
"""CPU tests for the Ulysses CP patching registry and the zero-patch guard.

* The validator must read ``ULYSSES_ATTENTION_MAP``, the wrapper registry derived from the
  ``UlyssesAttention`` subclass tree, rather than a copy — a class listed there without a wrapper
  would pass validation and then silently skip patching (local-chunk-only attention).
* ``apply_ulysses_attention`` must RAISE when zero attention layers were patched — a warning
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

from halo.distributed.context_parallel import model_rules
from halo.distributed.context_parallel.config import CPConfigError
from halo.distributed.context_parallel.families import ULYSSES_ATTENTION_MAP
from halo.distributed.context_parallel.ulysses_model import apply_ulysses_attention


def test_validator_reads_the_derived_registry():
    """The validator must not carry its own copy of the list — importing it from anywhere else
    lets the two drift."""
    assert model_rules.ULYSSES_ATTENTION_MAP is ULYSSES_ATTENTION_MAP


def test_every_wrapper_takes_the_one_constructor_signature_the_patcher_calls():
    """``apply_ulysses_attention`` builds every wrapper as ``wrapper_cls(module, cp_group,
    cp_size)``, so a family declaring a fourth parameter (or renaming one) is a TypeError at patch
    time for that family only — invisible until someone runs its GPU suite."""
    expected = ("original_attention", "cp_group", "cp_size")
    for name, wrapper_cls in sorted(ULYSSES_ATTENTION_MAP.items()):
        params = tuple(inspect.signature(wrapper_cls.__init__).parameters)[1:]
        assert params == expected, f"{name} -> {wrapper_cls.__name__} takes {params}"


def test_zero_patched_layers_raises():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())  # no supported attention class anywhere
    with pytest.raises(CPConfigError, match="No attention layers"):
        apply_ulysses_attention(model, cp_group=None, cp_size=2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
