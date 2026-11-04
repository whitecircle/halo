#!/usr/bin/env python
"""The Ulysses CP patching registry and the zero-patch guard: zero patched layers train on truncated context, so it
raises."""

from __future__ import annotations

import inspect

import pytest
import torch.nn as nn

from halo.distributed.context_parallel import model_rules
from halo.distributed.context_parallel.config import CPConfigError
from halo.distributed.context_parallel.families import ULYSSES_ATTENTION_MAP
from halo.distributed.context_parallel.ulysses_model import apply_ulysses_attention


def test_validator_reads_the_derived_registry():
    """Its own copy would let a class pass validation and silently skip patching."""
    assert model_rules.ULYSSES_ATTENTION_MAP is ULYSSES_ATTENTION_MAP


def test_every_wrapper_takes_the_one_constructor_signature_the_patcher_calls():
    """A family declaring a fourth parameter is a TypeError at patch time, invisible until its GPU suite runs."""
    expected = ("original_attention", "cp_group", "cp_size")
    for name, wrapper_cls in sorted(ULYSSES_ATTENTION_MAP.items()):
        params = tuple(inspect.signature(wrapper_cls.__init__).parameters)[1:]
        assert params == expected, f"{name} -> {wrapper_cls.__name__} takes {params}"


def test_zero_patched_layers_raises():
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU())  # No supported attention class anywhere.
    with pytest.raises(CPConfigError, match="No attention layers"):
        apply_ulysses_attention(model, cp_group=None, cp_size=2)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
