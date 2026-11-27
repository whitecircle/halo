#!/usr/bin/env python
"""The fused-GLU probes recognize the activation a real block carries. ``ACT2FN["silu"]`` is neither `nn.SiLU` nor
`F.silu`, so a stand-in would accept a predicate the real object fails."""

from __future__ import annotations

import functools

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from halo.kernels.fused_glu import is_gelu_tanh_activation, is_silu_activation, silu_mul_eager

# Every other name must be rejected; a new alias lands here only with a decision that the kernels may serve it.
SILU_ALIASES = frozenset({"silu", "swish"})
GELU_TANH_ALIASES = frozenset({"gelu_accurate", "gelu_fast", "gelu_new", "gelu_python_tanh", "gelu_pytorch_tanh"})


def test_every_registered_activation_is_classified_by_behaviour(subtests):
    """`hardswish`, `mish` and the `gelu_*` family are close enough that a sloppy tolerance would mis-fire the
    kernel."""
    assert ACT2FN, "transformers registers no activation"
    for name in sorted(ACT2FN.keys()):
        with subtests.test(name):
            assert is_silu_activation(ACT2FN[name]) is (name in SILU_ALIASES)


def test_every_registered_activation_is_classified_against_gelu_tanh_too(subtests):
    """Exact `gelu` agrees with the tanh approximation to ~1e-3, so anything but bitwise equality would arm GeGLU on
    the wrong model."""
    assert ACT2FN, "transformers registers no activation"
    for name in sorted(ACT2FN.keys()):
        with subtests.test(name):
            assert is_gelu_tanh_activation(ACT2FN[name]) is (name in GELU_TANH_ALIASES)


@pytest.mark.parametrize(
    "act_fn",
    [
        F.silu,
        nn.SiLU(),
        nn.SiLU(inplace=True),  # Mutates its input; the probe must still compare against a pristine reference.
        functools.partial(F.silu),
        lambda x: x * torch.sigmoid(x),
        lambda x: x / (1.0 + torch.exp(-x)),
    ],
    ids=["functional", "module", "module_inplace", "partial", "sigmoid_product", "logistic_quotient"],
)
def test_any_spelling_of_silu_is_accepted(act_fn):
    """Callers hold whatever the wrapped block hands them."""
    assert is_silu_activation(act_fn)


@pytest.mark.parametrize(
    "act_fn",
    [
        None,
        "silu",
        object(),
        lambda x: (_ for _ in ()).throw(RuntimeError("this activation needs kwargs")),
        lambda x: (F.silu(x), None),
        lambda x: F.silu(x).bfloat16(),
        lambda x: F.silu(x).sum(),
        lambda x: F.silu(x) + 1e-7,
    ],
    ids=["none", "string", "not_callable", "raises", "returns_tuple", "downcasts", "reduces", "off_by_epsilon"],
)
def test_anything_unrecognizable_fails_closed(act_fn):
    """Every failure answers False instead of raising out of the caller's `__init__`."""
    assert is_silu_activation(act_fn) is False


def test_an_ambient_meta_default_device_does_not_break_the_probe():
    """EP layers are built near meta-device scopes, and `torch.equal` has no Meta kernel."""
    with torch.device("meta"):
        assert is_silu_activation(ACT2FN["silu"]) is True
        assert is_silu_activation(ACT2FN["gelu"]) is False


def test_the_kernel_the_gate_guards_matches_the_activation_it_accepts():
    """The eager reference the kernel is checked against must reproduce `act_fn(gate) * up`."""
    act_fn = ACT2FN["silu"]
    torch.manual_seed(0)
    gate, up = torch.randn(8, 16), torch.randn(8, 16)

    assert torch.equal(silu_mul_eager(gate, up), act_fn(gate) * up)
    assert not torch.allclose(silu_mul_eager(gate, up), ACT2FN["gelu"](gate) * up)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
