#!/usr/bin/env python
"""The fused-GLU probes must recognize the activation a REAL transformers block carries.

The activation-hardcoding kernels (`fused_silu_mul`, `fused_gelu_tanh_mul`, DeepSeek-V4's compiled
clamped SwiGLU) are only valid where the wrapped block's activation really is the one they implement,
so the gates deciding that are load-bearing for throughput: answering False everywhere is numerically
invisible and silently drops every GLM-4 Lite / Laguna / DeepSeek-V4 / Gemma-4 MoE onto the eager
combine. Answering True wrongly is worse — it changes the activation on every expert.

A nominal test cannot hold that line. ``ACT2FN["silu"]`` is `transformers.activations.SiLUActivation`,
a bare `nn.Module` subclass that is neither `nn.SiLU` nor `F.silu`, so a hand-built `nn.SiLU()`
stand-in would accept a predicate the real object fails. These tests therefore probe the REAL registry
objects, and sweep the WHOLE registry for the negative side so a near-SiLU entry cannot be waved
through.

Run: ``python tests/cpu/kernels/test_silu_activation_probe.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import functools

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN

from src.kernels.fused_glu import is_gelu_tanh_activation, is_silu_activation, silu_mul_eager

# The only registry entries that resolve to each probed activation. Every other name must be rejected;
# a new alias lands here only alongside a deliberate decision that the kernels may serve it.
SILU_ALIASES = frozenset({"silu", "swish"})
GELU_TANH_ALIASES = frozenset({"gelu_accurate", "gelu_fast", "gelu_new", "gelu_python_tanh", "gelu_pytorch_tanh"})


def test_the_activation_a_real_block_carries_is_recognized():
    """`ACT2FN["silu"]` is what every HF MoE block actually holds, so the probe must accept it."""
    act_fn = ACT2FN["silu"]
    assert is_silu_activation(act_fn), (
        f"is_silu_activation rejected the real ACT2FN['silu'] ({type(act_fn).__name__}) — every "
        "SiLU-hardcoding kernel just silently turned itself off"
    )


def test_a_nominal_check_would_fail_so_the_probe_must_stay_behavioural():
    """Anti-vacuity for the test above: without this, a transformers release that re-aliased silu back
    to `nn.SiLU` would make the acceptance trivially true and stop proving anything."""
    act_fn = ACT2FN["silu"]
    assert not isinstance(act_fn, nn.SiLU), "ACT2FN['silu'] is an nn.SiLU again — re-check the probe's value"
    assert act_fn is not F.silu
    # ...and it is nonetheless numerically SiLU, which is the whole reason a behavioural probe works.
    probe = torch.tensor([-3.0, -0.5, 0.0, 0.5, 3.0])
    assert torch.equal(act_fn(probe), F.silu(probe))


@pytest.mark.parametrize("name", sorted(ACT2FN.keys()))
def test_every_registered_activation_is_classified_by_behaviour(name):
    """Swept over the whole registry, not a chosen few: `hardswish`, `mish` and the `gelu_*` family are
    close enough to SiLU that a sloppy tolerance would wave them through and mis-fire the kernel."""
    assert is_silu_activation(ACT2FN[name]) is (name in SILU_ALIASES)


@pytest.mark.parametrize("name", sorted(ACT2FN.keys()))
def test_every_registered_activation_is_classified_against_gelu_tanh_too(name):
    """The same sweep for the second kernel. Exact (erf) `gelu` is the dangerous neighbour here — it
    agrees with the tanh approximation to ~1e-3, so anything but bitwise equality would arm the GeGLU
    kernel on a model that does not use it and change every expert's activation silently.

    A name whose alias set is wrong fails LOUDLY here rather than at the throughput level: if a future
    transformers release re-points one of these aliases, this test names it.
    """
    assert is_gelu_tanh_activation(ACT2FN[name]) is (name in GELU_TANH_ALIASES)


def test_the_two_probes_never_accept_the_same_activation():
    """`resolve_fused_glu_mul` returns the FIRST matching kernel, so overlapping probes would make the
    dispatch order load-bearing instead of the activation."""
    assert not (SILU_ALIASES & GELU_TANH_ALIASES)
    for name in ACT2FN:
        act_fn = ACT2FN[name]  # index, don't iterate values: the registry mints per access
        assert not (is_silu_activation(act_fn) and is_gelu_tanh_activation(act_fn)), (
            f"ACT2FN[{name!r}] matched both probes — the fused-GLU dispatch is now order-dependent"
        )


@pytest.mark.parametrize(
    "act_fn",
    [
        F.silu,
        nn.SiLU(),
        nn.SiLU(inplace=True),  # mutates its input; the probe must still compare against a pristine reference
        functools.partial(F.silu),
        lambda x: x * torch.sigmoid(x),
        lambda x: x / (1.0 + torch.exp(-x)),
    ],
    ids=["functional", "module", "module_inplace", "partial", "sigmoid_product", "logistic_quotient"],
)
def test_any_spelling_of_silu_is_accepted(act_fn):
    """Callers hold whatever the wrapped block hands them — a function, a module, a partial."""
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
    """The kernels are an optimization: an activation the probe cannot vouch for must go eager, never
    get substituted. Every failure mode answers False instead of raising out of the caller's `__init__`."""
    assert is_silu_activation(act_fn) is False


def test_the_verdict_is_deterministic_and_per_activation():
    """`ACT2FN[...]` mints a fresh instance per access, so the predicate is asked once per experts
    module. Repeat calls must agree, and no verdict may carry across activations — the guard that keeps
    any future memoization from returning one family's answer for another's activation."""
    silu, gelu = ACT2FN["silu"], ACT2FN["gelu"]
    assert ACT2FN["silu"] is not silu  # anti-vacuity: the registry really does mint per access
    assert [is_silu_activation(silu) for _ in range(3)] == [True, True, True]
    assert is_silu_activation(gelu) is False
    assert is_silu_activation(ACT2FN["silu"]) is True  # a second, distinct SiLUActivation object
    assert is_silu_activation(silu) is True


def test_an_ambient_meta_default_device_does_not_break_the_probe():
    """EP wrappers are constructed near meta-device scopes (the lazy expert loader). The probe pins its
    own tensor to CPU, so an ambient `torch.device("meta")` must neither flip the verdict nor raise out
    of a layer's `__init__` — `torch.equal` has no Meta kernel."""
    with torch.device("meta"):
        assert is_silu_activation(ACT2FN["silu"]) is True
        assert is_silu_activation(ACT2FN["gelu"]) is False


def test_the_kernel_the_gate_guards_matches_the_activation_it_accepts():
    """Ties the gate to the thing it enables: the eager reference the kernel is checked against must
    reproduce `act_fn(gate) * up` for the activation the probe accepted."""
    act_fn = ACT2FN["silu"]
    torch.manual_seed(0)
    gate, up = torch.randn(8, 16), torch.randn(8, 16)

    assert torch.equal(silu_mul_eager(gate, up), act_fn(gate) * up)
    assert not torch.allclose(silu_mul_eager(gate, up), ACT2FN["gelu"](gate) * up)  # the combine is load-bearing


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
