#!/usr/bin/env python
"""CPU test for native EP expert-LoRA disable + fail-loud resume guard
(``src/distributed/expert_parallel/base_layer.py``).

Native grouped LoRA on EP experts is NOT peft-managed. For a correct KL reference pass the delta must
be removed; for resume the saved adapter keys must match the rebuilt layer exactly. This pins:

  * :func:`disable_expert_adapters` — context manager that truly removes the LoRA delta (output equals
    the base-only matmul) and restores each layer's PRIOR per-layer enabled flag, with re-entrancy.
  * :func:`make_disable_adapter_ep_aware` — wraps ``PeftModel.disable_adapter`` so it ALSO flips the
    EP layers' ``_expert_adapters_enabled``; idempotent; both flags restored on exit.
  * :meth:`EPMoELayerBase.load_expert_lora_state_dict` — RAISES on a key-set mismatch (the GptOss
    gate_up_proj-vs-gate_proj rename, or an empty state handed to a layer carrying adapters) and
    accepts an empty state only on a layer without expert adapters.

Builds a tiny ``EPMoELayerBase`` SUBCLASS that bypasses the EP/DeepEP ``__init__`` and sets only the
attributes the tested methods touch. ``_grouped_mm`` is overridden to a plain matmul so ``_expert_proj``
runs on CPU with no DeepEP.

Run: ``python tests/cpu/peft/test_ep_lora_kl_disable.py`` (or ``pytest -m cpu``).
"""

from __future__ import annotations

import contextlib

import pytest
import torch
from torch import nn

from src.distributed.expert_parallel.base_layer import (
    disable_expert_adapters,
    make_disable_adapter_ep_aware,
)
from tests.common.ep_stubs import StubEPLayerBase

# --------------------------------------------------------------------------- #
# Tiny EPMoELayerBase stub — bypasses the EP/DeepEP __init__.
# --------------------------------------------------------------------------- #

_E, _K, _N, _R = 2, 4, 4, 2  # experts, in-dim, out-dim, lora rank


class _StubEPLayer(StubEPLayerBase):
    """Minimal concrete EP layer for the LoRA-projection + resume-guard methods.

    Bypasses ``EPMoELayerBase.__init__`` (which builds EPConfig/DeepEP); sets only what
    ``_expert_proj`` / ``_expert_proj_single`` / ``load_expert_lora_state_dict`` read.
    ``_grouped_mm`` is overridden to a plain batched matmul so no DeepEP is needed.
    """

    def __init__(self, *, with_lora: bool = True, attrs=("gate_proj",)):
        super().__init__()
        self.expert_lora_scaling = 2.0
        self.expert_lora_dropout = nn.Identity()
        self._expert_adapters_enabled = True
        self.expert_start, self.expert_end = 0, _E
        self.expert_tp_size = 1
        self.expert_tp_rank = 0

        for attr in attrs:
            # Base expert weight [E, K, N] (matmul convention).
            setattr(self, attr, nn.Parameter(torch.randn(_E, _K, _N)))
            if with_lora:
                # NON-ZERO A and B so the delta is genuinely present (do not rely on B=0).
                a = nn.Parameter(torch.randn(_E, _K, _R))
                b = nn.Parameter(torch.randn(_E, _R, _N))
                setattr(self, f"{attr}_lora_A", a)
                setattr(self, f"{attr}_lora_B", b)
        self._expert_lora_attrs = frozenset(attrs) if with_lora else frozenset()

    def _grouped_mm(self, mat_a, mat_b, *, offs=None, lowp=True):
        return torch.bmm(mat_a, mat_b)


def _tokens():
    torch.manual_seed(0)
    return torch.randn(_E, 3, _K)  # [E, tokens_per_expert, K]


# --------------------------------------------------------------------------- #
# disable_expert_adapters
# --------------------------------------------------------------------------- #


def test_disable_removes_lora_delta_not_just_zero_b():
    """Inside the context the projection must equal the BASE-only matmul (delta gone), and OUTSIDE it
    must differ — proving the toggle, not a coincidental B=0."""
    layer = _StubEPLayer()
    x = _tokens()
    offs = torch.tensor([3, 6])

    base_only = layer._grouped_mm(x, layer.gate_proj)
    with_delta = layer._expert_proj(x, "gate_proj", offs, torch.float32)
    assert not torch.allclose(with_delta, base_only), "LoRA delta is zero — A/B not effective; test is vacuous"

    with disable_expert_adapters(layer):
        disabled = layer._expert_proj(x, "gate_proj", offs, torch.float32)
    assert torch.allclose(disabled, base_only, atol=1e-5), "delta not removed inside disable context"

    # Restored afterwards.
    again = layer._expert_proj(x, "gate_proj", offs, torch.float32)
    assert torch.allclose(again, with_delta, atol=1e-5), "delta not restored after context"


def test_disable_single_loop_path_also_removed():
    """The per-expert loop path (``_expert_proj_single``) must respect the same flag."""
    layer = _StubEPLayer()
    x = torch.randn(3, _K)
    base = x @ layer.gate_proj[0]
    full = layer._expert_proj_single(0, x, "gate_proj")
    assert not torch.allclose(full, base)
    with disable_expert_adapters(layer):
        assert torch.allclose(layer._expert_proj_single(0, x, "gate_proj"), base, atol=1e-5)


def test_disable_restores_prior_per_layer_state_not_blanket_true():
    """One layer is pre-set disabled; after the context every layer's flag equals its PRIOR value —
    a blanket re-enable would wrongly flip the pre-disabled layer to True."""
    on_layer = _StubEPLayer()
    off_layer = _StubEPLayer()
    off_layer._expert_adapters_enabled = False
    container = nn.ModuleList([on_layer, off_layer])

    with disable_expert_adapters(container):
        assert on_layer._expert_adapters_enabled is False
        assert off_layer._expert_adapters_enabled is False

    assert on_layer._expert_adapters_enabled is True
    assert off_layer._expert_adapters_enabled is False, "prior-disabled layer wrongly re-enabled"


def test_disable_reentrant_inner_exit_does_not_reenable():
    """Nested contexts: inner exit must NOT prematurely re-enable; only the outermost exit restores."""
    layer = _StubEPLayer()
    with disable_expert_adapters(layer):
        with disable_expert_adapters(layer):
            assert layer._expert_adapters_enabled is False
        # inner exit — must stay disabled because the outer context is still active
        assert layer._expert_adapters_enabled is False
    assert layer._expert_adapters_enabled is True


# --------------------------------------------------------------------------- #
# make_disable_adapter_ep_aware
# --------------------------------------------------------------------------- #


class _StubPeftModel(nn.Module):
    """Stub PeftModel: a real ``@contextmanager disable_adapter`` flipping a flag, plus EP layers
    reachable through ``.modules()``."""

    def __init__(self):
        super().__init__()
        self.peft_adapter_active = True
        self.ep = _StubEPLayer()

    @contextlib.contextmanager
    def disable_adapter(self):
        self.peft_adapter_active = False
        try:
            yield
        finally:
            self.peft_adapter_active = True


def test_ep_aware_disable_flips_both_flags_and_restores():
    pm = _StubPeftModel()
    make_disable_adapter_ep_aware(pm)

    assert pm.peft_adapter_active is True
    assert pm.ep._expert_adapters_enabled is True
    with pm.disable_adapter():
        assert pm.peft_adapter_active is False, "peft adapter not disabled"
        assert pm.ep._expert_adapters_enabled is False, "EP expert adapter not disabled"
    assert pm.peft_adapter_active is True
    assert pm.ep._expert_adapters_enabled is True


def test_ep_aware_is_idempotent():
    """Calling twice must not double-wrap: the marker is set and the second call returns early."""
    pm = _StubPeftModel()
    make_disable_adapter_ep_aware(pm)
    wrapped_once = pm.disable_adapter
    make_disable_adapter_ep_aware(pm)
    assert pm.disable_adapter is wrapped_once, "double-wrapped — idempotence broken"
    assert pm._ep_disable_adapter_patched is True
    # And it still works after the second call.
    with pm.disable_adapter():
        assert pm.ep._expert_adapters_enabled is False


# --------------------------------------------------------------------------- #
# load_expert_lora_state_dict — fail-loud key guard
# --------------------------------------------------------------------------- #


def test_load_raises_on_key_set_mismatch():
    """A layer adapting {gate_proj, down_proj} fed a state keyed for the GptOss-renamed gate_up_proj
    must RAISE — the alternative is silently rebuilding zero-init adapters."""
    layer = _StubEPLayer(attrs=("gate_proj", "down_proj"))
    # Saved state from a checkpoint with the fused gate_up_proj naming.
    bad_state = {
        "experts.gate_up_proj.lora_A": torch.randn(_E, _K, _R),
        "experts.gate_up_proj.lora_B": torch.randn(_E, _R, _N),
    }
    with pytest.raises(RuntimeError, match="Expert-LoRA resume mismatch"):
        layer.load_expert_lora_state_dict(bad_state)


def test_load_roundtrips_matching_keys():
    """A correctly-keyed state copies into the live adapters (the guard does not over-reject valid keys)."""
    layer = _StubEPLayer(attrs=("gate_proj",))
    new_a = torch.randn(_E, _K, _R)
    new_b = torch.randn(_E, _R, _N)
    layer.load_expert_lora_state_dict({"experts.gate_proj.lora_A": new_a, "experts.gate_proj.lora_B": new_b})
    assert torch.allclose(layer.gate_proj_lora_A.data, new_a)
    assert torch.allclose(layer.gate_proj_lora_B.data, new_b)


def test_load_empty_state_raises_the_descriptive_mismatch():
    """A layer that rebuilt adapters but received no saved keys is the same resume mismatch as a
    wrongly-keyed state, and must raise the actionable message — not the bare ``KeyError`` the copy
    loop would otherwise produce, which names neither the layer nor the remedy."""
    layer = _StubEPLayer(attrs=("gate_proj",))
    with pytest.raises(RuntimeError, match="Expert-LoRA resume mismatch"):
        layer.load_expert_lora_state_dict({})


def test_load_empty_state_on_a_layer_without_adapters_is_a_no_op():
    """The guard keys on the key SETS, so a layer with no expert-LoRA attrs accepts an empty state."""
    layer = _StubEPLayer(attrs=())
    layer.load_expert_lora_state_dict({})


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
