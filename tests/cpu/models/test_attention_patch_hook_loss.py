#!/usr/bin/env python
"""Attention patches whose transformers hook is missing must stop the load, not no-op.
Skipped, a masked batch runs an FA4 variant the warm-up does not compile, or a packed row attends across documents."""

import types

import pytest

from halo.models.attention import backend_setup, packed_position_ids
from halo.models.families import mistral4


def _attention_class() -> type:
    """A fresh class per test: an installed injection marks the class's forward for good."""

    class _Attention:
        def forward(self, hidden_states, position_ids=None):
            return hidden_states

    return _Attention


def test_missing_flash_kwargs_builder_raises(monkeypatch):
    monkeypatch.delattr(backend_setup.flash_utils, "_process_flash_attention_kwargs")
    with pytest.raises(AttributeError, match="_process_flash_attention_kwargs"):
        backend_setup.patch_transformers_flash_varlen_int_seqlen()


def test_injection_without_attention_registry_raises_and_leaves_the_forward():
    modeling = types.ModuleType("fake_modeling")
    owner = _attention_class()
    original_forward = owner.forward
    with pytest.raises(AttributeError, match="ALL_ATTENTION_FUNCTIONS"):
        packed_position_ids.install_packed_position_ids_injection(modeling, owner, lambda module: (module,))
    assert owner.forward is original_forward, "a stash nothing re-injects must not be installed"


def test_injection_on_a_forward_taking_position_ids_through_kwargs_raises_and_leaves_the_forward():
    """The stash's signature bind would find none there and hand every attention None, with the patch reading as
    installed."""

    class _KwargsOnly:
        def forward(self, hidden_states, **kwargs):
            return hidden_states

    modeling = types.ModuleType("fake_modeling")
    modeling.ALL_ATTENTION_FUNCTIONS = {}
    original_forward = _KwargsOnly.forward
    with pytest.raises(RuntimeError, match="takes no position_ids"):
        packed_position_ids.install_packed_position_ids_injection(modeling, _KwargsOnly, lambda module: (module,))
    assert _KwargsOnly.forward is original_forward
    assert modeling.ALL_ATTENTION_FUNCTIONS == {}


def test_injection_with_registry_wraps_it():
    modeling = types.ModuleType("fake_modeling")
    modeling.ALL_ATTENTION_FUNCTIONS = {}
    owner = _attention_class()

    assert packed_position_ids.install_packed_position_ids_injection(modeling, owner, lambda module: (module,)) is True
    assert isinstance(modeling.ALL_ATTENTION_FUNCTIONS, packed_position_ids.PositionIdsInjectingRegistry)


def test_missing_mistral4_attention_class_raises(monkeypatch):
    monkeypatch.setattr(mistral4, "m4", types.ModuleType("modeling_mistral4"))
    with pytest.raises(AttributeError, match="Mistral4Attention"):
        mistral4.patch_mistral4_flash_packed_position_ids()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
