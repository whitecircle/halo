#!/usr/bin/env python
"""Attention patches whose transformers hook is missing must warn, not no-op in silence.

Each patch fixes a defect that stays invisible when it is skipped: FA4's varlen backward recompiles
every step, or a packed row attends across document boundaries on flash attention. A transformers
upgrade that renames the private hook would otherwise drop the fix without a line in the log.

Run: pytest tests/cpu/models/test_attention_patch_hook_loss.py
"""

import logging
import types

import pytest
from accelerate import PartialState

from src.models.patches import attention

PartialState()  # the patches log through accelerate's logger

_LOGGER = "src.models.patches.attention"


def _attention_class() -> type:
    """A fresh class per test: an installed injection marks the class's forward for good."""

    class _Attention:
        def forward(self, hidden_states, position_ids=None):
            return hidden_states

    return _Attention


def _warnings(caplog, needle: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.WARNING and needle in r.getMessage()]


def test_missing_flash_kwargs_builder_warns(monkeypatch, caplog):
    monkeypatch.delattr(attention.flash_utils, "_process_flash_attention_kwargs")
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        attention.patch_transformers_flash_varlen_int_seqlen()
    assert _warnings(caplog, "recompiles on every step")


def test_injection_without_attention_registry_warns_and_leaves_the_forward(caplog):
    modeling = types.ModuleType("fake_modeling")
    owner = _attention_class()
    original_forward = owner.forward
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        installed = attention.install_packed_position_ids_injection(modeling, owner, lambda module: (module,))

    assert installed is False
    assert owner.forward is original_forward, "a stash nothing re-injects must not be installed"
    assert _warnings(caplog, "attend across document boundaries")


def test_injection_with_registry_wraps_it():
    modeling = types.ModuleType("fake_modeling")
    modeling.ALL_ATTENTION_FUNCTIONS = {}
    owner = _attention_class()

    assert attention.install_packed_position_ids_injection(modeling, owner, lambda module: (module,)) is True
    assert isinstance(modeling.ALL_ATTENTION_FUNCTIONS, attention.PositionIdsInjectingRegistry)


def test_missing_mistral4_attention_class_warns(monkeypatch, caplog):
    monkeypatch.setattr(attention, "m4", types.ModuleType("modeling_mistral4"))
    with caplog.at_level(logging.WARNING, logger=_LOGGER):
        attention.patch_mistral4_flash_packed_position_ids()
    assert _warnings(caplog, "a packed Mistral4 row would attend across document boundaries")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
