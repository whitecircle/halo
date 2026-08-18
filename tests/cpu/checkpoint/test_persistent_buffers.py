"""CPU tests for ``persistent_buffers`` — the shared save-path helper.

Every gathered checkpoint save (EP via ``_persistent_non_ep_buffers``, TP via ``get_tp_state_dict``,
plain FSDP2/CP via ``gather_full_state_dict``) routes non-parameter state through this helper. A
param-only save loop silently drops PERSISTENT buffers (e.g. a Gemma4 ``layer_scalar`` residual
scalar, vision ``std_scale``) — corrupting every layer on reload — while NON-persistent buffers
(rotary caches, causal masks) must be excluded (HF recomputes them). This is model-agnostic: any
architecture's persistent buffers must be captured and its non-persistent ones skipped.

    python tests/cpu/checkpoint/test_persistent_buffers.py
"""

import sys

import pytest
import torch
import torch.nn as nn

from src.models.structure import persistent_buffers


class _Block(nn.Module):
    """A layer with a param, a PERSISTENT buffer (a residual scalar), and a NON-persistent one."""

    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)
        self.register_buffer("layer_scalar", torch.ones(1))  # persistent (default) → must be saved
        self.register_buffer("rope_cache", torch.zeros(4), persistent=False)  # recomputed → must be skipped


class _Model(nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([_Block() for _ in range(n_layers)])
        self.register_buffer("top_scalar", torch.ones(1))  # persistent top-level buffer
        self.register_buffer("mask", torch.zeros(2), persistent=False)


def test_captures_persistent_skips_nonpersistent_and_params():
    m = _Model(n_layers=3)
    names = {name for name, _ in persistent_buffers(m)}
    # Every persistent buffer (top-level + per-layer) is captured.
    assert names == {"top_scalar", *(f"layers.{i}.layer_scalar" for i in range(3))}
    # Non-persistent buffers and parameters are NOT captured.
    assert not any("rope_cache" in n or "mask" in n or "proj" in n for n in names)


def test_matches_state_dict_minus_params():
    # The helper's contract: exactly the buffers in state_dict() (persistent) that aren't parameters.
    m = _Model(n_layers=2)
    param_names = {n for n, _ in m.named_parameters()}
    expected = set(m.state_dict().keys()) - param_names
    assert {name for name, _ in persistent_buffers(m)} == expected


def test_exclude_prefixes():
    # EP layers are gathered separately, so their buffers are excluded by prefix.
    m = _Model(n_layers=3)
    names = {name for name, _ in persistent_buffers(m, exclude_prefixes=("layers.1.",))}
    assert "layers.1.layer_scalar" not in names
    assert "layers.0.layer_scalar" in names and "top_scalar" in names


def test_returns_actual_buffer_tensors():
    m = _Model(n_layers=1)
    got = dict(persistent_buffers(m))
    assert torch.equal(got["layers.0.layer_scalar"], m.layers[0].layer_scalar)


def test_model_without_persistent_buffers_is_empty():
    # A plain model whose only buffers are non-persistent yields nothing (no spurious saves).
    class _Bare(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 2)
            self.register_buffer("cache", torch.zeros(2), persistent=False)

    assert list(persistent_buffers(_Bare())) == []


def test_never_builds_a_state_dict():
    """Asking which buffers are persistent must have NO side effects on the model.

    Under FSDP2 ``state_dict()`` reshards the module's parameters: after a forward with
    ``reshard_after_forward=False`` every rank holds plain unsharded tensors, and the rank that calls
    ``state_dict()`` drops back to ``DTensor(Shard(0))`` while its peers do not. Every caller of this
    helper runs it on the save rank alone, so deriving persistence by diffing a state dict split the
    ranks' sharding state — and the next op over those params (a PEFT adapter unmerge in the merged EP
    save) was then a collective on the writer and local on everyone else. The job desynchronized and
    died in DeepEP's teardown barrier, blaming DeepEP for a save-path defect.
    """
    calls = []

    class _Tattling(_Model):
        def state_dict(self, *args, **kwargs):
            calls.append(1)
            return super().state_dict(*args, **kwargs)

    model = _Tattling(n_layers=2)
    assert {name for name, _ in persistent_buffers(model)} == {
        "top_scalar",
        "layers.0.layer_scalar",
        "layers.1.layer_scalar",
    }
    assert not calls, "persistent_buffers() built a state_dict, which reshards FSDP2 parameters"


def test_shared_buffer_is_yielded_once():
    """One tensor bound into two modules must not be written twice under two names."""

    class _Shared(nn.Module):
        def __init__(self):
            super().__init__()
            self.a, self.b = nn.Module(), nn.Module()
            shared = torch.ones(2)
            self.a.register_buffer("scalar", shared)
            self.b.register_buffer("scalar", shared)

    names = [name for name, _ in persistent_buffers(_Shared())]
    assert names == ["a.scalar"], names


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
