"""``persistent_buffers``: a param-only save drops persistent buffers (Gemma4 ``layer_scalar``) and corrupts every
layer on reload; non-persistent ones are recomputed."""

import pytest
import torch
import torch.nn as nn

from halo.models.param_roles import persistent_buffers


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(4, 4, bias=False)
        self.register_buffer("layer_scalar", torch.ones(1))
        self.register_buffer("rope_cache", torch.zeros(4), persistent=False)


class _Model(nn.Module):
    def __init__(self, n_layers=3):
        super().__init__()
        self.layers = nn.ModuleList([_Block() for _ in range(n_layers)])
        self.register_buffer("top_scalar", torch.ones(1))
        self.register_buffer("mask", torch.zeros(2), persistent=False)


def test_captures_persistent_skips_nonpersistent_and_params():
    m = _Model(n_layers=3)
    names = {name for name, _ in persistent_buffers(m)}
    assert names == {"top_scalar", *(f"layers.{i}.layer_scalar" for i in range(3))}
    assert not any("rope_cache" in n or "mask" in n or "proj" in n for n in names)


def test_matches_state_dict_minus_params():
    # Exactly the buffers in state_dict() (persistent) that aren't parameters.
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
    class _Bare(nn.Module):
        def __init__(self):
            super().__init__()
            self.lin = nn.Linear(2, 2)
            self.register_buffer("cache", torch.zeros(2), persistent=False)

    assert list(persistent_buffers(_Bare())) == []


def test_never_builds_a_state_dict():
    """``state_dict()`` reshards under FSDP2, so calling it on the save rank alone would split ranks' sharding state
    and desync in DeepEP's teardown barrier."""
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
    """One tensor in two modules must not be written twice."""

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
    raise SystemExit(pytest.main([__file__, "-v"]))
