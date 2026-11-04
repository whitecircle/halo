"""The gathered-save preamble: every rank issues the same gathers, since resolving a DTensor is a mesh-wide
``full_tensor()``. Counted at the primitives, as real DTensors need a process group."""

import pytest
import torch
import torch.nn as nn

import halo.runtime.dtensor as dtensor_mod
from halo.checkpoint.parallel.writing import chunked_saveable_tensors


class _BufferModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)
        self.register_buffer("layer_scalar", torch.ones(4), persistent=True)
        self.register_buffer("rotary_cache", torch.ones(4), persistent=False)


def _count_gathers(monkeypatch):
    """All of them are collectives."""
    counter = {"n": 0}

    def counting(tensor, _real_resolve=dtensor_mod.full_cpu_tensor):
        counter["n"] += 1
        return _real_resolve(tensor)

    def counting_materialize(tensor):
        counter["n"] += 1
        return tensor

    monkeypatch.setattr(dtensor_mod, "full_cpu_tensor", counting)
    monkeypatch.setattr(dtensor_mod, "materialize_dtensor", counting_materialize)
    return counter


def test_retain_does_not_change_the_number_of_gathers(monkeypatch):
    """One gather for the parameter and one for the persistent buffer, chunk by chunk."""
    model = _BufferModel()

    counter = _count_gathers(monkeypatch)
    list(chunked_saveable_tensors(model, retain=True))
    retaining = counter["n"]

    counter = _count_gathers(monkeypatch)
    list(chunked_saveable_tensors(model, retain=False))
    non_retaining = counter["n"]

    assert retaining == non_retaining == 2, (
        f"asymmetric gather: writer issued {retaining} collectives, a non-writer {non_retaining} (expected 2 on both)"
    )


def test_only_the_retaining_rank_keeps_the_tensors(monkeypatch):
    """Peers join the collectives without holding a copy of the model."""
    model = _BufferModel()
    _count_gathers(monkeypatch)

    streamed = list(chunked_saveable_tensors(model, retain=True))
    assert {key for chunk in streamed for key in chunk} == {"fc.weight", "layer_scalar"}
    assert all(chunk == {} for chunk in chunked_saveable_tensors(model, retain=False))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
