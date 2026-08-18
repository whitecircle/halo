"""CPU tests for the gathered-save preamble: every rank must issue the SAME gather collectives.

A gathered save (FSDP2 / TP / CP) resolves parameters, persistent buffers and the neutralized GptOss
sinks, but only the writer keeps the host copies. Resolving a DTensor is ``full_tensor()`` — a
mesh-wide collective — so a leg placed behind the ``retain`` gate hangs the save: the writer enters
the all-gather and its peers never do, and the run dies on the NCCL watchdog with no diagnostic.
Buffers are the leg at risk (no DTensor persistent buffer exists in the shipped roster today, but the
bias-update balancing slot is already contemplated as one), so these tests count gather calls on both
sides of ``retain`` rather than waiting for a live hang to prove it.

Counting is done by replacing the gather primitives where every branch reads them — the shared
``checkpoint_write`` pair. The tensors themselves stay plain, since a real DTensor needs an initialized
process group.
"""

import sys

import pytest
import torch
import torch.nn as nn

import src.distributed.checkpoint.write as checkpoint_write_mod
from src.distributed.checkpoint.write import chunked_saveable_tensors, gather_saveable_tensors


class _BufferModel(nn.Module):
    """A parameter and a PERSISTENT buffer — the two legs a gathered save resolves."""

    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(4, 4, bias=False)
        self.register_buffer("layer_scalar", torch.ones(4), persistent=True)
        self.register_buffer("rotary_cache", torch.ones(4), persistent=False)


def _count_gathers(monkeypatch):
    """Count every gather any leg issues, retaining or not (all of them are collectives)."""
    counter = {"n": 0}

    def counting(tensor, _real_resolve=checkpoint_write_mod.resolve_param_tensor):
        counter["n"] += 1
        return _real_resolve(tensor)

    def counting_materialize(tensor):
        counter["n"] += 1
        return tensor

    monkeypatch.setattr(checkpoint_write_mod, "resolve_param_tensor", counting)
    monkeypatch.setattr(checkpoint_write_mod, "materialize_dtensor", counting_materialize)
    return counter


@pytest.mark.parametrize(
    ("gather", "expected"),
    [
        pytest.param(
            lambda model, retain: gather_saveable_tensors(model, retain=retain),
            2,  # one parameter + one persistent buffer
            id="fsdp2",
        ),
        pytest.param(
            lambda model, retain: list(chunked_saveable_tensors(model, retain=retain)),
            2,  # the streamed FSDP2 / CP / TP save, chunk by chunk
            id="streamed",
        ),
    ],
)
def test_retain_does_not_change_the_number_of_gathers(monkeypatch, gather, expected):
    """The writer and the non-writers must run the identical collective sequence."""
    model = _BufferModel()

    counter = _count_gathers(monkeypatch)
    gather(model, True)
    retaining = counter["n"]

    counter = _count_gathers(monkeypatch)
    gather(model, False)
    non_retaining = counter["n"]

    assert retaining == non_retaining == expected, (
        f"asymmetric gather: writer issued {retaining} collectives, a non-writer {non_retaining} "
        f"(expected {expected} on both)"
    )


def test_only_the_retaining_rank_keeps_the_tensors(monkeypatch):
    """The point of the gate: peers join the collectives without holding a copy of the model."""
    model = _BufferModel()
    _count_gathers(monkeypatch)

    assert set(gather_saveable_tensors(model, retain=True)) == {"fc.weight", "layer_scalar"}
    assert gather_saveable_tensors(model, retain=False) == {}
    streamed = list(chunked_saveable_tensors(model, retain=True))
    assert {key for chunk in streamed for key in chunk} == {"fc.weight", "layer_scalar"}
    assert all(chunk == {} for chunk in chunked_saveable_tensors(model, retain=False))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
