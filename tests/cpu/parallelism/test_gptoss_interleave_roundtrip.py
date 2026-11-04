#!/usr/bin/env python
"""GptOss's interleaved expert layout: the split and the re-interleave are inverses, on CPU since both call sites
are GPU-only."""

from __future__ import annotations

import pytest
import torch

from halo.models.moe.expert_layout import interleave_gate_up, split_interleaved_gate_up

E, H, M = 3, 5, 4  # experts, hidden, intermediate


@pytest.mark.parametrize("shape", [(E, H, M), (E, M), (M,)], ids=["weights", "biases", "flat"])
def test_split_is_the_inverse_of_interleave(shape):
    gate = torch.randn(*shape)
    up = torch.randn(*shape)
    recovered_gate, recovered_up = split_interleaved_gate_up(interleave_gate_up(gate, up))
    assert torch.equal(recovered_gate, gate)
    assert torch.equal(recovered_up, up)


def test_the_layout_is_interleaved_not_concatenated():
    """Block concatenation round-trips against its own inverse and is still the wrong bytes for vLLM's loader."""
    gate = torch.tensor([[1.0, 2.0]])
    up = torch.tensor([[10.0, 20.0]])
    assert torch.equal(interleave_gate_up(gate, up), torch.tensor([[1.0, 10.0, 2.0, 20.0]]))


def test_the_split_reads_the_halves_the_checkpoint_wrote():
    """Swapped parity trains the up projection through the gate's activation."""
    fused = torch.tensor([[1.0, 10.0, 2.0, 20.0, 3.0, 30.0]])
    gate, up = split_interleaved_gate_up(fused)
    assert torch.equal(gate, torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.equal(up, torch.tensor([[10.0, 20.0, 30.0]]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
