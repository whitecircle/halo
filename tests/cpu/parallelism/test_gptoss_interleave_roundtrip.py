#!/usr/bin/env python
"""GptOss's interleaved expert layout: the split and the re-interleave must be inverses.

``gate_up_proj`` stores the halves interleaved (``[g0, u0, g1, u1, ...]``) — every other family
stores them in contiguous blocks. Two helpers carry that layout for the whole family: the runtime
splits it at init (grouped-GEMM and ETP storage) and in the per-expert loop, and the gather and the
shard merge re-interleave it on the way out. Both call sites of the split are GPU-only (SM90+ or
live expert-TP groups) and the re-interleave is only ever compared against itself, so a flipped axis
or an off-by-one would first appear as a corrupted export from a run nobody can cheaply repeat.

Asserted here on CPU: the pair round-trips for weights and biases alike, the interleave really
interleaves (a block-concatenation would round-trip through a matching split and be wrong on disk),
and the split reads the halves upstream wrote.

Run: pytest tests/cpu/parallelism/test_gptoss_interleave_roundtrip.py
"""

from __future__ import annotations

import pytest
import torch

from src.distributed.expert_parallel.layers.gpt_oss import interleave_gate_up, split_interleaved_gate_up

E, H, M = 3, 5, 4  # experts, hidden, intermediate


@pytest.mark.parametrize("shape", [(E, H, M), (E, M), (M,)], ids=["weights", "biases", "flat"])
def test_split_is_the_inverse_of_interleave(shape):
    gate = torch.randn(*shape)
    up = torch.randn(*shape)
    recovered_gate, recovered_up = split_interleaved_gate_up(interleave_gate_up(gate, up))
    assert torch.equal(recovered_gate, gate)
    assert torch.equal(recovered_up, up)


def test_the_layout_is_interleaved_not_concatenated():
    """The half-and-half layout every other family uses would satisfy a round-trip against its own
    inverse and still be the wrong bytes on disk, which is what vLLM's GptOss loader reads."""
    gate = torch.tensor([[1.0, 2.0]])
    up = torch.tensor([[10.0, 20.0]])
    assert torch.equal(interleave_gate_up(gate, up), torch.tensor([[1.0, 10.0, 2.0, 20.0]]))


def test_the_split_reads_the_halves_the_checkpoint_wrote():
    """The runtime slices the hub's own interleaved tensor, so the split must key on the SAME parity
    the interleave writes — swapping them trains the up projection through the gate's activation."""
    fused = torch.tensor([[1.0, 10.0, 2.0, 20.0, 3.0, 30.0]])
    gate, up = split_interleaved_gate_up(fused)
    assert torch.equal(gate, torch.tensor([[1.0, 2.0, 3.0]]))
    assert torch.equal(up, torch.tensor([[10.0, 20.0, 30.0]]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
