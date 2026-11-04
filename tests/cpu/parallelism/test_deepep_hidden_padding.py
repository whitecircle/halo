#!/usr/bin/env python
"""DeepEP's ``hidden % 256`` transport padding: a wrong width feeds the FFN garbage columns silently. GptOss's 2880
pads to 3072; the EP correctness tests all run models whose hidden divides 256."""

from __future__ import annotations

import pytest
import torch

from halo.distributed.expert_parallel.deepep.backends import _ElasticBackend
from halo.distributed.expert_parallel.deepep.capacity import DEEPEP_HIDDEN_ALIGN

# The first pads (GptOss), the rest already conform.
GPTOSS_HIDDEN = 2880


class _StubDispatcher:
    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim


def _backend(hidden_dim: int) -> _ElasticBackend:
    return _ElasticBackend(_StubDispatcher(hidden_dim))


def test_the_alignment_is_the_kernels_documented_256():
    """A hardware contract (the TMA combine kernel), not a tunable."""
    assert DEEPEP_HIDDEN_ALIGN == 256


@pytest.mark.parametrize(
    ("hidden", "expected"),
    [
        (GPTOSS_HIDDEN, 3072),  # 2880 -> 11.25 blocks -> 12.
        (256, 256),
        (2048, 2048),
        (4096, 4096),
        (1, 256),
        (255, 256),
        (257, 512),  # One element over a block boundary costs a whole block.
        (5120, 5120),
    ],
)
def test_the_wire_width_rounds_up_to_a_whole_number_of_blocks(hidden, expected):
    backend = _backend(hidden)
    assert backend._padded_hidden == expected
    assert backend._padded_hidden % DEEPEP_HIDDEN_ALIGN == 0
    assert backend._padded_hidden >= hidden, "the wire must never be NARROWER than the real hidden"


@pytest.mark.parametrize("hidden", [GPTOSS_HIDDEN, 1, 255, 257])
def test_a_non_conforming_hidden_takes_the_padding_branch(hidden):
    assert _backend(hidden)._needs_pad is True


@pytest.mark.parametrize("hidden", [256, 512, 2048, 4096, 5120])
def test_a_conforming_hidden_skips_the_padding_branch(hidden):
    assert _backend(hidden)._needs_pad is False


def test_pad_then_slice_returns_the_original_tensor_exactly():
    torch.manual_seed(0)
    backend = _backend(GPTOSS_HIDDEN)
    tokens = torch.randn(37, GPTOSS_HIDDEN, dtype=torch.bfloat16)

    wire = backend._pad(tokens)
    assert wire.shape == (37, 3072), f"wire width wrong: {tuple(wire.shape)}"

    back = backend._slice(wire)
    assert back.shape == tokens.shape
    assert torch.equal(back, tokens), "the pad/slice round trip changed the payload"


def test_the_padding_region_is_zero_not_uninitialized():
    """The pad columns are summed into the combine, so uninitialized memory adds nondeterministic garbage."""
    backend = _backend(GPTOSS_HIDDEN)
    wire = backend._pad(torch.randn(8, GPTOSS_HIDDEN, dtype=torch.bfloat16))
    assert torch.count_nonzero(wire[:, GPTOSS_HIDDEN:]).item() == 0, "transport padding is not zeroed"


def test_slicing_a_wire_tensor_drops_only_the_padding():
    """The leading real columns."""
    backend = _backend(GPTOSS_HIDDEN)
    wire = torch.arange(3072, dtype=torch.float32).repeat(4, 1)
    sliced = backend._slice(wire)
    assert sliced.shape == (4, GPTOSS_HIDDEN)
    assert torch.equal(sliced[0], torch.arange(GPTOSS_HIDDEN, dtype=torch.float32))


def test_the_slice_is_a_row_strided_view_not_a_copy():
    """The receive side hands back a view of the padded wire buffer, with no copy per transport call."""
    backend = _backend(GPTOSS_HIDDEN)
    padded = backend._pad(torch.randn(5, GPTOSS_HIDDEN))
    sliced = backend._slice(padded)
    assert sliced.stride() == (padded.shape[-1], 1)
    assert sliced.data_ptr() == padded.data_ptr()


def test_the_pad_columns_are_zero_whatever_the_allocator_returned():
    """A recycled block's stale values must not reach the wire."""
    backend = _backend(GPTOSS_HIDDEN)
    tokens = torch.randn(7, GPTOSS_HIDDEN)
    padded = backend._pad(tokens)
    assert torch.equal(padded[:, :GPTOSS_HIDDEN], tokens)
    assert torch.count_nonzero(padded[:, GPTOSS_HIDDEN:]) == 0


def test_a_conforming_hidden_is_passed_through_untouched():
    """Identity, so conforming models pay nothing."""
    backend = _backend(4096)
    tokens = torch.randn(11, 4096)
    assert backend._pad(tokens) is tokens
    assert backend._slice(tokens) is tokens


@pytest.mark.parametrize("shape", [(1, GPTOSS_HIDDEN), (0, GPTOSS_HIDDEN), (3, 5, GPTOSS_HIDDEN)])
def test_the_round_trip_holds_for_the_shapes_a_dispatch_produces(shape):
    """A single token, a rank that received none, and the topk-expanded 3-D layout."""
    backend = _backend(GPTOSS_HIDDEN)
    tokens = torch.randn(*shape, dtype=torch.bfloat16)
    back = backend._slice(backend._pad(tokens))
    assert back.shape == tokens.shape
    assert torch.equal(back, tokens)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
