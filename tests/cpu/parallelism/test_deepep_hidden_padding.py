#!/usr/bin/env python
"""DeepEP's ``hidden % 256`` transport padding — the pad/slice pair must be a clean round trip.

The ElasticBuffer's TMA combine kernel requires ``hidden % 256 == 0``, so ``_ElasticBackend`` widens
non-conforming hidden on the wire and slices it back. The pair is applied symmetrically across
forward and backward (``dispatch``/``combine``/``combine_grad``/``dispatch_grad``) and is invisible
to callers — which is exactly why a regression in it is silent: a wrong width does not crash, it
feeds the expert FFN garbage columns, or drops real ones.

Nothing else pins it. ``_padded_hidden`` appears in the suite only as an arena cache key
(``tests/gpu/parallelism/ep/test_ep_shared_arena.py``), which is satisfied by any value at all, and
the EP correctness tests all run models whose hidden already divides 256, so the padding branch is
never taken there. GptOss's 2880 is the roster's live counter-example: it pads to 3072 on every
dispatch of every layer.

The backend reads only ``dispatcher.hidden_dim`` to size the padding, so these drive the real
``_ElasticBackend`` methods against a stub dispatcher — no DeepEP, no CUDA, no process group.

Run: python tests/cpu/parallelism/test_deepep_hidden_padding.py  (or pytest -m cpu)
"""

from __future__ import annotations

import pytest
import torch

from src.distributed.expert_parallel.config import DEEPEP_HIDDEN_ALIGN
from src.distributed.expert_parallel.dispatcher import _ElasticBackend

# Roster hidden sizes: the first pads (GptOss), the rest already conform.
GPTOSS_HIDDEN = 2880


class _StubDispatcher:
    """The only field ``_ElasticBackend``'s padding reads."""

    def __init__(self, hidden_dim: int):
        self.hidden_dim = hidden_dim


def _backend(hidden_dim: int) -> _ElasticBackend:
    return _ElasticBackend(_StubDispatcher(hidden_dim))


def test_the_alignment_is_the_kernels_documented_256():
    """The constant is a hardware contract (the TMA combine kernel), not a tunable."""
    assert DEEPEP_HIDDEN_ALIGN == 256


@pytest.mark.parametrize(
    ("hidden", "expected"),
    [
        (GPTOSS_HIDDEN, 3072),  # 2880 -> 11.25 blocks -> 12
        (256, 256),  # exact, single block
        (2048, 2048),  # exact, 8 blocks
        (4096, 4096),  # exact, 16 blocks
        (1, 256),  # smallest non-conforming
        (255, 256),
        (257, 512),  # one element over a block boundary costs a whole block
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
    """Anti-vacuity, and the cheap path: a conforming model must pay nothing."""
    assert _backend(hidden)._needs_pad is False


def test_pad_then_slice_returns_the_original_tensor_exactly():
    """The round trip is the whole contract: what a caller hands in is what it gets back."""
    torch.manual_seed(0)
    backend = _backend(GPTOSS_HIDDEN)
    tokens = torch.randn(37, GPTOSS_HIDDEN, dtype=torch.bfloat16)

    wire = backend._pad(tokens)
    assert wire.shape == (37, 3072), f"wire width wrong: {tuple(wire.shape)}"

    back = backend._slice(wire)
    assert back.shape == tokens.shape
    assert torch.equal(back, tokens), "the pad/slice round trip changed the payload"


def test_the_padding_region_is_zero_not_uninitialized():
    """The pad columns are summed into the combine on the far side. Uninitialized memory there
    would add garbage to real expert outputs — nondeterministically, so it would not reproduce."""
    backend = _backend(GPTOSS_HIDDEN)
    wire = backend._pad(torch.randn(8, GPTOSS_HIDDEN, dtype=torch.bfloat16))
    assert torch.count_nonzero(wire[:, GPTOSS_HIDDEN:]).item() == 0, "transport padding is not zeroed"


def test_slicing_a_wire_tensor_drops_only_the_padding():
    """The slice must take the LEADING real columns, not the trailing ones."""
    backend = _backend(GPTOSS_HIDDEN)
    wire = torch.arange(3072, dtype=torch.float32).repeat(4, 1)
    sliced = backend._slice(wire)
    assert sliced.shape == (4, GPTOSS_HIDDEN)
    assert torch.equal(sliced[0], torch.arange(GPTOSS_HIDDEN, dtype=torch.float32))


def test_the_sliced_tensor_is_contiguous():
    """A narrowing view is non-contiguous; the expert GEMM and the next collective both want a
    packed buffer, so ``_slice`` materializes one."""
    backend = _backend(GPTOSS_HIDDEN)
    sliced = backend._slice(backend._pad(torch.randn(5, GPTOSS_HIDDEN)))
    assert sliced.is_contiguous()


def test_a_conforming_hidden_is_passed_through_untouched():
    """No copy on the cheap path — identity, so the padding costs conforming models nothing."""
    backend = _backend(4096)
    tokens = torch.randn(11, 4096)
    assert backend._pad(tokens) is tokens
    assert backend._slice(tokens) is tokens


@pytest.mark.parametrize("shape", [(1, GPTOSS_HIDDEN), (0, GPTOSS_HIDDEN), (3, 5, GPTOSS_HIDDEN)])
def test_the_round_trip_holds_for_the_shapes_a_dispatch_produces(shape):
    """Padding is on the LAST dim: a single token, a rank that received none, and the topk-expanded
    3-D layout must all survive."""
    backend = _backend(GPTOSS_HIDDEN)
    tokens = torch.randn(*shape, dtype=torch.bfloat16)
    back = backend._slice(backend._pad(tokens))
    assert back.shape == tokens.shape
    assert torch.equal(back, tokens)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
