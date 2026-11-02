#!/usr/bin/env python
"""CPU test: the torch chunked delta rule over packed segments starts each one from its own state.

The CUDA kernel takes ``cu_seqlens`` with one ``initial_state`` row per segment; on CPU the toolkit's dispatch runs
the torch body once per segment and must hand each its own row. Two segments with distinct initial states, through
the dispatch-installed public function and through the fixup directly, each match a call on that segment alone, in
output and final state.

    python tests/cpu/kernels/test_delta_rule_segment_states.py
"""

import inspect

import pytest
import torch
from transformers.integrations import hub_kernels
from transformers.models.qwen3_5 import modeling_qwen3_5

from halo.kernels.hub_kernel_dispatch import _segmented_chunk_gated_delta_rule, ensure_device_aware_kernel_dispatch

HEADS, KEY_DIM, VALUE_DIM = 2, 8, 6
SEGMENTS = (11, 7)


def _segments_problem():
    generator = torch.Generator().manual_seed(0)
    tokens = sum(SEGMENTS)
    query = torch.randn(1, tokens, HEADS, KEY_DIM, generator=generator)
    key = torch.randn(1, tokens, HEADS, KEY_DIM, generator=generator)
    value = torch.randn(1, tokens, HEADS, VALUE_DIM, generator=generator)
    g = -torch.rand(1, tokens, HEADS, generator=generator)
    beta = torch.rand(1, tokens, HEADS, generator=generator)
    initial_state = torch.randn(len(SEGMENTS), HEADS, KEY_DIM, VALUE_DIM, generator=generator) * 3
    cu_seqlens = torch.tensor([0, SEGMENTS[0], tokens])
    return query, key, value, g, beta, initial_state, cu_seqlens


def _alone(body, query, key, value, g, beta, initial_state, cu_seqlens):
    """Each segment's ``(output, final state)`` from a call on that segment alone."""
    edges = cu_seqlens.tolist()
    return [
        body(
            query[:, start:end],
            key[:, start:end],
            value[:, start:end],
            g=g[:, start:end],
            beta=beta[:, start:end],
            initial_state=initial_state[segment : segment + 1],
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        for segment, (start, end) in enumerate(zip(edges[:-1], edges[1:], strict=True))
    ]


def _assert_segments_match(packed, alone, cu_seqlens):
    output, state = packed
    edges = cu_seqlens.tolist()
    for segment, (start, end) in enumerate(zip(edges[:-1], edges[1:], strict=True)):
        torch.testing.assert_close(output[:, start:end], alone[segment][0], rtol=0, atol=0)
        torch.testing.assert_close(state[segment : segment + 1], alone[segment][1], rtol=0, atol=0)


@pytest.fixture(scope="module")
def body():
    """The undecorated torch delta rule."""
    return inspect.unwrap(modeling_qwen3_5.torch_chunk_gated_delta_rule)


def test_premise_the_two_initial_states_lead_apart(body):
    """With either segment started from the other's state, its output moves, or the checks below compare nothing."""
    query, key, value, g, beta, initial_state, cu_seqlens = _segments_problem()
    alone = _alone(body, query, key, value, g, beta, initial_state, cu_seqlens)
    swapped = _alone(body, query, key, value, g, beta, initial_state.flip(0), cu_seqlens)
    assert all(not torch.allclose(a[0], s[0]) for a, s in zip(alone, swapped, strict=True))


def test_the_fixup_runs_each_segment_from_its_own_state(body):
    query, key, value, g, beta, initial_state, cu_seqlens = _segments_problem()
    packed = _segmented_chunk_gated_delta_rule(
        body,
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )
    _assert_segments_match(packed, _alone(body, query, key, value, g, beta, initial_state, cu_seqlens), cu_seqlens)


def test_the_dispatched_function_routes_a_cpu_call_through_the_fixup(body):
    """The torch body decorated as a modeling module decorates it once the toolkit's dispatch is installed, called
    as a GatedDeltaNet layer calls it with its varlen set."""
    ensure_device_aware_kernel_dispatch()
    dispatched = hub_kernels.use_kernel_func_from_hub_with_fallback("chunk_gated_delta_rule", "fla")(body)
    query, key, value, g, beta, initial_state, cu_seqlens = _segments_problem()
    packed = dispatched(
        query,
        key,
        value,
        g=g,
        beta=beta,
        initial_state=initial_state,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu_seqlens,
    )
    _assert_segments_match(packed, _alone(body, query, key, value, g, beta, initial_state, cu_seqlens), cu_seqlens)


def test_one_state_for_two_segments_is_refused(body):
    query, key, value, g, beta, initial_state, cu_seqlens = _segments_problem()
    with pytest.raises(ValueError, match="initial_state holds 1 states for 2 segments"):
        _segmented_chunk_gated_delta_rule(
            body, query, key, value, g=g, beta=beta, initial_state=initial_state[:1], cu_seqlens=cu_seqlens
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
