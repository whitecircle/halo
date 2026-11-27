"""Inkling 2's block attention residual patch: the upstream SDPA residual's outputs and input gradients, from one block
state to the full stack and for one and three queries, as plain matmuls; ``apply_inkling2_patches`` installs it."""

from types import SimpleNamespace

import pytest
import torch
from transformers.models.inkling2.modeling_inkling2 import Inkling2BlockAttentionResidual

from halo.models.families.inkling2 import (
    UPSTREAM_BLOCK_ATTENTION_RESIDUAL_FORWARD,
    _block_attention_residual_forward,
    apply_inkling2_patches,
)

HIDDEN = 64
# Inkling-2-Small's 48 layers in blocks of 6: the final residual reads all eight finished blocks.
BLOCK_COUNTS = (0, 1, 8)


def _scored(forward, module, block_states, partial):
    block_states.grad = partial.grad = None
    outputs = torch.stack(forward(module, block_states, partial))
    weights = torch.linspace(-1.0, 1.0, HIDDEN)
    (outputs * weights).sum().backward()
    return outputs.detach(), block_states.grad.clone(), partial.grad.clone()


@pytest.mark.parametrize("num_queries", [1, 3])
@pytest.mark.parametrize("blocks", BLOCK_COUNTS)
def test_the_patch_is_the_upstream_residual(num_queries, blocks):
    torch.manual_seed(0)
    module = Inkling2BlockAttentionResidual(SimpleNamespace(hidden_size=HIDDEN), num_queries=num_queries)
    torch.nn.init.normal_(module.query, std=1.0)
    block_states = torch.randn(2, 7, blocks, HIDDEN, requires_grad=True)
    partial = torch.randn(2, 7, HIDDEN, requires_grad=True)
    expected = _scored(UPSTREAM_BLOCK_ATTENTION_RESIDUAL_FORWARD, module, block_states, partial)
    actual = _scored(_block_attention_residual_forward, module, block_states, partial)
    assert actual[0].shape == (num_queries, 2, 7, HIDDEN)
    for got, want in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, want, rtol=1e-5, atol=1e-6)


def test_apply_installs_the_residual_patch():
    apply_inkling2_patches()
    assert Inkling2BlockAttentionResidual.forward is _block_attention_residual_forward


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
