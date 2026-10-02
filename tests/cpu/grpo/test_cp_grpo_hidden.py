#!/usr/bin/env python
"""CP's GRPO scorer must enter through the wrapper without constructing LM logits."""

from types import SimpleNamespace

import pytest
import torch
from torch import nn

import src.distributed.context_parallel.wrapper as cp_wrapper
from tests.common.cp_wrapper import unpatched_cp_wrapper

SEQ = 12
HIDDEN = 5


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.embeddings = nn.Embedding(32, HIDDEN)
        self.calls = []

    def forward(self, *, input_ids, attention_mask, position_ids, use_cache):
        self.calls.append(
            (
                input_ids.clone(),
                attention_mask.clone() if attention_mask is not None else None,
                position_ids.clone(),
                use_cache,
            )
        )
        return SimpleNamespace(last_hidden_state=self.embeddings(input_ids) + position_ids.unsqueeze(-1))


class _Head(nn.Module):
    def forward(self, hidden):
        raise AssertionError("the hidden-only path must not materialize vocabulary logits")


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.base_model = _Backbone()
        self.lm_head = _Head()


def _wrapper(model, cp_size, cp_rank):
    return unpatched_cp_wrapper(
        model, cp_size=cp_size, cp_rank=cp_rank, attention_layers=[SimpleNamespace(global_position_ids=None)]
    )


@pytest.mark.parametrize("cp_size", [1, 2, 4])
def test_hidden_only_scores_each_contiguous_chunk_at_global_positions(cp_size):
    """CP2/4 cover rank-ordering; each rank sees its shard but hooks see all positions."""
    torch.manual_seed(18)
    model = _Model()
    ids = torch.randint(1, 32, (2, SEQ))
    mask = torch.ones_like(ids)
    mask[1, -2:] = 0
    positions = torch.stack([torch.arange(SEQ), torch.arange(SEQ) + 17])
    expected = model.base_model.embeddings(ids) + positions.unsqueeze(-1)

    for rank in range(cp_size):
        wrapper = _wrapper(model, cp_size, rank)
        got = wrapper.forward_hidden_states(ids, attention_mask=mask, position_ids=positions)
        start = rank * (SEQ // cp_size)
        end = start + (SEQ // cp_size)
        torch.testing.assert_close(got, expected[:, start:end])
        seen_ids, seen_mask, seen_positions, use_cache = model.base_model.calls[-1]
        torch.testing.assert_close(seen_ids, ids[:, start:end])
        torch.testing.assert_close(seen_mask, mask[:, start:end])
        torch.testing.assert_close(seen_positions, positions[:, start:end])
        assert use_cache is False
        torch.testing.assert_close(wrapper._attention_layers[0].global_position_ids, positions)

    stitched = torch.cat(
        [
            _wrapper(model, cp_size, rank).forward_hidden_states(ids, attention_mask=mask, position_ids=positions)
            for rank in range(cp_size)
        ],
        dim=1,
    )
    torch.testing.assert_close(stitched, expected)


def test_hidden_only_reuses_cp_guards_and_does_not_accept_labels():
    model = _Model()
    wrapper = _wrapper(model, cp_size=2, cp_rank=0)
    ids = torch.ones(1, SEQ, dtype=torch.long)
    mask = torch.ones_like(ids)
    mask[:, :2] = 0
    with pytest.raises(ValueError, match="LEFT-padded"):
        wrapper.forward_hidden_states(ids, attention_mask=mask)
    with pytest.raises(ValueError, match="does not accept labels"):
        wrapper.forward_hidden_states(ids, labels=ids)
    with pytest.raises(ValueError, match="use_cache=False"):
        wrapper.forward_hidden_states(ids, use_cache=True)
    with pytest.raises(ValueError, match="divisible"):
        wrapper.forward_hidden_states(ids[:, :-1])


def test_hidden_only_opens_a_fresh_ep_capacity_scope(monkeypatch):
    # Entering the backbone directly bypasses the CausalLM's pre-forward hook. A stale DeepEP
    # capacity budget would otherwise be reused for the next row even when its routing differs.
    calls = []
    monkeypatch.setattr(cp_wrapper, "bump_forward_generation", lambda: calls.append(1))
    wrapper = _wrapper(_Model(), cp_size=2, cp_rank=0)
    ids = torch.ones(1, SEQ, dtype=torch.long)
    wrapper.forward_hidden_states(ids)
    wrapper.forward_hidden_states(ids)
    assert len(calls) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
