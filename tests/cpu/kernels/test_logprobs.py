#!/usr/bin/env python
"""CPU test: the chunked fp32 log-prob kernel and its chunk budget.

``selective_logprobs`` backs the pipeline last-stage losses and replaces TRL's bf16 log-softmax on the
DPO/KTO loss paths, which pass it the shifted ``logits[:, :-1]`` view. Chunking is only legitimate if
the numbers do not move, so values and grads are held BITWISE to the unchunked fp32 formulation, and
the chunks must keep fp32 state out of backward — the reason they exist.

    python tests/cpu/kernels/test_logprobs.py
"""

import pytest
import torch

import src.kernels.logprobs as logprob_kernels
from src.kernels.logprobs import logit_chunk_rows, selective_logprobs

VOCAB = 37


def _grad_fn_names(tensor: torch.Tensor) -> list[str]:
    """Every autograd node reachable from ``tensor``, by class name, each node once."""
    names, seen, stack = [], set(), [tensor.grad_fn]
    while stack:
        node = stack.pop()
        if node is None or node in seen:
            continue
        seen.add(node)
        names.append(type(node).__name__)
        stack.extend(next_fn for next_fn, _ in node.next_functions)
    return names


# The last case's budget exceeds every row: each row is one chunk, still checkpointed.
@pytest.mark.parametrize(("batch", "seq", "budget_rows"), [(2, 16, 4), (4, 33, 7), (2, 9, 1), (2, 16, 64)])
def test_selective_logprobs_match_the_full_plane(monkeypatch, batch, seq, budget_rows):
    """A row left out of the checkpoint keeps its fp32 log-softmax alive until backward — a whole fp32
    plane across the batch — and a slice per chunk zero-fills a whole-row gradient per chunk."""
    monkeypatch.setattr(logprob_kernels, "LOGIT_CHUNK_ELEMENTS", budget_rows * VOCAB)
    torch.manual_seed(batch * 100 + seq)
    base = torch.randn(batch, seq, VOCAB, dtype=torch.bfloat16)
    index = torch.randint(0, VOCAB, (batch, seq - 1))
    chunked_in = base.clone().requires_grad_(True)
    full_in = base.clone().requires_grad_(True)
    saved = []

    def record(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
        got = selective_logprobs(chunked_in[:, :-1], index)
    want = torch.log_softmax(full_in[:, :-1].float(), dim=-1).gather(-1, index.unsqueeze(-1)).squeeze(-1)

    assert got.dtype == torch.float32
    assert torch.equal(got, want), f"max diff {(got - want).abs().max()}"
    held = [tuple(t.shape) for t in saved if t.dtype == torch.float32 and t.dim() and t.size(-1) == VOCAB]
    assert not held, f"fp32 vocab-wide tensors held for backward: {held}"
    assert _grad_fn_names(got).count("SliceBackward0") == 1, "only the caller's shifted view may slice"
    got.sum().backward()
    want.sum().backward()
    assert torch.equal(chunked_in.grad, full_in.grad), f"max grad diff {(chunked_in.grad - full_in.grad).abs().max()}"


def test_selective_logprobs_refuses_a_top_k_index():
    """TRL's ``selective_log_softmax`` also gathers K log-probs per position; this one does not."""
    with pytest.raises(ValueError, match="one index per position"):
        selective_logprobs(torch.randn(2, 5, 11), torch.zeros(2, 5, 3, dtype=torch.long))


def test_chunk_budget_is_vocab_aware():
    """The held fp32 plane must stay bounded as the vocabulary grows.

    The chunk budget is what keeps the fp32 upcast off the memory peak, and the plane it slices is
    ``tokens x V``. Sizing it in token ROWS bounds nothing in particular: at gpt-oss's V=201k a
    4096-row chunk is 3.3 GB — the size of the entire bf16 plane it is supposed to be a fraction of —
    while at V=32k the same constant is 0.5 GB. Asserting a flat ceiling in BYTES is what catches a
    regression back to a row-count budget.
    """
    ceiling = logprob_kernels.LOGIT_CHUNK_ELEMENTS * 4  # fp32
    for vocab in (32_000, 151_936, 201_088, 262_144):
        rows = logit_chunk_rows(vocab)
        assert rows >= 1, f"vocab {vocab} must still take at least one row per chunk"
        assert rows * vocab * 4 <= ceiling, (
            f"vocab {vocab}: chunk holds {rows * vocab * 4 / 1e9:.2f} GB of fp32, over the "
            f"{ceiling / 1e9:.2f} GB budget — the chunk is sized in rows, not elements"
        )
    # A vocabulary past the whole budget must not round down to a zero-row (infinite-loop) chunk.
    assert logit_chunk_rows(logprob_kernels.LOGIT_CHUNK_ELEMENTS * 2) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
