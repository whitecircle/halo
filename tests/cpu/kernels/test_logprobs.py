#!/usr/bin/env python
"""The chunked fp32 log-prob kernel and its chunk budget: values and grads held bitwise to the unchunked fp32
formulation."""

import pytest
import torch
from trl.trainer.utils import selective_log_softmax

import halo.kernels.logprobs as logprob_kernels
from halo.kernels.logprobs import logit_chunk_rows, next_token_logprobs, selective_logprobs, token_logprobs
from tests.common.numerics import autograd_nodes

VOCAB = 37


# The last budget exceeds every row: each row is one chunk, still checkpointed.
@pytest.mark.parametrize(("batch", "seq", "budget_rows"), [(2, 16, 4), (4, 33, 7), (2, 9, 1), (2, 16, 64)])
def test_selective_logprobs_match_the_full_plane(monkeypatch, batch, seq, budget_rows):
    """A row outside the checkpoint keeps its fp32 log-softmax alive until backward, and a per-chunk slice
    zero-fills a whole-row gradient."""
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
    slices = [node for node in autograd_nodes(got) if type(node).__name__ == "SliceBackward0"]
    assert len(slices) == 1, "only the caller's shifted view may slice"
    got.sum().backward()
    want.sum().backward()
    assert torch.equal(chunked_in.grad, full_in.grad), f"max grad diff {(chunked_in.grad - full_in.grad).abs().max()}"


def test_a_row_mask_zeroes_the_masked_positions_as_trl_does():
    """TRL's DPO/KTO paths pass the shifted completion mask as ``row_mask``."""
    torch.manual_seed(0)
    base = torch.randn(3, 9, VOCAB, dtype=torch.bfloat16)
    index = torch.randint(0, VOCAB, (3, 8))
    row_mask = torch.randint(0, 2, (3, 8))
    ours_in = base.clone().requires_grad_(True)
    trl_in = base.float().requires_grad_(True)

    got = selective_logprobs(ours_in[:, :-1], index, row_mask=row_mask)
    want = selective_log_softmax(trl_in[:, :-1], index, row_mask=row_mask)

    torch.testing.assert_close(got, want)
    assert torch.equal(got[row_mask == 0], torch.zeros_like(got[row_mask == 0]))
    got.sum().backward()
    want.sum().backward()
    torch.testing.assert_close(ours_in.grad.float(), trl_in.grad, rtol=1e-2, atol=1e-2)
    assert not ours_in.grad[:, :-1][row_mask == 0].any(), "a masked position sent back a gradient"


def test_selective_logprobs_refuses_a_row_mask_of_another_shape():
    with pytest.raises(ValueError, match="row_mask"):
        selective_logprobs(torch.randn(2, 5, 11), torch.zeros(2, 5, dtype=torch.long), row_mask=torch.ones(2, 4))


def test_selective_logprobs_refuses_a_top_k_index():
    """TRL's ``selective_log_softmax`` also gathers K log-probs per position; this one does not."""
    with pytest.raises(ValueError, match="one index per position"):
        selective_logprobs(torch.randn(2, 5, 11), torch.zeros(2, 5, 3, dtype=torch.long))


def test_chunk_budget_is_vocab_aware():
    """At gpt-oss's V=201k a 4096-row chunk is 3.3 GB, the whole bf16 plane; a ceiling in bytes catches a row-count
    budget."""
    ceiling = logprob_kernels.LOGIT_CHUNK_ELEMENTS * 4
    for vocab in (32_000, 151_936, 201_088, 262_144):
        rows = logit_chunk_rows(vocab)
        assert rows >= 1, f"vocab {vocab} must still take at least one row per chunk"
        assert rows * vocab * 4 <= ceiling, (
            f"vocab {vocab}: chunk holds {rows * vocab * 4 / 1e9:.2f} GB of fp32, over the "
            f"{ceiling / 1e9:.2f} GB budget — the chunk is sized in rows, not elements"
        )
    # It must not round down to a zero-row (infinite-loop) chunk.
    assert logit_chunk_rows(logprob_kernels.LOGIT_CHUNK_ELEMENTS * 2) == 1


def _unchunked_token_logprobs(logits, labels):
    """Two full fp32 ``[B, S-1, V]`` planes, held at once."""
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~mask, 0)
    logps = torch.log_softmax(shift_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return logps, mask


# The last budget exceeds every row: each row is one chunk.
@pytest.mark.parametrize(("batch", "seq", "budget_rows"), [(2, 16, 4), (4, 33, 7), (2, 9, 1), (2, 16, 64)])
def test_chunked_token_logprobs_match_the_full_plane(monkeypatch, batch, seq, budget_rows):
    """Whole planes cost ~26 GB per microbatch at ``V=201088, S=8192`` on the stage carrying the head, why it
    chunks."""
    vocab = 37
    monkeypatch.setattr(logprob_kernels, "LOGIT_CHUNK_ELEMENTS", budget_rows * vocab)
    torch.manual_seed(batch * 100 + seq)
    base = torch.randn(batch, seq, vocab, dtype=torch.bfloat16)
    labels = torch.randint(0, vocab, (batch, seq))
    labels[:, :2] = -100
    labels[-1, :] = -100  # An inert all-ignore row, as the PP eval row-padding emits.

    chunked_in = base.clone().requires_grad_(True)
    full_in = base.clone().requires_grad_(True)
    got_lp, got_mask = token_logprobs(chunked_in, labels)
    want_lp, want_mask = _unchunked_token_logprobs(full_in, labels)

    assert got_lp.shape == want_lp.shape == (batch, seq - 1)
    assert torch.equal(got_mask, want_mask)
    assert torch.equal(got_lp, want_lp), f"max diff {(got_lp - want_lp).abs().max()}"

    (got_lp * got_mask).sum().backward()
    (want_lp * want_mask).sum().backward()
    assert torch.equal(chunked_in.grad, full_in.grad), f"max grad diff {(chunked_in.grad - full_in.grad).abs().max()}"


@pytest.mark.parametrize(
    "logits_shape, start",
    [((2, 4, VOCAB), 0), ((3, 4, VOCAB), -1), ((3, 4, VOCAB), 5), ((3, 10, VOCAB), 0)],
    ids=["rows", "negative-start", "past-the-end", "longer-than-labels"],
)
def test_next_token_logprobs_refuses_a_window_outside_the_labels(logits_shape, start):
    """Read off shapes alone, so no host sync."""
    labels = torch.zeros(3, 8, dtype=torch.long)
    with pytest.raises(ValueError, match="do not lie inside labels"):
        next_token_logprobs(torch.zeros(logits_shape), labels, start)
    next_token_logprobs(torch.zeros(3, 4, VOCAB), labels, 4)


def test_token_logprobs_drop_the_final_position_after_scoring():
    """Slicing the logits instead makes backward zero-fill and copy a whole ``[B, S, V]`` plane."""
    logits = torch.randn(2, 9, VOCAB, requires_grad=True)
    logps, _ = token_logprobs(logits, torch.randint(0, VOCAB, (2, 9)))
    sliced = [
        tuple(node._saved_self_sym_sizes) for node in autograd_nodes(logps) if type(node).__name__ == "SliceBackward0"
    ]
    assert sliced == [(2, 9)], f"slices in the graph, by input shape: {sliced}"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
