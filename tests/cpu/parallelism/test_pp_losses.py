#!/usr/bin/env python
"""CPU test: the pipeline CE — label-roll shift and chunked-checkpoint equivalence.

``causal_lm_token_loss`` rolls labels instead of slicing logits and chunks the fp32 CE under a
non-reentrant checkpoint. Both must be exact: values AND gradients must match the naive
sliced/monolithic formulation, including across the chunk boundary (the GPU gates run tiny
sequences that never exceed one chunk, so the multi-chunk path lives here).

    python tests/cpu/parallelism/test_pp_losses.py
"""

import pytest
import torch
import torch.nn.functional as F

import src.distributed.pipeline_parallel.losses as pp_losses
from src.distributed.pipeline_parallel.losses import causal_lm_token_loss


def _naive_reference(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)).float(),
        shift_labels.view(-1),
        ignore_index=-100,
        reduction="sum",
    )


def _make(batch: int, seq: int, vocab: int = 37, seed: int = 3):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(batch, seq, vocab, generator=g, requires_grad=True)
    labels = torch.randint(0, vocab, (batch, seq), generator=g)
    labels[:, :3] = -100  # ignored prefix
    return logits, labels


@pytest.mark.parametrize("batch,seq", [(2, 16), (3, 33)])
def test_single_chunk_matches_naive(batch, seq):
    logits, labels = _make(batch, seq)
    loss = causal_lm_token_loss(logits, labels)
    ref = _naive_reference(logits.detach().clone().requires_grad_(True), labels)
    assert torch.allclose(loss, ref, atol=1e-5)


def test_multi_chunk_values_and_grads_match(monkeypatch):
    # Several chunks with an uneven tail: the path a wrong chunk boundary breaks.
    monkeypatch.setattr(pp_losses, "_CE_CHUNK_ELEMENTS", 50 * 37)  # 50 rows at vocab 37
    logits, labels = _make(4, 40)  # 160 tokens -> 4 chunks (50/50/50/10)
    loss = causal_lm_token_loss(logits, labels)
    loss.backward()

    ref_logits = logits.detach().clone().requires_grad_(True)
    ref = _naive_reference(ref_logits, labels)
    ref.backward()

    assert torch.allclose(loss, ref, atol=1e-5)
    assert torch.allclose(logits.grad, ref_logits.grad, atol=1e-6), (
        f"max grad diff {(logits.grad - ref_logits.grad).abs().max()}"
    )


def test_chunk_budget_is_vocab_aware():
    """The held fp32 plane must stay bounded as the vocabulary grows.

    The chunk budget is what keeps the last stage's fp32 upcast off the memory peak, and the plane it
    slices is ``tokens x V``. Sizing it in token ROWS bounds nothing in particular: at gpt-oss's
    V=201k a 4096-row chunk is 3.3 GB — the size of the entire bf16 plane it is supposed to be a
    fraction of — while at V=32k the same constant is 0.5 GB. Asserting a flat ceiling in BYTES is
    what catches a regression back to a row-count budget.
    """
    ceiling = pp_losses._CE_CHUNK_ELEMENTS * 4  # fp32
    for vocab in (32_000, 151_936, 201_088, 262_144):
        rows = pp_losses._ce_chunk_rows(vocab)
        assert rows >= 1, f"vocab {vocab} must still take at least one row per chunk"
        assert rows * vocab * 4 <= ceiling, (
            f"vocab {vocab}: chunk holds {rows * vocab * 4 / 1e9:.2f} GB of fp32, over the "
            f"{ceiling / 1e9:.2f} GB budget — the chunk is sized in rows, not elements"
        )
    # A vocabulary past the whole budget must not round down to a zero-row (infinite-loop) chunk.
    assert pp_losses._ce_chunk_rows(pp_losses._CE_CHUNK_ELEMENTS * 2) == 1


def test_all_ignored_chunk_is_inert(monkeypatch):
    monkeypatch.setattr(pp_losses, "_CE_CHUNK_ELEMENTS", 8 * 37)  # 8 rows at vocab 37
    logits, labels = _make(1, 32)
    labels[:] = -100
    loss = causal_lm_token_loss(logits, labels)
    assert float(loss) == 0.0
    loss.backward()
    assert torch.count_nonzero(logits.grad) == 0


def _unchunked_token_logprobs(logits, labels):
    """The pre-chunking formulation: two full fp32 ``[B, S-1, V]`` planes, held at once."""
    shift_logits = logits[:, :-1, :].float()
    shift_labels = labels[:, 1:]
    mask = shift_labels != -100
    safe_labels = shift_labels.masked_fill(~mask, 0)
    logps = torch.log_softmax(shift_logits, dim=-1).gather(-1, safe_labels.unsqueeze(-1)).squeeze(-1)
    return logps, mask


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
def test_chunked_token_logprobs_match_the_full_plane(monkeypatch, batch, seq, budget_rows):
    """Per-token log-probs must be BITWISE identical to the unchunked formulation, grads included,
    and backward must hold no fp32 vocab-wide tensor.

    ``token_logprobs`` feeds every preference/RL PP adapter (DPO, KTO, SMPO, offline GRPO), and its
    kernel, ``selective_logprobs``, also replaces TRL's bf16 log-softmax on the DPO/KTO loss path; it
    takes the shifted ``logits[:, :-1]`` view as TRL passes it. Building the fp32 upcast and the
    ``log_softmax`` output as whole ``[B, S-1, V]`` planes costs ~26 GB per microbatch at
    ``V=201088, S=8192`` — which is why it chunks. Chunking is only legitimate if the numbers do not
    move, so this asserts equality rather than closeness. A row left out of the checkpoint keeps its
    fp32 log-softmax alive until backward, a whole fp32 plane across the batch.
    """
    vocab = 37
    monkeypatch.setattr(pp_losses, "_CE_CHUNK_ELEMENTS", budget_rows * vocab)
    torch.manual_seed(batch * 100 + seq)
    base = torch.randn(batch, seq, vocab, dtype=torch.bfloat16)
    labels = torch.randint(0, vocab, (batch, seq))
    labels[:, :2] = -100
    labels[-1, :] = -100  # an inert all-ignore row, as the PP eval row-padding emits

    chunked_in = base.clone().requires_grad_(True)
    full_in = base.clone().requires_grad_(True)
    saved = []

    def record(tensor):
        saved.append(tensor)
        return tensor

    with torch.autograd.graph.saved_tensors_hooks(record, lambda tensor: tensor):
        got_lp, got_mask = pp_losses.token_logprobs(chunked_in, labels)
    want_lp, want_mask = _unchunked_token_logprobs(full_in, labels)

    assert got_lp.shape == want_lp.shape == (batch, seq - 1)
    assert got_lp.dtype == torch.float32
    assert torch.equal(got_mask, want_mask)
    assert torch.equal(got_lp, want_lp), f"max diff {(got_lp - want_lp).abs().max()}"
    held = [tuple(t.shape) for t in saved if t.dtype == torch.float32 and t.dim() and t.size(-1) == vocab]
    assert not held, f"fp32 vocab-wide tensors held for backward: {held}"
    # Only the shifted view: a slice per chunk zero-fills a whole-row gradient per chunk in backward.
    assert _grad_fn_names(got_lp).count("SliceBackward0") == 1

    (got_lp * got_mask).sum().backward()
    (want_lp * want_mask).sum().backward()
    assert torch.equal(chunked_in.grad, full_in.grad), f"max grad diff {(chunked_in.grad - full_in.grad).abs().max()}"


def test_selective_logprobs_refuses_a_top_k_index():
    """TRL's ``selective_log_softmax`` also gathers K log-probs per position; this one does not."""
    with pytest.raises(ValueError, match="one index per position"):
        pp_losses.selective_logprobs(torch.randn(2, 5, 11), torch.zeros(2, 5, 3, dtype=torch.long))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
