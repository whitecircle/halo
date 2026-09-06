"""CPU tests for the chunked GRPO log-prob / entropy helpers (src/trainers/grpo/chunked_logprobs.py).

The chunked path must compute the SAME per-token log-probs and entropy as the full-logits path TRL
uses (``selective_log_softmax`` / ``entropy_from_logits`` over ``hidden @ lm_head.weight.T / temp``),
without materializing the ``[B, T, vocab]`` logits. These tests assert that equivalence to tight
tolerance and FAIL if the chunked math drifts from the reference.

    python tests/cpu/grpo/test_chunked_grpo_logprobs.py
"""

import sys

import pytest
import torch
from transformers import PretrainedConfig
from trl.trainer.utils import entropy_from_logits, selective_log_softmax

from src.distributed.runtime import materialize_dtensor
from src.trainers.grpo.mixins import chunked_logprobs
from src.trainers.grpo.mixins.chunked_logprobs import (
    _VOCAB_CHUNK,
    ChunkedGRPOLogprobsMixin,
    chunked_selective_log_softmax,
    chunked_selective_log_softmax_with_entropy,
    rows_forward_densely,
)

# Vocab spans multiple chunk tiles so the online-softmax accumulation across chunks is exercised.
VOCAB = _VOCAB_CHUNK * 2 + 137
HIDDEN = 32
B, T = 2, 16


def _ref_logits(hidden, weight, bias, temperature):
    logits = torch.matmul(hidden, weight.t())
    if bias is not None:
        logits = logits + bias
    return logits / temperature


def test_chunked_logps_match_full_path():
    torch.manual_seed(0)
    hidden = torch.randn(B, T, HIDDEN, dtype=torch.float32)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float32) * 0.05
    ids = torch.randint(0, VOCAB, (B, T))
    for temperature in (1.0, 0.7):
        ref = selective_log_softmax(_ref_logits(hidden, weight, None, temperature), ids)
        got = chunked_selective_log_softmax(hidden, weight, ids, None, temperature)
        assert got.shape == (B, T)
        torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


def test_chunked_logps_match_with_bias():
    torch.manual_seed(1)
    hidden = torch.randn(B, T, HIDDEN, dtype=torch.float32)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float32) * 0.05
    bias = torch.randn(VOCAB, dtype=torch.float32) * 0.1
    ids = torch.randint(0, VOCAB, (B, T))
    ref = selective_log_softmax(_ref_logits(hidden, weight, bias, 1.0), ids)
    got = chunked_selective_log_softmax(hidden, weight, ids, bias, 1.0)
    torch.testing.assert_close(got, ref, atol=1e-4, rtol=1e-4)


def test_chunked_logps_backward_matches():
    torch.manual_seed(2)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float32) * 0.05
    ids = torch.randint(0, VOCAB, (B, T))

    h_ref = torch.randn(B, T, HIDDEN, dtype=torch.float32, requires_grad=True)
    selective_log_softmax(_ref_logits(h_ref, weight, None, 1.0), ids).sum().backward()

    h_chunk = h_ref.detach().clone().requires_grad_(True)
    chunked_selective_log_softmax(h_chunk, weight, ids, None, 1.0).sum().backward()

    torch.testing.assert_close(h_chunk.grad, h_ref.grad, atol=1e-4, rtol=1e-4)


def test_fused_entropy_matches_full_path():
    # Both outputs are checked: a fused sweep can corrupt the diagnostic or the loss numerator alone.
    torch.manual_seed(3)
    hidden = torch.randn(B, T, HIDDEN, dtype=torch.float32)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float32) * 0.05
    bias = torch.randn(VOCAB, dtype=torch.float32) * 0.1
    ids = torch.randint(0, VOCAB, (B, T))
    for temperature in (1.0, 0.7):
        for b in (None, bias):
            ref_entropy = entropy_from_logits(_ref_logits(hidden, weight, b, temperature))
            ref_logps = selective_log_softmax(_ref_logits(hidden, weight, b, temperature), ids)
            logps, entropy = chunked_selective_log_softmax_with_entropy(hidden, weight, ids, b, temperature)
            assert entropy.shape == (B, T)
            assert not entropy.requires_grad  # detached diagnostic
            torch.testing.assert_close(entropy, ref_entropy, atol=1e-4, rtol=1e-4)
            torch.testing.assert_close(logps, ref_logps, atol=1e-4, rtol=1e-4)


def test_fused_backward_matches_plain_chunked():
    # Both delegate backward to Liger's recompute path over the same saved tensors, so atol/rtol are 0.
    torch.manual_seed(4)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float32) * 0.05
    ids = torch.randint(0, VOCAB, (B, T))

    h_plain = torch.randn(B, T, HIDDEN, dtype=torch.float32, requires_grad=True)
    chunked_selective_log_softmax(h_plain, weight, ids, None, 1.0).sum().backward()

    h_fused = h_plain.detach().clone().requires_grad_(True)
    logps, _entropy = chunked_selective_log_softmax_with_entropy(h_fused, weight, ids, None, 1.0)
    logps.sum().backward()

    torch.testing.assert_close(h_fused.grad, h_plain.grad, atol=0.0, rtol=0.0)


def test_materialize_passthrough_for_plain_tensors():
    t = torch.randn(4, 4)
    assert materialize_dtensor(t) is t
    assert materialize_dtensor(None) is None


class _DenseHarness(ChunkedGRPOLogprobsMixin):
    """Minimal harness exposing ``_dense_last_hidden_state`` with a backbone whose hidden state is a
    differentiable function of a single leaf param — so we can assert each row's backward reaches it.

    ``_get_last_hidden_state`` faithfully emulates TRL's: forward → drop the final position (the causal
    next-token shift) → keep the last ``logits_to_keep``. A single-token span therefore yields a
    0-length hidden, exactly the shape a naive empty-row connection breaks on."""

    def __init__(self, hidden_size: int):
        self.param = torch.zeros(hidden_size, requires_grad=True)

    def _get_last_hidden_state(self, model, input_ids, attention_mask, logits_to_keep):
        h = self.param.numel()
        seq_len = input_ids.size(1)
        full = (self.param + 1.0).reshape(1, 1, h).expand(1, seq_len, h)  # (1, L, H), connected to param
        full = full[:, :-1, :]  # drop final position (causal shift) — 1-token span → 0-length
        return full[:, -logits_to_keep:, :]


def _dense_batch():
    # Row 1 is a dummy padding row: n_comp == 0, so its 1-token span collapses to a 0-length hidden.
    input_ids = torch.tensor([[5, 6, 7], [5, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1], [1, 0, 0]])
    return input_ids, attention_mask, 2  # logits_to_keep


def test_dense_empty_row_stays_connected_to_graph():
    # A disconnected zeros row gets its backward pruned, so the FSDP/EP grad collectives vanish only on
    # the rank that padded more dummy rows — a NCCL deadlock. The shape assert covers the widened span.
    harness = _DenseHarness(hidden_size=8)
    input_ids, attention_mask, ltk = _dense_batch()

    out, n_comps = harness._dense_last_hidden_state(None, input_ids, attention_mask, ltk)
    assert out.shape == (2, ltk, 8)
    assert n_comps == [2, 0]

    harness.param.grad = None
    out[1].sum().backward()  # the dummy row alone must still reach the param
    assert harness.param.grad is not None and harness.param.grad.abs().sum() > 0, (
        "empty (n_comp==0) row is disconnected from the model graph — its backward would be pruned, "
        "desyncing FSDP/EP collectives across ranks"
    )


def test_dense_real_row_values_unchanged_by_fix():
    # Keeping the empty row connected must not perturb a real row's values.
    harness = _DenseHarness(hidden_size=8)
    input_ids, attention_mask, ltk = _dense_batch()
    out, _n_comps = harness._dense_last_hidden_state(None, input_ids, attention_mask, ltk)

    expected_real = harness.param.detach() + 1.0
    torch.testing.assert_close(out[0, 0], expected_real)
    torch.testing.assert_close(out[0, 1], expected_real)
    # Row 1 connects position 0 only; its value is irrelevant (masked downstream), position 1 stays zero.
    assert torch.count_nonzero(out[1, 1]) == 0


class _EmbeddingBackboneHarness(ChunkedGRPOLogprobsMixin):
    """Backbone = a position-free embedding lookup (drop the final position, keep the last
    ``logits_to_keep``), so a row trimmed to its real span and its padded twin must agree exactly on
    every real completion position; the head is a plain Linear. Records each backbone call's input
    shape and whether it ran mask-free."""

    temperature = 1.0

    def __init__(self, attn_impl: str, vocab: int = 40, hidden: int = 8):
        torch.manual_seed(0)
        self.embed = torch.nn.Embedding(vocab, hidden)
        self.model = torch.nn.Module()
        self.model.lm_head = torch.nn.Linear(hidden, vocab, bias=False)
        self.model.config = PretrainedConfig()
        self.model.config._attn_implementation = attn_impl
        self.model.get_output_embeddings = lambda: self.model.lm_head
        self.forward_calls: list[tuple[tuple[int, ...], bool]] = []

    def _get_last_hidden_state(self, model, input_ids, attention_mask, logits_to_keep):
        self.forward_calls.append((tuple(input_ids.shape), attention_mask is None))
        full = self.embed(input_ids)[:, :-1, :]
        return full[:, -logits_to_keep:, :]


def _ragged_batch():
    # row 0: prompt 3 + completion 4, no padding; row 1: left-pad 1 + prompt 2 + completion 2 + right-pad 2.
    input_ids = torch.tensor([[11, 12, 13, 21, 22, 23, 24], [0, 14, 15, 25, 26, 0, 0]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1, 1], [0, 1, 1, 1, 1, 0, 0]])
    completion_mask = torch.tensor([[1, 1, 1, 1], [1, 1, 0, 0]], dtype=torch.bool)
    return input_ids, attention_mask, completion_mask, 4


def test_rows_forward_densely_on_every_attention_path_at_batch_size_one():
    sdpa = _EmbeddingBackboneHarness("sdpa").model
    fa4 = _EmbeddingBackboneHarness("flash_attention_4").model
    assert rows_forward_densely(sdpa, 1) and not rows_forward_densely(sdpa, 2)
    assert rows_forward_densely(fa4, 1) and rows_forward_densely(fa4, 4)


def test_single_row_dense_sweep_matches_padded_batch_on_real_positions():
    # At batch size 1 the SDPA path forwards each row trimmed (mask-free) and sweeps only its real
    # completion tokens; the padded path is the reference on every real position, pads stay zero.
    input_ids, attention_mask, completion_mask, ltk = _ragged_batch()
    padded = _EmbeddingBackboneHarness("sdpa")
    logps_pad, ent_pad = padded._chunked_logps_impl(padded.model, input_ids, attention_mask, ltk, 2, True)
    assert padded.forward_calls == [((2, 7), False)]

    dense = _EmbeddingBackboneHarness("sdpa")
    logps_dense, ent_dense = dense._chunked_logps_impl(dense.model, input_ids, attention_mask, ltk, 1, True)
    assert dense.forward_calls == [((1, 7), True), ((1, 4), True)]

    torch.testing.assert_close(logps_dense[completion_mask], logps_pad[completion_mask])
    torch.testing.assert_close(ent_dense[completion_mask], ent_pad[completion_mask])
    assert torch.count_nonzero(logps_dense[~completion_mask]) == 0
    assert torch.count_nonzero(ent_dense[~completion_mask]) == 0
    assert torch.isfinite(logps_dense).all()

    logps_dense.sum().backward()
    assert dense.model.lm_head.weight.grad is not None and dense.embed.weight.grad is not None


def test_sweep_matches_full_path_across_sequence_and_vocab_tiles(monkeypatch):
    # Both loops must cross tile boundaries: sequence tiles carry independent online-softmax
    # accumulators, vocab tiles rescale them.
    monkeypatch.setattr(chunked_logprobs, "_SEQ_CHUNK", 5)
    monkeypatch.setattr(chunked_logprobs, "_VOCAB_CHUNK", 7)
    vocab, temperature = 23, 0.8
    torch.manual_seed(2)
    hidden = torch.randn(B, T, HIDDEN, requires_grad=True)
    weight = torch.randn(vocab, HIDDEN, requires_grad=True)
    bias = torch.randn(vocab)
    ids = torch.randint(0, vocab, (B, T))

    h_ref = hidden.detach().clone().requires_grad_(True)
    w_ref = weight.detach().clone().requires_grad_(True)
    ref_logits = _ref_logits(h_ref, w_ref, bias, temperature)
    ref_logps = selective_log_softmax(ref_logits, ids)
    ref_entropy = entropy_from_logits(ref_logits)
    ref_logps.sum().backward()

    logps, entropy = chunked_selective_log_softmax_with_entropy(hidden, weight, ids, bias, temperature)
    logps.sum().backward()

    torch.testing.assert_close(logps, ref_logps, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(entropy, ref_entropy, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(hidden.grad, h_ref.grad, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(weight.grad, w_ref.grad, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
