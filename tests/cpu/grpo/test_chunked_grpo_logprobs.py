"""CPU tests for the chunked GRPO log-prob / entropy helpers (src/trainers/grpo/chunked_logprobs.py).

The chunked path must compute the SAME per-token log-probs and entropy as the full-logits path TRL
uses (``selective_log_softmax`` / ``entropy_from_logits`` over ``hidden @ lm_head.weight.T / temp``),
without materializing the ``[B, T, vocab]`` logits. These tests assert that equivalence to tight
tolerance and FAIL if the chunked math drifts from the reference.

    python tests/cpu/grpo/test_chunked_grpo_logprobs.py
"""

import sys
from types import SimpleNamespace

import pytest
import torch
from transformers import LlamaConfig, LlamaForCausalLM, PretrainedConfig
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
    # Both run the same dual-chunked recompute backward over the same saved tensors, so atol/rtol are
    # 0; the plain entry only leaves the entropy accumulator out of the forward sweep.
    torch.manual_seed(4)
    weight = torch.randn(VOCAB, HIDDEN, dtype=torch.float32) * 0.05
    ids = torch.randint(0, VOCAB, (B, T))

    h_plain = torch.randn(B, T, HIDDEN, dtype=torch.float32, requires_grad=True)
    logps_plain = chunked_selective_log_softmax(h_plain, weight, ids, None, 1.0)
    logps_plain.sum().backward()

    h_fused = h_plain.detach().clone().requires_grad_(True)
    logps, _entropy = chunked_selective_log_softmax_with_entropy(h_fused, weight, ids, None, 1.0)
    logps.sum().backward()

    torch.testing.assert_close(logps, logps_plain, atol=0.0, rtol=0.0)
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


class _ModeResolutionHarness(ChunkedGRPOLogprobsMixin):
    """Records the sub-batch size ``_get_per_token_logps_and_entropies`` resolves for each callee."""

    def __init__(self):
        self.model = torch.nn.Module()
        self.model.config = PretrainedConfig()
        self.model.train()
        self.args = SimpleNamespace(per_device_train_batch_size=1, per_device_eval_batch_size=4)
        self._use_chunked_grpo_logprobs = True
        # is_main_process: TRL's @profiling_decorator on the overridden method reads it.
        self.accelerator = SimpleNamespace(unwrap_model=lambda module: module, is_main_process=False)
        self.resolved: list[int] = []

    def _chunked_logps(self, model, input_ids, attention_mask, logits_to_keep, batch_size, compute_entropy):
        self.resolved.append(batch_size)
        return torch.zeros(input_ids.size(0), logits_to_keep), None


def test_reference_logps_use_the_trainer_mode_not_the_frozen_model_flag():
    """A frozen reference model sits in eval for the whole run (``frozen_models.py`` eval()s it, TRL
    prepares it with evaluation_mode=True), so reading ITS training flag resolves the KL's reference
    forward to the eval batch size while the policy uses the train one. At the shipped env-GRPO shape
    (train 1, eval 4) that also lands the two on opposite sides of ``rows_forward_densely``: the policy
    on trimmed dense rows, the reference on the padded batch — a KL between two computations.
    """
    harness = _ModeResolutionHarness()
    reference = torch.nn.Module()
    reference.eval()
    ids, mask = torch.zeros(2, 6, dtype=torch.long), torch.ones(2, 6, dtype=torch.long)

    harness._get_per_token_logps_and_entropies(harness.model, ids, mask, 3)
    harness._get_per_token_logps_and_entropies(reference, ids, mask, 3)

    assert harness.resolved == [1, 1], (
        f"policy and reference resolved different sub-batch sizes {harness.resolved}; at train=1/eval=4 "
        "that puts them on opposite sides of the dense/padded split"
    )

    harness.model.eval()  # the trainer's own eval pass moves both
    harness._get_per_token_logps_and_entropies(reference, ids, mask, 3)
    assert harness.resolved[-1] == 4


class _RealBackboneHarness(ChunkedGRPOLogprobsMixin):
    """A real causal LM behind ``_get_last_hidden_state``, reproducing TRL's own slicing.

    The embedding harness above is position-free, so it cannot see what the dense path does to
    attention and RoPE: the trimmed row is forwarded mask-free at positions ``[0, len)`` while the
    padded row runs at positions offset by its left padding.
    """

    temperature = 1.0

    def __init__(self, attn_implementation: str):
        config = LlamaConfig(
            vocab_size=64,
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
        torch.manual_seed(0)
        self.model = LlamaForCausalLM(config).eval()
        self.model.config._attn_implementation = attn_implementation
        self.model.model.config._attn_implementation = attn_implementation

    def _get_last_hidden_state(self, model, input_ids, attention_mask, logits_to_keep):
        hidden = model.model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False).last_hidden_state
        return hidden[:, :-1, :][:, -logits_to_keep:, :]


@pytest.mark.parametrize("attn_implementation", ["sdpa", "eager"])
def test_dense_rows_match_the_padded_batch_on_a_real_attention_stack(attn_implementation):
    # The dense arm trims each row and drops its mask, so the row's causal mask and RoPE positions
    # both change. Only a real attention + RoPE stack can show that the log-probs are unaffected;
    # an off-by-one span, a mask-free row that still carries padding, or a mis-tiled head sweep
    # would all move them.
    input_ids, attention_mask, completion_mask, ltk = _ragged_batch()
    harness = _RealBackboneHarness(attn_implementation)

    with torch.no_grad():
        padded, _ = harness._chunked_logps_impl(harness.model, input_ids, attention_mask, ltk, 2, False)
        dense, _ = harness._chunked_logps_impl(harness.model, input_ids, attention_mask, ltk, 1, False)

    torch.testing.assert_close(dense[completion_mask], padded[completion_mask], atol=1e-5, rtol=1e-5)
    assert torch.count_nonzero(dense[~completion_mask]) == 0


def test_backward_matches_autograd_for_every_input_across_tiles(monkeypatch):
    # The recompute backward reconstructs each logits tile from the saved log_z, so every input's
    # gradient — the head bias included, whose only chain to the loss is through this Function — has
    # to carry the 1/temperature factor and land in the right vocab tile. An upstream gradient that
    # varies per token catches a per-row scale that .sum() would hide.
    monkeypatch.setattr(chunked_logprobs, "_SEQ_CHUNK", 5)
    monkeypatch.setattr(chunked_logprobs, "_VOCAB_CHUNK", 7)
    vocab, b, t, hidden_size = 23, 3, 11, 6
    torch.manual_seed(5)
    hidden0 = torch.randn(b, t, hidden_size)
    weight0 = torch.randn(vocab, hidden_size) * 0.3
    bias0 = torch.randn(vocab) * 0.2
    ids = torch.randint(0, vocab, (b, t))  # targets land in every vocab tile
    upstream = torch.randn(b, t)

    def leaves(with_bias):
        h = hidden0.clone().requires_grad_(True)
        w = weight0.clone().requires_grad_(True)
        return h, w, bias0.clone().requires_grad_(True) if with_bias else None

    for temperature in (1.0, 0.7):
        for with_bias in (False, True):
            ref_h, ref_w, ref_b = leaves(with_bias)
            (selective_log_softmax(_ref_logits(ref_h, ref_w, ref_b, temperature), ids) * upstream).sum().backward()

            got_h, got_w, got_b = leaves(with_bias)
            (chunked_selective_log_softmax(got_h, got_w, ids, got_b, temperature) * upstream).sum().backward()

            for got, ref in ((got_h, ref_h), (got_w, ref_w), (got_b, ref_b)):
                if ref is None:
                    continue
                assert got.grad is not None
                torch.testing.assert_close(got.grad, ref.grad, atol=1e-4, rtol=1e-4)


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
