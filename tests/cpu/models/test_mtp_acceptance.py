#!/usr/bin/env python
"""Acceptance on known distributions, and through the head's hooks on the real families: an eval forward scores each
draft against the trunk's distribution over the same target, and leaves the loss to the next token."""

import pytest
import torch

from halo.models.mtp.acceptance import score_drafts
from tests.common.mtp_heads import install_term, packed_rows, padded_rows, tiny_model_with_head

_FAMILIES = ("qwen3_5", "qwen3_5_moe_text", "qwen3_next", "glm4_moe_lite", "qwen3_5_moe")


def _log_probs(*rows: list[float]) -> torch.Tensor:
    return torch.tensor(rows).log()


def test_the_scores_on_known_distributions():
    p = _log_probs([0.5, 0.3, 0.2], [0.1, 0.1, 0.8])
    q = _log_probs([0.2, 0.7, 0.1], [0.1, 0.2, 0.7])
    # Overlaps 0.2 + 0.3 + 0.1 and 0.1 + 0.1 + 0.7. The drafts name tokens 1 and 2.
    scored = score_drafts(lambda rows: rows, p, q, torch.tensor([1, 0]), chunk_rows=8)
    assert float(scored.overlap_sum) == pytest.approx(0.6 + 0.9, abs=1e-6)
    assert float(scored.greedy_sum) == pytest.approx(0.3 + 0.8, abs=1e-6)
    assert float(scored.top1_sum) == 1
    assert int(scored.positions) == 2


def test_chunking_does_not_move_the_scores():
    generator = torch.Generator().manual_seed(0)
    p, q = (torch.randn(9, 5, generator=generator) for _ in range(2))
    targets = torch.randint(0, 5, (9,), generator=generator)
    whole = score_drafts(lambda rows: rows, p, q, targets, chunk_rows=9)
    split = score_drafts(lambda rows: rows, p, q, targets, chunk_rows=2)
    for first, second in zip(whole, split, strict=True):
        assert float(first) == pytest.approx(float(second), abs=1e-5)


def test_no_position_scores_zero():
    empty = torch.empty(0, 4)
    scored = score_drafts(lambda rows: rows, empty, empty, torch.empty(0, dtype=torch.long), chunk_rows=2)
    assert int(scored.positions) == 0 and float(scored.overlap_sum) == 0


def _held_out(name: str, rows: dict, draft=None):
    model, head = tiny_model_with_head(name)
    install_term(model, head, weight=0.7, loss_type="soft")
    if draft is not None:
        head.forward = draft
    return model.eval(), head, model(**rows)


def _next_hidden(_next_embedding, hidden, *_args, **_kwargs):
    return torch.cat([hidden[:, 1:], hidden[:, -1:]], dim=1)


def _own_hidden(_next_embedding, hidden, *_args, **_kwargs):
    return hidden


@pytest.mark.parametrize("name", _FAMILIES)
def test_a_draft_equal_to_the_trunks_next_step_scores_full_overlap(name):
    """The draft at t and the trunk at t+1 predict one token, so the pair must be the same distribution here. Paired one
    position off, this overlap collapses."""
    rows = padded_rows()
    _, _, output = _held_out(name, rows, draft=_next_hidden)
    scored = output.mtp_acceptance
    assert int(scored.positions) > 0
    assert float(scored.overlap_sum) / int(scored.positions) == pytest.approx(1.0, abs=1e-4)

    _, _, shifted = _held_out(name, rows, draft=_own_hidden)
    assert float(shifted.mtp_acceptance.overlap_sum) / int(shifted.mtp_acceptance.positions) < 0.9999


@pytest.mark.parametrize("name", _FAMILIES)
def test_an_eval_forward_scores_the_targets_the_train_term_scores(name):
    """Padded rows: a masked label and the padding are not positions."""
    rows = padded_rows()
    _, _, output = _held_out(name, rows)
    assert int(output.mtp_acceptance.positions) == int((rows["labels"][:, 2:] != -100).sum())


@pytest.mark.parametrize("name", ["qwen3_5_moe_text", "glm4_moe_lite"])
def test_packed_rows_score_no_target_across_a_document_start(name):
    rows = packed_rows()
    _, _, output = _held_out(name, rows)
    positions = rows["position_ids"]
    expected = (rows["labels"][:, 2:] != -100) & (positions[:, 1:-1] != 0)
    assert int(output.mtp_acceptance.positions) == int(expected.sum())
    assert int(expected.sum()) < int((rows["labels"][:, 2:] != -100).sum()), "the row has no target to mask"


@pytest.mark.parametrize("name", _FAMILIES)
def test_an_eval_forward_leaves_the_loss_and_runs_the_head_with_no_graph(name):
    """``eval_loss`` stays the next-token loss a run with no head logs, and nothing routes under the soft objective."""
    rows = padded_rows()
    model, head = tiny_model_with_head(name)
    bare = model.eval()(**rows).loss
    install_term(model, head, weight=0.7, loss_type="soft")
    outputs: list[torch.Tensor] = []
    handle = head.register_forward_hook(lambda _module, _args, output: outputs.append(output))
    scored = model(**rows)
    handle.remove()
    assert outputs and not outputs[0].requires_grad
    assert torch.equal(scored.loss, bare)
    assert not hasattr(scored, "mtp_loss_split")


def test_an_eval_forward_with_no_labels_runs_no_head():
    rows = padded_rows()
    rows.pop("labels")
    model, head = tiny_model_with_head("qwen3_5")
    install_term(model, head, weight=0.7)
    calls: list[int] = []
    handle = head.register_forward_hook(lambda *_args: calls.append(1))
    output = model.eval()(**rows)
    handle.remove()
    assert not calls and not hasattr(output, "mtp_acceptance")


def test_the_training_forward_right_after_scores_nothing():
    rows = padded_rows()
    model, head = tiny_model_with_head("qwen3_5")
    install_term(model, head, weight=0.7)
    model.eval()(**rows)
    output = model.train()(**rows)
    assert not hasattr(output, "mtp_acceptance") and hasattr(output, "mtp_loss_split")


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
