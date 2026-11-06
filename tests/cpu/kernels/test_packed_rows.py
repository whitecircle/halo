"""The packed-row gate helpers (tests/common/packed_rows.py) on a tiny CPU model, so a helper that backpropagates
under ``torch.no_grad`` fails here and not in the first GPU launch."""

import pytest
import torch

from tests.common import packed_rows as gate
from tests.common.tiny_models import TINY_DENSE_ATTENTION_FAMILIES, tiny_family_model


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return tiny_family_model(TINY_DENSE_ATTENTION_FAMILIES["llama"], overrides={"num_hidden_layers": 2}).eval()


def test_isolation_check_passes_when_documents_are_isolated(model):
    checks = {}
    gate.check_isolation(model, "sdpa", "position_ids", (12, 9, 7), checks)
    assert checks["sdpa_a_neighbour_does_not_move_the_documents_around_it"] is True


def test_isolation_check_detects_a_leak_across_documents(model, monkeypatch):
    """A row that hands the model no document boundary is one causal sequence: the changed second document moves the
    third, and the check must say so."""
    monkeypatch.setattr(gate, "emission_kwargs", lambda emission, position_ids, config: {})
    checks = {}
    gate.check_isolation(model, "sdpa", "position_ids", (12, 9, 7), checks)
    assert checks["sdpa_a_neighbour_does_not_move_the_documents_around_it"] is False


def test_step_returns_gradients_and_logits_carry_the_graph(model):
    documents = gate.build_documents((12, 9), model.config.vocab_size, "cpu")
    logits = gate.row_logits(model, documents, None)
    assert logits.requires_grad
    step_logits, grads = gate.row_step(model, documents, None)
    assert grads and torch.equal(step_logits, logits.detach().float())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
