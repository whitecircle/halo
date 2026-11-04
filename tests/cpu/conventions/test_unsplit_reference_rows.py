#!/usr/bin/env python
"""``run_unsplit_reference``'s per-row forwards score a packed row as its documents alone: each row runs without a
KV cache, since transformers reads restarting positions as documents only where no cache is live."""

import pytest
import torch

from halo.models.segment_markers import segment_marker_kwargs, segment_markers_for
from tests.common.packed_flash_varlen import set_attn_implementation
from tests.common.pp_equivalence import run_unsplit_reference, token_ce_sums
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

DOCUMENTS = (20, 27, 17)
# fp32, the packed row's chunks against each document's: reduction order alone.
REL_TOL = 1e-3
# A row that attends and carries state across its documents moves the decays' gradients by O(1).
CROSSING_REL_MIN = 1e-1


def _model():
    torch.manual_seed(0)
    model = tiny_family_model(TINY_MOE_FAMILIES["qwen3_5_moe_text"]).float()
    set_attn_implementation(model.config, "sdpa")
    return model


def _packed_row(model) -> tuple[torch.Tensor, torch.Tensor, dict]:
    positions = torch.cat([torch.arange(n) for n in DOCUMENTS]).unsqueeze(0)
    input_ids = torch.randint(2, model.config.vocab_size, positions.shape, generator=torch.Generator().manual_seed(1))
    labels = input_ids.masked_fill(positions == 0, -100)
    markers = {"position_ids": positions, **segment_marker_kwargs(positions, segment_markers_for(model.config))}
    return input_ids, labels, markers


def _documents_alone(model, input_ids: torch.Tensor, labels: torch.Tensor) -> dict[str, torch.Tensor]:
    """The packed row's gradients as each document alone gives them, under the packed row's token count."""
    model.zero_grad(set_to_none=True)
    starts = [sum(DOCUMENTS[:i]) for i in range(len(DOCUMENTS))]
    summed = sum(
        token_ce_sums(model(input_ids=input_ids[:, s : s + n]).logits, labels[:, s : s + n]).sum()
        for s, n in zip(starts, DOCUMENTS, strict=True)
    )
    (summed / int((labels[:, 1:] != -100).sum())).backward()
    grads = {name: param.grad.detach().clone() for name, param in model.named_parameters() if param.grad is not None}
    model.zero_grad(set_to_none=True)
    return grads


def _worst_rel(grads: dict[str, torch.Tensor], expected: dict[str, torch.Tensor]) -> float:
    assert grads.keys() == expected.keys()
    return max(float((grads[name] - ref).norm() / ref.norm().clamp_min(1e-30)) for name, ref in expected.items())


def test_a_packed_row_scores_as_its_documents_alone():
    model = _model()
    input_ids, labels, markers = _packed_row(model)
    expected = _documents_alone(model, input_ids, labels)
    reference = run_unsplit_reference(model, input_ids, labels, "cpu", 1, row_kwargs=[markers])
    assert _worst_rel(reference.grads, expected) < REL_TOL


def test_a_live_cache_crosses_the_documents():
    """The same row under a live cache crosses its documents, so the comparison above can fail."""
    model = _model()
    input_ids, labels, markers = _packed_row(model)
    expected = _documents_alone(model, input_ids, labels)
    reference = run_unsplit_reference(model, input_ids, labels, "cpu", 1, row_kwargs=[{**markers, "use_cache": True}])
    assert _worst_rel(reference.grads, expected) > CROSSING_REL_MIN


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
