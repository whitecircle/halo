#!/usr/bin/env python
"""The cross-entropy sum a caller scoring hidden states takes: the model's own head path, the same value and gradient
in chunks, and a frozen head out of the graph. Liger's fused kernel is Triton-only, so its row is a GPU one; here ``fused``
falls to the chunked path on a CPU tensor."""

import pytest
import torch
import torch.nn.functional as F

from halo.kernels.logprobs import LABEL_IGNORE_INDEX
from halo.model_loading.liger import cross_entropy
from halo.model_loading.liger.cross_entropy import head_token_ce_sum
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

_ROWS = 12


@pytest.fixture
def model():
    torch.manual_seed(0)
    return tiny_family_model(TINY_MOE_FAMILIES["qwen3_5_moe_text"]).float()


def _rows(model) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(1)
    hidden = torch.randn(_ROWS, model.config.hidden_size, generator=generator).requires_grad_()
    targets = torch.randint(0, model.config.vocab_size, (_ROWS,), generator=generator)
    targets[::3] = LABEL_IGNORE_INDEX
    return hidden, targets


def _expected(model, hidden, targets) -> torch.Tensor:
    return F.cross_entropy(model.lm_head(hidden), targets, ignore_index=LABEL_IGNORE_INDEX, reduction="sum")


def test_the_sum_is_the_unchunked_cross_entropy(model):
    hidden, targets = _rows(model)
    assert torch.isclose(head_token_ce_sum(model, fused=False)(hidden, targets), _expected(model, hidden, targets))


def test_chunks_keep_the_value_and_both_gradients(model, monkeypatch):
    hidden, targets = _rows(model)
    _expected(model, hidden, targets).backward()
    whole_hidden, whole_head = hidden.grad.clone(), model.lm_head.weight.grad.clone()
    model.zero_grad()
    hidden.grad = None
    monkeypatch.setattr(cross_entropy, "logit_chunk_rows", lambda _vocab: 5)
    head_token_ce_sum(model, fused=False)(hidden, targets).backward()
    assert torch.allclose(hidden.grad, whole_hidden)
    assert torch.allclose(model.lm_head.weight.grad, whole_head)


def test_a_frozen_head_takes_no_gradient_and_the_hidden_still_does(model):
    hidden, targets = _rows(model)
    value = head_token_ce_sum(model, fused=False)(hidden, targets, frozen_head=True)
    assert torch.isclose(value, _expected(model, hidden, targets))
    value.backward()
    assert model.lm_head.weight.grad is None
    assert hidden.grad.abs().sum() > 0


def test_an_empty_plane_is_a_zero_that_still_carries_the_graph(model):
    """A row too short for a target still owes the backward through the layer that produced it."""
    hidden = torch.empty(0, model.config.hidden_size, requires_grad=True)
    value = head_token_ce_sum(model, fused=False)(hidden, torch.empty(0, dtype=torch.long))
    assert value == 0 and value.requires_grad


def test_the_fused_request_on_a_cpu_tensor_takes_the_chunked_path(model):
    hidden, targets = _rows(model)
    assert torch.isclose(head_token_ce_sum(model, fused=True)(hidden, targets), _expected(model, hidden, targets))


def test_the_fused_kernel_runs_on_a_gpu_tensor_and_leaves_a_frozen_head_out(model, monkeypatch):
    """The call the kernel gets, on a stand-in CUDA tensor: the weight detached, and an empty plane kept away from it."""
    seen = []

    def kernel(hidden, weight, targets, bias, **kwargs):
        seen.append((weight.requires_grad, targets.numel()))
        return type("Result", (), {"loss": hidden.sum() * 0, "predicted_tokens": targets})()

    monkeypatch.setattr(cross_entropy, "liger_fused_linear_cross_entropy", kernel)
    hidden, targets = _rows(model)

    class _OnGpu(torch.Tensor):
        is_cuda = True

    plane = hidden.as_subclass(_OnGpu)
    score = head_token_ce_sum(model, fused=True)
    score(plane, targets, frozen_head=True)
    score(plane, targets)
    assert seen == [(False, _ROWS), (True, _ROWS)]
    empty = torch.empty(0, model.config.hidden_size).as_subclass(_OnGpu)
    score(empty, torch.empty(0, dtype=torch.long))
    assert len(seen) == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
