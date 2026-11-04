#!/usr/bin/env python
"""The hyper-connection kernel patches reach exactly the families whose class has the mixing matrix, the knob leaves
upstream's classes alone, and a model off CUDA runs upstream's own forward. The kernels' numerics are the GPU suite's."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import transformers
import transformers.models.deepseek_v4.modeling_deepseek_v4 as deepseek_v4_modeling
import transformers.models.glm5_next.modeling_glm5_next as glm5_next_modeling
from transformers import DeepseekV4Config, DeepseekV4ForCausalLM

from halo.model_loading.model_preparation import apply_family_patches
from halo.models import hyper_connections
from halo.models.hyper_connections import _FAMILIES, _PATCHED_ATTR, apply_hyper_connection_patches
from tests.common.models import TINY_DSV4_CONFIG

_PATCHED_CLASSES = tuple(owner for patches in _FAMILIES.values() for owner, _ in patches)
_GLM5_NEXT_CLASSES = (glm5_next_modeling.Glm5NextTextHyperConnection, glm5_next_modeling.Glm5NextTextDecoderLayer)
_DEEPSEEK_V4_CLASSES = (deepseek_v4_modeling.DeepseekV4HyperConnection, deepseek_v4_modeling.DeepseekV4DecoderLayer)


def _upstream(owner: type):
    forward = owner.forward
    return forward.__wrapped__ if getattr(forward, _PATCHED_ATTR, False) else forward


@pytest.fixture(autouse=True)
def upstream_classes(monkeypatch):
    """Each test starts from upstream's forwards, whatever an earlier one in this process installed."""
    monkeypatch.delenv("HALO_HC_KERNELS", raising=False)
    for owner in _PATCHED_CLASSES:
        monkeypatch.setattr(owner, "forward", _upstream(owner))


def _patched(owners: tuple[type, ...]) -> list[bool]:
    return [getattr(owner.forward, _PATCHED_ATTR, False) for owner in owners]


@pytest.mark.parametrize(
    ("config", "reached"),
    [
        pytest.param(transformers.Glm5NextConfig(), _GLM5_NEXT_CLASSES, id="glm5_next"),
        pytest.param(transformers.Glm5NextTextConfig(), _GLM5_NEXT_CLASSES, id="glm5_next_text"),
        pytest.param(DeepseekV4Config(), _DEEPSEEK_V4_CLASSES, id="deepseek_v4"),
        pytest.param(transformers.HYV4Config(), (), id="hy_v4"),
        pytest.param(transformers.Qwen3MoeConfig(), (), id="qwen3_moe"),
    ],
)
def test_the_family_patches_reach_only_the_classes_with_the_mixing_matrix(config, reached):
    apply_family_patches(config, None)
    assert _patched(_PATCHED_CLASSES) == [owner in reached for owner in _PATCHED_CLASSES]


def test_applying_twice_wraps_once():
    apply_hyper_connection_patches(DeepseekV4Config())
    installed = [owner.forward for owner in _DEEPSEEK_V4_CLASSES]
    apply_hyper_connection_patches(DeepseekV4Config())
    assert [owner.forward for owner in _DEEPSEEK_V4_CLASSES] == installed


def test_the_knob_leaves_upstream_alone(monkeypatch):
    monkeypatch.setenv("HALO_HC_KERNELS", "0")
    stock = [owner.forward for owner in _PATCHED_CLASSES]
    apply_hyper_connection_patches(transformers.Glm5NextConfig())
    apply_hyper_connection_patches(DeepseekV4Config())
    assert [owner.forward for owner in _PATCHED_CLASSES] == stock


def test_a_moved_upstream_signature_fails_at_load(monkeypatch):
    def moved(self, hidden_streams, cache):
        raise AssertionError("never called")

    monkeypatch.setattr(deepseek_v4_modeling.DeepseekV4HyperConnection, "forward", moved)
    with pytest.raises(RuntimeError, match="re-derive it"):
        apply_hyper_connection_patches(DeepseekV4Config())


def test_a_model_off_cuda_runs_upstream():
    """No kernel is reached, and every output and gradient is upstream's bit for bit."""

    def refused(*_args, **_kwargs):
        raise AssertionError("a kernel was reached off CUDA")

    def run() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        torch.manual_seed(7)
        model = DeepseekV4ForCausalLM(DeepseekV4Config(**TINY_DSV4_CONFIG, attn_implementation="eager"))
        ids = torch.randint(0, model.config.vocab_size, (2, 16), generator=torch.Generator().manual_seed(5))
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        return loss.detach(), {n: p.grad for n, p in model.named_parameters() if p.grad is not None}

    stock_loss, stock_grads = run()
    apply_hyper_connection_patches(DeepseekV4Config())
    assert all(_patched(_DEEPSEEK_V4_CLASSES))
    with (
        patch.object(hyper_connections, "project_streams", refused),
        patch.object(hyper_connections, "sinkhorn_knopp", refused),
        patch.object(hyper_connections, "collapse_streams", refused),
        patch.object(hyper_connections, "mix_streams", refused),
    ):
        loss, grads = run()
    assert torch.equal(loss, stock_loss)
    assert grads.keys() == stock_grads.keys()
    for name, grad in grads.items():
        assert torch.equal(grad, stock_grads[name]), name


def test_an_input_on_cuda_takes_the_replacement():
    """The dispatch alone, on a stand-in: the replacement runs on GPU in the kernel suite."""
    owner = type("Owner", (), {"forward": lambda self, hidden_states: "upstream"})
    hyper_connections._install(owner, lambda self, hidden_states: "replacement")
    assert owner().forward(SimpleNamespace(is_cuda=False)) == "upstream"
    assert owner().forward(SimpleNamespace(is_cuda=True)) == "replacement"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
