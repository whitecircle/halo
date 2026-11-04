#!/usr/bin/env python
"""The hyper-connection kernel patches reach exactly the families whose class has the mixing matrix, the knob leaves
upstream's classes alone, a model off CUDA runs upstream's own forward, and each replacement forward, over eager forms of
the kernels, is upstream's bit for bit. The kernels' numerics are the GPU suite's."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F
import transformers
import transformers.models.deepseek_v4.modeling_deepseek_v4 as deepseek_v4_modeling
import transformers.models.glm5_next.modeling_glm5_next as glm5_next_modeling
from torch.distributed.tensor import Shard, distribute_tensor
from transformers import DeepseekV4Config, DeepseekV4ForCausalLM

from halo.kernels.hyper_connection import project_streams
from halo.model_loading.model_preparation import apply_family_patches
from halo.models import hyper_connections
from halo.models.hyper_connections import _FAMILIES, _PATCHED_ATTR, apply_hyper_connection_patches
from halo.models.moe.family_specs import find_moe_family_spec
from tests.common.distributed import fake_process_group_mesh
from tests.common.models import TINY_DSV4_CONFIG
from tests.common.tiny_models import TINY_MOE_FAMILIES, tiny_family_model

_PATCHED_CLASSES = tuple(owner for patches in _FAMILIES.values() for owner, _ in patches)
_GLM5_NEXT_CLASSES = (glm5_next_modeling.Glm5NextTextHyperConnection, glm5_next_modeling.Glm5NextTextDecoderLayer)
_GENERATED_TOKENS = 4
_DEEPSEEK_V4_CLASSES = (
    deepseek_v4_modeling.DeepseekV4HyperConnection,
    deepseek_v4_modeling.DeepseekV4DecoderLayer,
    deepseek_v4_modeling.DeepseekV4HyperHead,
)


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


def test_a_projection_with_no_local_fn_is_refused():
    """The one kernel input read straight off a parameter. A launch would take its null address as valid."""
    stream = torch.zeros(1, 3, 2, 4)
    width = stream.shape[-2] * stream.shape[-1]
    with pytest.raises(RuntimeError, match="no local data"):
        project_streams(stream, torch.empty(8, width, device="meta"), 1e-5)
    with fake_process_group_mesh(0, 2) as mesh:
        sharded = distribute_tensor(torch.zeros(8, width), mesh, [Shard(0)], src_data_rank=None)
        with pytest.raises(RuntimeError, match="no local data"):
            project_streams(stream, sharded, 1e-5)


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
    assert owner().forward(SimpleNamespace(is_cuda=False, dtype=torch.bfloat16)) == "upstream"
    assert owner().forward(SimpleNamespace(is_cuda=True, dtype=torch.bfloat16)) == "replacement"
    assert owner().forward(SimpleNamespace(is_cuda=True, dtype=torch.float32)) == "replacement"


def test_an_fp64_input_on_cuda_stays_on_upstream():
    """The kernels compute in fp32, and an fp64 forward is what a numerics reference reaches for."""
    owner = type("Owner", (), {"forward": lambda self, hidden_states: "upstream"})
    hyper_connections._install(owner, lambda self, hidden_states: "replacement")
    assert owner().forward(SimpleNamespace(is_cuda=True, dtype=torch.float64)) == "upstream"


def _eager_projection(stream, fn, norm_eps):
    flat = stream.flatten(-2).float()
    return F.linear(flat * torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps), fn.float())


def _eager_sinkhorn(comb, iters, eps):
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return comb


def _eager_collapse(stream, weight):
    return (weight.unsqueeze(-1) * stream).sum(dim=-2).to(stream.dtype)


def _eager_mix(block, residual, post, comb):
    return post.unsqueeze(-1) * block.unsqueeze(-2) + torch.matmul(comb.transpose(-1, -2), residual)


def _forwards_on_eager_kernels(monkeypatch, model_type: str, mix=_eager_mix) -> None:
    monkeypatch.setattr(hyper_connections, "project_streams", _eager_projection)
    monkeypatch.setattr(hyper_connections, "sinkhorn_knopp", _eager_sinkhorn)
    monkeypatch.setattr(hyper_connections, "collapse_streams", _eager_collapse)
    monkeypatch.setattr(hyper_connections, "mix_streams", mix)
    for owner, fused in _FAMILIES[find_moe_family_spec(model_type)]:
        monkeypatch.setattr(owner, "forward", fused)


def _trace(model_type: str, dtype: torch.dtype, autocast: bool) -> list[torch.Tensor]:
    """The loss, every gradient and hidden state of a training step, then a greedy cached generate."""
    torch.manual_seed(7)
    model = tiny_family_model(TINY_MOE_FAMILIES[model_type]).to(dtype).train()
    vocab = model.config.get_text_config().vocab_size
    ids = torch.randint(0, vocab, (2, 16), generator=torch.Generator().manual_seed(5))
    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
        out = model(input_ids=ids, labels=ids, output_hidden_states=True, use_cache=False)
        out.loss.backward()
        generated = model.eval().generate(
            input_ids=ids, max_new_tokens=_GENERATED_TOKENS, do_sample=False, use_cache=True
        )
    grads = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    return [out.loss.detach(), *grads, *out.hidden_states, generated]


def _identical(got: list[torch.Tensor], want: list[torch.Tensor]) -> bool:
    return len(got) == len(want) and all(torch.equal(a, b) for a, b in zip(got, want, strict=True))


@pytest.mark.parametrize("model_type", ["glm5_next", "deepseek_v4"])
@pytest.mark.parametrize(
    ("dtype", "autocast"), [(torch.float32, False), (torch.bfloat16, False), (torch.float32, True)]
)
def test_each_replacement_is_upstreams_line_for_line(monkeypatch, model_type, dtype, autocast):
    """What a transformers bump can move under an unchanged signature: a new norm, cast or dropout in the body."""
    stock = _trace(model_type, dtype, autocast)
    _forwards_on_eager_kernels(monkeypatch, model_type)
    assert _identical(_trace(model_type, dtype, autocast), stock)


@pytest.mark.parametrize("model_type", ["glm5_next", "deepseek_v4"])
def test_the_line_for_line_check_sees_a_wrong_mix(monkeypatch, model_type):
    def untransposed(block, residual, post, comb):
        return post.unsqueeze(-1) * block.unsqueeze(-2) + torch.matmul(comb, residual)

    stock = _trace(model_type, torch.float32, False)
    _forwards_on_eager_kernels(monkeypatch, model_type, mix=untransposed)
    assert not _identical(_trace(model_type, torch.float32, False), stock)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
