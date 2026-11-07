#!/usr/bin/env python
"""Load-time kernel warm-ups of Flash-Next: the family patches register them, ``finalize_loaded_model`` runs them over
the loaded model, and each warms the layouts read off that model's modules, in its dtypes, once per layout, skipping
the knob off, a non-CUDA layer and a model without the layers. The kernels themselves run in ``tests/gpu/kernels``."""

import pytest
import torch
import transformers.models.qwen4_exp.modeling_qwen4_exp as qwen4_exp_modeling
from triton.runtime.errors import OutOfResources

from halo.kernels.block_sparse_attention import SUPPORTED_DTYPES
from halo.model_loading import finalization
from halo.model_loading.finalization import finalize_loaded_model
from halo.models import kernel_warmups
from halo.models.families.qwen4_exp import gated_residual, sparse_attention
from halo.models.families.qwen4_exp.forwards import apply_qwen4_exp_patches
from tests.common.qwen4_exp import QSA_LAYER_IDX, tiny_qwen4_exp_text_config
from tests.common.qwen4_exp_qsa import layout_config

CUDA = torch.device("cuda", 0)
HEADS, KV_HEADS, HEAD_DIM, RATIO = 6, 2, 32, 2
STREAMS, HIDDEN, RANK = 3, 24, 8


@pytest.fixture(autouse=True)
def _registered_warmups(monkeypatch):
    monkeypatch.setattr(kernel_warmups, "_warmups", [])
    apply_qwen4_exp_patches(tiny_qwen4_exp_text_config())


@pytest.fixture
def attention_model():
    config = layout_config(HEADS, KV_HEADS, HEAD_DIM, indexer_compress_ratio=RATIO, indexer_budget=16)
    return torch.nn.ModuleList([qwen4_exp_modeling.Qwen4ExpTextAttention(config, QSA_LAYER_IDX)])


@pytest.fixture
def residual_model():
    config = tiny_qwen4_exp_text_config(hc_count=STREAMS, hidden_size=HIDDEN, hc_lowrank=RANK)
    return torch.nn.ModuleList([qwen4_exp_modeling.Qwen4ExpTextGatedResidual(config)])


def _on_cuda(monkeypatch, module, layouts_name):
    """The module's own layouts, moved to a CUDA device the host does not have; the warm-ups run on it, their kernels
    replaced."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(finalization, "apply_fla_launch_options", lambda: None)
    real = getattr(module, layouts_name)
    monkeypatch.setattr(
        module, layouts_name, lambda model: {(CUDA, *layout[1:]) for layout in real(model)}
    )  # fmt: skip


@pytest.fixture
def qsa_warmed(monkeypatch):
    calls = []
    _on_cuda(monkeypatch, sparse_attention, "qsa_layouts")
    monkeypatch.setattr(sparse_attention, "resolve_qsa_kernel", lambda *arguments: True)
    monkeypatch.setattr(sparse_attention, "warm_block_sparse_kernels", lambda *arguments: calls.append(arguments))
    return calls


@pytest.fixture
def residual_warmed(monkeypatch):
    calls = []
    _on_cuda(monkeypatch, gated_residual, "gated_residual_layouts")
    monkeypatch.setattr(gated_residual, "warm_gated_residual_kernels", lambda *arguments: calls.append(arguments))
    return calls


def test_the_family_patches_register_both_warmups_once():
    apply_qwen4_exp_patches(tiny_qwen4_exp_text_config())
    assert kernel_warmups._warmups == [sparse_attention.warm_qsa_kernels, gated_residual.warm_gated_residual]


def test_finalizing_runs_every_registered_warmup_over_the_model(monkeypatch):
    seen = []
    monkeypatch.setattr(kernel_warmups, "_warmups", [seen.append])
    model = torch.nn.Linear(2, 2)
    finalize_loaded_model(model)
    assert seen == [model]


def test_the_qsa_layout_is_read_off_the_modules_not_the_release(attention_model):
    assert sparse_attention.qsa_layouts(attention_model) == {(torch.device("cpu"), HEADS, KV_HEADS, HEAD_DIM, RATIO)}


def test_finalizing_warms_the_qsa_kernels_of_the_models_layout_in_both_dtypes(
    qsa_warmed, attention_model, monkeypatch
):
    monkeypatch.setenv("HALO_QSA_KERNEL", "on")
    finalize_loaded_model(attention_model)
    assert qsa_warmed == [(CUDA, dtype, HEADS, KV_HEADS, HEAD_DIM, RATIO) for dtype in SUPPORTED_DTYPES]


def test_the_qsa_knob_off_warms_nothing(qsa_warmed, attention_model, monkeypatch):
    monkeypatch.setenv("HALO_QSA_KERNEL", "off")
    finalize_loaded_model(attention_model)
    assert qsa_warmed == []


def test_a_model_without_qsa_layers_warms_no_qsa_kernel(qsa_warmed, monkeypatch):
    monkeypatch.setenv("HALO_QSA_KERNEL", "on")
    finalize_loaded_model(torch.nn.Linear(2, 2))
    assert qsa_warmed == []


def test_a_qsa_layer_off_cuda_warms_nothing(attention_model, monkeypatch):
    calls = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(finalization, "apply_fla_launch_options", lambda: None)
    monkeypatch.setenv("HALO_QSA_KERNEL", "on")
    monkeypatch.setattr(sparse_attention, "warm_block_sparse_kernels", lambda *arguments: calls.append(arguments))
    finalize_loaded_model(attention_model)
    assert calls == []


def test_a_kernel_that_fails_to_build_runs_tiled_under_auto_and_raises_under_on(
    attention_model, qsa_warmed, monkeypatch
):
    def out_of_resources(*arguments):
        raise OutOfResources(300_000, 232_448, "shared memory")

    monkeypatch.setattr(sparse_attention, "warm_block_sparse_kernels", out_of_resources)
    monkeypatch.setattr(sparse_attention, "_failed_warmups", {})
    monkeypatch.setenv("HALO_QSA_KERNEL", "on")
    with pytest.raises(RuntimeError, match="failed to build.*HALO_QSA_KERNEL=auto"):
        finalize_loaded_model(attention_model)
    monkeypatch.setenv("HALO_QSA_KERNEL", "auto")
    finalize_loaded_model(attention_model)
    reason = sparse_attention.qsa_kernel_refusal(torch.bfloat16, CUDA, HEAD_DIM, HEADS, KV_HEADS, RATIO)
    assert reason is not None and "failed to build" in reason and "shared memory" in reason


def test_finalizing_warms_the_gated_residual_at_the_modules_own_shapes(residual_warmed, residual_model, monkeypatch):
    monkeypatch.setenv("HALO_GATED_RESIDUAL_KERNELS", "1")
    finalize_loaded_model(residual_model)
    eps = residual_model[0].hc_norm.eps
    assert residual_warmed == [(CUDA, torch.float32, torch.float32, STREAMS, HIDDEN, eps, RANK)]


def test_the_gated_residual_knob_off_or_no_residual_warms_nothing(residual_warmed, residual_model, monkeypatch):
    monkeypatch.setenv("HALO_GATED_RESIDUAL_KERNELS", "0")
    finalize_loaded_model(residual_model)
    monkeypatch.setenv("HALO_GATED_RESIDUAL_KERNELS", "1")
    finalize_loaded_model(torch.nn.Linear(2, 2))
    assert residual_warmed == []


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
