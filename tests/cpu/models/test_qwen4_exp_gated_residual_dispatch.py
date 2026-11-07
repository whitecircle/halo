#!/usr/bin/env python
"""When Flash-Next's gated residual runs on its kernels and what the kernel path computes, with the kernels replaced by
their eager forms: the knob, the truth table of modules and inputs that fall back to upstream's forward, the wiring
against upstream's arithmetic, and the layer routing. The kernels themselves run in
``tests/gpu/kernels/test_gated_residual_kernels.py``."""

import pytest
import torch
import torch.nn.functional as F
import transformers.models.qwen4_exp.modeling_qwen4_exp as qwen4_exp_modeling

from halo.models.families.qwen4_exp import forwards, gated_residual
from halo.models.families.qwen4_exp.gated_residual import (
    inject_block,
    is_gated_residual_fusable,
    patch_qwen4_exp_gated_residual,
    uses_gated_residual_kernels,
)
from tests.common.qwen4_exp import QSA_LAYER_IDX, rotary_embeddings, tiny_qwen4_exp_text_config

KNOB = "HALO_GATED_RESIDUAL_KERNELS"
STREAMS, HIDDEN = 4, 24


@pytest.fixture(autouse=True)
def _installed():
    patch_qwen4_exp_gated_residual()


@pytest.fixture
def module():
    config = tiny_qwen4_exp_text_config(hc_count=STREAMS, hidden_size=HIDDEN, hc_lowrank=8)
    torch.manual_seed(0)
    residual = qwen4_exp_modeling.Qwen4ExpTextGatedResidual(config)
    with torch.no_grad():
        residual.hc_norm.weight.normal_(std=0.3)
    return residual


def _stream(rows: int = 5, dtype=torch.float32) -> torch.Tensor:
    return torch.randn(2, rows, STREAMS * HIDDEN, generator=torch.Generator().manual_seed(1), dtype=dtype)


def test_the_knob_is_off_by_default_and_read_per_call(monkeypatch):
    monkeypatch.delenv(KNOB, raising=False)
    assert not uses_gated_residual_kernels()
    for value, expected in (("1", True), ("0", False), ("true", True), ("off", False)):
        monkeypatch.setenv(KNOB, value)
        assert uses_gated_residual_kernels() is expected


def _hooked(module):
    module.input_mix_weight_up.register_forward_hook(lambda *_: None)


def _prehooked(module):
    module.hc_norm.register_forward_pre_hook(lambda *_: None)


def _biased(module):
    module.block_inject_weight.bias = torch.nn.Parameter(torch.zeros(STREAMS))


def _grouped_off_hidden(module):
    module.hc_norm.group_size = HIDDEN // 2


def _subclassed(module):
    class _Linear(torch.nn.Linear):
        pass

    module.input_mix_weight_down.__class__ = _Linear


@pytest.mark.parametrize(
    ("mutate", "reason"),
    [
        (_hooked, "a forward hook"),
        (_prehooked, "a forward hook"),
        (_biased, "not a bias-free nn.Linear"),
        (_grouped_off_hidden, "hc_norm.group_size"),
        (_subclassed, "not a bias-free nn.Linear"),
        (lambda module: None, "off CUDA"),
    ],
    ids=["hook", "pre-hook", "bias", "group-size", "linear-subclass", "cpu-input"],
)
def test_every_unservable_module_names_its_reason(module, mutate, reason):
    mutate(module)
    assert reason in gated_residual._unfusable_reason(module, _stream())


def test_a_fusable_module_on_the_knob_off_is_not_fused(module, monkeypatch):
    monkeypatch.setattr(gated_residual, "_unfusable_reason", lambda *_: None)
    monkeypatch.setenv(KNOB, "0")
    assert not is_gated_residual_fusable(module, _stream())
    monkeypatch.setenv(KNOB, "1")
    assert is_gated_residual_fusable(module, _stream())


def _reference_norm(x, weight, eps, streams):
    grouped = x.float().view(x.shape[0], streams, -1)
    normed = (
        grouped * torch.rsqrt(grouped.pow(2).mean(-1, keepdim=True) + eps) * (1 + weight.float().view(streams, -1))
    )
    return normed.reshape_as(x).to(x.dtype)


def _reference_mix(normed, down, w_up, streams):
    up = F.linear(F.silu(down / streams), w_up)
    return (torch.sigmoid(up) * normed).view(normed.shape[0], streams, -1).mean(-2)


@pytest.fixture
def eager_kernels(monkeypatch):
    """The kernel entry points as their eager forms, called, with the module deemed servable."""
    calls = []

    def recording(name, fn):
        def run(*args):
            calls.append(name)
            return fn(*args)

        return run

    monkeypatch.setattr(gated_residual, "_unfusable_reason", lambda *_: None)
    monkeypatch.setattr(gated_residual, "grouped_rms_norm", recording("norm", _reference_norm))
    monkeypatch.setattr(gated_residual, "gate_mix", recording("mix", _reference_mix))
    monkeypatch.setenv(KNOB, "1")
    return calls


def test_the_kernel_path_computes_upstreams_arithmetic(module, eager_kernels):
    original = qwen4_exp_modeling.Qwen4ExpTextGatedResidual.forward.__wrapped__
    x = _stream().requires_grad_()
    block = torch.randn(2, 5, HIDDEN)
    outputs = {}
    for name, forward in (("stock", lambda m, s: original(m, s)), ("kernels", lambda m, s: m(s))):
        module.zero_grad()
        s = x.detach().clone().requires_grad_()
        mixed, stream, gates = forward(module, s)
        out = inject_block(stream, mixed + block, gates)
        out.pow(2).sum().backward()
        outputs[name] = (out, s.grad, {n: p.grad.clone() for n, p in module.named_parameters()})
    assert eager_kernels.count("norm") == 1 and eager_kernels.count("mix") == 1
    for kernel_value, stock_value in zip(outputs["kernels"][:2], outputs["stock"][:2], strict=True):
        torch.testing.assert_close(kernel_value, stock_value, rtol=1e-5, atol=1e-6)
    for name, grad in outputs["stock"][2].items():
        torch.testing.assert_close(outputs["kernels"][2][name], grad, rtol=1e-4, atol=1e-6)


def test_a_wrongly_wired_kernel_path_fails_the_comparison(module, eager_kernels, monkeypatch):
    monkeypatch.setattr(
        gated_residual, "gate_mix", lambda normed, down, w_up, streams: normed.view(-1, streams, HIDDEN).sum(-2)
    )
    original = qwen4_exp_modeling.Qwen4ExpTextGatedResidual.forward.__wrapped__
    x = _stream()
    with pytest.raises(AssertionError):
        torch.testing.assert_close(module(x)[0], original(module, x)[0], rtol=1e-5, atol=1e-6)


def test_the_combine_off_the_knob_is_upstreams_expression(monkeypatch):
    monkeypatch.delenv(KNOB, raising=False)
    hyper_input = _stream()
    out, gates = torch.randn(2, 5, HIDDEN), torch.rand(2, 5, STREAMS)
    expected = hyper_input + (out.unsqueeze(-2) * gates.unsqueeze(-1)).flatten(-2)
    assert torch.equal(inject_block(hyper_input, out, gates), expected)


def test_a_use_combine_free_residual_returns_the_mix_only(eager_kernels):
    config = tiny_qwen4_exp_text_config(hc_count=STREAMS, hidden_size=HIDDEN, hc_lowrank=8)
    head = qwen4_exp_modeling.Qwen4ExpTextGatedResidual(config, use_combine=False)
    out = head(_stream())
    assert out.shape == (2, 5, HIDDEN)


def test_install_is_idempotent_and_leaves_one_wrapper():
    forward = qwen4_exp_modeling.Qwen4ExpTextGatedResidual.forward
    patch_qwen4_exp_gated_residual()
    assert qwen4_exp_modeling.Qwen4ExpTextGatedResidual.forward is forward
    assert not hasattr(forward.__wrapped__, "__wrapped__")


class _MovedForward(qwen4_exp_modeling.Qwen4ExpTextGatedResidual):
    def forward(self, hyper_input, streams):
        """The hyper-connection of a release that takes another input."""


def test_a_moved_upstream_signature_fails_install(monkeypatch):
    monkeypatch.setattr(qwen4_exp_modeling, "Qwen4ExpTextGatedResidual", _MovedForward)
    with pytest.raises(RuntimeError, match="_MovedForward.forward takes"):
        patch_qwen4_exp_gated_residual()


@pytest.mark.parametrize(("knob_value", "routed"), [("0", False), ("1", True)])
def test_a_layer_without_ple_takes_the_document_body_only_with_the_kernels(monkeypatch, knob_value, routed):
    monkeypatch.setenv(KNOB, knob_value)
    forwards.patch_qwen4_exp_document_forwards()
    config = tiny_qwen4_exp_text_config()
    layer = qwen4_exp_modeling.Qwen4ExpTextDecoderLayer(config, QSA_LAYER_IDX)
    assert layer.ple is None
    called = []
    monkeypatch.setattr(
        forwards, "_document_decoder_forward", lambda layer, hidden, *_, **__: called.append(1) or hidden
    )
    hidden = torch.randn(1, 6, config.hc_count * config.hidden_size, generator=torch.Generator().manual_seed(0))
    positions = torch.arange(6).unsqueeze(0)
    layer(hidden, rotary_embeddings(config, hidden, positions), attention_mask=None)
    assert bool(called) is routed


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
