#!/usr/bin/env python
"""The flex-sliding attention (sliding layers on FlexAttention, mem-efficient-only global layers on matmul
attention within a memory budget) reproduces the SDPA model exactly in what it attends to.

Tiny random models run packed rows (three documents, per-document ``position_ids``) longer than their
sliding window, once under ``sdpa`` and once under the flex-sliding implementation: Gemma 4 (alternating
sliding and global layers, the process pinned to mem-efficient SDPA) and Mistral (a window on every
layer). The loss and every parameter gradient must agree: a block mask that leaked across a document
boundary, dropped the window, or went stale between two different masks changes them.
"""

import pytest
import torch
from accelerate import PartialState
from torch.nn.attention.flex_attention import flex_attention
from transformers import (
    Gemma4ForCausalLM,
    Gemma4TextConfig,
    GptOssConfig,
    MistralConfig,
    MistralForCausalLM,
    Qwen3MoeConfig,
)

from src.models.patches import flex_sliding_attention
from src.models.patches.attention import patch_sdpa_for_gemma4_long_seq
from src.models.patches.flex_sliding_attention import (
    FLEX_SLIDING,
    register_flex_sliding_attention,
    resolve_flex_sliding_attn_implementation,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

PartialState()  # the toolkit logs through accelerate's logger

WINDOW = 32


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a.double() - b.double()).norm() / b.double().norm().clamp(min=1e-30)).item()


def _config(attn: str) -> Gemma4TextConfig:
    config = Gemma4TextConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        global_head_dim=128,
        num_global_key_value_heads=1,
        sliding_window=WINDOW,
        layer_types=["sliding_attention", "sliding_attention", "full_attention", "full_attention"],
        enable_moe_block=False,
        hidden_size_per_layer_input=0,
        num_kv_shared_layers=0,
    )
    config._attn_implementation = attn
    return config


def _causal_module() -> torch.nn.Module:
    module = torch.nn.Module()
    module.is_causal = True
    return module


def _packed_batch(lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(sum(lengths))
    ids = torch.randint(3, 256, (1, sum(lengths)), generator=generator)
    position_ids = torch.cat([torch.arange(n) for n in lengths]).unsqueeze(0)
    return ids.cuda(), position_ids.cuda()


def _loss_and_grads(model, ids, position_ids):
    model.zero_grad(set_to_none=True)
    out = model(input_ids=ids, position_ids=position_ids, labels=ids)
    out.loss.backward()
    return out.loss.detach(), {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.grad is not None}


@pytest.mark.parametrize("lengths", [[100, 7, 45], [150]])
def test_flex_sliding_wiring_matches_sdpa(lengths, monkeypatch):
    """Exact check of what the sliding layers attend to: with the uncompiled (fp32-exact) FlexAttention
    the model must reproduce the SDPA model's loss and every gradient to fp32 round-off."""
    monkeypatch.setattr(flex_sliding_attention, "_compiled_flex", flex_attention)
    patch_sdpa_for_gemma4_long_seq()
    name = register_flex_sliding_attention()
    torch.manual_seed(0)
    reference = Gemma4ForCausalLM(_config("sdpa")).cuda().float()
    candidate = Gemma4ForCausalLM(_config(name)).cuda().float()
    candidate.load_state_dict(reference.state_dict())
    ids, position_ids = _packed_batch(lengths)

    ref_loss, ref_grads = _loss_and_grads(reference, ids, position_ids)
    loss, grads = _loss_and_grads(candidate, ids, position_ids)

    torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-6)
    assert grads.keys() == ref_grads.keys()
    for key in ref_grads:
        assert _rel(grads[key], ref_grads[key]) < 1e-3, key


def test_compiled_flex_kernel_matches_fp64_reference():
    """The compiled kernel the layers run, against an fp64 masked softmax over a packed sliding mask.
    Its fp32 dots run in TF32, so the bound is TF32's, well below any masking error (which is O(1))."""
    torch.manual_seed(0)
    lengths, heads, kv_heads, dim = [100, 7, 45], 4, 2, 64
    seq = sum(lengths)
    doc = torch.cat([torch.full((n,), i) for i, n in enumerate(lengths)]).cuda()
    pos = torch.arange(seq, device="cuda")
    dense = (doc[:, None] == doc[None]) & (pos[None] <= pos[:, None]) & (pos[:, None] - pos[None] < WINDOW)
    q = torch.randn(1, heads, seq, dim, device="cuda", requires_grad=True)
    k = torch.randn(1, kv_heads, seq, dim, device="cuda", requires_grad=True)
    v = torch.randn(1, kv_heads, seq, dim, device="cuda", requires_grad=True)
    grad = torch.randn(1, heads, seq, dim, device="cuda")
    block_mask = flex_sliding_attention.sliding_block_mask(dense[None, None], 1, seq, seq, WINDOW, dense.device)
    out = flex_sliding_attention._compiled_flex(q, k, v, block_mask=block_mask, scale=1.0, enable_gqa=True)
    out.backward(grad)

    q64, k64, v64 = (t.detach().double().requires_grad_(True) for t in (q, k, v))
    scores = q64 @ k64.repeat_interleave(heads // kv_heads, 1).transpose(-1, -2)
    expected = torch.softmax(scores.masked_fill(~dense, float("-inf")), -1) @ v64.repeat_interleave(
        heads // kv_heads, 1
    )
    expected.backward(grad.double())
    assert _rel(out, expected) < 2e-3
    for got, want in ((q.grad, q64.grad), (k.grad, k64.grad), (v.grad, v64.grad)):
        assert _rel(got, want) < 5e-3


def test_tuned_tiles_match_sdpa_at_gemma4_shapes():
    """The production sliding shape (16 query / 8 KV heads, head_dim 256, window 1,024, bf16) at a length
    that selects the tuned tiles on SM100+: output and input gradients must match mem-efficient SDPA on
    the dense mask to bf16 accuracy."""
    torch.manual_seed(0)
    seq, window = 2048, 1024
    q = torch.randn(1, 16, seq, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, 8, seq, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, 8, seq, 256, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn_like(q)
    pos = torch.arange(seq, device="cuda")
    dense = (pos[None] <= pos[:, None]) & (pos[:, None] - pos[None] < window)
    out, _ = flex_sliding_attention.flex_sliding_attention(
        _causal_module(), q, k, v, dense[None, None], scaling=1.0, sliding_window=window
    )
    out.backward(grad.transpose(1, 2))
    q2, k2, v2 = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    expected = torch.nn.functional.scaled_dot_product_attention(
        q2, k2.repeat_interleave(2, 1), v2.repeat_interleave(2, 1), attn_mask=dense, scale=1.0
    )
    expected.backward(grad)
    assert _rel(out, expected.transpose(1, 2)) < 2e-2
    for got, want in ((q.grad, q2.grad), (k.grad, k2.grad), (v.grad, v2.grad)):
        assert _rel(got, want) < 2e-2


def test_bf16_model_error_is_within_the_sdpa_noise_floor():
    """The production dtype end to end, compiled kernel included. bf16 gradients of two correct
    implementations already differ, so the bound is relative: against an fp32 SDPA reference, the flex
    model's bf16 error must stay within twice SDPA's own bf16 error, parameter by parameter."""
    patch_sdpa_for_gemma4_long_seq()
    name = register_flex_sliding_attention()
    torch.manual_seed(0)
    reference = Gemma4ForCausalLM(_config("sdpa")).cuda().float()
    ids, position_ids = _packed_batch([100, 7, 45])
    ref_loss, ref_grads = _loss_and_grads(reference, ids, position_ids)
    errors = {}
    for attn in ("sdpa", name):
        model = Gemma4ForCausalLM(_config(attn)).cuda()
        model.load_state_dict(reference.state_dict())
        loss, grads = _loss_and_grads(model.to(torch.bfloat16), ids, position_ids)
        errors[attn] = (abs(loss.item() - ref_loss.item()), {k: _rel(grads[k], ref_grads[k]) for k in ref_grads})
    sdpa_loss_err, sdpa_errs = errors["sdpa"]
    flex_loss_err, flex_errs = errors[name]
    assert flex_loss_err <= 2 * sdpa_loss_err + 1e-3
    for key, sdpa_err in sdpa_errs.items():
        assert flex_errs[key] <= 2 * sdpa_err + 1e-3, (key, flex_errs[key], sdpa_err)


def test_global_layers_fall_back_to_sdpa_past_the_budget(monkeypatch):
    """Past the score-memory budget the global layers must take SDPA, and the result must not change."""
    monkeypatch.setattr(flex_sliding_attention, "_compiled_flex", flex_attention)
    patch_sdpa_for_gemma4_long_seq()
    name = register_flex_sliding_attention()
    torch.manual_seed(0)
    reference = Gemma4ForCausalLM(_config("sdpa")).cuda().float()
    candidate = Gemma4ForCausalLM(_config(name)).cuda().float()
    candidate.load_state_dict(reference.state_dict())
    ids, position_ids = _packed_batch([100, 7, 45])
    calls = []
    real = flex_sliding_attention._matmul_global_attention
    monkeypatch.setattr(flex_sliding_attention, "_matmul_global_attention", lambda *a: calls.append(1) or real(*a))
    monkeypatch.setattr(flex_sliding_attention, "EAGER_GLOBAL_BUDGET_BYTES", 0)
    loss, _ = _loss_and_grads(candidate, ids, position_ids)
    assert calls == []
    torch.testing.assert_close(loss, _loss_and_grads(reference, ids, position_ids)[0], rtol=1e-5, atol=1e-6)
    monkeypatch.setattr(flex_sliding_attention, "EAGER_GLOBAL_BUDGET_BYTES", 2 * 2**30)
    _loss_and_grads(candidate, ids, position_ids)
    assert len(calls) == 2  # the two global layers of the tiny model


def test_bidirectional_modules_keep_sdpa():
    """The vision/audio towers attend bidirectionally; only causal layers may take the causal matmul path."""
    q = torch.randn(1, 4, 16, 32, device="cuda")
    k = torch.randn(1, 4, 16, 32, device="cuda")
    tower = torch.nn.Module()
    tower.is_causal = False
    out, _ = flex_sliding_attention.flex_sliding_attention(tower, q, k, k, None, scaling=1.0)
    expected = torch.nn.functional.scaled_dot_product_attention(q, k, k, scale=1.0).transpose(1, 2)
    torch.testing.assert_close(out, expected)


def test_block_mask_is_rebuilt_for_a_new_mask():
    """Two different packings in a row must not share a cached block mask."""
    register_flex_sliding_attention()
    first = torch.tril(torch.ones(1, 1, 64, 64, dtype=torch.bool, device="cuda"))
    second = first.clone()
    second[..., 40:, :40] = False  # a document boundary at 40
    a = flex_sliding_attention.sliding_block_mask(first, 1, 64, 64, WINDOW, first.device)
    del first
    b = flex_sliding_attention.sliding_block_mask(second, 1, 64, 64, WINDOW, second.device)
    assert a is not b
    assert flex_sliding_attention.sliding_block_mask(second, 1, 64, 64, WINDOW, second.device) is b


def _mistral_config(attn: str) -> MistralConfig:
    config = MistralConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
        sliding_window=WINDOW,
    )
    config._attn_implementation = attn
    return config


@pytest.mark.parametrize("lengths", [[100, 7, 45], [150]])
def test_a_non_gemma_sliding_model_matches_sdpa(lengths, monkeypatch):
    """Nothing in the path is Gemma's: a Mistral model (one window on every layer, flash SDPA left enabled)
    reproduces the SDPA model's loss and every gradient with uncompiled FlexAttention."""
    monkeypatch.setattr(flex_sliding_attention, "_compiled_flex", flex_attention)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
    name = register_flex_sliding_attention()
    torch.manual_seed(0)
    reference = MistralForCausalLM(_mistral_config("sdpa")).cuda().float()
    candidate = MistralForCausalLM(_mistral_config(name)).cuda().float()
    candidate.load_state_dict(reference.state_dict())
    ids, position_ids = _packed_batch(lengths)
    calls = []
    real = flex_sliding_attention._compiled_flex
    monkeypatch.setattr(flex_sliding_attention, "_compiled_flex", lambda *a, **k: calls.append(1) or real(*a, **k))

    ref_loss, ref_grads = _loss_and_grads(reference, ids, position_ids)
    loss, grads = _loss_and_grads(candidate, ids, position_ids)

    assert len(calls) == 2  # both layers slide
    torch.testing.assert_close(loss, ref_loss, rtol=1e-5, atol=1e-6)
    for key in ref_grads:
        assert _rel(grads[key], ref_grads[key]) < 1e-3, key


def test_a_call_carrying_attention_sinks_keeps_sdpa(monkeypatch):
    """FlexAttention is not given sinks here, so a sink-carrying call (GptOss's ``s_aux``) must reach the
    registered SDPA implementation with the sinks intact."""
    seen = {}

    def spy(module, query, key, value, attention_mask, **kwargs):
        seen.update(kwargs)
        return query.transpose(1, 2), None

    monkeypatch.setitem(flex_sliding_attention.ALL_ATTENTION_FUNCTIONS._global_mapping, "sdpa", spy)
    q = torch.randn(1, 4, 64, 32, device="cuda")
    sinks = torch.zeros(4, device="cuda")
    flex_sliding_attention.flex_sliding_attention(_causal_module(), q, q, q, None, sliding_window=WINDOW, s_aux=sinks)
    assert seen.get("s_aux") is sinks


def test_the_resolver_picks_models_by_what_they_attend_with():
    """Sliding layers or wide heads select the flex-sliding implementation on an SDPA run; a model with
    neither, a sinks model, a non-SDPA run, and the opt-out keep what the run resolved."""
    gemma4 = _config("sdpa")
    assert resolve_flex_sliding_attn_implementation(gemma4, "sdpa") == FLEX_SLIDING
    assert resolve_flex_sliding_attn_implementation(_mistral_config("sdpa"), "sdpa") == FLEX_SLIDING
    assert resolve_flex_sliding_attn_implementation(GptOssConfig(), "sdpa") == "sdpa"  # sinks: SDPA on every call
    dense_window_off = Qwen3MoeConfig(use_sliding_window=False, sliding_window=4096)
    assert resolve_flex_sliding_attn_implementation(dense_window_off, "sdpa") == "sdpa"
    assert resolve_flex_sliding_attn_implementation(gemma4, "flash_attention_4") == "flash_attention_4"


def test_the_opt_out_keeps_sdpa(monkeypatch):
    monkeypatch.setenv("HALO_FLEX_SLIDING", "0")
    assert resolve_flex_sliding_attn_implementation(_mistral_config("sdpa"), "sdpa") == "sdpa"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
